"""Routing, DSA selection and KDA gate analyses on a patch-0710 dump (``routedump.load``), numpy only.

    python3 routing.py <dump dir> [prompts.json] [--expert-bytes N | --preset exl3-4bpw-rank|nvfp4-rank] [--out x.md]

Sections (markdown on stdout or --out):
  A. cross-context routing agreement on the shared chunk (prompts.json: {"S", "prompts", "meta": {name: s_start}}):
     per MoE layer, mean Jaccard of the top-8 sets, weighted overlap (sum of min of the normalized weights over the
     common experts) and top-1 expert agreement;
  B. DSA selection stability: consecutive recorded rows of a request (Jaccard, share of a row's pools in the previous
     row's set) and, with prompts.json, across contexts on the S-relative pools of the same S row;
  C. expert temporal locality: overlap of a row's set with the next row's, and coverage by the union of the last w rows;
  D. predictability: layer L -> L+1 (an empirical co-occurrence predictor), token-keyed prior, previous row's set,
     layer popularity, fitted on the rows outside every context's S and scored on A+S's S rows when the probe's
     prompts match (else even / odd requests, or a single request's halves; the report prints how many held-out rows
     repeat an 8-token context seen in training); recall@k
     and the prefetch headroom in bytes per row and layer (bytes correctly in flight before the router, bytes wasted);
  E. KDA gates: per layer, the decay horizon 1 / (1 - exp(mean log gate)) in tokens (median and p10/p90 over heads),
     and the cross-context deviation of the per-head log gates on S rows.

Every number this prints is computed from the dump given; nothing is a claim about decode speed. Prefill rows are a
proxy for decode tokens (the same model and router on the same kind of text) [H].
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import routedump

EXPERTS = 288
# routed expert bytes a rank reads per layer (gate, up, down of 4096 x 2048), TP=2 splitting the width [C]:
PRESETS = {
    "exl3-4bpw-rank": 3 * 4096 * 2048 * 4 // 8 // 2,            # 6,291,456 B (EXL3 4.0 bpw trellis, no scale overhead)
    "nvfp4-rank": 3 * 4096 * 2048 * (4 * 16 + 8) // 16 // 8 // 2,  # 7,077,888 B (e2m1 + one e4m3 scale per 16)
}
PAIRS = [("A+S", "B+S"), ("S", "A+S"), ("L+S", "A+S")]


# -- set helpers ------------------------------------------------------------------------------------------------------
def onehot(sets: np.ndarray, n: int = EXPERTS) -> np.ndarray:
    """[..., k] ids (-1: none) -> [..., n] bool membership."""

    out = np.zeros(sets.shape[:-1] + (n,), bool)
    idx = np.where(sets >= 0, sets, n)
    flat = out.reshape(-1, n)
    pad = np.zeros((flat.shape[0], n + 1), bool)
    np.put_along_axis(pad, idx.reshape(-1, sets.shape[-1]).astype(np.int64), True, axis=1)
    flat[:] = pad[:, :n]
    return out


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Membership rows [..., n] -> Jaccard [...] (1 where both are empty)."""

    inter = (a & b).sum(-1)
    union = (a | b).sum(-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def weighted(ea, wa, eb, wb) -> np.ndarray:
    """Sum over common experts of min(normalized weight): [R, L, 8] x2 -> [R, L]."""

    def dense(e, w):
        w = w / np.maximum(w.sum(-1, keepdims=True), 1e-30)
        out = np.zeros(e.shape[:-1] + (EXPERTS + 1,), np.float64)
        np.put_along_axis(out, np.where(e >= 0, e, EXPERTS).astype(np.int64), w, axis=-1)
        return out[..., :EXPERTS]

    return np.minimum(dense(ea, wa), dense(eb, wb)).sum(-1)


def top1(e, w) -> np.ndarray:
    return np.take_along_axis(e, w.argmax(-1)[..., None], -1)[..., 0]


# -- prompts ----------------------------------------------------------------------------------------------------------
def name_requests(d: dict, P: dict | None) -> dict[str, tuple[int, int]]:
    """prompt name -> (first row of S in the dump, S rows), matching each request's tokens to prompts.json."""

    if P is None:
        return {}
    nS = len(P["S"])
    got = {}
    for first, n in d["requests"]:
        toks = d["token"][first:first + n]
        for name, m in P["meta"].items():
            if m["len"] == n and np.array_equal(toks, np.asarray(P["prompts"][name], np.int32)):
                got[name] = (first + m["s_start"], nS)
    return got


def pairs(names: dict, P: dict | None) -> list[tuple[str, str]]:
    want = P.get("pairs", PAIRS) if P else []
    return [(x, y) for x, y in want if x in names and y in names]


# -- A. cross-context routing ------------------------------------------------------------------------------------------
def routing_agreement(d: dict, names: dict, P: dict | None) -> dict:
    out = {}
    E, W = d["experts"], d["weights"]
    for x, y in pairs(names, P):
        (ax, n), (ay, _) = names[x], names[y]
        ex, wx, ey, wy = E[ax:ax + n], W[ax:ax + n], E[ay:ay + n], W[ay:ay + n]
        out[f"{x} vs {y}"] = {
            "jaccard": jaccard(onehot(ex), onehot(ey)).mean(0),
            "weighted": weighted(ex, wx, ey, wy).mean(0),
            "top1": (top1(ex, wx) == top1(ey, wy)).mean(0),
            "jaccard_head64": jaccard(onehot(ex[:64]), onehot(ey[:64])).mean(0),
        }
    return out


# -- B. DSA selections ----------------------------------------------------------------------------------------------
def selection_stability(d: dict, names: dict, P: dict | None) -> dict:
    out = {"consecutive": {}, "cross": {}}
    req = d["request"]
    for layer, (sel, rows) in sorted(d["dsa_sel"].items()):
        order = np.argsort(rows, kind="stable")
        sel, rows = sel[order], rows[order]
        same = req[rows[1:]] == req[rows[:-1]]
        if same.any():
            a, b = sel[:-1][same], sel[1:][same]
            js, kept = [], []
            for p, q in zip(a, b):
                p, q = p[p >= 0], q[q >= 0]
                i = np.intersect1d(p, q, assume_unique=True).size
                js.append(i / max(1, np.union1d(p, q).size))
                kept.append(i / max(1, q.size))
            out["consecutive"][layer] = {"jaccard": float(np.mean(js)), "in_previous": float(np.mean(kept)),
                                         "pairs": int(same.sum()), "gap": float(np.median(rows[1:][same] - rows[:-1][same]))}
        for x, y in pairs(names, P):
            (ax, n), (ay, _) = names[x], names[y]
            sx, sy = d["pos"][ax], d["pos"][ay]
            if sx % 4 or sy % 4:
                continue                                    # S-relative pools exist only for 4-aligned S starts
            px = {int(r - ax): s for r, s in zip(rows, sel) if ax <= r < ax + n}
            py = {int(r - ay): s for r, s in zip(rows, sel) if ay <= r < ay + n}
            js, ins = [], []
            for k in sorted(set(px) & set(py)):
                a, b = px[k], py[k]
                a = a[a >= 0] - sx // 4
                b = b[b >= 0] - sy // 4
                ins.append(((a >= 0).mean() + (b >= 0).mean()) / 2)
                a, b = a[a >= 0], b[b >= 0]
                js.append(np.intersect1d(a, b).size / max(1, np.union1d(a, b).size))
            if js:
                out["cross"].setdefault(f"{x} vs {y}", {})[layer] = {"jaccard_in_S": float(np.mean(js)),
                                                                     "share_in_S": float(np.mean(ins)), "rows": len(js)}
    return out


# -- C. temporal locality ---------------------------------------------------------------------------------------------
def locality(d: dict, windows=(1, 4, 16, 64)) -> dict:
    E = d["experts"]
    oh = onehot(E)                                           # [R, L, 288]
    out = {}
    for w in windows:
        cov = []
        for first, n in d["requests"]:
            if n <= w:
                continue
            x = oh[first:first + n].astype(np.int32)
            c = np.cumsum(x, 0)
            prev = c[w - 1:n - 1] - np.concatenate([np.zeros((1,) + c.shape[1:], np.int32), c[:n - w - 1]])
            hit = ((prev > 0) & oh[first + w:first + n]).sum(-1) / 8.0
            cov.append(hit)
        if cov:
            out[w] = np.concatenate(cov).mean(0)
    distinct = {}
    for w in (16, 128, 2048):
        u = [onehot(E[f + a:f + a + w].transpose(1, 0, 2).reshape(E.shape[1], -1)).sum(-1)
             for f, n in d["requests"] for a in range(0, n - w + 1, w)]
        if u:
            distinct[w] = np.mean(u, 0)
    return {"coverage": out, "distinct": distinct}


# -- D. predictability and prefetch headroom ---------------------------------------------------------------------
def split(d: dict, names: dict | None = None, holdout: str = "A+S") -> tuple[np.ndarray, np.ndarray]:
    """Train and held-out rows. With the probe's prompts matched (``names``): held out, the S rows of ``holdout``;
    trained on, every row outside every context's S (the shared chunk repeats across requests, so any other split
    would test on text it trained on). Else even / odd requests, or the two halves of a single request."""

    req = d["request"]
    nreq = len(d["requests"])
    valid = (d["experts"] >= 0).all((1, 2))
    if names and holdout in names:
        inS = np.zeros(len(req), bool)
        for first, n in names.values():
            inS[first:first + n] = True
        test = np.zeros(len(req), bool)
        first, n = names[holdout]
        test[first:first + n] = True
        return ~inS & valid, test & valid
    if nreq >= 2:
        train = (req % 2 == 0) & valid
    else:
        train = (np.arange(len(req)) < len(req) // 2) & valid
    return train, ~train & valid


def topk_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """[..., n] -> [..., k] indices of the k largest (ties to the lower id)."""

    return np.argsort(-scores, axis=-1, kind="stable")[..., :k]


def recall(pred: np.ndarray, oh_true: np.ndarray) -> np.ndarray:
    return np.take_along_axis(oh_true, pred, -1).sum(-1) / 8.0


def repeated_context(d: dict, train: np.ndarray, test: np.ndarray, n: int = 8) -> float:
    """Share of held-out rows whose last ``n`` tokens also end some training row: text the predictors saw."""

    t = d["token"].astype(np.int64)
    key = np.zeros(len(t), np.uint64)
    for j in range(n):
        sh = np.concatenate([np.full(j, -1, np.int64), t[:len(t) - j]])
        key = key * np.uint64(1000003) + (sh + 1).astype(np.uint64)
    return float(np.isin(key[test], key[train]).mean()) if test.any() else 0.0


def predictors(d: dict, ks=(8, 16, 32), names: dict | None = None) -> dict:
    E = d["experts"]
    train, test = split(d, names)
    L = E.shape[1]
    out: dict = {}
    if not train.any() or not test.any() or L == 0:
        return out
    oh = onehot(E).astype(np.float32)                         # [R, L, 288]
    pop = oh[train].sum(0)                                    # [L, 288]
    te = oh[test]
    res = {name: {k: [] for k in ks} for name in ("popularity", "layer L-1", "token", "previous row", "random")}
    for k in ks:
        res["popularity"][k] = recall(np.broadcast_to(topk_scores(pop, k), te.shape[:2] + (k,)), te > 0).mean(0)
        res["random"][k] = np.full(L, min(k, EXPERTS) / EXPERTS)
    # layer L-1 -> L: co-occurrence counts C[l][a][b] over the training rows
    lay = np.full((L, len(ks)), np.nan)
    for l in range(1, L):
        C = oh[train, l - 1].T @ oh[train, l]                 # [288, 288]
        C = C / np.maximum(C.sum(1, keepdims=True), 1)
        pred = topk_scores(te[:, l - 1] @ C + 1e-6 * pop[l], max(ks))   # popularity breaks ties among unseen experts
        for j, k in enumerate(ks):
            lay[l, j] = recall(pred[:, :k], te[:, l] > 0).mean()
    for j, k in enumerate(ks):
        res["layer L-1"][k] = lay[:, j]
    # token-keyed prior: counts of (layer, expert) over training rows of the same token; unseen -> popularity
    tok = d["token"]
    ttr, tte = tok[train], tok[test]
    order = np.argsort(ttr, kind="stable")
    uniq, starts = np.unique(ttr[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    trows = oh[train][order]
    pos_of = {int(t): (s, e) for t, s, e in zip(uniq, starts, ends)}
    seen = np.array([int(t) in pos_of for t in tte])
    tk = {k: np.zeros((len(tte), L)) for k in ks}
    kmax = max(ks)
    for i, t in enumerate(tte):
        span = pos_of.get(int(t))
        pred = topk_scores((trows[span[0]:span[1]].sum(0) if span else 0) + 1e-6 * pop, kmax)
        for k in ks:
            tk[k][i] = recall(pred[:, :k], te[i] > 0)
    for k in ks:
        res["token"][k] = tk[k].mean(0)
    out["token_seen_share"] = float(seen.mean())
    # previous row of the same request, same layer (no fitting): its 8 experts, then popularity
    req = d["request"]
    prev_ok = np.zeros(len(req), bool)
    prev_ok[1:] = req[1:] == req[:-1]
    idx = np.nonzero(test & prev_ok)[0]
    if idx.size:
        pred = topk_scores(oh[idx - 1] + 1e-6 * pop, max(ks))
        for k in ks:
            res["previous row"][k] = recall(pred[..., :k], oh[idx] > 0).mean(0)
    out["recall"] = res
    out["rows"] = {"train": int(train.sum()), "test": int(test.sum())}
    out["repeated_8gram"] = repeated_context(d, train, test)
    return out


def headroom(pred: dict, expert_bytes: int) -> dict:
    """Per predictor and k: bytes a row-layer could have in flight correctly before its router runs, and wasted."""

    out = {}
    for name, by_k in pred.get("recall", {}).items():
        for k, r in by_k.items():
            r = np.asarray(r, float)
            if r.size == 0 or np.isnan(r).all():
                continue
            m = float(np.nanmean(r))
            out[(name, k)] = {"hit_bytes": m * 8 * expert_bytes, "wasted_bytes": (k - m * 8) * expert_bytes,
                              "hit_share": m}
    out["all_miss_bytes"] = 8 * expert_bytes
    return out


# -- E. KDA gates -----------------------------------------------------------------------------------------------------
def kda_horizons(d: dict, names: dict, P: dict | None) -> dict:
    g = d["kda_gate"]
    if g is None:
        return {}
    ok = ~np.isnan(g).any((1, 2))
    lg = g[ok].astype(np.float64)                            # [R', Lk, H], natural log of the decay, <= 0
    head = np.median(lg, 0)                                 # [Lk, H]
    hor = 1.0 / np.maximum(1.0 - np.exp(head), 1e-12)
    out = {"layers": d["kda_layers"], "p50": np.median(hor, 1), "p10": np.percentile(hor, 10, 1),
           "p90": np.percentile(hor, 90, 1), "positive": float((lg > 0).mean())}
    cross = {}
    for x, y in pairs(names, P):
        (ax, n), (ay, _) = names[x], names[y]
        a, b = g[ax:ax + n].astype(np.float64), g[ay:ay + n].astype(np.float64)
        cross[f"{x} vs {y}"] = np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-12), axis=(0, 2))
    out["cross"] = cross
    return out


# -- report -----------------------------------------------------------------------------------------------------------
def report(d: dict, P: dict | None, expert_bytes: int) -> str:
    lines = []
    w = lines.append
    names = name_requests(d, P)
    ml = d["moe_layers"]
    w(f"rows {len(d['token'])}, requests {len(d['requests'])}, MoE layers {len(ml)}, DSA layers recorded "
      f"{sorted(d['dsa_sel'])}, KDA layers {len(d['kda_layers'])}; prompts matched: {sorted(names)}")
    if ml and names:
        A = routing_agreement(d, names, P)
        if A:
            w("\n## A. Routing agreement on S across contexts (mean over S rows)\n")
            keys = list(A)
            w("| layer | " + " | ".join(f"{k} jaccard / weighted / top1 / head64 jaccard" for k in keys) + " |")
            w("|---|" + "---|" * len(keys))
            for j, l in enumerate(ml):
                w(f"| {l} | " + " | ".join(f"{A[k]['jaccard'][j]:.3f} / {A[k]['weighted'][j]:.3f} / {A[k]['top1'][j]:.3f} / "
                                         f"{A[k]['jaccard_head64'][j]:.3f}" for k in keys) + " |")
    if d["dsa_sel"]:
        B = selection_stability(d, names, P)
        w("\n## B. DSA selection stability\n")
        w("| layer | consecutive jaccard | share in previous | pairs | median gap (rows) |")
        w("|---|---|---|---|---|")
        for l, v in B["consecutive"].items():
            w(f"| {l} | {v['jaccard']:.3f} | {v['in_previous']:.3f} | {v['pairs']} | {v['gap']:.0f} |")
        for k, by in B["cross"].items():
            w(f"\n{k}: S-relative pools (jaccard, share of the selection inside S, rows)\n")
            w("| layer | jaccard in S | share in S | rows |")
            w("|---|---|---|---|")
            for l, v in by.items():
                w(f"| {l} | {v['jaccard_in_S']:.3f} | {v['share_in_S']:.3f} | {v['rows']} |")
    if ml:
        C = locality(d)
        w("\n## C. Expert temporal locality (share of a row's 8 experts in the union of the previous w rows)\n")
        ws = list(C["coverage"])
        dw = list(C["distinct"])
        w("| layer | " + " | ".join(f"w={x}" for x in ws) + " | " + " | ".join(f"distinct in {x} rows" for x in dw) + " |")
        w("|---|" + "---|" * (len(ws) + len(dw)))
        for j, l in enumerate(ml):
            w(f"| {l} | " + " | ".join(f"{C['coverage'][x][j]:.3f}" for x in ws) + " | "
              + " | ".join(f"{C['distinct'][x][j]:.1f}" for x in dw) + " |")
        D = predictors(d, names=names)
        if D:
            w(f"\n## D. Predictability (recall of the router's 8 among k predicted; train {D['rows']['train']} rows, "
              f"held out {D['rows']['test']}; held-out tokens seen in training {D['token_seen_share']:.3f}; held-out "
              f"rows whose last 8 tokens end a training row {D['repeated_8gram']:.3f})\n")
            names_ = list(D["recall"])
            ks = list(D["recall"]["popularity"])
            w("| layer | " + " | ".join(f"{n} @{k}" for n in names_ for k in ks) + " |")
            w("|---|" + "---|" * (len(names_) * len(ks)))
            for j, l in enumerate(ml):
                cells = []
                for n in names_:
                    for k in ks:
                        r = np.asarray(D["recall"][n][k])
                        cells.append("" if r.size == 0 or np.isnan(r[j]) else f"{r[j]:.3f}")
                w(f"| {l} | " + " | ".join(cells) + " |")
            H = headroom(D, expert_bytes)
            w(f"\nPrefetch headroom per row and MoE layer at {expert_bytes:,} B an expert (all 8 missed: "
              f"{H['all_miss_bytes'] / 1e6:.1f} MB):\n")
            w("| predictor | k | hit share | correct bytes in flight (MB) | wasted bytes (MB) |")
            w("|---|---|---|---|---|")
            for (n, k), v in ((key, v) for key, v in H.items() if isinstance(key, tuple)):
                w(f"| {n} | {k} | {v['hit_share']:.3f} | {v['hit_bytes'] / 1e6:.1f} | {v['wasted_bytes'] / 1e6:.1f} |")
    K = kda_horizons(d, names, P)
    if K:
        w(f"\n## E. KDA decay horizon in tokens, 1 / (1 - exp(median log gate)) per head (share of positive log "
          f"gates, expected 0: {K['positive']:.4f})\n")
        cr = list(K["cross"])
        w("| layer | p10 | p50 | p90 | " + " | ".join(f"{k} rel dev" for k in cr) + " |")
        w("|---|---|---|---|" + "---|" * len(cr))
        for j, l in enumerate(K["layers"]):
            w(f"| {l} | {K['p10'][j]:.1f} | {K['p50'][j]:.1f} | {K['p90'][j]:.1f} | "
              + " | ".join(f"{K['cross'][k][j]:.3f}" for k in cr) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("prompts", nargs="?")
    ap.add_argument("--expert-bytes", type=int)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="exl3-4bpw-rank")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    d = routedump.load(a.dump)
    P = json.load(open(a.prompts)) if a.prompts else None
    text = report(d, P, a.expert_bytes or PRESETS[a.preset])
    if a.out:
        open(a.out, "w").write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
