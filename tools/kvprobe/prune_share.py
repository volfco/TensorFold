"""TF_GLM_INDEX_PRUNE's skip share on real keys, offline (numpy only): which whole KV pages a page bound excludes from a
DSA index scan, next to the share no bound could beat, for the current device bound and tighter ones.

    python3 prune_share.py KEEP_STEM QUERIES.npz [QUERIES.npz ...] [--rows 64] [--seeds 8] [--rank 8] [--json OUT]
    python3 prune_share.py --synthetic        (a self-check on a synthetic keep: every bound sound, the order kept)

KEEP_STEM: a TF_GLM_KV_DUMP keep (kvdump.py); its DSA caches' pooled index keys (`Keep.pooled(j)`, pool p covering
positions 4p..4p+3, 128 pools a page) are the keys. QUERIES.npz: sampled indexer rows of the same prompt, arrays
    q      [R, H, 128]  the index queries (cuda_dsa_block's `qi` rows): bf16 codes (uint16) or floats
    w      [R, H]       the head weights (the index projection's last H columns, `ikr[row][128:]`): bf16 codes or floats
    pos    [R]          each row's position (the row sees pools 0 .. (pos + 1) // 4 - 1)
    cache  [R]          the row's DSA cache index j in the keep (its layer's; Keep.dsa_caches)
and optionally scalars `scale` (default 128 ** -0.5) and `wscale` (default 1 / sqrt(H)), as cuda_sparse passes them.
A row's pool score is sum_h wscale w_h relu(scale q_h . k) (glm_index_scores.cuh); its selection the 512 best pools.

Per DSA cache and position bin it prints the median over rows of the share of whole visible pages excluded:
    oracle      the page's best pool scores strictly below the row's 512th best score (no sound bound does better)
    box/8       the device bound (glm_index_prune.cuh page_bound: per-dimension min and max of the page's keys, the
                relu-monotone dot bound per head, margin 2^-8 of the magnitude)
    box/12      the same with a margin of 2^-12
    head        each head's exact maximum (and minimum) of q_h . k over the page's keys, then the same weighting:
                the best any bound can do that treats the 32 heads separately (what the box loses to the box, not
                to the heads)
    pca         the page's mean, its top --rank principal directions with the coefficients' ranges, and a residual
                ball; min(box, pca) per head (both are sound), margin 2^-8
each as two numbers: `device` (the cut as the device takes it: the --seeds pages of largest bound and the partial page
scored, the 512th best of their pools the threshold) / `exact` (the row's own 512th best score as the threshold). A
bound below a page's best score is counted as unsound and fails the run.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvdump

PP, POOL, K, D = 128, 4, 512, 128  # pools a page, tokens a pool, pools a row selects, index key width
BINS = ((0, 16384, "<16k"), (16384, 65536, "16k-64k"), (65536, 131072, "64k-128k"), (131072, 1 << 40, "128k+"))
BOUNDS = ("box/8", "box/12", "head", "pca")


def floats(a):
    a = np.asarray(a)
    return kvdump.bf16(a) if a.dtype == np.uint16 else a.astype(np.float32)


class Pages:
    """One DSA cache's whole pages: keys [pages, 128, 128], box (lo, hi) and the principal-direction bound's parts."""

    def __init__(self, keys, rank):
        n = keys.shape[0] // PP
        self.keys = keys[:n * PP].reshape(n, PP, D).astype(np.float64)
        self.lo, self.hi = self.keys.min(axis=1), self.keys.max(axis=1)
        self.mu = self.keys.mean(axis=1)
        c = self.keys - self.mu[:, None, :]
        _, _, vt = np.linalg.svd(c, full_matrices=False)  # vt [pages, 128, 128]: rows the directions
        self.u = np.transpose(vt[:, :rank, :], (0, 2, 1))  # [pages, 128, rank]
        a = np.matmul(c, self.u)  # coefficients [pages, 128, rank]
        self.alo, self.ahi = a.min(axis=1), a.max(axis=1)
        rest = c - np.matmul(a, np.transpose(self.u, (0, 2, 1)))
        self.rho = np.sqrt((rest ** 2).sum(axis=2)).max(axis=1)  # the residual ball's radius


def weigh(up, down, w, scale):
    """sum_h of w_h relu(scale x_h), the dot's upper end for w_h >= 0 and its lower end otherwise: [pages]."""
    return np.where(w >= 0, w * np.maximum(scale * up, 0), w * np.maximum(scale * down, 0)).sum(axis=1)


def row_bounds(pg, keys, q, w, scale, npool):
    """One row's page bounds (each name in BOUNDS: [whole pages]), its pool scores and its pages' best scores."""
    whole = min(npool // PP, pg.lo.shape[0])
    q = q.astype(np.float64)
    dots = keys[:npool].astype(np.float64) @ q.T  # [npool, H]
    scores = (w * np.maximum(scale * dots, 0)).sum(axis=1)
    best = scores[:whole * PP].reshape(whole, PP).max(axis=1)
    lo, hi = pg.lo[:whole], pg.hi[:whole]
    qp, qn = np.maximum(q, 0), np.minimum(q, 0)
    up, down = hi @ qp.T + lo @ qn.T, lo @ qp.T + hi @ qn.T  # [whole, H]
    mag = (np.maximum(np.abs(lo), np.abs(hi)) @ np.abs(q).T * np.abs(w) * scale).sum(axis=1)
    tiny = np.abs(w).sum() * 1e-33 + 1e-30
    box = weigh(up, down, w, scale)
    d = dots[:whole * PP].reshape(whole, PP, -1)
    head = weigh(d.max(axis=1), d.min(axis=1), w, scale)
    qu = np.einsum("hd,pdr->phr", q, pg.u[:whole])  # [whole, H, rank]
    perp = np.sqrt(np.maximum((q * q).sum(axis=1)[None, :] - (qu * qu).sum(axis=2), 0)) * pg.rho[:whole, None]
    qmu = pg.mu[:whole] @ q.T
    a, b = qu * pg.alo[:whole, None, :], qu * pg.ahi[:whole, None, :]
    pup = qmu + np.maximum(a, b).sum(axis=2) + perp
    pdown = qmu + np.minimum(a, b).sum(axis=2) - perp
    pca = weigh(np.minimum(up, pup), np.maximum(down, pdown), w, scale)
    out = {"box/8": box + mag / 256 + tiny, "box/12": box + mag / 4096 + tiny, "head": head + mag / 256 + tiny,
           "pca": pca + mag / 256 + tiny}
    return out, scores, best


def kth(x, k=K):
    return np.partition(x, x.size - k)[x.size - k] if x.size >= k else -np.inf


def device_cut(bound, scores, whole, seeds):
    """The threshold the device's cut takes: the 512th best pool among the seeds' (largest bounds, ties to the lower
    page) and the visible pools past the whole pages; -inf (nothing skipped) when they hold fewer than 512."""
    order = np.lexsort((np.arange(whole), -bound))[:seeds]
    pools = np.concatenate([scores[g * PP:(g + 1) * PP] for g in order] + [scores[whole * PP:]])
    return kth(pools), order


def measure(keep, queries, rows=64, seeds=8, rank=8, rng=None):
    rng = rng or np.random.default_rng(0)
    caches = {j: None for j in keep.dsa_caches}
    acc, unsound = {}, 0
    for qf in queries:
        z = np.load(qf)
        h = z["q"].shape[1]
        scale = float(z["scale"]) if "scale" in z else D ** -0.5
        wscale = float(z["wscale"]) if "wscale" in z else 1 / np.sqrt(h)
        cache, pos = z["cache"].astype(int), z["pos"].astype(np.int64)
        for j in np.unique(cache):
            if int(j) not in caches:
                raise SystemExit(f"{qf}: cache {j} is not one of the keep's DSA caches {sorted(caches)}")
            at = np.flatnonzero(cache == j)
            if rows and at.size > rows:
                at = np.sort(rng.choice(at, rows, replace=False))
            if caches[int(j)] is None:
                keys = kvdump.bf16(keep.pooled(int(j)))
                caches[int(j)] = (keys, Pages(keys, rank))
            keys, pg = caches[int(j)]
            for r in at:
                npool = (int(pos[r]) + 1) // POOL
                if npool > keys.shape[0]:
                    raise SystemExit(f"{qf}: row at {pos[r]} sees {npool} pools, the keep holds {keys.shape[0]}")
                whole = npool // PP
                if npool <= K or whole <= seeds:
                    continue
                w = floats(z["w"][r]).astype(np.float64) * np.float32(wscale)
                b, scores, best = row_bounds(pg, keys, floats(z["q"][r]), w, scale, npool)
                t = kth(scores)
                row = {"oracle": float(np.mean(best < t))}
                for name in BOUNDS:
                    unsound += int(np.count_nonzero(b[name] < best))
                    cut, seed = device_cut(b[name], scores, whole, seeds)
                    skip = b[name] < cut
                    skip[seed] = False
                    row[name] = (float(np.mean(skip)), float(np.mean(b[name] < t)))
                label = next(l for lo, hi, l in BINS if lo <= pos[r] < hi)
                acc.setdefault((int(j), label), []).append(row)
    return acc, unsound


def table(acc):
    out = {}
    order = [b[2] for b in BINS]
    for (j, label), rs in sorted(acc.items(), key=lambda x: (x[0][0], order.index(x[0][1]))):
        e = {"rows": len(rs), "oracle": float(np.median([r["oracle"] for r in rs]))}
        for name in BOUNDS:
            e[name] = [float(np.median([r[name][i] for r in rs])) for i in (0, 1)]
        out[f"cache {j} {label}"] = e
    return out


def report(t, unsound):
    print("median share of whole visible pages excluded (device cut / exact 512th best)")
    print(f"{'cache, position':<22}{'rows':>6}{'oracle':>8}" + "".join(f"{n:>16}" for n in BOUNDS))
    for name, e in t.items():
        print(f"{name:<22}{e['rows']:>6}{e['oracle']:>8.3f}" + "".join(f"{e[n][0]:>8.3f}/{e[n][1]:<7.3f}" for n in BOUNDS))
    print(f"unsound page bounds: {unsound}" + (" (a bound below its page's best score: a bug)" if unsound else ""))


def synthetic(tmp):
    """A keep of clustered pages and topical query rows: every bound sound, oracle >= head >= box."""
    rng = np.random.default_rng(7)
    pos, h = 160_000, 32
    pools = pos // POOL + 2
    topics = rng.normal(size=(24, D))
    page_topic = rng.integers(0, 24, pools // PP + 1)
    keys = 2 * topics[page_topic[np.arange(pools) // PP]] + 0.5 * rng.normal(size=(pools, D))
    codes = (keys.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    stem = os.path.join(tmp, "glm-rank0-keep1")
    kvdump.write(stem, pos=pos, kv_lora=64, i_dim=D, heads=1, kda_layers=0, latents=[np.zeros((pos, 64), np.uint16)],
                 pooled=[codes])
    n = 24
    rp = rng.integers(40_000, pos - 1, n)
    q = 0.6 * topics[rng.integers(0, 24, n)][:, None, :] + rng.normal(size=(n, h, D))
    w = np.abs(rng.normal(size=(n, h))) * np.where(rng.random((n, h)) < 0.25, -1, 1)
    qf = os.path.join(tmp, "rows.npz")
    np.savez(qf, q=q.astype(np.float32), w=w.astype(np.float32), pos=rp, cache=np.zeros(n, np.int64))
    acc, unsound = measure(kvdump.Keep(stem), [qf], rows=0)
    return table(acc), unsound


def main(argv):
    if len(argv) >= 2 and argv[1] == "--synthetic":
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            t, unsound = synthetic(tmp)
        report(t, unsound)
        return 1 if unsound else 0
    opts = {"--rows": "64", "--seeds": "8", "--rank": "8", "--json": None}
    args, it = [], iter(argv[1:])
    for a in it:
        if a in opts:
            opts[a] = next(it)
        else:
            args.append(a)
    if len(args) < 2:
        print(__doc__)
        return 2
    acc, unsound = measure(kvdump.Keep(args[0]), args[1:], int(opts["--rows"]), int(opts["--seeds"]), int(opts["--rank"]))
    t = table(acc)
    report(t, unsound)
    if opts["--json"]:
        json.dump({"table": t, "unsound": unsound}, open(opts["--json"], "w"), indent=1)
    return 1 if unsound else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
