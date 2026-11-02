"""Compare a TF_GLM_KDA_REPLAY_CHECK pair of dumps: the stored KDA state of a kept GLM entry against the state a replay
of its last W tokens rebuilt (zig/src/families/glm/cuda_kda_replay.zig writes both as head-only keeps).

    python3 replay_check.py STORED_STEM REPLAYED_STEM [--window W] [--horizons ROUTING_MD] [--json OUT]
    python3 replay_check.py --dir DUMP_DIR [--horizons ROUTING_MD] [--json OUT]

A stem is `<dir>/kda-replay-rank<r>-keep<id>-w<W>-stored` (or `-replayed`) without its `.bin/.parts/.plan`; `--dir` takes
every stored/replayed pair in a directory. Per KDA layer and head: relative L2 of the replayed recurrent state against
the stored one (norm(replayed - stored) / norm(stored)), max abs error, bit identity; the same for the short-conv
history (a head's q, k and v channel blocks over the three rows). With `--horizons` (routing.py's report, section E:
`| layer | p10 | p50 | p90 | ...`, the decay horizon h in tokens per head, quantiles over heads) each layer also gets
the direct-term bound (1 - 1/h)^W at its p50 and p90 horizon: the share of the pre-window state that survives W gates of
mean decay 1 - 1/h. The bound covers only the recurrence's own decay; deeper layers also see their inputs change
through the replayed layers below them, so their error can exceed it. Every reader fault raises kvdump.DumpError.
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvdump, kvstate

STEM = re.compile(r"kda-replay-rank(\d+)-keep(\d+)-w(\d+)-(stored|replayed)\.plan$")


def head_stats(x, y):
    """x stored, y replayed: [heads, ...] float32. Per head rel L2, max abs, bit identity."""
    a = x.reshape(x.shape[0], -1).astype(np.float64)
    b = y.reshape(y.shape[0], -1).astype(np.float64)
    d = b - a
    ref = np.linalg.norm(a, axis=1)
    err = np.linalg.norm(d, axis=1)
    rel = np.where(ref > 0, err / np.where(ref > 0, ref, 1), np.where(err == 0, 0.0, np.inf))
    same = np.all(x.reshape(x.shape[0], -1).view(np.uint32) == y.reshape(y.shape[0], -1).view(np.uint32), axis=1)
    return rel, np.abs(d).max(axis=1), same


def conv_heads(c, heads):
    """[3, conv_c] conv history to [heads, 3 rows x 3 blocks x 128]: head h's q, k and v channels."""
    return c.reshape(3, 3, heads, 128).transpose(2, 0, 1, 3).reshape(heads, -1)


def read_horizons(path):
    """routing.py's section E table: {layer: (p10, p50, p90)} in tokens."""
    out, on = {}, False
    with open(path) as f:
        for line in f:
            if line.startswith("## "):
                on = line.startswith("## E.")
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if on and len(cells) >= 4 and cells[0].isdigit():
                out[int(cells[0])] = tuple(float(c) for c in cells[1:4])
    if not out:
        raise kvdump.DumpError(f"{path}: no section E horizon rows")
    return out


def bound(h, w):
    """Share of the state from before a W-token window left after W gates of mean decay 1 - 1/h."""
    return float((1.0 - 1.0 / h) ** w) if h > 1 else 0.0


def q(v):
    v = np.asarray(v, np.float64)
    return {"p50": float(np.quantile(v, 0.5)), "p90": float(np.quantile(v, 0.9)), "max": float(v.max())}


def compare(stored_stem, replayed_stem, window=None, horizons=None):
    s, r = kvdump.Keep(stored_stem), kvdump.Keep(replayed_stem)
    for k in ("rank", "world", "pos"):
        if s.meta[k] != r.meta[k]:
            raise kvdump.DumpError(f"{k} differs: stored {s.meta[k]}, replayed {r.meta[k]}")
    if s.shape != r.shape:
        raise kvdump.DumpError("the two dumps' shapes differ")
    if window is None:
        m = STEM.search(stored_stem + ".plan")
        if not m:
            raise kvdump.DumpError(f"{stored_stem}: no -w<W>- in the name; pass --window")
        window = int(m.group(3))
    heads = s.shape["heads"]
    _, names = kvstate.layer_names(s)
    out = {"stored": stored_stem, "replayed": replayed_stem, "rank": s.rank, "pos": s.pos, "window": window,
           "heads": heads, "layers": []}
    for li, layer in enumerate(names):
        rel, mx, same = head_stats(s.kda_state(li), r.kda_state(li))
        crel, cmx, csame = head_stats(conv_heads(s.conv(li), heads), conv_heads(r.conv(li), heads))
        row = {"kda": li, "layer": layer, "rec_rel": q(rel), "rec_maxabs": float(mx.max()), "rec_same": int(same.sum()),
               "conv_rel": q(crel), "conv_maxabs": float(cmx.max()), "conv_same": int(csame.sum()),
               "heads": [{"head": h, "rec_rel": float(rel[h]), "rec_maxabs": float(mx[h]), "rec_same": bool(same[h]),
                          "conv_rel": float(crel[h]), "conv_maxabs": float(cmx[h]), "conv_same": bool(csame[h])}
                         for h in range(heads)]}
        if horizons is not None and layer in horizons:
            p10, p50, p90 = horizons[layer]
            row["horizon"] = {"p10": p10, "p50": p50, "p90": p90}
            row["bound"] = {"p50": bound(p50, window), "p90": bound(p90, window)}
        out["layers"].append(row)
    allrel = np.array([h["rec_rel"] for l in out["layers"] for h in l["heads"]])
    out["total"] = {"heads": int(allrel.size), "rec_same": sum(l["rec_same"] for l in out["layers"]),
                    "conv_same": sum(l["conv_same"] for l in out["layers"]), "rec_rel": q(allrel)}
    return out


def pairs(directory):
    found = {}
    for name in os.listdir(directory):
        m = STEM.match(name)
        if m:
            found.setdefault(m.groups()[:3], {})[m.group(4)] = os.path.join(directory, name[:-len(".plan")])
    return [(v["stored"], v["replayed"]) for k, v in sorted(found.items()) if len(v) == 2]


def g(v):
    return "inf" if v == float("inf") else f"{v:.3g}"


def markdown(res):
    lines = [f"### rank {res['rank']}, pos {res['pos']}, window {res['window']} ({os.path.basename(res['stored'])})", "",
             "| KDA layer | heads bit-identical | rec rel p50 | p90 | max | rec max abs | conv identical | conv rel p50 | max "
             "| horizon p50 / p90 | bound p50 / p90 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in res["layers"]:
        hz, b = l.get("horizon"), l.get("bound")
        hz_txt = "%.1f / %.1f" % (hz["p50"], hz["p90"]) if hz else "-"
        b_txt = "%s / %s" % (g(b["p50"]), g(b["p90"])) if b else "-"
        lines.append(f"| {l['layer']} | {l['rec_same']}/{res['heads']} | {g(l['rec_rel']['p50'])} | {g(l['rec_rel']['p90'])} "
                     f"| {g(l['rec_rel']['max'])} | {g(l['rec_maxabs'])} | {l['conv_same']}/{res['heads']} "
                     f"| {g(l['conv_rel']['p50'])} | {g(l['conv_rel']['max'])} "
                     f"| {hz_txt} | {b_txt} |")
    t = res["total"]
    lines += ["", f"All heads: {t['rec_same']}/{t['heads']} bit-identical, rec rel p50 {g(t['rec_rel']['p50'])}, "
                  f"p90 {g(t['rec_rel']['p90'])}, max {g(t['rec_rel']['max'])}; conv {t['conv_same']}/{t['heads']} identical.", ""]
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--dir")
    ap.add_argument("--window", type=int)
    ap.add_argument("--horizons")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.dir:
        todo = pairs(a.dir)
        if not todo:
            raise kvdump.DumpError(f"{a.dir}: no stored/replayed pairs")
    elif len(a.stems) == 2:
        todo = [tuple(a.stems)]
    else:
        ap.error("two stems (stored, replayed) or --dir")
    hz = read_horizons(a.horizons) if a.horizons else None
    results = [compare(x, y, a.window, hz) for x, y in todo]
    for r in results:
        print(markdown(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
