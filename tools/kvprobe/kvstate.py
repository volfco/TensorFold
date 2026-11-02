"""Cross-context comparison of the GLM engine's stored state (TF_GLM_KV_DUMP keeps), the follow-up the hidden-tap probe
could not make (docs/kv-cross-context-probe-2026-11-01.md, H13): MLA latent rows, DSA index rows (ring keys, pooled
keys), the KDA recurrent state per head and the short-conv history, for the same chunk S after different prefixes.

    python3 kvstate.py PROBE_JSON OUT_PREFIX [--rank R] [--bench PATH/tf-kv-codec-bench] [--pairs X:Y,...]

PROBE_JSON is what kv_probe_send.py writes: per capture its content type, context label, S's start and length in the
prompt's ids, and the keep stems each rank's dump directory got for it. Writes OUT_PREFIX.json and OUT_PREFIX.md.
Metrics follow analyze.py: relative L2 per row (norm(x - y) / norm(y)), cosine, and order-0 byte-plane entropy of the
codes alone (raw), XOR'd with the other context's same-token row (xor), or int16 differences (delta).
"""
import json, subprocess, sys
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import kvdump

HEAD = 64  # the first tokens of S, which see mostly the prefix
PAIRS = [("A+S", "B+S"), ("S", "A+S"), ("L+S", "A+S")]


def q(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "p50": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "mean": float(v.mean())}


def rel_l2(x, y):
    return np.linalg.norm(x - y, axis=-1) / np.maximum(np.linalg.norm(y, axis=-1), 1e-30)


def cosine(x, y):
    return (x * y).sum(-1) / np.maximum(np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-30)


def entropy_bits(b):
    c = np.bincount(np.asarray(b, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def plane_ratio(codes):
    """Order-0 modeled bytes over raw bytes, one byte plane a code byte (kv_codec's plain model)."""
    codes = np.asarray(codes)
    if codes.dtype == np.uint8:
        return entropy_bits(codes) / 8.0
    lo = (codes & 0xFF).astype(np.uint8)
    hi = (codes >> 8).astype(np.uint8)
    return (entropy_bits(lo) + entropy_bits(hi)) / 16.0


def entropy3(cx, cy):
    xor = cx ^ cy
    if cx.dtype == np.uint8:
        delta = (cx.astype(np.int16) - cy.astype(np.int16)).astype(np.uint8)
    else:
        delta = (cx.astype(np.int32) - cy.astype(np.int32)).astype(np.uint16)
    return {"raw": plane_ratio(cx), "xor": plane_ratio(xor), "delta": plane_ratio(delta)}


def rows_metrics(x, y, cx=None, cy=None, head=HEAD):
    d = rel_l2(x, y)
    out = {"rel_l2": q(d), "head": q(d[:head]), "body": q(d[head:]), "cosine": q(cosine(x, y)),
           "identical_rows": int(np.all(cx == cy, axis=1).sum()) if cx is not None else None}
    if len(y) > 1:
        perm = np.random.default_rng(0).permutation(len(y))
        out["other_token"] = q(rel_l2(y[perm], y))
    if cx is not None:
        out["entropy"] = entropy3(cx, cy)
    return out


def spectrum(m):
    """Singular values of a matrix, energy fractions and effective rank (exp of the entropy of s^2 / sum s^2)."""
    s = np.linalg.svd(np.asarray(m, dtype=np.float64), compute_uv=False)
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return {"eff_rank": 0.0, "rank90": 0, "rank99": 0, "top": [], "energy8": 0.0, "energy32": 0.0}
    p = e / tot
    nz = p[p > 0]
    c = np.cumsum(p)
    return {"eff_rank": float(np.exp(-(nz * np.log(nz)).sum())), "rank90": int(np.searchsorted(c, 0.90) + 1),
            "rank99": int(np.searchsorted(c, 0.99) + 1), "energy8": float(c[min(7, len(c) - 1)]),
            "energy32": float(c[min(31, len(c) - 1)]), "top": [float(v) for v in (s[:8] / s[0])]}


def kda_layer(sx, sy):
    """Per head: deviation of X's state from Y's, Y's spectrum and the difference's."""
    heads = []
    for h in range(sy.shape[0]):
        a, b = sx[h].astype(np.float64), sy[h].astype(np.float64)
        nb = np.linalg.norm(b)
        heads.append({"rel": float(np.linalg.norm(a - b) / max(nb, 1e-30)), "norm": float(nb),
                      "state": spectrum(b), "diff": spectrum(a - b)})
    agg = lambda key, sub=None: q([(h[key][sub] if sub else h[key]) for h in heads])
    return {"rel": agg("rel"), "eff_rank": agg("state", "eff_rank"), "rank90": agg("state", "rank90"),
            "diff_eff_rank": agg("diff", "eff_rank"), "diff_energy8": agg("diff", "energy8"),
            "diff_energy32": agg("diff", "energy32"), "heads": heads}


def kda_single(s):
    heads = [spectrum(s[h]) for h in range(s.shape[0])]
    return {"eff_rank": q([h["eff_rank"] for h in heads]), "rank90": q([h["rank90"] for h in heads]),
            "rank99": q([h["rank99"] for h in heads]), "norm": q([float(np.linalg.norm(s[h])) for h in range(s.shape[0])])}


def layer_names(k, layers=45, dsa_first=3, dsa_every=4):
    """Global layer ids of the DSA caches and KDA layers (GLM-5.3-Flash: DSA at 3, 7, ..., 43)."""
    dsa = [l for l in range(layers) if l >= dsa_first and (l - dsa_first) % dsa_every == 0]
    kda = [l for l in range(layers) if l not in dsa]
    if len(dsa) != len(k.dsa_caches) or len(kda) != k.shape["kda_layers"]:
        return list(range(len(k.dsa_caches))), list(range(k.shape["kda_layers"]))
    return dsa, kda


class Capture:
    """One prompt's keep on one rank, with S's place in it."""

    def __init__(self, entry, rank):
        self.type, self.context = entry["type"], entry["context"]
        self.s_start, self.s_len, self.suffix = entry["s_start"], entry["s_len"], entry.get("suffix_len", 0)
        stems = entry["keeps"].get(str(rank), [])
        keeps = [kvdump.Keep(s) for s in stems]
        if not keeps:
            raise kvdump.DumpError(f"{self.type}/{self.context}: no keep on rank {rank}")
        self.keep = max(keeps, key=lambda k: k.pos)  # the deepest mark the request kept
        self.covered = max(0, min(self.s_len, self.keep.pos - self.s_start))
        self.after = self.keep.pos - (self.s_start + self.s_len)  # tokens between S's end and the keep


def compare(x, y):
    kx, ky = x.keep, y.keep
    n = min(x.covered, y.covered)
    out = {"rows": n, "pos": [kx.pos, ky.pos], "after_s": [x.after, y.after], "latent": [], "ring": [], "pooled": [],
           "kda": [], "conv": []}
    dsa, kda = layer_names(ky)
    for j, layer in zip(ky.dsa_caches, dsa):
        lx, cx = kx.latent(j)
        ly, cy = ky.latent(j)
        a, b = slice(x.s_start, x.s_start + n), slice(y.s_start, y.s_start + n)
        if n > 0:
            out["latent"].append({"layer": layer, **rows_metrics(lx[a], ly[b], cx[a], cy[b])})
        # index keys in the rings: S's rows that both rings still hold
        ikx, _, fx = kx.rings(j)
        iky, _, fy = ky.rings(j)
        if ikx is not None:
            lo = max(fx - x.s_start, fy - y.s_start, 0)
            hi = min(fx + len(ikx) - x.s_start, fy + len(iky) - y.s_start, n)
            if hi > lo:
                ax = ikx[x.s_start + lo - fx: x.s_start + hi - fx]
                ay = iky[y.s_start + lo - fy: y.s_start + hi - fy]
                out["ring"].append({"layer": layer, "from": int(lo), **rows_metrics(kvdump.bf16(ax), kvdump.bf16(ay), ax, ay, head=max(0, HEAD - lo))})
        # pooled keys: pools inside S in both, when both starts sit at the same place in a pool
        if x.s_start % 4 == y.s_start % 4 and n >= 8:
            px, py = kx.pooled(j), ky.pooled(j)
            m = (n - (-x.s_start) % 4) // 4  # whole pools from the first pool boundary in S
            ax = px[(x.s_start + 3) // 4:][:m]
            ay = py[(y.s_start + 3) // 4:][:m]
            m = min(len(ax), len(ay))
            if m > 0:
                out["pooled"].append({"layer": layer, **rows_metrics(kvdump.bf16(ax[:m]), kvdump.bf16(ay[:m]), ax[:m], ay[:m], head=HEAD // 4)})
    # the recurrent state and conv history are at the keep: comparable only at the same distance past S
    if x.after == y.after and x.covered == x.s_len and y.covered == y.s_len:
        for li, layer in enumerate(kda):
            out["kda"].append({"layer": layer, **kda_layer(kx.kda_state(li), ky.kda_state(li))})
            out["conv"].append({"layer": layer, "rel_l2": q(rel_l2(kx.conv(li), ky.conv(li)))})
    else:
        out["kda_skipped"] = "the keeps sit at different distances past S, or before its end"
    return out


def single(c):
    k = c.keep
    dsa, kda = layer_names(k)
    out = {"pos": k.pos, "kda": [], "latent": []}
    for li, layer in enumerate(kda):
        out["kda"].append({"layer": layer, **kda_single(k.kda_state(li))})
    for j, layer in zip(k.dsa_caches, dsa):
        _, codes = k.latent(j)
        out["latent"].append({"layer": layer, "raw": plane_ratio(codes[:k.pos])})
    return out


def bench(path, stem):
    r = subprocess.run([path, "--file", stem + ".bin"], capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("KVBENCH")]
    return {"exit": r.returncode, "lines": lines}


def analyze(probe, rank=0, pairs=PAIRS, bench_path=None):
    caps = {}
    for e in probe["captures"]:
        if str(rank) in e["keeps"]:
            caps[(e["type"], e["context"])] = Capture(e, rank)
    types = sorted({t for t, _ in caps})
    result = {"rank": rank, "types": {}}
    for t in types:
        r = {"pairs": {}, "single": {}}
        for (tt, ctx), c in caps.items():
            if tt == t:
                r["single"][ctx] = single(c)
                if bench_path:
                    r["single"][ctx]["bench"] = bench(bench_path, c.keep.stem)
        for x, y in pairs:
            if (t, x) in caps and (t, y) in caps:
                r["pairs"][f"{x} vs {y}"] = compare(caps[(t, x)], caps[(t, y)])
        result["types"][t] = r
    return result


def f3(m, key="p50"):
    return "n/a" if not m or m.get("n", 0) == 0 else f"{m[key]:.3f}"


def markdown(result):
    out = [f"# Stored-state cross-context comparison (rank {result['rank']})\n"]
    for t, r in result["types"].items():
        out.append(f"## {t}\n")
        for k, p in r["pairs"].items():
            out.append(f"### {k}: {p['rows']} rows of S, keeps at {p['pos']}, {p['after_s']} tokens past S\n")
            out.append("| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |")
            out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            ring = {m["layer"]: m for m in p["ring"]}
            pool = {m["layer"]: m for m in p["pooled"]}
            for m in p["latent"]:
                e = m["entropy"]
                out.append(f"| {m['layer']} | {f3(m['rel_l2'])} | {f3(m['rel_l2'], 'p90')} | {f3(m['head'])} | {f3(m['body'])} | "
                           f"{f3(m['cosine'])} | {f3(m.get('other_token'))} | {e['raw']:.3f} | {e['xor']:.3f} | {e['delta']:.3f} | "
                           f"{f3(ring.get(m['layer'], {}).get('rel_l2'))} | {f3(pool.get(m['layer'], {}).get('rel_l2'))} |")
            if p["kda"]:
                out.append("\n| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |")
                out.append("|---|---|---|---|---|---|---|---|---|")
                conv = {m["layer"]: m for m in p["conv"]}
                for m in p["kda"]:
                    out.append(f"| {m['layer']} | {f3(m['rel'])} | {f3(m['rel'], 'p90')} | {f3(m['eff_rank'])} | {f3(m['rank90'])} | "
                               f"{f3(m['diff_eff_rank'])} | {f3(m['diff_energy8'])} | {f3(m['diff_energy32'])} | {f3(conv[m['layer']]['rel_l2'])} |")
            else:
                out.append(f"\nKDA: {p.get('kda_skipped', 'none')}")
            out.append("")
        for ctx, s in r["single"].items():
            if "bench" in s:
                out.append(f"bench {ctx}: exit {s['bench']['exit']}")
                out.extend("    " + l for l in s["bench"]["lines"])
        out.append("")
    return "\n".join(out) + "\n"


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    probe = json.load(open(argv[1]))
    rank, bench_path, pairs = 0, None, PAIRS
    rest = argv[3:]
    while rest:
        k = rest.pop(0)
        if k == "--rank":
            rank = int(rest.pop(0))
        elif k == "--bench":
            bench_path = rest.pop(0)
        elif k == "--pairs":
            pairs = [tuple(p.split(":")) for p in rest.pop(0).split(",")]
        else:
            raise SystemExit(f"unknown option {k}")
    result = analyze(probe, rank, pairs, bench_path)
    json.dump(result, open(argv[2] + ".json", "w"), indent=1)
    open(argv[2] + ".md", "w").write(markdown(result))
    print("wrote", argv[2] + ".json", argv[2] + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
