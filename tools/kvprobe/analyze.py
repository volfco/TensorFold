"""Cross-context hidden-state probe: same token chunk S under different prefixes, every layer's tap compared.

Reads a patches/0430 draft dump (index.jsonl + shard-*.bin): kind p segments in request order. Writes a JSON of
per-layer metrics and a markdown summary."""
import json, sys, math
from collections import defaultdict
import numpy as np

dump = sys.argv[1]
P = json.load(open(sys.argv[2]))
out_json, out_md = sys.argv[3], sys.argv[4]
S = np.asarray(P["S"], dtype=np.int32)
nS = len(S)
meta = json.load(open(f"{dump}/meta.json"))
D = meta["hidden"]

# ---- reassemble prefill rows per request (a request starts at start == 0) ---------------------------------------
recs = [json.loads(l) for l in open(f"{dump}/index.jsonl")]
recs = [r for r in recs if r["kind"] == "p"]
reqs = []
for r in recs:
    if r["start"] == 0:
        reqs.append([])
    reqs[-1].append(r)
shards = {}
def arr(r, name):
    off, dtype, shape = r["arrays"][name]
    f = shards.setdefault(r["shard"], np.memmap(f"{dump}/{r['shard']}", dtype=np.uint8, mode="r"))
    dt = {"bfloat16": np.uint16, "int32": np.int32, "<i4": np.int32, "float16": np.float16, "float32": np.float32}.get(dtype, dtype)
    n = int(np.prod(shape)) * np.dtype(dt).itemsize
    return f[off:off + n].view(dt).reshape(shape)

by_name = {}
for segs in reqs:
    toks = np.concatenate([arr(r, "tokens") for r in segs])
    T = len(toks)
    nlayers = len(segs[0]["tap_layers"])
    for name, m in P["meta"].items():
        if m["len"] == T and np.array_equal(toks, np.asarray(P["prompts"][name], dtype=np.int32)):
            by_name[name] = (segs, toks)
            break
    else:
        print(f"unmatched request of {T} tokens (tokens {toks[:4]}...)")
assert set(by_name) == set(P["meta"]), set(by_name)
layers = reqs[0][0]["tap_layers"]
L = len(layers)
print(f"requests: {list(by_name)}, layers {L}, D {D}")

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def taps_of(name, layer_slot):
    """[nS, D] bf16 bit patterns of S's rows at tap slot `layer_slot` for request `name`."""
    segs, toks = by_name[name]
    s0 = P["meta"][name]["s_start"]
    rows = []
    for r in segs:
        a, n = r["start"], r["n"]
        lo, hi = max(a, s0), min(a + n, s0 + nS)
        if lo >= hi:
            continue
        t = arr(r, "taps")                      # [n, L*D]
        rows.append(np.asarray(t[lo - a:hi - a, layer_slot * D:(layer_slot + 1) * D]))
    x = np.concatenate(rows)
    assert x.shape == (nS, D), x.shape
    return x

def hidden_of(name):
    segs, toks = by_name[name]
    s0 = P["meta"][name]["s_start"]
    rows = []
    for r in segs:
        a, n = r["start"], r["n"]
        lo, hi = max(a, s0), min(a + n, s0 + nS)
        if lo >= hi:
            continue
        rows.append(np.asarray(arr(r, "hidden")[lo - a:hi - a]))
    return np.concatenate(rows)

def rel_l2(x, y):
    return np.linalg.norm(x - y, axis=1) / np.maximum(np.linalg.norm(y, axis=1), 1e-30)
def cosine(x, y):
    return (x * y).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
def q(v):
    return {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(np.max(v))}
def entropy_bits(bytes_):
    c = np.bincount(bytes_.ravel(), minlength=256).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())
def plane_ratio(u16):
    """order-0 modeled bytes / raw bytes for a bf16 block coded as two byte planes (the kv_codec plain model)."""
    lo = (u16 & 0xFF).astype(np.uint8); hi = (u16 >> 8).astype(np.uint8)
    return (entropy_bits(lo) + entropy_bits(hi)) / 16.0

pairs = [("A+S", "B+S", "cross-prefix, same position (primary)"),
         ("S", "A+S", "no prefix vs prefix (position shift + context)"),
         ("L+S", "A+S", "long prefix (8192) vs short (1024)")]
rng = np.random.default_rng(0)
perm = rng.permutation(nS)
head = 64                                              # the first tokens of S see mostly the prefix
result = {"meta": {"layers": layers, "D": D, "nS": nS, "pairs": pairs, "head_tokens": head, "dump": meta.get("env", {})}, "layers": []}
prev = {}
for i, layer in enumerate(layers):
    cur = {n: taps_of(n, i) for n in by_name}
    curf = {n: bf16_to_f32(v) for n, v in cur.items()}
    row = {"layer": layer}
    ref = curf["A+S"]
    row["baseline_other_token"] = q(rel_l2(ref[perm], ref))          # a different token of S, same context
    row["baseline_next_token"] = q(rel_l2(ref[1:], ref[:-1]))        # the neighbouring token, same context
    row["row_norm_mean"] = float(np.linalg.norm(ref, axis=1).mean())
    for x, y, _ in pairs:
        k = f"{x} vs {y}"
        d = rel_l2(curf[x], curf[y])
        row[k] = {"rel_l2": q(d), "rel_l2_head": q(d[:head]), "rel_l2_body": q(d[head:]), "cosine": q(cosine(curf[x], curf[y])),
                  "bit_identical_rows": int(np.all(cur[x] == cur[y], axis=1).sum())}
        if prev:                                                      # the residual update the layer added
            ux, uy = curf[x] - prev[x], curf[y] - prev[y]
            row[k]["update_rel_l2"] = q(rel_l2(ux, uy))
            row[k]["update_cosine"] = q(cosine(ux, uy))
            row[k]["update_norm_over_state"] = float((np.linalg.norm(uy, axis=1) / np.linalg.norm(curf[y], axis=1)).mean())
        # lossless dedup proxy: bf16 rows of x coded alone, XOR'd against y's same-token row, or int16-delta'd
        raw = plane_ratio(cur[x])
        xr = plane_ratio(cur[x] ^ cur[y])
        dl = plane_ratio((cur[x].astype(np.int32) - cur[y].astype(np.int32)).astype(np.uint16))
        row[k]["entropy"] = {"raw": raw, "xor": xr, "delta": dl}
    result["layers"].append(row)
    prev = curf
    print(f"layer {layer}: A+S vs B+S rel_l2 p50 {row['A+S vs B+S']['rel_l2']['p50']:.4f} cos p50 {row['A+S vs B+S']['cosine']['p50']:.4f} "
          f"other-token p50 {row['baseline_other_token']['p50']:.3f} xor/raw {row['A+S vs B+S']['entropy']['xor']:.3f}/{row['A+S vs B+S']['entropy']['raw']:.3f}", flush=True)

# final-normed row (what the MTP head / lm_head read)
hf = {n: bf16_to_f32(hidden_of(n)) for n in by_name}
result["final_normed"] = {f"{x} vs {y}": {"rel_l2": q(rel_l2(hf[x], hf[y])), "cosine": q(cosine(hf[x], hf[y]))} for x, y, _ in pairs}
json.dump(result, open(out_json, "w"), indent=1)

with open(out_md, "w") as f:
    f.write(f"# Cross-context hidden-state probe (GLM-5.3-Flash EXL3, {L} tap layers, S = {nS} tokens)\n\n")
    for x, y, why in pairs:
        k = f"{x} vs {y}"
        f.write(f"## {k}: {why}\n\n| layer | state rel L2 p50 | p90 | head64 p50 | body p50 | cos p50 | update rel L2 p50 | update/state | other-token p50 | next-token p50 | raw | xor | delta |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for row in result["layers"]:
            r = row[k]
            up = r.get("update_rel_l2", {}).get("p50", float("nan")); uo = r.get("update_norm_over_state", float("nan"))
            e = r["entropy"]
            f.write(f"| {row['layer']} | {r['rel_l2']['p50']:.4f} | {r['rel_l2']['p90']:.4f} | {r['rel_l2_head']['p50']:.4f} | {r['rel_l2_body']['p50']:.4f} | {r['cosine']['p50']:.4f} | {up:.4f} | {uo:.3f} | {row['baseline_other_token']['p50']:.3f} | {row['baseline_next_token']['p50']:.3f} | {e['raw']:.3f} | {e['xor']:.3f} | {e['delta']:.3f} |\n")
        fn = result["final_normed"][k]
        f.write(f"\nfinal-normed row: rel L2 p50 {fn['rel_l2']['p50']:.4f}, p90 {fn['rel_l2']['p90']:.4f}, cosine p50 {fn['cosine']['p50']:.4f}\n\n")
print("wrote", out_json, out_md)
