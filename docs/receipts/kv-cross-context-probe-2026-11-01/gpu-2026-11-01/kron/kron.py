"""Kronecker (nearest-Kronecker-product, Van Loan) against SVD on a real kept GLM state, plus the lossless floor.

    python3 -P kron.py <keep stem> [caches, e.g. 0,5,10] [out.json]

Per KDA head (fp32 128x128): energy captured by k Kronecker terms (A m1xn1 (x) B m2xn2, the SVD of the rearranged
matrix) against a rank-r SVD at the same stored-parameter budget. Per DSA latent block (bf16 [T, 512]): the same, and
the lossless bound of "structured prediction + exact residual" (bf16 codes mapped to a monotone integer, order-0
entropy of the residual, byte-split as a plane codec sees it) against the raw codes' entropy. Also the dead half of the
double-buffered recurrent state and the window scratch fields: are they stale, and what do they cost.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/kvprobe/TensorFold/tools/kvprobe"))
import kvdump


def entropy_bits(x):
    x = np.asarray(x).ravel()
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def plane_bits_u16(codes):
    """Order-0 entropy, high byte plus low byte, of uint16 codes (what a byte-plane codec can reach)."""
    c = np.asarray(codes, np.uint16).ravel()
    return entropy_bits(c >> 8) + entropy_bits(c & 0xFF)


def plane_bits_f32(a):
    b = np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint8).reshape(-1, 4)
    return sum(entropy_bits(b[:, i]) for i in range(4))


def monotone(codes):
    """bf16 code -> integer ordered like the value (negative codes map below zero)."""
    c = np.asarray(codes, np.int64)
    return np.where(c < 0x8000, c, 0x8000 - c)


def to_bf16(v):
    u = np.ascontiguousarray(np.asarray(v, np.float32)).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def energy_curve_svd(M):
    s = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    e = s * s
    return np.cumsum(e) / e.sum()


def energy_curve_nkp(M, m1, m2, n1, n2):
    """Energy of the k-term nearest Kronecker product, k = 1..: singular values of the Van Loan rearrangement."""
    M = np.asarray(M, np.float64)
    assert M.shape == (m1 * m2, n1 * n2), (M.shape, m1, m2, n1, n2)
    R = M.reshape(m1, m2, n1, n2).transpose(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)
    if R.shape[0] >= R.shape[1]:
        e = np.sort(np.linalg.eigvalsh(R.T @ R))[::-1].clip(0)
    else:
        e = np.sort(np.linalg.eigvalsh(R @ R.T))[::-1].clip(0)
    return np.cumsum(e) / e.sum()


def at_budget(curve, per_term, budget):
    k = int(budget // per_term)
    if k <= 0:
        return 0.0
    return float(curve[min(k, len(curve)) - 1])


def q(xs, p):
    xs = np.sort(np.asarray(xs, np.float64))
    return float(xs[min(len(xs) - 1, int(p * len(xs)))])


def kda_part(keep, budgets=(1024, 2048, 4096, 8192, 16384)):
    s = keep.shape
    nk, heads, dk, dv = s["kda_layers"], s["heads"], s["dk"], s["dv"]
    shapes = {"svd": None, "kron16x16(x)8x8": (16, 8, 16, 8), "kron8x8(x)16x16": (8, 16, 8, 16),
              "kron32x32(x)4x4": (32, 4, 32, 4), "kron4x4(x)32x32": (4, 32, 4, 32)}
    res = {name: {b: [] for b in budgets} for name in shapes}
    rank90, rank99 = [], []
    floor_live, floor_dead, dead_rel, dead_zero, dead_equal = [], [], [], [], []
    bf16_rel, bf16_rank16_rel = [], []
    o, n = keep.fields["rec"]
    per = s["rec_layer"]
    t0 = time.time()
    for li in range(nk):
        live = keep.kda_state(li)
        dead = np.asarray(keep._bytes(o + ((1 - keep.cur[li]) * nk + li) * per, per)).view(np.float32).reshape(heads, dk, dv)
        floor_live.append(plane_bits_f32(live))
        floor_dead.append(plane_bits_f32(dead))
        nl = np.linalg.norm(live)
        dead_rel.append(float(np.linalg.norm(dead - live) / nl) if nl > 0 else float("nan"))
        dead_zero.append(float(np.mean(dead == 0)))
        dead_equal.append(float(np.mean(dead.view(np.uint32) == live.view(np.uint32))))
        rb = kvdump.bf16(to_bf16(live))
        bf16_rel.append(float(np.linalg.norm(rb - live) / nl) if nl > 0 else float("nan"))
        for h in range(heads):
            M = live[h]
            cs = energy_curve_svd(M)
            rank90.append(int(np.searchsorted(cs, 0.90) + 1))
            rank99.append(int(np.searchsorted(cs, 0.99) + 1))
            for b in budgets:
                res["svd"][b].append(at_budget(cs, dk + dv, b))
            for name, sh in shapes.items():
                if sh is None:
                    continue
                m1, m2, n1, n2 = sh
                cn = energy_curve_nkp(M, m1, m2, n1, n2)
                for b in budgets:
                    res[name][b].append(at_budget(cn, m1 * n1 + m2 * n2, b))
    out = {"layers": nk, "heads": heads, "seconds": round(time.time() - t0, 1),
           "rank90_p50_p90_max": [q(rank90, .5), q(rank90, .9), max(rank90)],
           "rank99_p50_p90_max": [q(rank99, .5), q(rank99, .9), max(rank99)],
           "energy_at_budget": {name: {str(b): {"p10": q(v[b], .1), "p50": q(v[b], .5), "p90": q(v[b], .9)} for b in budgets}
                                for name, v in res.items()},
           "lossless_floor_bits_per_fp32_live_p50": q(floor_live, .5), "lossless_floor_bits_per_fp32_dead_p50": q(floor_dead, .5),
           "dead_half": {"rel_l2_vs_live_p50_max": [q(dead_rel, .5), max(dead_rel)], "zero_frac_p50": q(dead_zero, .5),
                         "bit_equal_frac_p50": q(dead_equal, .5), "bytes_per_rank": nk * per},
           "bf16_round_rel_l2_p50_max": [q(bf16_rel, .5), max(bf16_rel)]}
    return out


def latent_part(keep, caches, ranks=(16, 32, 64, 128, 256), pred_ranks=(32, 64, 128)):
    out = {}
    T, D = keep.pos, keep.shape["kv_lora"]
    for j in caches:
        t0 = time.time()
        X, codes = keep.latent(j)
        X = X[:T]
        codes = np.asarray(codes[:T])
        X64 = X.astype(np.float64)
        U, S, Vt = np.linalg.svd(X64, full_matrices=False)
        e = S * S
        cs = np.cumsum(e) / e.sum()
        r = {"rows": int(T), "svd_energy": {str(k): float(cs[k - 1]) for k in ranks},
             "svd_cost_bits_per_value": {str(k): 16.0 * k * (T + D) / (T * D) for k in ranks}}
        # Kronecker shapes: feature-side (each row = a_t (x) b, n1 groups of n2), the within-pool shape, and a mixed one
        kron = {}
        for n1, n2 in ((64, 8), (32, 16), (16, 32), (8, 64)):
            cn = energy_curve_nkp(X64, T, 1, n1, n2)
            per = T * n1 + n2
            kron[f"row=a(x)b n1={n1} n2={n2}"] = {"terms": {str(k): float(cn[k - 1]) for k in (1, 2, 4, 8) if k <= len(cn)},
                                                 "cost_bits_per_value_per_term": 16.0 * per / (T * D),
                                                 "svd_rank_of_same_cost": n1}
        if T % 4 == 0:
            cn = energy_curve_nkp(X64, T // 4, 4, D, 1)
            kron["pool4: A[T/4,512](x)b[4,1]"] = {"terms": {str(k): float(cn[k - 1]) for k in (1, 2, 4) if k <= len(cn)},
                                                 "cost_bits_per_value_per_term": 16.0 * (T * D / 4 + 4) / (T * D),
                                                 "svd_rank_of_same_cost": 128}
            cn = energy_curve_nkp(X64, T // 4, 4, D // 4, 4)
            kron["mixed: A[T/4,128](x)B[4,4]"] = {"terms": {str(k): float(cn[k - 1]) for k in (1, 2, 4, 8, 16) if k <= len(cn)},
                                                 "cost_bits_per_value_per_term": 16.0 * (T * D / 16 + 16) / (T * D),
                                                 "svd_rank_of_same_cost": 32}
        r["kronecker"] = kron
        # lossless: raw codes against prediction + exact residual
        key = monotone(codes)
        r["raw_bits_per_value"] = {"plane_order0": plane_bits_u16(codes), "symbol_order0": entropy_bits(codes)}
        pr = {}
        for k in pred_ranks:
            # decodable: the factors a decoder would hold are bf16 (U_k S_k and V_k^T rounded), the prediction formed from them
            Ub = kvdump.bf16(to_bf16((U[:, :k] * S[:k]).astype(np.float32))).astype(np.float64)
            Vb = kvdump.bf16(to_bf16(Vt[:k].astype(np.float32))).astype(np.float64)
            P = Ub @ Vb
            pc = to_bf16(P.astype(np.float32))
            d = key - monotone(pc)
            resid_energy = float(np.sum((X64 - P) ** 2) / np.sum(X64 ** 2))
            du = (d + 65536).astype(np.uint32)  # non-negative symbols for the byte split
            pr[str(k)] = {"residual_energy": resid_energy,
                          "residual_plane_bits": entropy_bits((du >> 8) & 0xFF) + entropy_bits(du & 0xFF) + entropy_bits(du >> 16),
                          "residual_symbol_bits": entropy_bits(d),
                          "factor_bits_per_value": 16.0 * k * (T + D) / (T * D),
                          "abs_resid_p50_codes": float(np.median(np.abs(d))),
                          "exact_frac": float(np.mean(d == 0))}
        r["predict_then_code"] = pr
        r["seconds"] = round(time.time() - t0, 1)
        out[str(j)] = r
        print(f"cache {j} done in {r['seconds']} s", flush=True)
    return out


def scratch_part(keep):
    out = {}
    for f in ("proj", "save_k", "save_v", "save_g", "save_b", "conv"):
        b = np.asarray(keep.field(f))
        out[f] = {"bytes": int(b.size), "zero_frac": float(np.mean(b == 0))}
    return out


def main():
    stem = os.path.expanduser(sys.argv[1])
    caches = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,5,10").split(",")]
    keep = kvdump.Keep(stem)
    classes = {}
    for p in keep.parts:
        classes[p["class"]] = classes.get(p["class"], 0) + p["len"]
    rep = {"stem": stem, "pos": keep.pos, "rank": keep.rank, "classes_bytes": classes, "fields": {k: v[1] for k, v in keep.fields.items()},
           "scratch": scratch_part(keep)}
    print(json.dumps({k: rep[k] for k in ("pos", "classes_bytes", "fields", "scratch")}, indent=1), flush=True)
    rep["kda"] = kda_part(keep)
    print(json.dumps(rep["kda"], indent=1), flush=True)
    rep["latent"] = latent_part(keep, caches)
    print(json.dumps(rep["latent"], indent=1), flush=True)
    if len(sys.argv) > 3:
        with open(os.path.expanduser(sys.argv[3]), "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
