"""A TF_GLM_KV_LIVE_HALF=1 run's keeps against the whole-head run's, keep id by keep id (one rank's dump dirs): the
.bin smaller by exactly the dead half, every class outside the arena head byte-identical, the head's bytes before and
after the hole identical to the full head's, the live KDA states and the conv history identical through the reader.

    python3 -P compare_half.py DUMP_BOTH DUMP_LIVE [OUT.json]
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/kvprobe/TensorFold/tools/kvprobe"))
import kvdump


def same_bytes(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b))


def main(a, b, out=None):
    full = {k: s for _, k, s in kvdump.stems(a)}
    half = {k: s for _, k, s in kvdump.stems(b)}
    ids = sorted(set(full) & set(half))
    rep = {"keeps": {}, "ok": bool(ids), "compared": len(ids), "only_full": sorted(set(full) - set(half)), "only_half": sorted(set(half) - set(full))}
    for k in ids:
        F, H = kvdump.Keep(full[k]), kvdump.Keep(half[k])
        s = F.shape
        nk, per = s["kda_layers"], s["rec_layer"]
        r = {"pos": F.pos, "half": H.half, "cur": "".join(map(str, H.cur)), "full_len": F.len, "half_len": H.len}
        r["len_delta_ok"] = bool(H.half is not None and F.len - H.len == nk * per)
        fp = [p for p in F.parts if p["class"] != "prefix"]
        hp = [p for p in H.parts if p["class"] != "prefix"]
        same = len(fp) == len(hp) and all(x["class"] == y["class"] and x["len"] == y["len"] for x, y in zip(fp, hp))
        if same:
            for x, y in zip(fp, hp):
                if not same_bytes(np.asarray(F.raw[x["off"]:x["off"] + x["len"]]), np.asarray(H.raw[y["off"]:y["off"] + y["len"]])):
                    same = False
                    r["first_differing_part"] = x["class"]
                    break
        r["other_classes_identical"] = bool(same)
        if H.half is not None:
            h, hole, hl = H.half
            f0 = F.parts[0]
            head_f = np.asarray(F.raw[f0["off"]:f0["off"] + f0["len"]])
            p0, p1 = H.parts[0], H.parts[1]
            r["head_before_hole_identical"] = same_bytes(np.asarray(H.raw[p0["off"]:p0["off"] + p0["len"]]), head_f[:hole])
            r["head_after_hole_identical"] = same_bytes(np.asarray(H.raw[p1["off"]:p1["off"] + p1["len"]]), head_f[hole + hl:hole + hl + p1["len"]])
            r["live_states_identical"] = all(np.array_equal(F.kda_state(li), H.kda_state(li)) for li in range(nk))
            r["conv_identical"] = all(np.array_equal(F.conv(li), H.conv(li)) for li in range(nk))
            dead = head_f[hole:hole + hl]
            r["dead_half_in_full_run_zero_frac"] = float(np.mean(dead == 0))
        ok = r["len_delta_ok"] and same and all(r.get(x) for x in ("head_before_hole_identical", "head_after_hole_identical", "live_states_identical", "conv_identical"))
        r["ok"] = bool(ok)
        rep["ok"] = rep["ok"] and bool(ok)
        rep["keeps"][k] = r
        print(k, json.dumps(r), flush=True)
    print("ALL OK" if rep["ok"] else f"NOT OK (keeps compared: {len(ids)})")
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
