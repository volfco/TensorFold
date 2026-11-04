"""Score the production decode-row dumps offline: draft_eval's sources over every request in dump order, plus a
per-request breakdown of Recall's copies (whole-reply copies against ordinary replies).

    python3 -P recall_prod.py DUMP_DIR OUT.json [--k 8,16]
"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/kvprobe/TensorFold/tools/kvprobe"))
import draft_eval as D


def per_request(reqs, source, k):
    """One evaluate() a request with a persistent source: per-request coverage and strict acceptance at k."""
    rows = []
    for q in reqs:
        r = D.evaluate([q], source, ks=(k,))[k]
        rows.append({"index": q["index"], "n": r["positions"], "proposals": r["proposals"], "strict": r["strict"],
                     "loose": r["loose"], "coverage": r["coverage"], "per_pos": r["per_position_strict"]})
    return rows


def classify(rows):
    """Whole copies: 80 % of positions covered with 8 or more accepted a position; partial: some copy landed; none."""
    out = {"whole": {"requests": 0, "rows": 0, "strict": 0}, "partial": {"requests": 0, "rows": 0, "strict": 0},
           "none": {"requests": 0, "rows": 0, "strict": 0}}
    for r in rows:
        c = "whole" if (r["coverage"] >= 0.8 and r["per_pos"] >= 8) else ("partial" if r["strict"] > 0 else "none")
        out[c]["requests"] += 1
        out[c]["rows"] += r["n"]
        out[c]["strict"] += r["strict"]
    total = sum(v["rows"] for v in out.values())
    for v in out.values():
        v["row_share"] = v["rows"] / max(1, total)
    return out


def main(dump, out, ks=(8, 16)):
    t0 = time.time()
    reqs = D.load_dump(dump)
    rows = sum(len(q["tokens"]) for q in reqs)
    print(f"loaded {len(reqs)} requests, {rows} decode rows in {time.time() - t0:.0f} s", flush=True)
    rep = {"dump": dump, "requests": len(reqs), "rows": rows, "sources": {}, "per_request": {}}
    for name, gate in (("recall", False), ("recall", True), ("context", False), ("engine", False), ("engine", True)):
        t1 = time.time()
        src = D.make_source(name, reqs, gate)
        res = D.evaluate(reqs, src, ks=ks)
        label = name + ("-gated" if gate else "")
        rep["sources"][label] = {str(k): v for k, v in res.items()}
        print(f"{label}: " + ", ".join(f"k={k} coverage {v['coverage']:.3f} strict/proposal {v['mean_strict']:.2f} strict/position {v['per_position_strict']:.3f}"
                                       for k, v in res.items()) + f"  ({time.time() - t1:.0f} s)", flush=True)
        json.dump(rep, open(out, "w"), indent=1)
    for name in ("recall", "context"):
        t1 = time.time()
        src = D.make_source(name, reqs, False)
        pr = per_request(reqs, src, 16)
        cls = classify(pr)
        rep["per_request"][name] = {"classes": cls, "rows": pr}
        print(f"per-request {name} k=16: " + "; ".join(f"{c}: {v['requests']} requests, {v['row_share']:.3f} of rows, {v['strict']} strict" for c, v in cls.items())
              + f"  ({time.time() - t1:.0f} s)", flush=True)
        json.dump(rep, open(out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ks = tuple(int(x) for x in (sys.argv[sys.argv.index("--k") + 1] if "--k" in sys.argv else "8,16").split(","))
    main(sys.argv[1], sys.argv[2], ks)
