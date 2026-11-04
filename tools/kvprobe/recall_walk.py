"""A decode walk over the production decode-row dumps: at each position a source proposes a copy; the tokens it lands
(strictly, against the reply's own greedy picks) are skipped as one round, else one token a round. Reports the share
of decode tokens landed by copies, rounds against one-token decoding, and per-request classes.

    python3 -P recall_walk.py DUMP_DIR OUT.json
"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/kvprobe/TensorFold/tools/kvprobe"))
import draft_eval as D


def walk(reqs, source, k):
    per = []
    for q in reqs:
        toks, greedy = list(q["tokens"]), list(q["greedy"])
        source.start(q)
        n, i, landed, rounds, copies = len(toks), 0, 0, 0, 0
        while i < n:
            d = source.propose(toks[: i + 1], k)
            m = D.accepted(d, toks, greedy, i, True) if d else 0
            if hasattr(source, "observe") and d:
                source.observe(m)
            if m > 0:
                landed += m
                copies += 1
            i += m + 1
            rounds += 1
        source.finish(q)
        per.append({"index": q["index"], "n": n, "landed": landed, "rounds": rounds, "copies": copies,
                    "share": landed / max(1, n)})
    return per


def summarize(per):
    n = sum(r["n"] for r in per)
    landed = sum(r["landed"] for r in per)
    rounds = sum(r["rounds"] for r in per)
    copies = sum(r["copies"] for r in per)
    cls = {"whole": [], "partial": [], "none": []}
    for r in per:
        cls["whole" if r["share"] >= 0.8 else "partial" if r["landed"] > 0 else "none"].append(r)
    return {"tokens": n, "landed": landed, "landed_share": landed / max(1, n), "rounds": rounds,
            "rounds_share_of_tokens": rounds / max(1, n), "copies": copies, "landed_per_copy": landed / max(1, copies),
            "classes": {c: {"requests": len(v), "token_share": sum(r["n"] for r in v) / max(1, n),
                            "landed_share_within": sum(r["landed"] for r in v) / max(1, sum(r["n"] for r in v))}
                        for c, v in cls.items()}}


def main(dump, out):
    t0 = time.time()
    reqs = D.load_dump(dump)
    rep = {"dump": dump, "requests": len(reqs), "rows": sum(len(q["tokens"]) for q in reqs), "walks": {}}
    for name, gate, k in (("context", False, 7), ("recall", False, 7), ("recall", True, 7), ("engine", False, 7),
                          ("engine", True, 7), ("engine", False, 16), ("recall", False, 16)):
        label = f"{name}{'-gated' if gate else ''}-k{k}"
        src = D.make_source(name, reqs, gate)
        per = walk(reqs, src, k)
        s = summarize(per)
        rep["walks"][label] = {"summary": s, "per_request": per}
        c = s["classes"]
        print(f"{label:16s} landed {s['landed_share']:.3f} of {s['tokens']} tokens, {s['landed_per_copy']:.2f} a copy, rounds/token {s['rounds_share_of_tokens']:.3f}; "
              f"whole {c['whole']['requests']} req {c['whole']['token_share']:.3f} of tokens; partial {c['partial']['requests']} req "
              f"{c['partial']['token_share']:.3f} landing {c['partial']['landed_share_within']:.3f} within; none {c['none']['requests']} req", flush=True)
        json.dump(rep, open(out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
