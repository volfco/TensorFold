"""Greedy next-token agreement across contexts for the shared chunk's rows (the dump's top-1 ids and log-probs)."""
import json, sys, numpy as np
dump, P = sys.argv[1], json.load(open(sys.argv[2]))
recs = [json.loads(l) for l in open(f"{dump}/index.jsonl") if json.loads(l)["kind"] == "p"]
reqs = []
for r in recs:
    if r["start"] == 0: reqs.append([])
    reqs[-1].append(r)
shards = {}
def arr(r, name):
    off, dtype, shape = r["arrays"][name]
    f = shards.setdefault(r["shard"], np.memmap(f"{dump}/{r['shard']}", dtype=np.uint8, mode="r"))
    dt = {"bfloat16": np.uint16, "int32": np.int32, "float16": np.float16}[dtype]
    return f[off:off + int(np.prod(shape)) * np.dtype(dt).itemsize].view(dt).reshape(shape)
nS = len(P["S"]); got = {}
for segs in reqs:
    toks = np.concatenate([arr(r, "tokens") for r in segs])
    for name, m in P["meta"].items():
        if len(toks) == m["len"] and np.array_equal(toks, np.asarray(P["prompts"][name], dtype=np.int32)):
            ids = np.concatenate([arr(r, "ids")[:, 0] for r in segs]); lp = np.concatenate([arr(r, "lp")[:, 0].astype(np.float32) for r in segs])
            s0 = m["s_start"]; got[name] = (ids[s0:s0 + nS], lp[s0:s0 + nS])
S = np.asarray(P["S"])
for x, y in [("A+S", "B+S"), ("S", "A+S"), ("L+S", "A+S")]:
    same = got[x][0] == got[y][0]
    print(f"{x} vs {y}: greedy pick agrees on {same.mean()*100:.1f}% of S rows (head64 {same[:64].mean()*100:.1f}%, body {same[64:].mean()*100:.1f}%); "
          f"mean |dlogp| of top-1 {np.abs(got[x][1] - got[y][1]).mean():.3f} nats")
# teacher-forced accuracy of the top-1 against S's actual next token, per context
for n, (ids, lp) in got.items():
    print(f"{n}: top-1 equals the actual next token on {(ids[:-1] == S[1:]).mean()*100:.1f}% of S rows, mean top-1 logp {lp.mean():.3f}")
