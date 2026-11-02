"""Position curve, confidence dependence and channel concentration from the probe dump."""
import json, sys, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "/dev/null", "/dev/null"]
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "analyze.py")).read().split("pairs = [")[0])
def lp_of(name):
    segs, toks = by_name[name]; s0 = P["meta"][name]["s_start"]; rows = []
    for r in segs:
        a, n = r["start"], r["n"]; lo, hi = max(a, s0), min(a + n, s0 + nS)
        if lo < hi: rows.append(np.asarray(arr(r, "lp")[lo - a:hi - a, 0]).astype(np.float32))
    return np.concatenate(rows)
bins = [(0, 8), (8, 32), (32, 128), (128, 512), (512, 1024), (1024, 2048)]
print("deviation A+S vs B+S by position in S (rel L2 p50 per bin):")
print("layer | " + " | ".join(f"{a}-{b}" for a, b in bins))
conf = lp_of("A+S")                              # log-prob of the greedy pick: high = confident
hi_conf, lo_conf = conf > np.quantile(conf, 0.75), conf < np.quantile(conf, 0.25)
S = np.asarray(P["S"]); 
uniq, counts = np.unique(S, return_counts=True); freq = dict(zip(uniq, counts)); f = np.array([freq[t] for t in S])
frequent, rare = f >= np.quantile(f, 0.75), f <= 1
for i in (0, 3, 9, 15, 21, 27, 33, 39, 44):
    a = bf16_to_f32(taps_of("A+S", i)); b = bf16_to_f32(taps_of("B+S", i)); d = rel_l2(a, b)
    print(f"{layers[i]:2d} | " + " | ".join(f"{np.median(d[x:y]):.3f}" for x, y in bins)
          + f" || confident-quartile p50 {np.median(d[hi_conf]):.3f} vs unsure-quartile {np.median(d[lo_conf]):.3f}"
          + f" || frequent tokens {np.median(d[frequent]):.3f} vs singletons {np.median(d[rare]):.3f}")
print("\nchannel concentration of the state (A+S): share of row energy in the top 1 / 8 / 64 channels (mean over rows), and the top channel's index")
for i in (0, 3, 9, 21, 33, 42, 43, 44):
    a = bf16_to_f32(taps_of("A+S", i)); e = a * a; tot = e.sum(1, keepdims=True)
    srt = -np.sort(-e, axis=1)
    top = np.bincount(np.argmax(e, axis=1), minlength=D); ch = int(np.argmax(top))
    print(f"layer {layers[i]:2d}: {np.mean(srt[:, :1].sum(1) / tot[:, 0]):.3f} / {np.mean(srt[:, :8].sum(1) / tot[:, 0]):.3f} / {np.mean(srt[:, :64].sum(1) / tot[:, 0]):.3f}; "
          f"channel {ch} is the max channel on {top[ch] / nS * 100:.0f}% of rows, mean |x| there {np.abs(a[:, ch]).mean():.1f} vs row rms {np.sqrt(tot.mean() / D):.2f}")
