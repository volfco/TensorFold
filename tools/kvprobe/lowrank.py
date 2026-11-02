"""Is the cross-context difference structured? Rank and sparsity of D = tap(A+S) - tap(B+S) over S's 2048 rows."""
import json, sys, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "/dev/null", "/dev/null"]
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "analyze.py")).read().split("pairs = [")[0])
ranks = (8, 32, 128, 512)
print("layer | energy of D in rank 8/32/128/512 | same for the state A | energy in top 5%/20% of |D| coords | update energy in rank 32/128")
prev = None
for i in range(0, len(layers)):
    a = bf16_to_f32(taps_of("A+S", i)); b = bf16_to_f32(taps_of("B+S", i)); d = a - b
    if i % 3 and layers[i] not in (1, 44):
        prev = (a, b); continue
    ev = lambda m: np.sort(np.linalg.eigvalsh((m.astype(np.float64).T @ m.astype(np.float64))))[::-1].clip(0)
    ed, ea = ev(d), ev(a)
    fr = lambda e, r: e[:r].sum() / e.sum()
    srt = np.sort(np.abs(d), axis=1)[:, ::-1] ** 2
    top = lambda f: (srt[:, :int(f * d.shape[1])].sum(1) / srt.sum(1)).mean()
    extra = ""
    if prev is not None:
        ud = (a - prev[0]) - (b - prev[1]); su = ev(ud)
        extra = f" | {fr(su,32):.2f}/{fr(su,128):.2f}"
    print(f"{layers[i]:2d} | {fr(ed,8):.2f}/{fr(ed,32):.2f}/{fr(ed,128):.2f}/{fr(ed,512):.2f} | {fr(ea,8):.2f}/{fr(ea,32):.2f}/{fr(ea,128):.2f}/{fr(ea,512):.2f} | {top(0.05):.2f}/{top(0.20):.2f}{extra}", flush=True)
    prev = (a, b)
