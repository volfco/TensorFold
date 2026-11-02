"""Long-context DSA selection analysis on a patch-0710 dump, by position bin: prunable whole pages before the row
(no selected pool in the page), selection stability between consecutive recorded rows (the dump's stride apart), and
how far back the selected pools reach (share of the 2048 selected tokens older than 8k / 32k / 128k tokens)."""
import sys, json, numpy as np
sys.path.insert(0, "/home/colum/kvprobe/TensorFold/tools/kvprobe")
import routedump
rd = routedump.load(sys.argv[1])
pos_all = rd["pos"]
bins = [(2048, 16384), (16384, 65536), (65536, 131072), (131072, 204800)]
page_rows, pool = 512, 4
out = {}
print("| layer | position bin | rows | prunable pages p50 / mean | consecutive jaccard | share in previous | selected tokens older than 8k / 32k / 128k |")
print("|---|---|---|---|---|---|---|")
for layer, (sel, at) in sorted(rd["dsa_sel"].items()):
    pos = pos_all[at]
    order = np.argsort(pos); sel, pos = sel[order], pos[order]
    for lo, hi in bins:
        idx = np.where((pos >= lo) & (pos < hi))[0]
        if len(idx) < 2: continue
        fr, jac, share, far = [], [], [], {8192: [], 32768: [], 131072: []}
        for j, i in enumerate(idx):
            x = sel[i]; x = x[x >= 0]; p = int(pos[i]); pages = p // page_rows
            if pages > 0:
                hit = np.unique(x * pool // page_rows); fr.append(1 - np.count_nonzero(hit < pages) / pages)
            if j > 0:
                y = sel[idx[j - 1]]; y = y[y >= 0]
                inter = len(np.intersect1d(x, y)); jac.append(inter / max(1, len(np.union1d(x, y)))); share.append(inter / max(1, len(x)))
            age = p - x * pool
            for k in far: far[k].append(float(np.mean(age > k)))
        row = {"rows": len(idx), "prunable_p50": float(np.median(fr)) if fr else None, "prunable_mean": float(np.mean(fr)) if fr else None,
               "jaccard": float(np.mean(jac)), "share_prev": float(np.mean(share)), "older": {k: float(np.mean(v)) for k, v in far.items()}}
        out[f"{layer}:{lo}-{hi}"] = row
        print(f"| {layer} | {lo//1024}k-{hi//1024}k | {row['rows']} | {row['prunable_p50']:.3f} / {row['prunable_mean']:.3f} | {row['jaccard']:.3f} | {row['share_prev']:.3f} | {row['older'][8192]:.2f} / {row['older'][32768]:.2f} / {row['older'][131072]:.2f} |", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
