# Kronecker factorization against the SVD on kept states (2026-11-02, CPU)

Session record: `docs/kv-cross-context-probe-2026-11-01.md` section 11.18. Question: can Kronecker products pull the
entropy out of a cache entry and shrink it on disk.

- `kron.py`: the measurement (numpy only). Per KDA head (fp32 128x128): energy captured by k nearest-Kronecker-product
  terms (the SVD of the Van Loan rearrangement) against a rank-r SVD at the same stored-parameter budget. Per DSA
  latent block (bf16 [T, 512]): the same for feature-side, within-pool and mixed shapes, and the lossless bound of
  "prediction from bf16 factors plus the exact residual" (bf16 codes as monotone integers, order-0 entropy, byte-split
  and as symbols) against the raw codes. Also the dead half of the double-buffered recurrent state and the window
  scratch fields: zero fraction, bit-equality and relative difference against the live half.
- `kron2-keep9.json` / `.log`: `~/kvprobe/native/dump/glm-rank0-keep9` (20,480 tokens, rank 0), caches 0, 5, 10.
- `kron2-keep4.json` / `.log`: `glm-rank0-keep4` (8,192 tokens, rank 0), caches 0, 5, 10.

Run on spark0 beside production: `systemd-run --user --scope -p MemoryMax=3G nice -n 19 python3 -P kron.py <stem>
0,5,10 out.json` (about 2.5 minutes a keep; the KDA part 6 s). A first run with float64 factors (not kept) agreed
with these within 0.001 on every prediction figure.

Headline (both keeps): at 4,096 stored parameters a head (4x) the SVD keeps 0.983 of a head's energy at the median,
the 16x16 (x) 8x8 Kronecker 0.627, the 32x32 (x) 4x4 0.433; on the rows a rank-128 SVD keeps 0.77 to 0.79 where the
Kronecker shapes of the same cost keep 0.27 to 0.60; predicting the rows at rank 128 and coding the exact residual
saves 0.2 to 0.3 bits a value as symbols (none as planes) against 4.1 bits of factors. The dead half of the recurrent
state is stale (relative difference 0.57 from the live half, no word equal, not zero) and the window scratch is all
zeros in both keeps.
