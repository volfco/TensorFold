# Cross-context hidden-state probe on GLM-5.3-Flash (session record, 2026-11-01)

Status: research record, single capture, descriptive. Nothing here is a served receipt or a qualified result.
Tags follow the sibling research project's convention: [O] observed, [C] calculated from observed values,
[H] hypothesis, [U] untested. Raw tables, the prompts' token ids and the scripts are in
`docs/receipts/kv-cross-context-probe-2026-11-01/` and `tools/kvprobe/`.

## 1. Questions asked, and the short answers

1. *Can a single token's computation be cached as the difference between the incoming and outgoing hidden state, and
   reused across sessions or prefixes?* No. Past the embedding there is no prefix-independent per-token value, and the
   per-layer difference is the most context-dependent of the three representations (state, update, final row) [O].
2. *Do the Spark nodes allow measuring this?* Yes; the capture ran on the production ring with an existing dump hook
   (section 4) and the per-layer numbers are in section 6.
3. *Is there partial caching: structure the cached value so only a difference is computed, or make a later recompute
   cheaper because half the work was done?* No for this model: the cross-context difference is dense and higher-rank
   than the state it perturbs, so a correction costs a full matmul, and the nonlinearities break linearity anyway. The
   only exact partial cache is the whole prefix state, which production already resumes [O][C].
4. *Other dimensions, drafter caching, hints from cache entries, carrying information between rounds?* Assessed in
   section 8; everything there is [H] or [U] except where a number from section 6 is cited.

## 2. Literature (web search 2026-11-01; only LinearKV was read beyond its abstract)

- EPIC, position-independent caching, ICML 2025: prefill chunks in isolation, splice, recompute a fixed set of tokens
  per chunk. https://arxiv.org/html/2410.15332v1 (slides https://icml.cc/media/icml-2025/Slides/43926.pdf)
- CacheBlend: recompute the 10 to 15 % of tokens with the largest KV deviation.
  https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/
- KVLink, CacheSlide, CacheTune, Cache-Craft, CacheClip, ProphetKV: later variants, summarized by
  https://arxiv.org/pdf/2605.24022 (adaptive KV reuse) and https://arxiv.org/pdf/2606.06256 (RedKnot).
- LinearKV (arXiv 2608.11231, July 2026): position-independent caching for hybrid models (Mamba-2, Gated DeltaNet).
  Each 512-token chunk is prefilled alone and cached as its attention KV plus each recurrent layer's end state. Exactly
  composing recurrent states through the decay transitions is worse than seeding every recurrent layer with the last
  matched chunk's local state and replaying about 20 % of positions in order (Granite Mamba-2: 46.6 % vs 86.8 % of full
  quality under EPIC's selector; GDN models tie). TTFT 0.46 to 0.62x of full prefill at 32k. The only result that
  addresses a KDA-style state. https://arxiv.org/abs/2608.11231
- DeltaKV (residual coding of KV against nearest historical rows, 29 % of original size, same context):
  https://arxiv.org/html/2602.08005v1. C2KV (cross-context reuse with residual mean pooling):
  https://arxiv.org/html/2607.17715v1. Conduit (residual-stream restoration for cached visual tokens):
  https://arxiv.org/html/2609.05821v1. "The Residual Stream Is All You Need" (store hidden rows, recompute K/V):
  https://arxiv.org/html/2603.19664. RelayCaching (decoding KV reuse across agents): https://arxiv.org/pdf/2603.13289.
- Relevance: MLA's 512-wide latent is already the "store the hidden row, recompute K/V at use" form, and this model's
  attention is NoPE (`qk_rope_head_dim` 0), so position re-rotation is not an obstacle; context is.

## 3. Hypotheses and verdicts

| # | Statement | Verdict | Evidence (section 6 unless noted) |
|---|---|---|---|
| H1 | A per-token value past layer 0 exists that does not depend on the prefix | Refuted [O] | Layer 0 output deviates 2.4 % at the median across two 1024-token prefixes, zero bit-identical rows at every layer |
| H2 | Caching the per-layer difference (tap_l minus tap_l-1) is more reusable than the state | Refuted [O] | The update deviates more than the state at nearly every layer (0.34 vs 0.13 at layer 12, 0.72 vs 0.45 at layer 21); only the last few layers invert |
| H3 | The cross-context difference is low-rank or sparse, so a correction is cheaper than recompute | Refuted [O] | Rank 128 of 2048 holds 34 to 38 % of the difference's energy in layers 21 to 33, less than the state's own; top 20 % of coordinates hold 65 %, the dense-Gaussian figure |
| H4 | Cross-prefix reuse with boundary recompute (CacheBlend/EPIC style) is viable on this checkpoint | Not supported [O] | At layer 22, 98 % of chunk rows deviate by more than 0.3 and 53 % by more than 0.5; the body of the chunk, not its boundary, carries the deviation |
| H5 | Deviation is a boundary effect that decays to zero inside the chunk | Refuted [O] | Position curve: layer 21 falls from 1.19 (first 8 tokens) to 0.39 (tokens 1024 to 2048) and plateaus; layer 0 reaches 0.016 |
| H6 | Deviation grows monotonically with depth | Refuted [O] | Plateau of 0.55 to 0.61 through layers 23 to 33, then a fall to 0.41 at layer 44; the last layer's own update deviates only 0.26 |
| H7 | Context-stable tokens are the ones the model is confident about | Refuted [O] | Confident and unsure quartiles deviate alike (0.44 vs 0.47 at layer 21) |
| H8 | Same-token rows across contexts are similar enough for lossless residual coding to pay | Partly [O] | XOR against the other context's row cuts byte-plane entropy 0.67 to 0.42 at layer 0 and 0.53 at layer 9; no gain past layer 23. Hidden rows only, not the stored latent |
| H9 | A different prefix changes the model's prediction on the shared chunk materially | [O] | Greedy pick agrees on 77 % of chunk positions, 61 % in the first 64; teacher-forced on prompt rows |
| H10 | Position shift alone (chunk at 0 vs at 1024) adds deviation | [O] | First 64 tokens: 0.198 at layer 0 vs 0.098 cross-prefix; empty KDA state and conv history at position 0 |
| H11 | Prefix length matters | [O] | 8192-token prefix vs 1024: 0.67 at the plateau vs 0.59; agreement 72 % vs 77 % |
| H12 | The residual stream has outlier channels that distort row metrics | [O] | Flat until layer 42; at layer 44 one channel is the maximum on 97 % of rows and holds 20 % of the energy. Final-normed metrics are the trustworthy last-layer numbers |
| H13 | Hidden-tap results transfer to the MLA latent and the KDA state the engine stores | [U] | The tap is the mean of the 4 hyper-connection streams; the latent is a learned stream combination projected to 512 dims; the KDA state was not captured |
| H14 | Reusing only the first layers (0 to 2, deviation 2 to 8 %) across contexts is an acceptable approximation | [U] | Those layers are about 5 % of per-token work; downstream error not measured |

## 4. Setup and environment

- Model: GLM-5.3-Flash EXL3 4-bit (`neko-legends/GLM-5.3-Flash-Uncensored-EXL3`), 45 layers (34 KDA linear-attention,
  11 DeepSeek sparse attention at layers 3, 7, ..., 43), hidden 4096, kv_lora_rank 512, 64 heads, qk_nope 256,
  qk_rope 0 (NoPE), 288 routed experts top-8 plus one shared, hyper-connections with 4 streams, first 3 MLPs dense,
  indexer top-2048 with 4-token pools.
- Serving stack on the two GB10 nodes (spark0 rank 0, spark1 rank 1, TP=2 over a direct link): the Python/Torch engine
  of the `glm53-tensorfold-spark` project (vendored TensorFold 0.3.4 plus 77 patches, image `glm53-tensorfold:dev`),
  latent fp8 KV, 1M context, batch 4, DFlash2 drafter. This is not the Zig native engine in this repository; the Zig
  engine's `TF_GLM_KV_DUMP` hook (uncommitted, `zig/src/families/glm/cuda_snapshot_codec.zig`) was not usable because
  the ring does not run it.
- Capture hook: the stack's patch 0430 draft dump (`GLM53_TF_DRAFT_DUMP`) writes, per committed prefill row, the token,
  the final-normed row, the mean of the 4 residual streams after each listed layer (`GLM53_TF_DRAFT_DUMP_TAP_LAYERS`),
  and the top-k log-probabilities. All 45 layers were tapped in bf16: 376,842 B per token.
- Probe config (`config/kvprobe.env` on spark0, a copy of the production config with these overrides): CONTEXT and
  `GLM53_TF_KV_POOL_TOKENS` 131072 (memory headroom: production sits at 114 of 121 GB), no drafter, canary and warmups
  off, a separate request log, the dump knobs above with `WHAT=prefill`, `TOPK=1`, `TAPS=bf16`. Production config
  untouched; its container environment saved for comparison.
- Prompts (`tools/kvprobe/make_prompts.py`, token ids via the server's `/tokenize`, sent as flat id lists on
  `/v1/completions` with `max_tokens` 1 and temperature 0): a shared 2048-token prose chunk S from a results document,
  under four contexts: S alone; A+S with a 1024-token README prefix; B+S with a 1024-token shell-script prefix; L+S
  with an 8192-token prefix. The first tokens of the four differ, so no prefix sharing between requests. All four
  prefilled cold (`cached_tokens` 0); prefill took 10.4, 6.5, 5.7 and 17.9 s.
- Dump: 10 segments, 18,432 prefill rows, two shards, 6.95 GB, under the session store on spark0 (root-owned).

## 5. Process, including what went wrong

1. Orientation found no KV dump path in the serving stack's native engine; the Zig hook exists but the ring runs the
   Python stack. Patch 0430's tap dump turned out to be exactly the needed capture, so no engine change was made.
2. Prompts were built against the live tokenizer before any restart.
3. The restart onto the probe config was refused by the Claude Code permission classifier as interfering with a
   running workload. The user ran it by hand (`CONFIG=config/kvprobe.env scripts/serve.sh restart`); ready in 74 s.
   The restore to production also has to be run by hand. The ring stayed on the probe config for the rest of the
   session.
4. The dump landed in a per-host subdirectory of the configured directory (`<dir>/<host>-<rank>-<epoch>/`), which the
   verification step did not expect; the row count was then verified directly from `index.jsonl` (18,432 = 2048 +
   3072 + 3072 + 10240), with no "writer fell behind" line in the rank-0 log.
5. `docker rm -f` sends SIGTERM and Python's `atexit` close does not run; `index.jsonl` and the shards are written as
   segments complete, so the data is complete, but `meta.json` lacks the closing statistics.
6. The analysis ran on spark0's CPU with numpy over memory-mapped shards. A full SVD of each 2048x4096 difference
   matrix was far slower than expected on the Grace cores (over 20 minutes without a first line); Gram-matrix
   eigenvalues replaced it. Two tooling traps: `timeout` on the outer ssh killed the remote job, and `grep | tee` hid
   all progress because grep buffers when its output is a pipe. The rerun used `setsid nohup python3 -u`.
7. Analyses in order: per-layer deviation and entropy (`analyze.py`), greedy-pick agreement (`agree.py`),
   recompute-set fractions (`recompute.py`), rank and sparsity (`lowrank.py`), position curve, confidence and
   frequency dependence, channel concentration (`extra.py`).

## 6. Measurements

Quantities. "State" is the tap: the mean of the 4 hyper-connection streams after a layer, bf16. "Update" is
tap_l minus tap_l-1 (the layer's residual contribution; not available for layer 0, whose input is the embedding).
"Final-normed" is the row the lm_head reads. Relative L2 is norm(x - y) / norm(y) per row with y the second context;
p50 and p90 are over the 2048 rows of S. "Head 64" is the first 64 tokens of S, "body" the rest. Baselines: "other
token" is a random other row of S in the same context, "next token" the neighbouring row. Entropy columns are the
order-0 modeled bytes over raw bytes for the bf16 rows coded as two byte planes (the plain model in
`zig/src/core/kv_codec.zig`): "raw" codes the row alone, "xor" codes the row XOR the same token's row from the other
context, "delta" the int16 difference. Layer type: KDA or DSA (full attention).

### 6.1 Primary pair, A+S vs B+S (same position 1024, different 1024-token prefix) [O]

| layer | type | state p50 | state p90 | update p50 | head 64 p50 | body p50 | cosine p50 | other-token p50 | next-token p50 | raw | xor | delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | KDA | 0.024 | 0.065 | n/a | 0.098 | 0.024 | 1.000 | 0.96 | 0.94 | 0.671 | 0.420 | 0.438 |
| 1 | KDA | 0.078 | 0.178 | 0.073 | 0.285 | 0.077 | 0.997 | 1.22 | 1.11 | 0.667 | 0.523 | 0.534 |
| 2 | KDA | 0.023 | 0.058 | 0.023 | 0.091 | 0.023 | 1.000 | 1.13 | 1.07 | 0.664 | 0.415 | 0.432 |
| 3 | DSA | 0.048 | 0.113 | 0.086 | 0.144 | 0.047 | 0.999 | 1.22 | 1.19 | 0.669 | 0.476 | 0.490 |
| 4 | KDA | 0.049 | 0.117 | 0.099 | 0.154 | 0.048 | 0.999 | 1.22 | 1.19 | 0.670 | 0.479 | 0.493 |
| 5 | KDA | 0.052 | 0.120 | 0.136 | 0.159 | 0.050 | 0.999 | 1.22 | 1.18 | 0.670 | 0.483 | 0.496 |
| 6 | KDA | 0.065 | 0.152 | 0.205 | 0.188 | 0.063 | 0.998 | 1.22 | 1.17 | 0.669 | 0.504 | 0.516 |
| 7 | DSA | 0.068 | 0.157 | 0.130 | 0.193 | 0.066 | 0.998 | 1.22 | 1.18 | 0.669 | 0.508 | 0.519 |
| 8 | KDA | 0.072 | 0.164 | 0.189 | 0.203 | 0.070 | 0.998 | 1.22 | 1.18 | 0.669 | 0.512 | 0.523 |
| 9 | KDA | 0.084 | 0.185 | 0.250 | 0.249 | 0.082 | 0.997 | 1.23 | 1.18 | 0.669 | 0.526 | 0.536 |
| 10 | KDA | 0.111 | 0.229 | 0.245 | 0.304 | 0.109 | 0.994 | 1.25 | 1.17 | 0.669 | 0.549 | 0.557 |
| 11 | DSA | 0.126 | 0.258 | 0.321 | 0.340 | 0.124 | 0.992 | 1.24 | 1.16 | 0.669 | 0.559 | 0.566 |
| 12 | KDA | 0.131 | 0.265 | 0.337 | 0.350 | 0.128 | 0.992 | 1.24 | 1.16 | 0.669 | 0.562 | 0.569 |
| 13 | KDA | 0.142 | 0.285 | 0.357 | 0.369 | 0.139 | 0.990 | 1.25 | 1.16 | 0.669 | 0.568 | 0.574 |
| 14 | KDA | 0.158 | 0.314 | 0.375 | 0.403 | 0.155 | 0.988 | 1.25 | 1.14 | 0.669 | 0.577 | 0.582 |
| 15 | DSA | 0.194 | 0.379 | 0.409 | 0.458 | 0.190 | 0.982 | 1.27 | 1.16 | 0.670 | 0.592 | 0.596 |
| 16 | KDA | 0.214 | 0.417 | 0.449 | 0.487 | 0.210 | 0.978 | 1.27 | 1.15 | 0.670 | 0.598 | 0.602 |
| 17 | KDA | 0.243 | 0.466 | 0.487 | 0.528 | 0.238 | 0.971 | 1.28 | 1.14 | 0.670 | 0.607 | 0.609 |
| 18 | KDA | 0.274 | 0.527 | 0.505 | 0.546 | 0.270 | 0.964 | 1.27 | 1.11 | 0.669 | 0.616 | 0.618 |
| 19 | DSA | 0.346 | 0.655 | 0.632 | 0.669 | 0.340 | 0.941 | 1.29 | 1.12 | 0.668 | 0.630 | 0.632 |
| 20 | KDA | 0.391 | 0.728 | 0.672 | 0.742 | 0.385 | 0.925 | 1.29 | 1.12 | 0.667 | 0.636 | 0.639 |
| 21 | KDA | 0.451 | 0.790 | 0.715 | 0.776 | 0.443 | 0.902 | 1.30 | 1.13 | 0.665 | 0.643 | 0.646 |
| 22 | KDA | 0.514 | 0.862 | 0.711 | 0.817 | 0.507 | 0.872 | 1.30 | 1.10 | 0.665 | 0.652 | 0.654 |
| 23 | DSA | 0.584 | 0.924 | 0.767 | 0.901 | 0.577 | 0.834 | 1.31 | 1.13 | 0.667 | 0.662 | 0.661 |
| 24 | KDA | 0.586 | 0.900 | 0.753 | 0.878 | 0.579 | 0.834 | 1.30 | 1.11 | 0.668 | 0.663 | 0.661 |
| 25 | KDA | 0.573 | 0.875 | 0.701 | 0.876 | 0.567 | 0.842 | 1.30 | 1.08 | 0.667 | 0.662 | 0.659 |
| 26 | KDA | 0.571 | 0.849 | 0.746 | 0.845 | 0.566 | 0.844 | 1.29 | 1.06 | 0.668 | 0.662 | 0.659 |
| 27 | DSA | 0.611 | 0.879 | 0.819 | 0.868 | 0.606 | 0.830 | 1.29 | 1.04 | 0.667 | 0.665 | 0.661 |
| 28 | KDA | 0.586 | 0.860 | 0.723 | 0.847 | 0.581 | 0.842 | 1.28 | 1.04 | 0.665 | 0.662 | 0.659 |
| 29 | KDA | 0.559 | 0.844 | 0.687 | 0.828 | 0.553 | 0.857 | 1.27 | 1.02 | 0.664 | 0.658 | 0.657 |
| 30 | KDA | 0.554 | 0.830 | 0.621 | 0.817 | 0.549 | 0.858 | 1.27 | 1.02 | 0.663 | 0.656 | 0.657 |
| 31 | DSA | 0.585 | 0.855 | 0.726 | 0.840 | 0.582 | 0.842 | 1.27 | 1.03 | 0.663 | 0.658 | 0.659 |
| 32 | KDA | 0.579 | 0.855 | 0.650 | 0.838 | 0.574 | 0.844 | 1.28 | 1.06 | 0.664 | 0.658 | 0.659 |
| 33 | KDA | 0.565 | 0.847 | 0.655 | 0.830 | 0.561 | 0.850 | 1.27 | 1.04 | 0.666 | 0.657 | 0.659 |
| 34 | KDA | 0.450 | 0.698 | 0.396 | 0.667 | 0.444 | 0.906 | 1.15 | 0.91 | 0.669 | 0.644 | 0.645 |
| 35 | DSA | 0.490 | 0.744 | 0.724 | 0.712 | 0.486 | 0.891 | 1.14 | 0.91 | 0.668 | 0.647 | 0.649 |
| 36 | KDA | 0.446 | 0.680 | 0.404 | 0.654 | 0.443 | 0.908 | 1.16 | 0.95 | 0.666 | 0.641 | 0.644 |
| 37 | KDA | 0.453 | 0.695 | 0.705 | 0.658 | 0.449 | 0.905 | 1.16 | 0.96 | 0.666 | 0.642 | 0.645 |
| 38 | KDA | 0.451 | 0.693 | 0.628 | 0.659 | 0.446 | 0.905 | 1.16 | 0.95 | 0.665 | 0.642 | 0.645 |
| 39 | DSA | 0.490 | 0.727 | 0.754 | 0.705 | 0.485 | 0.889 | 1.17 | 0.96 | 0.664 | 0.646 | 0.650 |
| 40 | KDA | 0.492 | 0.737 | 0.613 | 0.707 | 0.487 | 0.887 | 1.18 | 0.96 | 0.664 | 0.646 | 0.650 |
| 41 | KDA | 0.488 | 0.715 | 0.599 | 0.689 | 0.485 | 0.887 | 1.20 | 0.96 | 0.663 | 0.645 | 0.650 |
| 42 | KDA | 0.480 | 0.660 | 0.544 | 0.702 | 0.478 | 0.889 | 1.17 | 0.89 | 0.662 | 0.644 | 0.648 |
| 43 | DSA | 0.493 | 0.662 | 0.558 | 0.709 | 0.489 | 0.883 | 1.18 | 0.93 | 0.662 | 0.647 | 0.651 |
| 44 | KDA | 0.410 | 0.547 | 0.262 | 0.575 | 0.406 | 0.919 | 1.08 | 0.87 | 0.667 | 0.648 | 0.648 |

### 6.2 S vs A+S (chunk at position 0 with no prefix, vs at 1024) [O]

| layer | state p50 | state p90 | update p50 | head 64 p50 | body p50 | cosine p50 |
|---|---|---|---|---|---|---|
| 0 | 0.030 | 0.092 | n/a | 0.198 | 0.029 | 1.000 |
| 1 | 0.077 | 0.196 | 0.071 | 0.370 | 0.076 | 0.997 |
| 3 | 0.054 | 0.133 | 0.094 | 0.236 | 0.052 | 0.999 |
| 6 | 0.071 | 0.181 | 0.210 | 0.314 | 0.070 | 0.998 |
| 9 | 0.087 | 0.212 | 0.239 | 0.365 | 0.085 | 0.996 |
| 12 | 0.140 | 0.286 | 0.374 | 0.447 | 0.138 | 0.990 |
| 15 | 0.234 | 0.420 | 0.544 | 0.619 | 0.231 | 0.973 |
| 18 | 0.318 | 0.566 | 0.571 | 0.730 | 0.314 | 0.949 |
| 21 | 0.492 | 0.836 | 0.764 | 0.937 | 0.485 | 0.875 |
| 24 | 0.608 | 0.916 | 0.776 | 1.015 | 0.603 | 0.802 |
| 27 | 0.608 | 0.888 | 0.772 | 0.990 | 0.602 | 0.804 |
| 30 | 0.553 | 0.843 | 0.620 | 0.924 | 0.549 | 0.839 |
| 33 | 0.578 | 0.869 | 0.674 | 0.944 | 0.572 | 0.824 |
| 36 | 0.452 | 0.672 | 0.404 | 0.777 | 0.448 | 0.893 |
| 39 | 0.482 | 0.707 | 0.649 | 0.800 | 0.478 | 0.878 |
| 42 | 0.473 | 0.650 | 0.537 | 0.746 | 0.468 | 0.885 |
| 44 | 0.404 | 0.551 | 0.266 | 0.629 | 0.401 | 0.918 |

### 6.3 L+S vs A+S (8192-token prefix vs 1024) [O]

| layer | state p50 | state p90 | update p50 | head 64 p50 | body p50 | cosine p50 |
|---|---|---|---|---|---|---|
| 0 | 0.034 | 0.086 | n/a | 0.115 | 0.033 | 0.999 |
| 1 | 0.112 | 0.224 | 0.102 | 0.341 | 0.110 | 0.994 |
| 3 | 0.084 | 0.168 | 0.149 | 0.189 | 0.083 | 0.997 |
| 6 | 0.110 | 0.215 | 0.331 | 0.234 | 0.108 | 0.994 |
| 9 | 0.130 | 0.248 | 0.334 | 0.283 | 0.127 | 0.992 |
| 12 | 0.188 | 0.335 | 0.455 | 0.391 | 0.185 | 0.983 |
| 15 | 0.273 | 0.473 | 0.582 | 0.502 | 0.269 | 0.963 |
| 18 | 0.362 | 0.637 | 0.635 | 0.648 | 0.357 | 0.934 |
| 21 | 0.543 | 0.873 | 0.804 | 0.827 | 0.537 | 0.849 |
| 24 | 0.670 | 0.982 | 0.850 | 0.940 | 0.665 | 0.772 |
| 27 | 0.672 | 0.983 | 0.841 | 0.948 | 0.665 | 0.771 |
| 30 | 0.613 | 0.934 | 0.684 | 0.905 | 0.606 | 0.811 |
| 33 | 0.622 | 0.943 | 0.723 | 0.934 | 0.616 | 0.805 |
| 36 | 0.487 | 0.765 | 0.428 | 0.760 | 0.482 | 0.879 |
| 39 | 0.527 | 0.812 | 0.730 | 0.783 | 0.521 | 0.860 |
| 42 | 0.532 | 0.742 | 0.608 | 0.777 | 0.527 | 0.857 |
| 44 | 0.445 | 0.589 | 0.278 | 0.607 | 0.442 | 0.901 |

### 6.4 Final-normed row and greedy picks [O]

| pair | rel L2 p50 | rel L2 p90 | cosine p50 |
|---|---|---|---|
| A+S vs B+S | 0.452 | 0.597 | 0.898 |
| S vs A+S | 0.461 | 0.635 | 0.895 |
| L+S vs A+S | 0.505 | 0.670 | 0.873 |

```
A+S vs B+S: greedy pick agrees on 77.0% of S rows (head64 60.9%, body 77.5%); mean |dlogp| of top-1 0.222 nats
S vs A+S: greedy pick agrees on 75.0% of S rows (head64 59.4%, body 75.6%); mean |dlogp| of top-1 0.235 nats
L+S vs A+S: greedy pick agrees on 72.4% of S rows (head64 57.8%, body 72.8%); mean |dlogp| of top-1 0.271 nats
S: top-1 equals the actual next token on 48.1% of S rows, mean top-1 logp -0.910
A+S: top-1 equals the actual next token on 49.0% of S rows, mean top-1 logp -0.894
B+S: top-1 equals the actual next token on 48.2% of S rows, mean top-1 logp -0.940
L+S: top-1 equals the actual next token on 50.2% of S rows, mean top-1 logp -0.830
```

Recompute set (rows whose state deviates beyond a threshold, A+S vs B+S) [O]: layer 22: 52.8 % over 0.5, 98.0 % over
0.3, 100 % over 0.2; layer 44: 17.5 % over 0.5, 91.0 % over 0.3, 99.3 % over 0.2.

### 6.5 Position within S, confidence and token frequency (A+S vs B+S, state rel L2 p50) [O]

| layer | 0-8 | 8-32 | 32-128 | 128-512 | 512-1024 | 1024-2048 | confident quartile | unsure quartile | frequent tokens | singleton tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.219 | 0.120 | 0.074 | 0.046 | 0.028 | 0.016 | 0.024 | 0.031 | 0.034 | 0.020 |
| 3 | 0.269 | 0.174 | 0.110 | 0.075 | 0.050 | 0.035 | 0.050 | 0.052 | 0.052 | 0.048 |
| 9 | 0.519 | 0.290 | 0.170 | 0.124 | 0.090 | 0.066 | 0.086 | 0.091 | 0.092 | 0.085 |
| 15 | 0.888 | 0.538 | 0.375 | 0.262 | 0.198 | 0.161 | 0.192 | 0.211 | 0.222 | 0.188 |
| 21 | 1.192 | 0.897 | 0.675 | 0.582 | 0.440 | 0.388 | 0.441 | 0.470 | 0.527 | 0.399 |
| 27 | 1.129 | 0.919 | 0.815 | 0.704 | 0.611 | 0.555 | 0.602 | 0.623 | 0.660 | 0.556 |
| 33 | 1.238 | 0.908 | 0.731 | 0.635 | 0.559 | 0.522 | 0.570 | 0.580 | 0.610 | 0.527 |
| 39 | 1.130 | 0.773 | 0.650 | 0.549 | 0.479 | 0.456 | 0.494 | 0.504 | 0.536 | 0.463 |
| 44 | 1.166 | 0.584 | 0.534 | 0.445 | 0.407 | 0.388 | 0.411 | 0.408 | 0.386 | 0.416 |

Confidence quartiles are by the log-probability of the greedy pick in A+S. Frequent tokens are the top quartile by
count within S; singletons occur once.

### 6.6 Rank and sparsity of the difference D = tap(A+S) minus tap(B+S) over the 2048 rows [O]

Energy fractions from the Gram matrix's eigenvalues (float64). Layer 44 is omitted from interpretation: its rows are
dominated by one channel (6.7), so rank 8 captures everything.

| layer | energy of D in rank 8 / 32 / 128 / 512 | same for the state A+S | energy in top 5 % / 20 % of coordinates | update difference, rank 32 / 128 |
|---|---|---|---|---|
| 0 | 0.54/0.73/0.88/0.96 | 0.53/0.64/0.81/0.98 | 0.29/0.66 |  |
| 1 | 0.62/0.81/0.94/0.99 | 0.38/0.53/0.72/0.93 | 0.28/0.65 | 0.79/0.92 |
| 3 | 0.26/0.46/0.71/0.93 | 0.34/0.52/0.80/0.98 | 0.28/0.65 | 0.44/0.70 |
| 6 | 0.21/0.39/0.63/0.88 | 0.33/0.50/0.76/0.97 | 0.28/0.65 | 0.36/0.60 |
| 9 | 0.19/0.36/0.59/0.86 | 0.31/0.47/0.74/0.96 | 0.28/0.65 | 0.27/0.51 |
| 12 | 0.15/0.30/0.54/0.83 | 0.30/0.45/0.69/0.92 | 0.28/0.65 | 0.22/0.46 |
| 15 | 0.11/0.24/0.47/0.80 | 0.27/0.41/0.65/0.90 | 0.28/0.65 | 0.20/0.44 |
| 18 | 0.12/0.24/0.46/0.78 | 0.26/0.40/0.62/0.88 | 0.28/0.65 | 0.23/0.44 |
| 21 | 0.07/0.17/0.38/0.72 | 0.23/0.35/0.56/0.83 | 0.28/0.65 | 0.12/0.31 |
| 24 | 0.07/0.16/0.34/0.69 | 0.22/0.34/0.53/0.80 | 0.28/0.65 | 0.14/0.33 |
| 27 | 0.09/0.18/0.37/0.71 | 0.27/0.40/0.58/0.83 | 0.28/0.65 | 0.23/0.45 |
| 30 | 0.10/0.18/0.37/0.70 | 0.30/0.42/0.61/0.84 | 0.28/0.65 | 0.21/0.38 |
| 33 | 0.11/0.20/0.38/0.70 | 0.30/0.42/0.60/0.84 | 0.28/0.65 | 0.24/0.40 |
| 36 | 0.16/0.26/0.45/0.75 | 0.45/0.58/0.74/0.91 | 0.28/0.65 | 0.26/0.45 |
| 39 | 0.17/0.27/0.46/0.75 | 0.44/0.56/0.72/0.89 | 0.28/0.65 | 0.46/0.63 |
| 42 | 0.22/0.33/0.51/0.78 | 0.45/0.58/0.73/0.90 | 0.28/0.65 | 0.31/0.50 |
| 44 | 1.00/1.00/1.00/1.00 | 0.91/0.94/0.96/0.99 | 0.31/0.67 | 1.00/1.00 |

### 6.7 Channel concentration of the state (A+S) [O]

| layer | top 1 / 8 / 64 channels' share of row energy | dominant channel | rows where it is the max | mean abs there | row rms |
|---|---|---|---|---|---|
| 0 | 0.011 / 0.054 / 0.165 | 2500 | 71 % | 0.0 | 0.00 |
| 3 | 0.004 / 0.023 / 0.120 | 2551 | 7 % | 0.1 | 0.03 |
| 9 | 0.004 / 0.022 / 0.120 | 925 | 4 % | 0.0 | 0.02 |
| 21 | 0.004 / 0.022 / 0.119 | 2162 | 1 % | 0.1 | 0.05 |
| 33 | 0.004 / 0.022 / 0.119 | 378 | 1 % | 0.4 | 0.31 |
| 42 | 0.005 / 0.025 / 0.123 | 378 | 38 % | 2.9 | 0.84 |
| 43 | 0.008 / 0.033 / 0.134 | 2520 | 34 % | 4.2 | 4.10 |
| 44 | 0.201 / 0.239 / 0.356 | 1448 | 97 % | 41.5 | 2.83 |

### 6.8 Costs observed [O]

Cold prefill of 3072 tokens took 5.7 to 6.5 s on the ring under the probe config (about 500 tokens/s including the
one-token decode), 10240 tokens 17.9 s. Per token and layer the matmuls on the hidden state dominate: about 150 M
multiply-adds in the attention or KDA projections and about 225 M in the eight routed experts plus the shared expert
[C from the config]; the context mixing at a 3k context is about 25 M [C]. The production request log on spark0 shows
exact prefix resumes of tens of thousands of tokens (one entry: 68,687 of 68,687).

## 7. Caveats

- One prose chunk, one checkpoint, one capture per cell; no replication, no confidence intervals.
- Hidden taps, not the stored quantities (H13). The entropy figures are for 4096-wide bf16 hidden rows; production
  stores 512-wide fp8 latents, the Zig engine bf16 latents.
- Greedy agreement is teacher-forced on prompt rows, not downstream answer quality, and a different prefix is
  supposed to change predictions; the 23 % disagreement is the context signal a cross-prefix reuse would discard.
- The DSA indexer has `indexer_rope_interleave` true, so "NoPE" holds for attention keys and values, not necessarily
  for the indexer.
- The serving stack's prefill modes (fast, lean, row-split, pipelined) claim bit-equality with the exact path; the
  taps come from whichever ran. All four requests ran the same mode.

## 8. Assessment of the follow-up questions (all [H] or [U] unless a section 6 number is cited)

Open dimensions worth measuring next:
- The stored quantities under the same four contexts: MLA latent rows and the KDA state per head (spectrum, effective
  rank, per-head decay horizon from the gates). The Zig engine's `TF_GLM_KV_DUMP` captures exactly these classes
  (latent, scale, ring, pooled, prefix arena with the KDA state) and `tf-kv-codec-bench --file` screens them; this
  needs the Zig engine running on a GB10, which means stopping the production ring.
- Content type: code, JSON tool output, chat-template boilerplate, which agent traffic is mostly made of.
- Expert routing agreement across contexts, and indexer selection stability across contexts and across decode steps.

Drafter caching: the drafter's inputs are the taps the target just computed, so a hidden-state cache does not help
it. What does: token-record drafting from cached text (the Zig engine's Recall, `docs/recall.md`, already does this
across sessions; the 77 % greedy agreement is a rough acceptance floor for repeated prose), prior completions as
drafts on regenerate or relay, and recording real traffic (patch 0430's purpose) to refit the drafter on the served
distribution. Depth adaptation from acceptance statistics already exists in the serving stack.

Hints from cache entries (post-processable by an external agent): per-page bounds on the pooled indexer keys for an
exact prune of the index scan (Quest-style upper bounds; the Zig pages already hold the pool-key rows); per-token
selection counts to demote never-selected latent rows to host memory; selection-set reuse across consecutive decode
steps (approximate, needs the quality gate); a token-keyed expert prior to prefetch expert tiles before the router
runs (the serving stack's patch 0460 measured +2.4 % single-stream from in-layer L2 prefetch); the last round's touched
pages as a restore order for a resumed session; compaction and demotion between turns rather than during prefill.

Continuous sessions: nothing in a previous round's hidden states makes the next round's prefill of new tokens
cheaper; exact prefix resume is the whole gain and is in production.

Prior art to build on: `docs/kv-cache-compression-plan.md` and `docs/atlas-kv-cache-compression.md` in this repo; the
sibling research project in `~/Workspace/mindreader` (not in this repo): `RESEARCH.md` Possibilities 2 to 5 (drafter
hints, continuation retrieval, prefix retention, expert prefetch, with the unified-memory caveat that there is no
host-to-GPU transfer to hide on a Spark), `FURTHER_RESEARCH.md` R3 (predict FFN weight blocks), R5 (selective prefill
repair), R10 (restore KV from saved layer inputs), R11 (reuse sparse-attention selection work), reports 03 (cache and
restoration), 09 (hybrid state: rollback vs checkpoint vs bounded replay), 10 (sparse attention, oracle-first), 17
(long context), and 89 (KV-cache compressibility of the served fp8 cache: MLA latent R 1.219 under zstd-3 with the gain
in the exponent field, KDA state gain in the high byte plane), whose evidence discipline (`AGENTS.md` there) this
record follows.

## 9. Artifacts

- spark0: the dump under the session store directory `kvprobe-20261101/<host>-1-<epoch>/` (root-owned, 6.95 GB);
  `~/kvprobe/` with the prompts, scripts, `result.json`, `result.md`, `summary.md`, `agree.txt`, `lowrank.txt`,
  `extra.txt`, the saved production container environment, the probe restart log; `config/kvprobe.env` in the serving
  project; the probe request log `kvprobe-requests.jsonl` in the session store.
- This repo (uncommitted): this file; `docs/receipts/kv-cross-context-probe-2026-11-01/` (raw tables, `prompts.json`
  with the token ids); `tools/kvprobe/` (the scripts; `lowrank.py`, `extra.py`, `recompute.py` reuse `analyze.py`'s
  loaders by executing its top half, a shortcut worth replacing with a module).
- Reproduction: start the ring on a config with the dump knobs of section 4; `python3 tools/kvprobe/make_prompts.py
  <text root> prompts.json`; `python3 tools/kvprobe/send_probe.py prompts.json <dump dir>/index.jsonl`; then
  `analyze.py <dump dir> prompts.json result.json result.md`, `agree.py`, `recompute.py`, `lowrank.py`, `extra.py` with
  `<dump dir> prompts.json`. Run long analyses detached (`setsid nohup python3 -u`), never through `grep | tee`.

## 10. Status

The ring stayed on the probe config (131072-token context, no drafter, every prefill tapped) for about 2.5 hours
after the capture, then the user restored production by hand (`scripts/serve.sh restart` with no `CONFIG`): ready in
100 s, canary passed (3 probes, 5.71 tokens a round, 70 tok/s decode), 1M context advertised. The restored rank-0
container environment equals the saved pre-probe one in every key except the launch timestamp and
`GLM53_TF_SESSION_DISK_GIB` (128 before, 380 after), which comes from the production config file as it stood at
restart, not from the probe.


## 11. GPU runs on the Spark ring, 2026-11-01 evening (deployment of the follow-up tooling)

Scope: the tooling from `tools/kvprobe/STATUS.md` (the four tracks) built and run on the two GB10 nodes with the
user's authorization to stop and restart the production ring. Everything below is [O] unless marked.

### 11.1 Deployment

- The working tree (this repo, including the uncommitted codec and the agent's changes) was synced to spark0 and built
  there with the Zig 0.17.0 aarch64 toolchain and flags from the October kernel harness (`zig build native
  kv-codec-bench kernel-contract-fatbins glm-expert-prefetch-bench install -Doptimize=ReleaseFast -Dnvcc=/usr/local/
  cuda/bin/nvcc -Dsm=121 -j2`, about 10 minutes under a 14 GiB cgroup cap while production still served). The built
  `zig-out` was copied to spark1 (same sha256 on both nodes). The two Track 4 kernels were compiled by hand with nvcc
  13.0 for sm_121: no spills (ring down d2/d3/d4 96/124/128 registers, gate-up d4 165).
- The Zig engine served the Mia-AiLab EXL3 4bpw checkpoint from the NFS hub at TP=2 over the direct link (the recipe's
  NVFP4 checkpoint is not on the nodes), `--context 32768 --parallel 1 --prompt-cache-gib 16 --no-drafts`, with
  `TF_GLM_HOST_CACHE=1 TF_GLM_KV_CODEC=1 TF_GLM_KV_LEDGER=1 TF_GLM_KV_DUMP=<dir>`. Each rank loaded in 228 s (82.65
  GiB of weights). Launch scripts: `~/kvprobe/native/` on spark0 (`launch.sh`, `start-ranks.sh`, `stop-ranks.sh`).
- The serving-stack probe image with patch 0710 (`glm53-tensorfold:route`) was built from a copy of the serving
  project and shipped to spark1 without touching the production image.

### 11.2 Pre-engine GPU checks

- `tf-cuda-test glm-exl3`: PASS (decode and prompt slot rows within 0.186 of their float64 tolerance).
- `tf-cuda-test kernel-contract-live`: 168 problems over 1378 registered launches, in the families qwen35_kv8 (7),
  qwen35 (5), nvfp4_experts_tile (2), qwen35_keep (1). The October baseline binary (1.1.9, 1344 launches) reports the
  same 168 problems in the same families, so none is new; the 34 added launches (the pending prefetch and ring
  records) pass the live limits.
- `kernel-contract-limits` over the built fatbins plus the two hand-built ones: many problems across every family in
  the way the directory was assembled (symlinked fatbins); not a usable signal and not compared to a baseline.
- Track 4 bench (`tf-glm-expert-prefetch-bench`, dims 4096, width 1024 per rank, 288 experts, 7 reps with the first
  discarded), synthetic weights [O]:

| variant | rows 1 median ms | rows 4 | rows 8 | bits |
|---|---|---|---|---|
| stock | 0.3195 | 1.0056 | 1.8882 | SAME |
| prefetch-lines | 0.3164 | 1.0265 | 1.9108 | SAME |
| prefetch-bulk | 0.3185 | 1.0015 | 1.9551 | SAME |
| ring2 | 0.3236 | 1.0047 | 1.8872 | SAME |
| ring3 | 0.3092 | 0.9912 | 1.9261 | SAME |
| ring4 | 0.3154 | 0.9912 | 1.9272 | SAME |
| ring3+prefetch-bulk | 0.3092 | 0.9892 | 1.9735 | SAME |

  Every variant reproduces the stock kernel's output hash at every row count, which is the bit-equality assertion the
  prototypes make. Differences in time are within run-to-run noise (ring3 is about 3 % under stock at rows 1 and 4
  and about 2 % over it at rows 8); no speed claim follows, as the track's own gate requires a served A/B.

### 11.3 Track 1: the stored quantities across contexts (H13)

16 cold prefills (4 content types x S, A+S, B+S, L+S; prompts of 6.3k to 14.8k tokens, S of 1739 to 2075 tokens at
the same offset modulo the 2048-row grid), 16 keeps per rank with `.bin`, `.parts` and the new `.plan` manifest
beside each, 4.5 GB a rank, `KVLEDGER` lines for every keep. Ledger totals on rank 0:

| class | bytes over 16 keeps | share | order-0 modeled | ratio |
|---|---|---|---|---|
| prefix (KDA states, conv history, arena head) | 2,610,089,984 | 55.0 % | 2,154,545,514 | 0.826 |
| latent (MLA rows, 11 layers) | 1,660,944,384 | 35.0 % | 1,107,528,966 | 0.667 |
| ring (decode rings) | 369,098,752 | 7.8 % | 247,660,798 | 0.671 |
| pooled (DSA pool keys) | 103,899,136 | 2.2 % | 69,080,000 | 0.665 |
| tail | 135,168 | 0.0 % | 89,742 | 0.664 |
| total | 4,744,167,424 | | 3,578,905,020 | 0.754 |

The prefix class (the KDA states and conv history) is 55 % of a kept entry on this checkpoint at these lengths, the
MLA latent 35 %; the order-0 byte-plane model (the codec's plain mode) prices the entry at 0.754 of raw, with the
latent at 0.667.

Rank 1's analysis (latent rows are replicated on both ranks; KDA heads are split, so these are rank 1's 32 heads a
layer). A+S vs B+S, same position, different 2k-token prefix, p50 over S's rows or heads:

| DSA layer | prose rel L2 / cos | code rel L2 / cos | json rel L2 / cos | boilerplate rel L2 / cos | prose xor / raw |
|---|---|---|---|---|---|
| 3 | 0.09 / 1.00 | 0.11 / 0.99 | 0.12 / 0.99 | 0.09 / 1.00 | 0.53 / 0.67 |
| 7 | 0.13 / 0.99 | 0.13 / 0.99 | 0.14 / 0.99 | 0.14 / 0.99 | 0.56 / 0.67 |
| 11 | 0.14 / 0.99 | 0.15 / 0.99 | 0.21 / 0.98 | 0.20 / 0.98 | 0.57 / 0.66 |
| 15 | 0.22 / 0.98 | 0.26 / 0.97 | 0.38 / 0.93 | 0.38 / 0.93 | 0.60 / 0.66 |
| 19 | 0.34 / 0.94 | 0.40 / 0.92 | 0.65 / 0.79 | 0.64 / 0.80 | 0.63 / 0.66 |
| 23 | 0.38 / 0.93 | 0.45 / 0.90 | 0.71 / 0.74 | 0.71 / 0.75 | 0.64 / 0.66 |
| 27 | 0.37 / 0.93 | 0.46 / 0.89 | 0.72 / 0.74 | 0.71 / 0.75 | 0.63 / 0.66 |
| 31 | 0.28 / 0.96 | 0.35 / 0.94 | 0.51 / 0.87 | 0.48 / 0.89 | 0.61 / 0.66 |
| 35 | 0.27 / 0.96 | 0.37 / 0.93 | 0.54 / 0.86 | 0.51 / 0.87 | 0.61 / 0.66 |
| 39 | 0.29 / 0.96 | 0.40 / 0.92 | 0.58 / 0.83 | 0.55 / 0.85 | 0.62 / 0.66 |
| 43 | 0.34 / 0.94 | 0.42 / 0.91 | 0.66 / 0.78 | 0.53 / 0.86 | 0.63 / 0.66 |

| KDA layer | prose rel / eff. rank / rank90 | code rel / eff. rank / rank90 | json rel / eff. rank / rank90 | boilerplate rel / eff. rank / rank90 |
|---|---|---|---|---|
| 1 | 0.01 / 2.8 / 3 | 0.01 / 3.2 / 3 | 0.01 / 3.0 / 3 | 0.01 / 3.1 / 3 |
| 5 | 0.07 / 1.8 / 2 | 0.08 / 1.9 / 2 | 0.04 / 2.2 / 2 | 0.07 / 2.2 / 2 |
| 9 | 0.08 / 3.4 / 4 | 0.13 / 3.5 / 4 | 0.08 / 3.9 / 4 | 0.09 / 3.6 / 4 |
| 13 | 0.14 / 3.9 / 4 | 0.18 / 4.1 / 6 | 0.15 / 4.5 / 6 | 0.15 / 5.1 / 7 |
| 17 | 0.22 / 4.2 / 5 | 0.24 / 4.8 / 5 | 0.19 / 4.4 / 6 | 0.23 / 5.7 / 7 |
| 21 | 0.31 / 7.0 / 10 | 0.33 / 6.9 / 10 | 0.31 / 7.0 / 10 | 0.40 / 8.1 / 11 |
| 25 | 0.28 / 6.3 / 8 | 0.25 / 7.1 / 10 | 0.25 / 6.7 / 10 | 0.36 / 8.4 / 11 |
| 29 | 0.29 / 5.8 / 7 | 0.24 / 5.7 / 8 | 0.23 / 5.4 / 8 | 0.33 / 6.3 / 9 |
| 33 | 0.24 / 6.7 / 8 | 0.22 / 6.7 / 8 | 0.22 / 6.5 / 8 | 0.31 / 6.9 / 10 |
| 37 | 0.22 / 5.4 / 6 | 0.20 / 5.1 / 6 | 0.21 / 5.6 / 6 | 0.29 / 5.7 / 6 |
| 41 | 0.30 / 8.6 / 12 | 0.29 / 8.2 / 12 | 0.29 / 8.1 / 12 | 0.36 / 8.3 / 11 |
| 44 | 0.22 / 5.4 / 6 | 0.20 / 5.3 / 6 | 0.20 / 5.8 / 7 | 0.26 / 5.9 / 8 |

The other pairs, and the short-conv history:

| type | pair | latent rel L2 p50 at layers 3 / 23 / 43 | KDA rel p50 at layers 1 / 21 / 44 | conv rel p50 at 1 / 21 / 44 |
|---|---|---|---|---|
| prose | A+S vs B+S | 0.09 / 0.38 / 0.34 | 0.01 / 0.31 / 0.22 | 0.004 / 0.254 / 0.219 |
| prose | S vs A+S | 0.10 / 0.38 / 0.34 | 0.01 / 0.30 / 0.20 | 0.003 / 0.246 / 0.190 |
| prose | L+S vs A+S | 0.10 / 0.43 / 0.36 | 0.01 / 0.33 / 0.24 | 0.004 / 0.267 / 0.200 |
| code | A+S vs B+S | 0.11 / 0.45 / 0.42 | 0.01 / 0.33 / 0.20 | 0.006 / 0.252 / 0.166 |
| code | S vs A+S | 0.14 / 0.40 / 0.37 | 0.01 / 0.33 / 0.19 | 0.004 / 0.234 / 0.160 |
| code | L+S vs A+S | 0.09 / 0.36 / 0.33 | 0.01 / 0.35 / 0.19 | 0.007 / 0.286 / 0.167 |
| json | A+S vs B+S | 0.12 / 0.71 / 0.66 | 0.01 / 0.31 / 0.20 | 0.004 / 0.352 / 0.131 |
| json | S vs A+S | 0.14 / 0.65 / 0.58 | 0.01 / 0.28 / 0.18 | 0.004 / 0.272 / 0.137 |
| json | L+S vs A+S | 0.09 / 0.64 / 0.60 | 0.01 / 0.31 / 0.21 | 0.005 / 0.331 / 0.145 |
| boilerplate | A+S vs B+S | 0.09 / 0.71 / 0.53 | 0.01 / 0.40 / 0.26 | 0.005 / 0.310 / 0.188 |
| boilerplate | S vs A+S | 0.12 / 0.65 / 0.50 | 0.01 / 0.36 / 0.30 | 0.003 / 0.245 / 0.153 |
| boilerplate | L+S vs A+S | 0.08 / 0.65 / 0.50 | 0.01 / 0.39 / 0.22 | 0.005 / 0.253 / 0.183 |

Readings [C]:
- The MLA latent is more context-stable than the hidden taps were: prose plateaus at 0.38 (taps 0.6), code at 0.46.
  JSON tool output and chat boilerplate deviate the most (0.71), the opposite of the expectation that structured text
  is token-identity-driven. The XOR-against-other-context entropy gain on latent rows is 21 % at layer 3 and gone by
  layer 19, as on the taps.
- The KDA recurrent state after a keep 4.2k tokens past S deviates far less than the taps: 0.01 at layer 1, 0.2 to
  0.4 deep. Position shift and prefix length barely move it (0.01 at layer 1 in every pair).
- The KDA state per head has an effective rank of 2 to 12 of 128 (rank for 90 % of energy at most 14) in every layer
  and content type, and the cross-context difference is itself low-rank (59 to 99 % of its energy in rank 8). This is
  the largest lever the probe found for the entry's biggest class, and it is a lossy one: a rank-r factorization of
  each head's 128x128 state would need the plan's Phase 3 quality gate. Whether the rank stays this low at much
  longer positions is [U].
- The conv history deviates 0.004 at layer 1 and 0.13 to 0.35 deep.

Rank 0's analysis (its 32 heads) agrees: KDA state deviation 0.01 at layer 1 and 0.21 to 0.36 at layers 21 and 44,
effective rank 1.9 to 9.7 with a median of 4.2 to 5.1 per type, rank for 90 % of energy at most 14 [O].

Codec bench on the 16 rank-0 keeps (`tf-kv-codec-bench --file`, 259 to 358 MB each, 4.74 GB in all; ratio = stored /
raw, bit-exact round trips) [O]:

| codec | ratio min / median / max | encode GB/s (median) | decode GB/s (median) |
|---|---|---|---|
| plain (byte planes + rANS), 1 thread | 0.707 / 0.718 / 0.724 | 0.32 | 1.21 |
| plain, 8 threads | same | 1.74 | 6.37 |
| atlas (column codecs by model), 8 threads | 0.706 / 0.717 / 0.723 | 0.52 | 6.11 |
| atlas-exhaustive, 8 threads | same as atlas | 0.15 | 6.17 |
| deflate-6 | 0.815 / 0.819 / 0.821 | 0.06 | 0.18 |
| ClickHouse ALP-RD | 0.814 / 0.833 / 0.841 | 0.19 | 2.04 |
| ClickHouse Gorilla / DoubleDelta / FPC | 0.94 | | |

The plan's Phase 1 acceptance (at least three captured shapes, actual serialized sizes, round trips) is met: a real
kept state compacts to 0.72 of raw with the plain codec, the Atlas column codecs gain 0.001 of ratio (a tenth of a percentage point) at 3x the encode
cost, and deflate reaches only 0.82. On the Grace cores a 1 GiB state keeps in about 0.6 s and restores in 0.16 s at
8 threads.

### 11.4 Track 3: Recall on the Zig engine (proxy)

Three distinct chat prompts, each sent twice at temperature 0 with `--recall` on an empty record and drafts on. The
second reply of each prompt is byte-identical to the first (same token sha256) and takes a third of the time:

| prompt | reply | seconds | rounds | accepted / drafted | tokens sha256 (first 12) |
|---|---|---|---|---|---|
| 0 | 1 | 26.28 | 148 | 172/254 | 39cc098df2f1 |
| 0 | 2 | 9.86 | 41 | 278/278 | 39cc098df2f1 |
| 1 | 1 | 22.48 | 119 | 200/274 | 3b871668f3b0 |
| 1 | 2 | 10.27 | 42 | 277/282 | 3b871668f3b0 |
| 2 | 1 | 27.56 | 155 | 164/274 | bc433a0af5f0 |
| 2 | 2 | 10.19 | 42 | 277/280 | bc433a0af5f0 |

The record's counters after the run: 1269 ids, 120 copy proposals, 811 tokens proposed, 811 accepted; the 16-token band's landing rate 0.887. This validates the mechanism on this build (copies verified by the target, identical tokens); it is the regenerate best case, not the agent-workload yield the offline estimate (`STATUS-track3.md`: about 7 % of rounds at most) is about. Confound: the second reply also benefits from the prompt cache, though these prompts are below the keep grid (`cached_tokens` 0).

### 11.5 Track 1 item 3 and Track 2/4 inputs: the routing dump (first attempt failed; see 11.8 for the retry)

The probe ring with the 0710 image started on the route config (lean, pipelined and multi-slot prefill off as the
patch requires, 131072 context) and exhausted memory on both nodes before serving: rank 1 went from 35 GiB available
after its weights to 0 within two minutes, rank 0 the same (both kernels log NVRM "Out of memory" at 21:06:00 to
21:06:02, the engine's own allocations failing), and spark0 stopped answering ssh. Stopping rank 1 at 21:08 freed
spark1 (117 GiB available). Spark0 stayed hung: its journal shows tasks blocked for 122 to 245 s on locks held by the
engine process up to 21:45:25. At 21:46 both nodes came up from a reset with no shutdown in either journal and no
crash dump on either (`/var/crash` holds only kdump's lock file), so the reset was external to both kernels: a power
event or an operator reset of both nodes; spark0's hang was caused by this probe config, spark1's reset was not.
The reset lost rank 0's running analysis and the capture script. The retry in 11.8 used `CONTEXT=32768`, `GLM53_TF_KV_POOL_TOKENS=32768`,
`GLM53_TF_PREFILL_ROWS_MAX=1024`, `GLM53_TF_BATCH=1`, `GLM53_TF_VISION=0` and a memory guard on both nodes.

### 11.6 Incident: the direct link after the reboot

After the reboot all four CX7 ports on both nodes stayed in `Polling` with `Negotiation failure` (PHY FW): no
carrier, no fabric addresses, so neither the production ring nor any TP=2 run could start. What was done, in order:
`ip link` flaps on both ends (no change); `mstlink -a TG` port toggles (no change); forcing 200G_4X with
autonegotiation off on both ends (no change); `mstfwreset -l 3` on spark0's two devices run concurrently (the
device is multi-host, a sequential reset times out on the sync) and the same on spark1, which rebooted spark1;
restoring autonegotiation on spark0 with `mstlink --speeds <all>` and a toggle. After that last step all four ports
read `Active / LinkUp / 200G`, the netplan addresses came back, both nodes ping over both links and `rdma link`
shows the cabled ports ACTIVE. The cable is a passive 1 m QSFP28 DAC; the "insufficient power on the PCIe slot (27W)"
message appears in every boot including the working ones and is not the cause. Which step fixed it is not
established [U]; the sequence is recorded so the next person does not repeat the firmware reset on a serving node.
Spark1's second reboot (about 22:02) was caused by that firmware reset, an action of the operator agent; it too left
no crash dump.

### 11.7 Production restore

`scripts/serve.sh start` from the production checkout with a clean environment: memory gate passed (115 / 116 GiB
free), ready, canary ok (3 probes, 5.71 tokens a round, 68.3 tok/s decode), 1M context, 4 slots; both containers up;
the rank-0 container environment equals the 20:03 reference in every key but the launch timestamp.

What stays on the nodes: the native dumps and analyses under `~/kvprobe/native/` on both nodes (4.5 GB a rank), the
built tree `~/kvprobe/TensorFold/`, the probe image `glm53-tensorfold:route` on both nodes, the project copy
`~/kvprobe/glm53-build/`, the Recall record `~/kvprobe/native/recall/`. Copies of the results are in
`docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/`.

### 11.8 Routing dump, second attempt (completed)

With the memory-safe config and a guard on each node that removes the containers under 6 GiB available, the probe
ring (image `glm53-tensorfold:route`, patch 0710) started in 444 s and held at 94 of 121 GB on both nodes. The four
hidden-tap prompts (S, A+S, B+S, L+S of the first probe, same token ids) prefilled cold in 5.2, 4.1, 4.0 and 12.4 s;
the dump holds 22 segments, 18,432 prefill rows, with `route_ids`, `route_w`, `kda_lg` on every segment and
`dsa_sel` on the 13 segments past position 2050. Outage 22:25 to 22:34; production restored by the same script
(canary ok, warmups at 4096 and 16384 tokens done, container environment equal to the reference but the timestamp).
First-run checks: the share of positive KDA log gates is 0.0000 as expected; rows per request equal the prompt
lengths; the routing weights sum to the routed scale 2.5 on every row and layer (min 2.4999993, max 2.5000007). Analysis: `tools/kvprobe/routing.py` (report copied to the receipts folder as `routing.md`). All [O]:

- **Routing agreement on S across prefixes (A+S vs B+S), mean over S's rows.** Top-8 Jaccard 0.83 at layer 3,
  0.72 at 7, 0.60 at 20, 0.44 to 0.55 through layers 25 to 39, 0.65 at 44; top-1 expert agreement 0.99 at layer 3,
  0.63 to 0.70 in the middle, 0.80 at 44; weighted overlap 0.94 to 0.61 to 0.78. The first 64 tokens of S agree less
  (Jaccard 0.3 to 0.6). An 8192-token prefix lowers every figure by 0.03 to 0.06 more. With two 8-sets a Jaccard of 0.44 to
  0.55 means 2.7 to 3.1 of the 8 experts differ: about a third of a token's experts in the middle layers depend on
  the prefix [C].
- **DSA selection stability.** Consecutive recorded rows (16 apart) share 0.53 to 0.67 of their 512 selected pools.
  Inside S the selection is nearly prefix-independent for the same prefix length: A+S vs B+S Jaccard 0.90 to 0.94
  on S's pools, but 0.44 to 0.69 against the 8192-token prefix, where the top-2048 budget spreads over more context.
- **Expert temporal locality.** A row's 8 experts are in the union of the previous 16 rows' sets 47 to 75 % of the
  time, of the previous 64 rows 84 to 90 %; 2048 rows touch 263 to 288 of the 288 experts. There is no small
  working set to pin.
- **Predictability, recall of the router's 8 among k predicted (held-out 2048 rows).** Popularity 0.05 to 0.25 at
  k=8; the previous row's set 0.05 to 0.37; a token-keyed prior 0.22 to 0.51 at k=8 and 0.46 to 0.72 at k=32; the
  layer-to-layer co-occurrence predictor 0.26 to 0.47 at k=8 and 0.48 to 0.78 at k=32. Oracle headroom at 6.3 MB an
  expert a rank: the best k=8 predictor puts 17 MB of correct bytes in flight against 33 MB wasted per row and
  layer (hit share 0.34); at k=32, 31.5 MB correct against 170 MB wasted (0.63). A routed-expert prefetch would
  fetch three to six bytes for every useful one at the accuracy these predictors reach (33 / 17 at k=8, 170 / 31.5
  at k=32) [C].
- **KDA decay horizon** (1 / (1 - exp(median log gate)) per head): median 1.4 to 8.6 tokens by layer, p10 about 1
  to 3, p90 4 to 40. Most heads forget within ten tokens; the longest-memory heads keep a few dozen. Gate statistics
  deviate 0.00 to 0.13 across prefixes.
- **Prunable pages.** The upper bound from recorded selections is 0 % at every layer (p50 0.0, mean under 0.001):
  with 2048 pools of 4 tokens selected, every page of these 2k to 10k-token prompts holds a selected pool. The
  prune can only pay past about 8k tokens of context; it needs a long-context capture to measure [U].

Readings [C]: the short decay horizons explain both the low cross-context deviation of the KDA state and its low
effective rank (a decayed sum of the last few tokens' rank-1 updates), and they make bounded input replay of the
last few dozen tokens (the sibling project's report 09, E-H1) the natural alternative to storing the 55 % KDA share
of a cold entry, at the cost of exactness in the longest-memory heads [H]. The routing numbers close the
routed-expert prefetch question on this checkpoint unless a better predictor appears: locality is weak and the
cheap predictors catch a third of the picks at k=8.

### 11.9 Long-context routing dump (capture 3, completed)

One 204,800-token prompt (the repository's own documentation, 29 files, tokenized through the production ring) on the
probe ring with 262144 context and pool, selections recorded every 64 rows: prefill 254 s (806 tokens/s), 201 segments,
3167 recorded selection rows a layer, 2.9 GB; memory 96 of 121 GB on both nodes; outage 23:00 to 23:07, production
restored by the script (canary and warmups ok, environment equal to the reference but the timestamp). All [O]:

- **Prunable pages, by position of the query row** (share of the whole 512-row pages before the row that hold none of
  its 2048 selected 4-token pools; an exact bound on what a page prune with sound per-page key bounds could skip):

| query position | layer 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2k to 16k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16k to 64k | 0.36 | 0.01 | 0.06 | 0.12 | 0.11 | 0.13 | 0.12 | 0.11 | 0.09 | 0.06 | 0.07 |
| 64k to 128k | 0.61 | 0.18 | 0.33 | 0.39 | 0.41 | 0.40 | 0.41 | 0.39 | 0.36 | 0.30 | 0.33 |
| 128k to 200k | 0.75 | 0.48 | 0.65 | 0.64 | 0.61 | 0.56 | 0.60 | 0.61 | 0.59 | 0.47 | 0.57 |

  Medians over the rows in each bin. Past 128k tokens an exact prune skips about half to three quarters of the pages
  of the index scan, past 64k about a third; below 16k nothing. The hint's device path is now worth building for long
  contexts [C].
- **Selection reach.** At query positions past 128k, 59 to 85 % of a row's selected tokens are older than 8k tokens,
  21 to 69 % older than 32k, up to 23 % older than 128k: the indexer reaches across the whole context, so a recency
  window is not a substitute for the index.
- **Never-selected pools.** Over the 3167 sampled rows a layer, only 1 to 7 % of the prompt's pools are never selected
  (layer 7: 50 %, the one layer whose selections stay local), and 3 to 14 % are selected by at most one sampled row.
  Demoting rows by selection counts has little room on this prompt; the "cold rows" idea is closed unless decode-time
  selections turn out much narrower [C].
- **Stability between rows 64 apart** is low at long context (Jaccard 0.06 to 0.27, share of the previous row's set
  0.11 to 0.41), so this capture says nothing about reuse between consecutive decode steps; that needs a stride-1
  decode-row dump.
- Raw tables: `gpu-2026-11-01/longctx/` in the receipts folder (`longctx.log`, `prunable.log`, the script).

### 11.10 KDA replay restore on the ring (the hypothesis of section 11.8 tested; refuted as implemented)

Feature: `TF_GLM_KDA_REPLAY=W` with `TF_GLM_KDA_REPLAY_CHECK=1` (`zig/src/families/glm/cuda_kda_replay.zig`,
`tools/kvprobe/STATUS-replay.md`): on a restore, the attention rows are restored, the arena head (KDA states and conv
history) is zeroed, the last W prompt tokens are replayed over the stored rows with the DSA writes suppressed, and the
rebuilt head is compared with the stored one, which then serves. Run on the native ring with the prose A+S chat body
(8528 tokens, keep at position 8192), three windows, one restart each; replies with the check on equal the first
reply's tokens in every run (sha `95db6f07`), and the same body served by the eager and lazy runs without any replay
knob gave the same sha, which is the runbook's knob-free pass condition; the check mode itself is sound. All [O]:

| window W | heads bit-identical (of 1088 a rank) | state rel L2 p50 | p90 | max | conv rel max | layer 0 p50 | layer 1 p50 / p90 | layer 9 p50 | layer 21 p50 | layer 44 p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 1 (rank 0), 6 (rank 1) | 0.87 | 0.98 | 1.45 | 1.10 | - | 0.19 / 0.77 | 0.45 | 0.89 | 0.91 |
| 256 | 10, 12 | 0.73 | 0.91 | 1.17 | 0.85 | - | 0.08 / 0.56 | 0.24 | 0.76 | 0.84 |
| 512 | 14, 16 | 0.67 | 0.88 | 1.15 | 0.80 | 4e-8 (14 of 32 heads bit-identical) | 0.06 / 0.42 | 0.19 | 0.69 | 0.80 |

Replay form (`TF_GLM_KDA_REPLAY_KEEP=1`, W = 256): the keep shrank from 272 MiB to 78 MiB a rank (the arena head
gone), the restore's replay took 1.41 to 1.43 s, and the reply changed at token 10 of 14 ("done" against "Done."),
sha `77acd313` against `95db6f07`.

Reading [C]: a zero initial state plus a window of up to 512 tokens does not rebuild the state. Layer 0, whose
inputs are the context-free embeddings, converges (relative error 4e-8 at W = 512); every deeper layer's error is set
by its inputs, the hidden states of the replayed tokens, which are wrong while the lower layers' states are still
converging, so the error compounds with depth (0.06 at layer 1, 0.69 at layer 21, 0.80 at layer 44 at W = 512) and
falls only slowly with W. The per-head mean gate horizons of section 11.8 (the direct-term bound 1e-11 to 1e-167 at
W = 512) do not describe this: the horizon that matters is the slowest channel of a head, not the mean, and the
input error dominates anyway. The cross-context deviation of the stored state (0.2 to 0.4, section 11.3) is smaller
than the zero-init replay error, so any prefix is a better starting point than none.

Decision (see `decisions.md`): the replay form is not adopted; the feature stays as an experiment behind its flags.
The one variant still open [H]: replay from a coarse stored state (fp8 or a low-rank factor of each head) instead of
zero, where the replay only has to correct the recent tokens' contribution; untested.

### 11.11 The codec in the engine: eager compaction (served, four chat bodies)

`TF_GLM_HOST_CACHE=1 TF_GLM_KV_CODEC=1` (eager, the default when the codec is on), no drafts, four chat bodies of 6.3k
to 8.7k tokens sent once each and then again [O]: keeps 4, compacted 4 on the prompt pass, raw 1,112,510,464 B,
stored 809,068,310 B (ratio 0.727), pages 532 to 387, `encode_ms` 6089 inside the keeps (about 1.5 s on the prompt
pass per keep, as logged), `decode_ms` 1770 over 4 restores, `duplicate` 15.7 MB; the second sends
resumed 8192, 6144, 8192 and 8192 tokens from the compacted entries and every reply's tokens equal the first send's
(sha `95db6f07` for all four bodies, whose final turn is the same question). The 1.5 s a keep on the prompt pass is
what lazy compaction is meant to move off the critical path (11.12). The eager timers exceed the lazy run's for
identical bytes by about 5x both ways (`encode_ms` 6089 against `lazy_encode_ms` 1173, `decode_ms` 1770 against 343),
so they include something besides the codec's own work (a wait, a copy or contention with the prompt pass); not
explained [U]. The user-visible claim, that lazy takes about 1.5 s a keep off the prompt pass, stands either way.

### 11.12 Lazy compaction with intermixed entries and pinned workers (served)

`TF_GLM_KV_CODEC_LAZY=1 TF_GLM_KV_CODEC_LAZY_MS=2000 TF_GLM_KV_CODEC_CPUS=5-9` on top of the eager settings, the
same four bodies sent once, a 15 s pause, then again [O]. The start line reports `mode=lazy delay_ms=2000 threads=5
cpus=5-9 (outside the engine's 0-4)`: the session's cpuset is CPUs 0 to 9, so the engine's `taskset -c 0-4,10-14`
leaves it on 0 to 4 and the workers took the other five; a list naming CPUs outside the cpuset (15-19 on the first
try) is refused by name at start. Each keep returned raw and was compacted on the workers 2.28 to 2.30 s later
(`compacted keep=N ... after_ms`), its pages trimmed from 136 to 98 or 99 (124 to 91) and returned to the pool
(`trimmed ... free=`), and the prompt cache's charge for the four states settled at 774 MiB, the compacted size.
Totals: keeps 4, compacted 4, restores 4 from compacted entries, raw 1,112,510,464 B, stored 809,068,310 B (ratio
0.727, byte for byte what the eager run stored), pages 532 to 387, `encode_ms` 0.0 on the prompt pass and
`lazy_encode_ms` 1173 on the workers, `decode_ms` 343, cancels 0. The second sends resumed 8192 / 6144 / 8192 / 8192
tokens and every reply's tokens equal the first send's and the eager run's (sha `95db6f07`).

What this served run did not exercise: a restore from a still-raw or part-compacted entry (the host tests cover both),
the TP page agreement under a rank-skewed schedule beyond these four keeps, and the extra gate exchange's cost; a
first lazy launch died at once because it started 5 s after the production containers were removed, before the GPU
was released (a 30 s pause fixed it).

Reading [C]: with lazy compaction the prompt pass pays nothing for the codec (1.5 s a keep on the prompt pass in
the eager run, by its own timer), the
compaction runs on cores the operator chooses, and the saving (27 % of pages here) arrives about `LAZY_MS` plus the
encode after the keep. This is the Phase 2 served evidence the plan asked for, at four keeps; a receipt in the
repository's style needs the longer run in `tools/kvprobe/RUNBOOK-lazy.md`.

### 11.13 One more incident, and the state of the nodes at the end

The full routing report on the 204,800-token dump (`routing.py`, section 11.9) was killed by the kernel's OOM killer
at 23 GB of resident memory while production served beside it; the victim was the analysis process (UID 1000), the
engine survived, and the position-binned analysis ran instead. Analyses on dumps of that size go under a memory cap.

Production has been restored four more times by the capture and validation scripts (23:07, 00:23, 00:26, 00:40),
each with the canary and both warmups passing and the container environment equal to the 20:03 reference but the
launch timestamp. Left on the nodes beyond section 11.7: the long-context dump (2.9 GB), the replay check dumps
(six head pairs under `~/kvprobe/replay/`), the lazy and eager logs under `~/kvprobe/lazy/`, the rebuilt tree with
both features (binary `afa14593…` on both nodes), and the route config files in the project copy.

### 11.14 KDA head stored as fp8 (the coarse-init variant, served)

`TF_GLM_KDA_REPLAY_INIT=fp8` (`zig/src/families/glm/cuda_kda_coarse.zig`): a replay-form keep stores the arena head
as e4m3 with one fp32 power-of-two scale per row of 128, 19,675,424 B a rank in place of the 194 MiB head; a restore
dequantizes it, then optionally replays W tokens on top. Same body and procedure as 11.10. All [O]:

| variant | state rel L2 p50 / p90 / max over 1088 heads a rank | conv rel max | bit-identical heads |
|---|---|---|---|
| fp8 copy alone (W = 0) | 0.0264 / 0.0281 / 0.0395 (rank 0), 0.0264 / 0.0283 / 0.0450 (rank 1) | 0.034 / 0.039 | 0 (every head is rounded) |
| fp8 copy, then replay 64 tokens | 0.373 / 0.761 / 1.49 | 1.39 | 1 to 5 |
| fp8 copy, then replay 256 tokens | 0.329 / 0.620 / 1.31 | 1.09 | 10 to 12 |

The copy alone is within e4m3's rounding (relative error 2.6 % at the median, 4.5 % at worst, uniform across layers);
replaying on top of it makes things worse at every window, because the copy already holds those tokens and the
replay applies them again (the agent's reading, now measured). The fp8 replay form at W = 0, served: the keep is 94
MiB a rank (272 full, 78 for the zero form), its ledger shows the prefix class at 19,675,424 B, the restore's
dequantization is not separately timed (`ms=0.00`), and the three replies, two of them restored from the fp8 copy,
equal the first reply's tokens (sha `95db6f07`), the same tokens the knob-free, eager and lazy runs produced.

Reading [C]: storing the KDA head in fp8 is a lossy rung with a bounded error (2.6 % relative per head, no
compounding, no reply change in this one sample) that cuts a kept entry by 65 % at this length; it is the cold-tier
form worth taking to the Phase 3 quality gate (`tensorfold kl` on a prompt set, several bodies, both greedy and
sampled). The replay mechanism itself is closed: neither a zero start nor a coarse start converges to the stored
state within 512 tokens.

### 11.15 The exact page prune on the device (`TF_GLM_INDEX_PRUNE=1`: kernel gates, timing, served)

The device path of the prune of 11.9 (`zig/src/families/glm/cuda_index_prune_dev.zig`, kernels in
`zig/kernels/cuda/glm_index_prune.cuh`; contract and host results in `tools/kvprobe/STATUS-prune.md`) ran on the GB10
for the first time on 2026-11-02: rank 0 alone for the kernel checks, the ring for the served comparison. Raw logs in
`docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/prune/`. All [O] unless marked.

- Build: `glm_index.cu` compiled with nvcc 13.0 for sm_121 at the first try. `tf_glm_index_scores`' sm_121 compile
  record is unchanged (124 registers, 32,768 bytes of static shared memory, bound 256, "record current"), so turning
  it into `scores_body<false>` moved nothing. The four prune launches printed no RECORD line: `registry.zig` imports
  `glm_index_prune_pending.zig` for its tests only and does not append its records to the set the limits pass walks,
  so the contract's compile records (gate 5) are still missing.
- `tf-cuda-test glm-index-check`: PASS (the existing index kernels' bytes unchanged).
- `tf-cuda-test glm-index-prune`: 65 of 65 PASS. At 32,768, 131,072 and 204,800 tokens, seven key cases and 512, 8
  and 1 query rows, the pruned selection equals the full scan's byte for byte (the 512-row lines also under
  `TF_GLM_INDEX_CUDA=1`'s `tf_glm_index_pools`), and the control (strips left unwritten) changes the selection, 441 of
  505 pages skipped there. Skip shares on the synthetic keys: 0 in most cases; 0.10 to 0.14 for clustered keys with
  positive weights at 512 rows (0.07 to 0.17 at 8 rows, 0.065 at 1 row and 204k); 0.16 to 0.20 for tied pages at 512
  rows and 0.34 at 1 row (32k and 131k). These show that the mechanism prunes, not the share on real queries.
- Contract runner: `tf_glm_index_scores_prune` (every page planned, pages out of order) passes repeat (five runs
  byte-identical), shapes (2 shapes bit-equal to `tf_glm_scores`, the kernel it replaces) and sums (exact at k 128);
  on the non-finite gate it prints the same line as `tf_glm_scores` and `tf_glm_index_scores` ("1024 of 16384
  outputs differ in class: finite where float64 gives nan"), the known finding those kernels carry (relu is
  `fmaxf(x, 0)`, so a NaN dot scores 0 where float64 keeps the NaN), counted as known, not failed: the run's summary
  is 503 kernels passing gates 1 to 4, 0 failed, 7 known findings, 338 controls red as planted. (I first read this
  line as a new failure; the author's reading of the full log corrected it.) `tf_glm_prune_bounds` passes gates 1
  to 3 (4 non-finite and 252 finite outputs in the float64 classes).
- Timing (`glm-index-prune timing`: wall clock around the whole select, the median of five alternating runs; two
  runs of the harness agree within 10 %), microseconds a query row:

| tokens | 512 rows: full / pruned | 8 rows: full / pruned | 1 row: full / pruned |
|---|---|---|---|
| 32,768 | 1.4 / 1.8 (x0.79) | 5.1 to 5.3 / 13.3 to 13.6 (x0.39) | 24.3 to 26.5 / 83.9 to 88.1 (x0.29) |
| 131,072 | 7.5 / 7.9 (x0.96) | 13.3 to 14.6 / 30.0 to 31.5 (x0.45) | 49.1 to 53.4 / 170.0 to 174.5 (x0.30) |
| 204,800 | 11.5 to 11.8 / 11.9 to 12.1 (x0.97) | 19.5 to 21.1 / 44.5 to 46.1 (x0.46) | 67.5 to 79.8 / 235.5 to 249.8 (x0.30) |

  The pruned scan is slower at every length and row count on these keys. Reading [C]: at one row the extra 60 us
  (32k) to 170 us (204k) matches four extra kernel launches' CPU-side latency on an idle stream (plan, planned
  scores, cut, planned scores again; 10 to 15 us each) rather than GPU work; GLM's CUDA decode in this tree is not
  graph-captured, so that latency is a served cost (11 DSA layers x 170 us x 255 rounds is about 0.5 s of the 200k
  document's 32 s decode below, where the flag-on ring measured 0.3 to 1.0 s slower, within the noise); at 512 rows
  the two scans are within 3 % with 0 to 14 % of the pages skipped.
- Served (the ring, `PROBE_CONTEXT=245760`, host cache on, codec off, no drafts, greedy, 256 tokens a reply, the
  model thinking): two documents (31,213 and 205,171 prompt tokens), three turns each (first; the same prompt again,
  a restore of the kept state with its strips recomputed; a follow-up question), the flag off and then on:

| document, turn | reply tokens' sha (off = on) | TTFT off / on (s) | decode of 256 tokens off / on (s) |
|---|---|---|---|
| 32k first | 08343d686f81 | 48.1 / 47.4 | 31.3 / 31.1 |
| 32k again (restored) | 08343d686f81 | 1.49 / 1.50 | 31.5 / 31.4 |
| 32k follow-up | be1b97a0825b | 1.46 / 1.47 | 31.8 / 31.1 |
| 200k first (174,451 new tokens) | e69767d174a7 | 266.9 / 263.5 | 31.7 / 32.0 |
| 200k again (restored) | e69767d174a7 | 1.66 / 1.63 | 31.6 / 31.8 |
| 200k follow-up | 0099ad2ee888 | 1.53 / 1.46 | 31.3 / 32.3 |

  All six token streams are identical between the rings (the hashes cover the thinking tokens; `max_tokens` cut every
  reply inside its thinking). The times are within this ring's run-to-run noise (about 1 s on a 32 s decode, 3 s on a
  265 s prefill). The engine logs no prune statistic, so the share of pages skipped on these real keys is not known
  (the capture of 11.9 bounds it at 0.47 to 0.75 past 128k tokens).

Reading [C]: the device path is exact where it was measured (the kernel oracle, the contract's byte gates against the
kernel it replaces, six served replies), and the restore path recomputes the strips correctly (the restored turns
match). What it cannot be on this ring is a speed lever of any size at 200k tokens: a decode step costs 124 ms at
both 32k and 200k (the expert weights, not attention), and a step's whole index scan is 11 layers x 70 to 80 us,
under 1 % of it; the 200k document's prefill ran at 1.5 ms a token, of which the scan is 0.13 ms a row at the end
and about half that on average, so even a 0.75 skip would return 2 to 3 % of a long prefill, and the launch overhead
has to go first to see any of it. The lever grows with context (the scan is linear in it, the expert cost is not)
and on hardware where the expert layers are cheap next to the scan. Decision: the flag stays off by default; the
mechanism is the upstream candidate (exact by construction, checked by an oracle, an emulation and the contract),
the performance claim is not. Follow-ups sent to the author and delivered the same morning (commit 909c4f48f,
merged as 50bccc592, untimed on a GPU yet): a pruned select is three launches (the plan kernel runs 32 pages a
block with the seeds picked by each row's last block, the cut is folded into the first planned-scores pass, idle
CTAs exit before loading their rows), `TF_GLM_INDEX_PRUNE_MIN_PAGES` (default 256 whole pages, 131k tokens, so
shorter selects take the plain scan; `glm-index-prune` forces 0), the timing line reports wall and CUDA-event GPU
time with the pages skipped, `INDEXPRUNE` lines after each prompt pass and at teardown with the skip share on real
keys, and the pending records in the set the limits pass walks. The non-finite case went into the host emulation
(11 cases, 0 selections differ). The next GB10 pass re-runs the checks and the timing and serves the 200k document
with the threshold at 0 for the skip share.

Round 2 on the GPU (binaries `bin-94b8c703`, 06:57 UTC; raw logs in `.../gpu-2026-11-01/prune2/`) [O]:

- The limits pass now walks the three launches and prints their records (sm_121): `tf_glm_index_scores_prune` 122
  registers, 35,072 bytes of static shared memory, bound 256; `tf_glm_prune_bounds` 36 registers, no shared memory,
  bound 128; `tf_glm_prune_plan` 53 registers, 32,008 bytes, bound 256; `tf_glm_index_scores` unchanged (124, 32,768,
  256). The lines are in the receipts for the maintainer gate (the other SMs are unbuilt here, so each record reads
  "stale" until pasted).
- `glm-index-check` PASS; `glm-index-prune` 65 of 65 PASS with the same skip shares as round 1 and the control caught.
- Timing, now wall and GPU (CUDA events) side by side, microseconds a row, two harness runs agreeing:

| tokens | 512 rows: full / pruned (GPU) | 8 rows | 1 row |
|---|---|---|---|
| 32,768 | 1.4 / 1.8 (x0.77) | 5.0 / 14.3 (x0.35) | 21.8 / 60.6 to 62.9 (x0.36) |
| 131,072 | 7.5 / 7.8 (x0.96) | 13.6 / 26.0 (x0.52) | 48.6 to 60.7 / 99.6 to 116.0 (x0.50) |
| 204,800 | 11.5 to 11.7 / 11.8 (x0.98) | 19.4 to 20.2 / 33.9 to 34.7 (x0.57) | 64.9 to 77.2 / 120.0 to 138.5 (x0.55) |

  The single-row path came down from 86 to 61 us at 32k and from 240 to 125 us at 204k (three launches instead of
  four, the plan in parallel), but it is still slower than the plain scan at every length, and the GPU column shows
  why my round-1 reading was wrong too: the wall and GPU times differ by only 4 to 5 us, so the extra cost is GPU
  time of the small plan and planned-scores kernels (each a few blocks with a fixed tail of about 20 us), not host
  launch latency. On synthetic keys the skip share is 0 to 0.34; the served share on real keys is below.
- Served, round 2 (`TF_GLM_INDEX_PRUNE=1 TF_GLM_INDEX_PRUNE_MIN_PAGES=0`, the same two documents, three turns each,
  against a flag-off ring on the same binaries; `prune2/byte-identity.txt`, `indexprune-r0-lines.log`): every one
  of the six replies differs from the flag-off ring (32k: `d4fc55851afe`, `a188d1d6947d`, `1ad5939d3053` against
  `08343d686f81`, `08343d686f81`, `be1b97a0825b`; 200k: `69954a77181e`, `69954a77181e`, `8828b545ac76` against
  `e69767d174a7`, `e69767d174a7`, `0099ad2ee888`), the first prompt pass included, and the restored repeat of the
  32k prompt differs from its own first reply. The flag-off replies equal round 1's, so the default path is
  unchanged and round 2's pruned path is what broke. At the same time the new counters say the prune skipped
  nothing: `INDEXPRUNE rank=0 total min_pages=0 selects=1075404 pages=229200510 skipped=0 share=0.0000`, every
  pass line and rank 1 alike. Two findings in one: (a) round 2 is not exact in the engine although `glm-index-prune`
  passes at 512, 8 and 1 rows (the engine's chunk shapes, the folded cut, the counters or the threshold-0
  admission are the suspects; sent to a new agent for round 3); (b) the counters saw zero skips over 229 million
  page visits on real keys at 32k and 200k tokens [O]. Whether that zero is the bound (per-dimension minimum and
  maximum over a page's 128 pooled keys with a 2^-8 margin, which may never fall below a row's k-th best key on
  real keys) or the same bug suppressing the plan is undecided [H]: the counters come from the build whose served
  path is wrong. The capture of 11.9 counted 0.47 to 0.75 of the pages past 128k as holding no selected pool, which
  is what an oracle can see, not what a bound can prove; the offline bound-share measurement on the kept pooled
  keys (round 3's second task) separates the two. Timings equal the flag-off ring's within noise (32k first turn
  79.25 vs 79.23 s, 200k 294.9 vs 297.6 s).

Reading [C], after round 2: the exact prune stands on round 1's evidence at commit 2b61900ad (65 of 65, the
contract, six served replies identical), and round 2 on main is a regression to fix before the device path can be
served again; the finding that will decide the feature is the share on real keys, which is zero on the round-2
build and unattributed until the offline measurement says whether the interval bound can prune real keys at all
or needs replacing (per-head bounds, a low-rank or clustered page bound, a smaller margin). The flag stays off by
default; the upstream candidate is the exactness machinery as of 2b61900ad (strips, oracle, emulation, contract
records), with no performance claim.

Round 3 (a fresh agent, the same morning; commit f45dc1c02, merged as e8227ae9e, host-tested only) found the served
bug in host code, not in a kernel: round 2 placed the plan's row counters and the cut's group counters after the
plan bytes at an offset that depends on the select's rows and pages, zeroed once at arena allocation and cleared
by each launch's last block; a select with fewer rows or pages than the one before it on the same scratch (the
493-row last chunk of the 32k prompt after 2048-row chunks, the MTP layer's one row fewer, a shorter request after
a longer one) found its counters inside the previous select's plan bytes, so no plan block saw itself as last, no
page became a candidate (hence `skipped=0`), pages left at their old value were scored in neither pass and the
select read the previous layer's scores there, and the cut never ran for the group. The emulation now runs six
selects on one shared scratch and reproduces round 2's layout as a planted fault (31, 49, 1, 1 and 8 selections
differ from the second select on); with the counters at fixed places at the head of the scratch (512 rows, 64
groups, 2,304 bytes, `n > 512` refused by name) nothing differs and every counter returns to 0. `glm-index-prune`
gains seven "one scratch" lines at the engine's shapes (2048, 493, 492 and 1024 rows, a decode row at 131,071 then
one at 31,213, a window of 8) that have not run on a GPU; tests 22 of 22 host, 456 of 456 build steps. So the zero
skip share of round 2 [O] was the bug, not the bound [C]: with the plan never written, no page could be skipped. The
bound's real-key share is still unmeasured: the offline tool (`tools/kvprobe/prune_share.py`, oracle, the box
bound at two margins, per-head and PCA bounds, sound-by-construction checks) needs the 200k keep plus a capture of
the index queries and head weights that no dump records yet (an env-flagged capture in `cuda_dsa_block.attend` is
the next change). Its synthetic run says the box over 128 dimensions excludes 0 of the pages where the oracle
excludes 0.95 and per-head maxima 0.92 [O, synthetic]. The next GB10 pass (a rebuild, `glm-index-prune`, the
served pair with the threshold at 0) is the operator's call after five restarts today.

### 11.16 Recall's yield on production decode rows (the first 36 minutes)

Production restarted at 02:51 UTC with the serving stack's decode-row draft dump (patches/0430: decode rows only, the
top-32 ids a row, no taps, 8,388 B a token, 100 GiB cap) and ran until 03:27, when the validation window stopped it
(resumed at 05:47 for the week). The dump holds 178 flush segments, 39,560 decode rows, chained into 32 requests by
position (`tools/kvprobe/draft_eval.py` gained a decode-only loader: the prompt rows are not in the dump, so context
copies see the reply alone and Recall's record holds replies only, a lower bound for both). The rows' token at i + 1
equals row i's top-1 pick at 99.7 % of positions (greedy decoding with a few sampled divergences). Outputs in
`docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/recall-yield/` (aggregates only; the dump stays on
the node). All [O] unless marked.

The window's traffic (the request log): 5 canary probes; one thinking reply of 2,386 tokens to a 12k prompt; three
replies to the same 59-token prompt in one conversation (15,663 tokens, cancelled; then 6,000 and 12,000 to the
length cap); and 21 small requests at 03:25 to 03:27 with prompts of 2.6k to 4.2k tokens sharing cached prefixes
(1,344, 2,560 or 3,392 tokens), thinking on, 13 to 864 decode tokens each.

| draft source (k = 16) | coverage of positions | accepted of 16 a proposal (strict) | accepted a position |
|---|---|---|---|
| Recall (record of earlier replies, 8-id keys, 16 matched ids) | 0.461 | 15.55 | 7.16 |
| engine order (own context first, then Recall) | 0.514 | 13.82 | 7.10 |
| own context (within the reply) | 0.077 | 4.03 | 0.31 |
| 1-gram prior fitted on the other requests | 0.978 | 0.53 | 0.52 |

The pricing gate (`recall.Rates`) withheld nothing: gated and ungated results are identical. The breakdown: the two
later replies to the repeated prompt (18,000 rows, 46 % of the window) are copied whole from the first (coverage 1.0,
15.5 accepted a position, the regenerate case of 11.4); over the other 21,560 rows Recall proposes at about 1.2 % of
the positions, in 5 of the 27 requests, with 7 to 14 of 16 accepted where it does.

Reading [C]: in this window Recall's gain is the regenerate case: a prompt sent again (a cancelled request re-sent,
a benchmark, a retry) decodes at 16 tokens a round instead of the drafter's 3.6, and had Recall been on, the window's
rounds would have fallen by about a third, almost all of it from those two replies. On the ordinary agent traffic
the copyable spans are about 1 % of the tokens (replies only; prompt-quoting copies are not in this bound), a per
cent of rounds at most, consistent with the earlier estimate (at most 7 %). What decides whether Recall pays beyond
the regenerate case is the share of re-sent prompts in a week of real traffic, which the running dump will give;
the session-store walk (`recall_yield.py --store`, replies against every earlier kept conversation) is the
complementary measure.

The session-store walk ran beside production at 06:41 over the node's kept conversations (100 entries, 2,212,096
ids, 522,299 of them new past each entry's longest common prefix with an earlier one, 9,678 of those written by the
assistant; `recall-yield/session-store-walk-k7.json`): the conversation's own context copy lands 20.7 % of the
assistant tokens (522 copies, 3.84 ids a copy, 18 % landing nothing), Recall across entries lands none (0 copies of
16 matching ids). The assistant's share of what the store keeps is small (the entries are prompts, tool output and
history), so this is a thin sample, but it agrees with the decode-row dump: on ordinary agent traffic Recall's
cross-conversation copies are rare, and its gain is the re-sent prompt.

### 11.17 The codec receipt: off, eager, eager-unpinned and lazy, nine turns each (2026-11-02)

`tools/kvprobe/lazy_receipt.py`'s conversation served on the ring (`PROBE_CONTEXT=65536`, host cache on, the ledger
and the dump on in every configuration, 48-token greedy replies, 30 s pauses, lazy delay 20 s): six main turns of
2,200-token pieces of the repository's docs after a 3,072-token first piece, then a raw-keep turn, a raw restore
and a partial restore, one serve a configuration. The receipt is `docs/receipts/glm-kv-codec-lazy-2b61900a/`
(README, records, SHA256SUMS, written by `receipt.py`). All [O] unless marked.

- Replies: identical in all nine turns across the four configurations (token sha256), restores from raw, compacted
  and intermixed entries included.
- Codec totals of rank 0 over nine keeps of 3,102 to 20,881 tokens:

| configuration | ratio | pages | encode on the prompt pass | compaction on the workers | decode (8 restores) | copies | keep I/O (ledger, dump) |
|---|---|---|---|---|---|---|---|
| eager, workers on CPUs 5-9 | 0.692 | 1358 -> 942 | 2,353 ms (0.26 s a keep) | 0 | 637 ms | 651 ms | 4,171 ms |
| eager, workers unpinned | 0.692 | 1358 -> 942 | 14,280 ms (1.59 s a keep) | 0 | 3,485 ms | 681 ms | 4,153 ms |
| lazy, workers on CPUs 5-9 | 0.688 | 958 -> 660 | 0 | 1,696 ms (7 compactions, 0.24 s each) | 417 ms | 78 + 202 ms | 4,483 ms |

- The 5x of 11.11 is explained: workers left unpinned on the engine's cores are 6x slower at the same bytes, in the
  encode and in the decode alike; pinned eager and lazy encode at the same rate. The timers were right, the cores
  were wrong.
- Pages: lazy trimmed seven times and freed 298 pages (`free=6660` at the end); eager gives pages back inside the
  keep; off frees nothing.
- Time to first token a turn, off / eager / lazy (s): 5.77 / 6.70 / 6.41, 5.54 / 6.75 / 6.29, 5.42 / 6.90 / 6.54,
  6.04 / 7.54 / 6.96, 6.16 / 7.86 / 7.14, 6.15 / 7.95 / 7.22, 7.02 / 9.07 / 8.38 (the raw-keep turn), 4.33 / 6.65 /
  5.85 (a raw restore), 4.37 / 6.82 / 5.87 (a partial restore); eager unpinned 7.96 to 12.91. Lazy costs 0.6 to
  1.5 s a turn over off and eager 0.9 to 2.5 s, both growing with the kept state.
- Reading [C]: the off configuration wrote no ledger and no dump (both act only with the codec on; its dump
  directory is empty), so those deltas carry the receipt's own instrumentation (keep I/O about 0.5 s a keep,
  2.7 GB of dumps a configuration) besides the codec's time on the prompt pass, which in lazy mode is the restore's
  decode (0.05 s a restore) and its copies (0.03 s). About half of lazy's delta is outside the codec's timers. The
  uninstrumented pair (off and lazy with no ledger and no dump, the production form) is queued in the next outage
  and decides the mode's real cost, the TP gate exchange included; until then the recipe's recommendation (lazy
  when the codec is on) stands on the encode moving off the prompt pass, and the default stays eager.
- Rank 1 prints no totals under SIGTERM (the stop script's signal); rank 0's totals are the ones compared.

The uninstrumented pair ran in the next outage (binaries `bin-94b8c703`, 07:22 to 07:59 UTC; `records/plain-*`
of the receipt). A first attempt with `TF_GLM_KV_LEDGER=0` still wrote the dump, because the probe launcher
supplies a default `TF_GLM_KV_DUMP` (`keep_io_ms=4638.8`: the dump alone is the 0.5 s a keep); with a
`PROBE_NO_DUMP=1` switch added to the launcher on both nodes, lazy with no ledger and no dump has
`keep_io_ms=0.0`, replies identical to off in all nine turns, and a time to first token equal to off within the
run's noise: 5.75 / 5.80, 5.57 / 5.62, 5.46 / 5.52, 6.15 / 6.10, 6.27 / 6.19, 6.12 / 6.18, 7.15 / 7.24 (raw-keep),
4.35 / 4.51 (raw restore), 4.52 / 4.61 (partial) s, lazy / off. Reading [C]: in its production form the codec in
lazy mode costs the prompt pass nothing measurable on this ring at 3k to 21k tokens (restores of compacted entries
included), the TP gate exchange is inside the same noise (under 0.1 s a turn), and every TTFT delta seen earlier
was the dump's write. Together with the 0.69 ratio and the pages given back, this is the receipt the lazy mode
needed; the default can move to lazy at the maintainer's gate.

### 11.18 Kronecker factorization of kept states against the SVD, and the dead half of the recurrent state (CPU, 2026-11-02)

The question was whether Kronecker products could pull the entropy out of a cache entry and shrink it on disk.
Measured on two rank-0 keeps of the native dump (`~/kvprobe/native/dump/glm-rank0-keep9`, 20,480 tokens, and
`keep4`, 8,192 tokens; bf16 latent rows, fp32 KDA state: the Zig keep format, not the serving stack's fp8 session
store), on the CPU beside production under a 3 GB cap, no GPU and no outage. Script, JSON and logs:
`docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/kron/`. "Kronecker" is the nearest Kronecker
product of Van Loan and Pitsianis: the k-term sum of A_i (x) B_i is the rank-k SVD of the rearranged matrix, so each
shape is compared with the plain SVD at the same number of stored parameters. All [O] unless marked.

Recurrent state (one fp32 128 x 128 matrix a head, 1,088 heads a rank), energy captured, median over heads (p10 at
the 4x budget in parentheses):

| stored parameters a head | SVD | Kronecker A 16x16 (x) B 8x8 | Kronecker A 32x32 (x) B 4x4 |
|---|---|---|---|
| 1,024 (16x smaller) | 0.862 | 0.291 | no term fits |
| 2,048 (8x) | 0.943 | 0.444 | 0.186 |
| 4,096 (4x) | 0.983 (0.864) | 0.627 (0.482) | 0.433 (0.312) |
| 8,192 (2x) | 0.998 | 0.829 | 0.718 |
| 16,384 (the matrix itself) | 1.000 | 0.979 | 0.984 |

The 8k keep is within 0.01 of every cell. Rank for 90 % of a head's energy: p50 6, p90 20, max 49; for 99 %: p50 20,
p90 51, max 82 (11.3's shorter keeps gave at most 14 for 90 % and a median of 16 with a maximum of 83 for 99 %: the
99 % figures agree, the 90 % tail is longer here). Reading [C]: the state is a decayed sum of outer products, low-rank and
not separable on a grid of its indices, so a Kronecker term is a poor basis for it; at the full 16,384 parameters it
still misses 2 %.

Latent rows (bf16 [T, 512], caches 0, 5 and 10), energy captured at matched cost, the 20k keep (the 8k keep within
0.01):

| model | cost, bits a value | energy, cache 0 / 5 / 10 |
|---|---|---|
| SVD rank 64 | 2.05 | 0.611 / 0.632 / 0.639 |
| Kronecker row = a (x) b, 64 groups of 8, one term | 2.0 | 0.134 / 0.141 / 0.137 |
| SVD rank 128 | 4.1 | 0.786 / 0.765 / 0.780 |
| Kronecker row = a (x) b, 64 groups of 8, two terms | 4.0 | 0.265 / 0.272 / 0.271 |
| Kronecker A[T/4, 512] (x) b[4, 1], one term (a pool's rows share a direction) | 4.0 | 0.410 / 0.595 / 0.552 |
| Kronecker A[T/4, 128] (x) B[4, 4], four terms | 4.0 | 0.410 / 0.595 / 0.552 |
| SVD rank 256 | 8.2 | 0.929 / 0.893 / 0.919 |

The row shapes with 32, 16 and 8 groups are lower still. The rows are close to full rank: half their energy needs
rank 32 to 64 of 512 and 90 % needs about 256.

Lossless: a structural prediction with its exact residual coded. The factors a decoder holds are rounded to bf16, the
prediction is rounded to bf16, the residual is the difference of the codes' monotone integers; order-0 entropies,
byte-split as a plane codec sees them and as whole symbols (the 20k keep; the 8k keep within 0.02):

| | bits a value, cache 0 / 5 / 10 |
|---|---|
| raw codes, byte-split (the codec's latent ratio 0.667 is 10.7 of 16) | 10.69 / 10.59 / 10.60 |
| raw codes, whole symbols | 10.47 / 10.47 / 10.48 |
| residual after a rank-128 prediction, byte-split | 11.43 / 11.51 / 11.47 |
| residual after a rank-128 prediction, whole symbols | 10.15 / 10.27 / 10.22 |
| the rank-128 factors themselves | 4.10 |
| residual after a rank-32 prediction, whole symbols; its factors | 11.02 / 10.94 / 10.92; 1.03 |

The rank-128 prediction leaves 21 to 24 % of the energy, a median residual of 89 to 101 bf16 codes, and is exact for
0.4 % of the values; it saves 0.2 to 0.3 bits a value as symbols and nothing as planes (the residual spreads over the
exponent byte), against 4.1 bits a value of factors; rank 32 saves less than nothing. Reading [C]: the floor is the
mantissa. Of the 10.6 bits a bf16 row value costs, 7 are its stored mantissa bits; of the fp32 state's 27.3 bits a
value (order-0 planes, 27.3 for the live half and 27.4 for the dead one), 23 are mantissa. A structural model can only
shrink the residual's magnitude, which saves exponent bits, and a residual at 46 % of the signal's magnitude can save
about 1.1 of them at best; no model pays for its own factors on these tensors. As a lossy rung Kronecker is dominated
by the plain SVD on both tensors, and the SVD by the precision rungs: rounding the live state to bf16 is a relative L2
error of 0.0017 at the median and 0.0023 at worst over the 1,088 heads (2x on the state, nothing to store besides),
e4m3 0.026 (11.14: 4x, served identical in one sample).

The dead half. A keep stores both halves of the double-buffered recurrent state (`rec`, 142.6 MB a rank;
`cuda_snapshot_layout.zig` line 134) and the window scratch (`proj`, `save_k`, `save_v`, `save_g`, `save_b`, 18 MB).
In both keeps the scratch is all zeros, and the dead half is neither zero nor a copy: relative L2 difference from the
live half 0.57 at the median over layers (0.89 max), no word bit-equal. The codec's ledger shows what that costs
(the receipt of 11.17, rank 0, `KVLEDGER ... class=prefix`): the run's first keep models the prefix class at 83.5 MB
with 54.8 % of its words zero; every later keep models it at 134.6 to 135.0 MB with 11.0 % zero (the scratch). When
the dead half is zero and when it is stale follows from the code [C]: a fresh sequence's `begin` zeroes both halves
(`cuda_state.zig` line 439, from `cuda_engine.zig` line 319); every prefill chunk reads `recAt(cur)`, writes
`recAt(1 - cur)` and flips the layer's bit (`cuda_forward.zig` line 276), every committed decode window does the
same, and a restore copies the entry's bytes back. So the dead half is zero only in a keep taken after a single
prefill chunk since the slot's reset (the receipt's 3,072-token first turn was one chunk); a longer prompt, a
continuation or any decode leaves it stale, and the stale half rides through every later keep and restore. (An
earlier version of this paragraph said a prompt pass does not flip the halves; the mixer's per-chunk flip shows it
does.) The stale half is 71.3 MB raw
and 51 MB modeled a rank a keep, the only prefix bytes the codec cannot remove: on the ledger's ratios 16 % of the
20k-token entry (314 to 262 MB) and 24 % of the 8k one (216 to 164 MB) [C]. By code reading [C] the dead half is never read after a
restore: `keepAs` refuses while a window is pending (`cuda_snapshots.zig`), the forward reads `recAt(cur)` and writes
`recAt(1 - cur)` (`cuda_forward.zig` lines 264 and 273), and commit's partial-accept replay reads `cur` and the
scratch into `1 - cur` before flipping, all inside a window that begins after the restore. Excluding it is a plan
change in `cuda_snapshot_layout.zig` (write the half the `cur` bits select, restore it to the live slot), the form the
fp8 head of 11.14 already takes (its 19.7 MB head holds one half), to be verified with the receipt harness of 11.17
(byte-identical replies; an outage) [U].

### 11.19 The live-half keep, served (`TF_GLM_KV_LIVE_HALF=1`, 2026-11-02)

The snapshot-plan change 11.18 asked for, implemented as commit 94d789153: a keep carries the KDA half the layers'
parity selects and leaves the other out (the arena head in two prefix parts around it), a restore builds its plan
from the kept state's own parity and leaves the slot's dead half as it was, the dump's manifest records the hole
(`half <h> <offset> <bytes>`, `kvdump.py` maps past it), the settings fingerprint carries the flag (layout 16), and
the replay forms are refused with it. Run on the ring from 04:38:57 to 04:59:15 UTC with production stopped and
restored; receipt `docs/receipts/glm-kv-live-half-94d78915/` (binary `a0082920…` on both ranks): the codec receipt's
nine-turn conversation under the lazy codec on five pinned workers with the ledger and the dump on, the whole head
(`both`, the flag at 0) then the live half (`live`), the same binary, compared with each other and with 11.17's plain
off run. All [O] unless marked.

- Replies: identical in every turn across off, both and live (the nine reply hashes of 11.17).
- The keeps: every live keep is exactly 71,303,168 bytes smaller than its whole-head twin, its classes outside the
  head byte-identical, its head's bytes before and after the hole identical, its live states and conv history
  identical through the reader (nine of nine keeps on each rank). In the whole-head run the dead half was all zero at
  keep 1 (position 2,048, one prefill chunk) and 0.4 % zero at keeps 2 to 9, and the parity alternated with the chunk
  count (2,048-row chunks), as 11.18's corrected reading says.
- Ledger, prefix class a keep (rank 0; rank 1 within 0.02 MB): 163,130,624 to 91,827,456 bytes; modeled 134.6 to
  135.0 MB to 71.5 to 71.8 MB at keeps 2 to 9, 83.5 to 71.5 MB at keep 1 (the zero half still cost 12 MB modeled).
- Codec totals (rank 0, lazy): raw 2,002,762,752 to 1,503,640,576 (seven compacted keeps times 71,303,168 exactly),
  stored 1,377,548,400 to 1,010,740,121 (26.6 % less), ratio 0.6878 to 0.6722, pages 958 to 660 against 720 to 486
  after compaction (26 % fewer held), 298 against 234 freed; lazy_encode 1689 to 1307 ms, decode 403 to 320,
  copy 72 to 57, lazy_copy 196 to 142, keep_io 4070 to 3130.
- Time to first token, both to live, seconds: 6.40 to 6.16, 6.31 to 6.08, 6.30 to 6.14, 6.92 to 6.72, 7.20 to 6.88,
  7.35 to 6.96, 8.42 to 7.93 (raw-keep), 5.70 to 5.44 (raw restore), 5.86 to 5.55 (partial). Reading [C]: 0.2 to
  0.5 s a turn lower with the instrumentation on, which the timers account for (the dump's write, the ledger's read
  and the restores' decode shrink with the bytes); without it the prompt pass saves the copy and the decode only, tens
  of milliseconds a turn. The saving that matters is the 71 MB a rank a keep of raw host pages and the 63 MB of
  modeled bytes a stale keep, with every other kept byte and every reply unchanged.
- Not covered [U]: drafts were off, so commit's partial-accept replay (`keep < pending`, which reads `recAt(cur)` and
  the window scratch into `recAt(1 - cur)`) did not run; one conversation, TP=2, no concurrent streams. The build
  node's `test-cuda-host` fails three unrelated steps whose generated inputs it lacks; every test that compiled
  passed (the log names none, so the ranks' fingerprint test is not identified in it), and a missing switch case in
  `cuda_ranks.zig` surfaced only in the native build, which compiles the file whole.
- The flag stays off by default until the maintainer's gate (`docs/recipes/glm-5.3-flash-cuda.md`).

## 12. Outcome tags (2026-11-02)

Successful, candidates for upstreaming: the lossless codec on real kept states (11.3), lazy compaction with
intermixed entries and pinned workers (11.12; the receipt 11.17), the exact page prune of the DSA index scan (host consumer plus the
long-context bound, 11.9; the device path as of commit 2b61900ad exact on the GPU and served byte-identical,
11.15, with no speed claim; its round 2 on main is a served regression with round 3 open), and Recall (11.4; first production sample 11.16). Candidate for the quality gate: the KDA head stored as fp8 without replay (11.14).
Experiments not adopted: the KDA replay restore (11.10, 11.14), the routed-expert prefetch and decode ring (11.2),
cross-prefix reuse and residual coding (sections 6 to 8). Closed by measurement: selection-count demotion (11.9),
Kronecker factorization and structured-prediction coding of kept states (11.18). Successful, candidate for
upstreaming: the live-half keep (`TF_GLM_KV_LIVE_HALF=1`, 11.19: served byte-identical, every keep 71 MB a rank
smaller, the receipt `glm-kv-live-half-94d78915`).
