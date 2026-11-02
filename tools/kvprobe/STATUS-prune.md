# Exact page prune of the DSA index scan: status

Tags as in `STATUS.md`: [O] observed, [U] untested. Runbook for the GPU: `RUNBOOK-prune.md`. Evidence that motivates it:
`docs/kv-cross-context-probe-2026-11-01.md` 11.9 (prunable pages 0 below 16k tokens, about a third at 64k to 128k,
0.47 to 0.75 past 128k, in every DSA layer).

## GB10 round 1 (2026-11-02) and what changed after it

- Observed [O]: `glm-index-prune` 65 of 65 PASS, control caught; 32k served turns byte-identical off/on. Contract:
  `tf_glm_index_scores_prune` (every page planned) prints `FAIL nonfinite ... 1024 of 16384` followed by `KNOWN
  nonfinite ... relu is fmaxf(x, 0)`: the same line and count as `tf_glm_scores` and `tf_glm_index_scores` (the spec
  carries tf_glm_scores' known finding; SUMMARY 0 fail, 7 known). Not a prune bug: the planned kernel propagates as the
  kernel it replaces. The emulation now plants the same values (a NaN query value, +inf/-inf/NaN keys): selections and
  every written score equal the full scan's, prompt and decode tiling.
- Observed [O]: the pruned select was slower at every measured length on synthetic keys (1 row: x0.29 at 32k and 131k;
  wall clock, eager launches). GLM's CUDA decode is not graph-captured in this tree (no capture in
  `zig/src/families/glm/`), so the launch latency is served cost too.
- Changed: (1) the plan runs as cdiv(pages, 32) blocks a row (it was one block a row, serial over the pages), the row's
  last block (an atomic counter it resets) picks the seeds; (2) the cut is no launch of its own: the first planned-scores
  pass's last CTA of each 8-row group cuts its rows and writes -inf for the skipped pages; the planned-scores CTAs with no
  planned tile exit before loading their queries; the cut ranks the first 16 scored pages' pools (any subset of
  scored pools is sound) with a warp-parallel digit search. A select is 3 launches (plan, scores with the cut,
  scores), was 4. (3) `TF_GLM_INDEX_PRUNE_MIN_PAGES` (0 to 1,024, default 256 = 131,072 tokens, a typo refused by name
  at start): a select prunes only when its last row has that many whole pages. (4) Served logs: `INDEXPRUNE rank=R pass
  tokens=T selects=S pages=P skipped=K share=X` at the end of each prompt pass (counts since the previous line: that
  pass's chunks and the decode steps before it), and `INDEXPRUNE rank=R total min_pages=M ...` at teardown; counts are
  u64, all DSA layers together. (5) `glm-index-prune timing` prints wall and CUDA-event (GPU) microseconds a row and the
  pruned runs' skip share on each `TIME` line. (6) The pending records join `cuda_pending`, so
  `kernel-contract-limits` walks them (the cut's record is gone with its launch).
- Awaits GB10: the new timing (wall and GPU), RECORD lines for the three launches, and a default for the knob from them.

## Contract

- Behaviour, behind `TF_GLM_INDEX_PRUNE=1` (unset or 0: off; anything else refused by name at settings,
  `TF_GLM_INDEX_PRUNE: 0 or 1`). Off, no byte of any default path changes: page layout, kv_pool sizing, kernels'
  arithmetic, launches and selections are as before (`tf_glm_index_scores` became `scores_body<false>`, the same
  source arithmetic; its bytes are checked by `glm-index-check` on the GPU [U]).
- On: each DSA cache's part of a KV page (every DSA layer and the MTP layer) gains a 512-byte strip after its 128
  pooled keys: each of the 128 index-key dimensions' minimum, then maximum, over the page's pooled keys, bf16 (exact:
  a min or max of bf16 values; NaN when the column holds one). The page grows from 12 x 557,056 to 12 x 557,568 bytes
  on GLM-5.3-Flash (13,056 to 13,068 bytes a token), padded to whole latent rows for any layer count; kv_pool sizes
  pages from the layout, so the pool holds proportionally fewer pages (0.09 %).
- Strips are written by `tf_glm_prune_bounds` right after `tf_glm_pool_keys` (indexUpdate) for every page the write
  completes (prompt chunks, decode and verify windows, MTP rows), and for every whole page of a state restored from
  the prompt cache (`cuda_snapshots.restore`, raw or codec entries; an entry carries no strips). A rollback that
  rewinds into a page leaves it partial; it is never pruned until a later write completes it again and rewrites its
  strip. A cache on striped pages whose Set lacks the kernels refuses the write by name (`IndexPruneNoKernels`) rather
  than leave a stale strip.
- A select on striped pages over a pool bucket past 4,096 (contexts past 16,384 tokens; at most 1,024 pages a row,
  contexts to 524,288 tokens) runs: `tf_glm_prune_plan` (a block a row: each whole visible page's bound; the 8 pages of
  largest bound, and every page that cannot be bounded, planned for scoring), `tf_glm_index_scores_prune` (the planned
  pages scored with tf_glm_index_scores' arithmetic, the candidates written -inf), `tf_glm_prune_cut` (the row's k-th
  best order key among the scored pools; a candidate whose bound's key is strictly below it stays skipped, the rest are
  planned), `tf_glm_index_scores_prune` again (the rest scored), then the unchanged select (`tf_glm_select_rows`, its
  wide form, or `tf_glm_index_pools` under `TF_GLM_INDEX_CUDA=1`) over the whole row. Shorter buckets, windows (no
  pages) and unaligned caches take the plain scan.
- Refusals by name at engine start: `IndexPruneSettings` (setting and kernels disagree), `IndexPruneNeedsPages`
  (sequences not on pages: one rank needs `TF_KV_PAGES=1`), `IndexPruneCapacity` (more than 1,024 pages a row),
  `IndexPruneNoKernels` (an image without them). Ranks agree through the fingerprint (`index_prune`, layout 15); a
  page of other bytes on a follower is `RankMismatch` as before.
- Exactness (the argument): the planned scores of a pool are bit for bit the full scan's (same kernel body, same
  arithmetic), so the cut's k-th best key among the scored pools is at most the full row's k-th best key. The bound of
  a page is at least the kernel's score of every pool on it: positive head weights take the dot's upper end
  (sum of max(q lo, q hi)), negative weights its lower end (relu monotone, scale positive), every product of bf16
  values exact, and a margin of 2^-8 of the terms' magnitude plus absolute terms covers the tensor-core fp32 sums, the
  scale, the 32-head sum and the bound's own fp32 sum (each well under 2^-14 of that magnitude); a non-finite strip,
  query or weight, or a magnitude of 1e30 or more, gives NaN, which never prunes. So a skipped pool's key is strictly
  below the full row's k-th best key: it is not selected by the full scan, and as -inf it is not selected by the
  pruned one; every pool at or above the k-th best key is scored and keeps its bits, so the k-th best key, the count
  above it and the ties at it (resolved to the lower pool by the unchanged select) are the same. The selection, its
  order and the token lists are therefore identical, ties included.
- Owning modules: `zig/kernels/cuda/glm_index_prune.cuh`, `glm_index_scores.cuh` (`scores_body<PRUNE>`),
  `glm_index.cu`; `zig/src/families/glm/cuda_index_prune_dev.zig` (launches, rules, admission),
  `cuda_index_prune_ref.zig` (host mirror on the device layout), `cuda_index_prune.zig` (setting, host consumer);
  one-line hooks in `cuda_sparse.zig`, `cuda_state.zig` (layout), `cuda_pages.zig`, `cuda_dsa_block.zig`,
  `cuda_latent.zig`, `cuda_glue.zig`, `cuda_forward.zig`, `cuda_engine.zig`, `cuda_snapshots.zig`, `cuda_ranks.zig`.

## What passed (x86 host, Zig 0.17.0, no GPU, no nvcc) [O]

- `zig build test-glm-index-host`: 21 of 21, including the host oracle (280 cases, unsound bound caught), the device
  layout mirror (strips written chunk by chunk then decode-sized, nine cases x 3 seeds: clustered, positive weights,
  random keys, tied pages, every key equal, a NaN key, k 512 with 8 seeds, 32 heads, a short context; selection equal,
  pages skipped and not all), both planted faults caught (negative weights bounded from above; strips left zero), the
  strip rule (min, max, NaN, written only once whole), the page layout (512 bytes a part, whole latent rows), the
  launch rules and the refusals.
- `zig build test-glm-index-prune-emulate`: the .cuh kernels themselves compiled as host C++ (threads as OS threads,
  the mma as emulate.cpp models it), the engine's sequence of launches: strips after chunked writes equal the host's;
  nine cases, every selection equal to the full scan's (`tf_glm_index_scores` + stable top-K), no stray score, 730 of
  2,990 whole pages skipped; both planted faults caught. `zig build test-glm-index-emulate` (the existing kernels after
  the template change) passes.
- `cuda_ranks_test`: the setting's parse, default and refusal, and the fingerprint naming `index_prune`.
- Exact commands and counts (Zig 0.17.0 on x86, 2026-11-01):
  - `zig build test-glm-index-host`: 21 of 21 tests passed.
  - `zig build test-glm-index-host test-cuda-host compile-glm-host compile-cuda-tests test-glm-index-emulate
    test-glm-index-prune-emulate`: 294 of 294 steps, 1,914 of 1,922 tests passed (8 skipped), exit 0.
  - The prune emulation's own output (`zig c++ -std=c++20 -O1 -Itools/glm_index_emulate
    tools/glm_index_emulate/prune.cpp`, then run): 9 cases, 0 selections differ, 0 stray scores, 730 of 2,990 whole
    pages skipped; both planted faults caught; `PASS`.
  - `zig build test`: 452 of 452 steps, 1,902 of 1,920 tests passed (18 skipped), exit 0.
  - `python3 tools/zig/lean_check.py` against the base's output, `tools/zig/lean_new.py`: 569 problems, 574 on the
    base, 0 new.

## GPU results, 2026-11-02 (GB10, sm_121, nvcc 13.0; session record 11.15) [O]

- Compiles: `glm_index.cu` built with nvcc at the first try; `tf_glm_index_scores`' sm_121 record unchanged (124
  registers, 32,768 static, bound 256, "record current"). The limits pass printed no RECORD line for the four prune
  launches: `registry.zig` imports the pending file for its tests only and does not append its records to the set
  the pass walks (open; sent to the author).
- `glm-index-check` PASS; `glm-index-prune` 65 of 65 PASS at 32k, 131k and 204k tokens, 7 cases, 512 / 8 / 1 rows,
  the control caught (441 of 505 pages skipped with strips unwritten changes the selection). Skip shares on the
  synthetic keys 0 to 0.20 (0.34 for tied pages at one row).
- Contract: `tf_glm_index_scores_prune` passes repeat, shapes (bit-equal to `tf_glm_scores`) and sums, fails the
  non-finite gate ("1024 of 16384 outputs differ in class; first at 0: finite where float64 gives nan"; open);
  `tf_glm_prune_bounds` passes gates 1 to 3.
- Timing (wall clock a query row, median of five): pruned slower everywhere on the synthetic keys: 512 rows x0.79
  (32k) to x0.97 (204k); 8 rows x0.39 to x0.46; 1 row x0.29 to x0.30 (86 vs 25 us at 32k, 240 vs 70 us at 204k).
  Reading: launch latency of the four extra kernels on an idle stream, not GPU work; not measured inside a graph.
- Served off vs on (ring, 245,760 context, host cache on, greedy, 256 tokens): six replies (32k and 200k documents;
  first, restored, follow-up) byte-identical; TTFT 48.1 / 47.4 s (32k) and 266.9 / 263.5 s (200k), decode 31 to 32 s
  either way: within noise. The engine logs no skip share.
- Bound on the lever (calculated): a decode step is 124 ms at 32k and at 200k on this ring; the index scan of a step
  is 11 x 70 to 80 us, under 1 %; the scan is about 4 % of a long prefill's time a token. The flag stays off by
  default; the exact mechanism is the candidate, not a speed claim on GB10.

## Open decisions

- The margin (2^-8 of the magnitude, as the host consumer's) is generous; 2^-12 would still be 4x the worst tensor-core
  sum error and prune more. Measure the skip share both ways on real queries before tightening.
- Seeds (8 pages) and the threshold (pools past 4,096): set from the long-context dump, not measured; the plan's cost
  (a bound per page per row on CUDA cores) is about 2 % of the full scan's MACs. Both are constants in
  `cuda_index_prune_dev.zig`.
- The skip share on real queries is not logged by the engine (the cut can count it: `Scratch.prune_stats`, a host read
  per select). The served run of 2026-11-02 therefore has no skip share; a totals line at teardown and an `info:`
  line per prompt pass were requested of the author.
- Strips live in the page (the pool's layout) rather than a side buffer: they follow the page through the table,
  their bytes are counted by the pool, and ranks compare page bytes. Kept states do not carry them (recomputed at
  restore from the rows, 400 pages x 12 caches at 204k tokens).
- The decode path uses the same extra launches a select; at short contexts (the plain scan below 16k tokens)
  nothing changes. The timing run answered the rest: the extra launches cost more than the pages skipped at every
  measured length on synthetic keys (x0.29 at one row); fewer launches on the single-row path, GPU-time timing and
  a threshold knob are with the author.
