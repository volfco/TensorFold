# Exact page prune of the DSA index scan: status

Tags as in `STATUS.md`: [O] observed, [U] untested. Runbook for the GPU: `RUNBOOK-prune.md`. Evidence that motivates it:
`docs/kv-cross-context-probe-2026-11-01.md` 11.9 (prunable pages 0 below 16k tokens, about a third at 64k to 128k,
0.47 to 0.75 past 128k, in every DSA layer).

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

## Awaits the GPU [U]

- Compile of `glm_index.cu` with nvcc (written without one): `tf_glm_index_scores_prune`, `tf_glm_prune_bounds`,
  `_plan`, `_cut`; their compile records (pending in `zig/src/core/contract/glm_index_prune_pending.zig`, every SM
  failing qualification until measured), and `tf_glm_index_scores`' record unchanged.
- `tf-cuda-test glm-index-check` (the existing kernels' bytes) and `glm-index-prune` (exactness at 32k, 131k, 204k
  with the glue's select, and for 512-row blocks also under `TF_GLM_INDEX_CUDA=1`'s `tf_glm_index_pools`; skip
  shares; control), `glm-index-prune timing`.
- The contract runner: `tf_glm_index_scores_prune` with a plan of 0s as a replacement of `tf_glm_scores`,
  `tf_glm_prune_bounds` gates 1 to 3, its swapped-table control red.
- Served byte identity off/on at about 32k and 200k context with a restore (RUNBOOK step 4), and the served effect on
  prefill and decode time.

## Open decisions

- The margin (2^-8 of the magnitude, as the host consumer's) is generous; 2^-12 would still be 4x the worst tensor-core
  sum error and prune more. Measure the skip share both ways on real queries before tightening.
- Seeds (8 pages) and the threshold (pools past 4,096): set from the long-context dump, not measured; the plan's cost
  (a bound per page per row on CUDA cores) is about 2 % of the full scan's MACs. Both are constants in
  `cuda_index_prune_dev.zig`.
- The skip share on real queries is not logged by the engine (the cut can count it: `Scratch.prune_stats`, a host read
  per select). A diagnostic switch would need a host sync per layer; not added.
- Strips live in the page (the pool's layout) rather than a side buffer: they follow the page through the table,
  their bytes are counted by the pool, and ranks compare page bytes. Kept states do not carry them (recomputed at
  restore from the rows, 400 pages x 12 caches at 204k tokens).
- `HINTS.md` ("Not built: the device path ...") and `STATUS.md` Track 2 still describe the device path as unbuilt:
  another session edits `tools/kvprobe/` in the main checkout, so this branch leaves them for the lead to update.
- The decode path uses the same three extra launches a select; at short contexts (the plain scan below 16k tokens)
  nothing changes. Whether the plan's extra launches pay at 32k (where few pages prune) is for the timing run.
