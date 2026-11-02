# Cache-entry hint sidecars

An external agent post-processes a kept state (or its `TF_GLM_KV_DUMP` capture) and writes a sidecar next to the
entry. The engine-side reader is `zig/src/core/kv_hints.zig` (tests: `zig build test-kv-hints`); the post-processor
is `tools/kvprobe/hints.py` (tests: `python3 tools/kvprobe/test_hints.py`). Both build the same golden file byte for
byte (its sha256 is pinned in both test files). Nothing in the engine reads a sidecar yet; the first consumer is the
host prototype of the exact page prune (`zig/src/families/glm/cuda_index_prune.zig`).

## Format, version 1 (little-endian throughout)

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `TFKVHNT1` |
| 8 | 4 | version, 1 |
| 12 | 4 | header bytes, 192 |
| 16 | 4 | kept-state layout version (the dump manifest's `tfkvplan` version, 1) |
| 20 | 4 | page rows (GLM: 512 tokens) |
| 24 | 4 | pool (GLM: 4 tokens) |
| 28 | 4 | index key width `i_dim` |
| 32 | 4 | DSA layers (caches) |
| 36 | 4 | section count n |
| 40 | 8 | token count: the kept state's position |
| 48 | 8 | token hash: Zig `std.hash.Wyhash.hash(0, ids as u32 little-endian)` over those tokens |
| 56 | 32 | model: sha256 of `config.json` (the rank fingerprint's `config`) |
| 88 | 32 | checkpoint digest (the rank fingerprint's `checkpoint`, as 32 bytes) |
| 120 | 8 | file bytes, checksum included |
| 128 | 4 | routed experts (GLM-5.3-Flash: 288) |
| 132 | 4 | MoE layers (42) |
| 136 | 56 | reserved, zero |
| 192 | 32 n | section table: kind u32, layer u32, count u32, width u32, offset u64, bytes u64 |
| ... | | payloads, each 8-byte aligned, in table order, not overlapping |
| end - 8 | 8 | Wyhash(0) of every byte before it |

Sections:

| kind | layer | count x width | payload |
|---|---|---|---|
| 1 bounds | DSA cache index | whole pages x 4 i_dim | per page: the minimum then the maximum of each index-key dimension over its pooled keys, bf16 |
| 2 selected | DSA cache index | pools (token count / pool) x 4 | u32: how often the indexer selected the pool in the recorded rows |
| 3 hot | all (0xFFFFFFFF) | n x 4 | u32 page indices, a restore order: the last recorded round's touched pages first, then the rest ascending |
| 4 experts | all | n x 24 | token u32, MoE layer u16, count u16 (at most 8), expert ids u16[8]: a token-keyed expert prior |

The reader refuses by name, before any section is used: `HintTruncated` (short file or recorded length differs),
`HintMagic`, `HintVersion`, `HintHeader`, `HintReserved`, `HintChecksum`, `HintModel`, `HintCheckpoint`, `HintLayout`,
`HintPageSize` (page rows or pool), `HintShape` (i_dim, DSA layers, experts, MoE layers), `HintTokenCount`,
`HintTokens`, `HintSection` (table past the file, payload out of order, misaligned, out of bounds, count x width not
its bytes, more bound pages than whole pages, a selection count not one a pool), `HintKind`, `HintLayer`, `HintBounds`
(a minimum above its maximum or a non-finite bound), `HintHotPage` (a page the entry does not hold), `HintExpert` (an
expert id, MoE layer or count out of range). Every refusal has a test that plants exactly that fault.

## The exact page prune (bounds)

The DSA indexer scores a pool as `sum_h w_h * relu(scale * q_h . k)` (`tools/glm_index_emulate/emulate.cpp`
`model_score`; the head weights can be negative) and keeps the best 512 pools. For a page with per-dimension bounds
`[lo, hi]`, `q_h . k` lies in `[sum_d min(q lo, q hi), sum_d max(q lo, q hi)]`; relu is monotone and the scale
positive, so a positive weight takes the upper end and a negative weight the lower end. A page whose bound (plus a
rounding margin) is strictly below the running 512th best score holds no selected pool; skipping it leaves the
selected set identical under any tie rule that depends only on score and pool. The host consumer scans pages in
descending bound order and stops there; its oracle test compares against the full scan on clustered, unclustered,
tied, negative-weight and edge-k cases and checks that the variant bounding negative weights from above is caught.

Not built: the device path (a bounds pass and a page-list launch of `tf_glm_index_scores`), so `TF_GLM_INDEX_PRUNE=1`
is refused by name on a device run (`error.IndexPruneHostOnly`). It is a kernel change and needs the maintainer gate.
What decides whether to build it: the prunable fraction on real queries. `hints.py --estimate` gives an upper bound
from a routing dump's recorded selections (the fraction of whole pages holding no selected pool); the exact fraction
needs the indexer's query rows and head weights, which neither dump records yet (a few hundred rows per layer would
do; an extension of the serving-stack patch 0710 or a Zig-side capture).

## Design notes (no engine code)

Demotion of never-selected latent rows to host memory (selected counts).
- Mechanism: rows of pools the indexer never selected over a window move to a host tier; a later selection of one
  faults it back before the sparse attention reads it.
- Measurement that would justify it: on routing dumps of real sessions, per DSA layer, the fraction of pools never
  selected over the last N decode steps and the probability that such a pool is selected within the next M steps
  (the refetch rate). It pays only if a large never-selected fraction coexists with a refetch rate whose cost (host
  to device on a discrete card; on GB10's unified memory there is no transfer to save, only capacity) stays below
  the capacity it frees. Decode-row selections are not recorded yet (patch 0710 records prefill rows).

Selection reuse across consecutive decode steps.
- Mechanism: skip the indexer on step t+1 and reuse step t's set, or rescore only pools near the set.
- It is approximate: the selection changes, so it needs the Phase 3 quality gate (frozen margins, KL against the
  exact path), not just a bit check.
- Measurement: Jaccard of consecutive rows' selections (routing.py section B reports it for prefill rows), the
  indexer's share of decode time on GB10 (nsys), and the KL of the reused path on a fixed probe set.

Restore order from the hot pages.
- Mechanism: a resumed session restores (or decodes, under the KV codec) the last round's touched pages first and
  starts decoding while the rest arrive.
- Measurement: restore time against the first decode step's start under `TF_GLM_KV_CODEC` (KVCODEC decode_ms in the
  rank-0 log) on GB10; it matters only if restore sits on the critical path of time to first token, which the
  served receipts must show first. Today a restore completes before the prompt pass continues.

Token-keyed expert prior.
- Feeds the expert-tile prefetch experiment (Track 4): the prior's recall@8 on held-out rows is reported by
  `tools/kvprobe/routing.py` section D; prefetching on a prior only helps if the router-to-first-expert-load gap
  is a stall (on unified memory the expert read is DRAM bandwidth, not a transfer).


## Device path (2026-11-02)

The exact page prune now has a device path behind `TF_GLM_INDEX_PRUNE=1` (`zig/src/families/glm/cuda_index_prune_dev.zig`, kernels in `zig/kernels/cuda/glm_index_prune.cuh`): the per-page bounds live as a 512-byte strip in each DSA cache's part of a KV page (written with the pooled keys and recomputed after a restore), and selects over more than 4,096 pools run a bounds-first scan whose selection equals the full scan's bit for bit (argument and checks in `STATUS-prune.md`). The sidecar's page bounds remain the form an external post-processor can produce for kept entries; the engine's strips are the live form. On the GB10 (2026-11-02) the device path is exact (65 of 65 kernel cases, six served replies byte-identical off and on at 32k and 200k) but not yet a speedup: the index scan is under 1 % of a decode step at 200k on this ring and the pruned single-row select pays four extra launches (session record 11.15).
