# KV-cache compression plan (from the Atlas research)

This plan applies the Atlas KV-cache compression research
([atlas-kv-cache-compression.md](atlas-kv-cache-compression.md)) to TensorFold.
It is a measure-first plan: every Atlas number is a single-capture, different-engine observation,
and TensorFold's exactness contracts are not negotiable at any rung.
Performance context: [atlas-performance-review.md](atlas-performance-review.md).

## Goal and scope

Goal: shrink the retained KV/prefix state TensorFold keeps for GLM-5.3-Flash conversations, and
quantify the honest cost/quality trade of each rung, behind the existing receipts discipline.
Scope is deliberately narrow: GLM-5.3-Flash (CUDA first, Metal where the same code paths exist) and
the lossless rungs only until Phase 2 proves allocator-visible savings.
Family-general work (Qwen, Nemotron, Flash Next pools) follows only after GLM evidence exists.

## Ground state (verified 2026-10-29)

- One KV page pool: `zig/src/core/kv_pool.zig` (`Layout.token_bytes`/`page_tokens`, `kv_pool.split`,
  lease/refcount semantics; live KV is never evicted, only store/prompt-cache holders reclaim).
- GLM CUDA pages: `zig/src/families/glm/cuda_pages.zig` + `zig/src/families/glm/cuda_state.zig`
  (`stateSizes`/`make`; latent cache rows + DSA pool-key rows per page; layout comment 13,056 B/token).
- GLM latent formats: `zig/src/families/glm/cuda_latent.zig` `cacheBytes` is 512×bf16 (1,024 B) or
  512×e4m3 + one fp32 scale (516 B); `--kv-cache fp8` works on dense windows only and is refused on
  pages by name (`error.Fp8OnPages`, `zig/src/families/glm/cuda_start.zig` `paging()`).
- Persistence tiers already exist and are not this plan's target: `--learn-conversations` /
  `--spill-gib` row stores (`zig/src/core/rows_store.zig`, `zig/src/sessions/*.zig`) and the RAM
  prompt cache (`zig/src/core/prompt_cache.zig`, radix match in `prompt_radix.zig`).
- Weight quantization (`docs/quantization.md`) is a separate surface; `--kv-cache fp8` is the only
  KV quantization today and is refused on pages (no pages path exists).
- No offline KV dump/capture tool exists (nearest: `GLM_CAPTURE` in `zig/tests/glm_run.zig`, a
  ≤16-row dev oracle). No KV compression hook exists.

## Non-goals

- No hot-path decode compression: Atlas measured decode as weight-bandwidth-bound, not
  KV-bandwidth-bound, and its fp8-KV speed story died on a same-binary control
  ([atlas-performance-review.md](atlas-performance-review.md) "KV cache precision").
- No remote KV servers, cluster coordinators or NVMe tier changes here; host-RAM tiers and
  prompt-cache crowding are issues #1511 and #1510, conversation persistence is #1312.
- No lossy rung ships without Phase 3 gates; no paper quantizer (TurboQuant/KVComp) is scheduled:
  their quality claims are model-evals on other architectures, their Atlas error figures are
  proxy-scope-conflicted (B0 defect 5), and truncation rungs are directly scorable with
  `tensorfold kl`. Revisit only behind a matched-scope TensorFold measurement
  ([atlas-kv-cache-compression.md](atlas-kv-cache-compression.md) "Paper quantizers").

## Implementation status (2026-11-01)

Lossless block codec (byte planes + order-0 rANS, `zig/src/core/kv_codec.zig`, `kv_rans.zig`), the Atlas column-codec
candidates (`kv_xform.zig`), the Phase 0 ledger and raw dump with a plan manifest (`kv_ledger.zig`,
`cuda_snapshot_codec.zig`, `cuda_snapshot_layout.zig` `manifest`), and GLM kept-state compaction in host pages
(`TF_GLM_KV_CODEC=1` with `TF_GLM_HOST_CACHE=1`) are written, host-tested and were run in the engine on the two GB10
nodes on 2026-11-01 (`docs/kv-cross-context-probe-2026-11-01.md` section 11). Phase 0's ledger and Phase 1's
acceptance are met on 16 real kept states (below). Phase 2: lazy compaction on a pinned worker pool with raw and
compacted entries intermixed is written, host-tested and served on the ring (`tools/kvprobe/STATUS-lazy.md`; the
prompt pass pays no encode, the workers run on the operator's cores, restores from compacted entries are byte-exact);
the repository-style receipt is `docs/receipts/glm-kv-codec-lazy-2b61900a/` (2026-11-02: nine turns in four configurations, replies identical, 0.24 s a compaction on pinned workers; without ledger and dump, lazy's TTFT equals off's within noise, so the codec in lazy mode is free on the prompt pass). The replay restore that rebuilds the KDA state instead of
storing it ran and is not adopted (`tools/kvprobe/STATUS-replay.md`, session record 11.10). The decode path is
untouched (Non-goals). 2026-11-02: `TF_GLM_KV_LIVE_HALF=1` (a keep carries the live KDA half alone,
`cuda_snapshot_layout.zig`) is written, host-tested and served byte-identical on the ring (session record 11.19,
receipt `docs/receipts/glm-kv-live-half-94d78915/`), off by default.

## Findings on the GB10 ring (2026-11-01, one capture per cell, [O] unless marked)

Kept GLM-5.3-Flash states (EXL3 4bpw, TP=2, BF16 latent, 16 keeps of 6k to 15k-token prompts, 4.74 GB on rank 0):

| class | share of a kept entry | order-0 modeled ratio |
|---|---|---|
| prefix (KDA recurrent states, conv history, arena head) | 55.0 % | 0.826 |
| latent (MLA rows, 11 layers) | 35.0 % | 0.667 |
| ring (decode rings) | 7.8 % | 0.671 |
| pooled (DSA pool keys) | 2.2 % | 0.665 |

Codec on the same keeps (`tf-kv-codec-bench --file`, bit-exact round trips): plain 0.718 of raw (min 0.707, max
0.724), encode 1.74 GB/s and decode 6.4 GB/s at 8 Grace threads (0.32 / 1.21 at one); the Atlas column codecs 0.717
at 3x the encode cost (`decisions.md` 2026-11-01: plain stays the default); deflate-6 0.819; ClickHouse ALP-RD 0.83,
Gorilla/DoubleDelta/FPC 0.94. A 1 GiB state keeps in about 0.6 s and restores in 0.16 s at 8 threads.

The KDA state, the largest class, is the one lever past lossless: per head it has an effective rank of 2 to 12 of 128
(rank for 90 % of energy at most 14; for 99 %, median 16 but up to 83), it deviates only 0.01 to 0.4 across different
prefixes, and the serving stack's gate dump gives per-head decay horizons with a median of 1.4 to 8.6 tokens and a p90
of 4 to 40. The replay restore built on this (drop the prefix class, rebuild it by replaying the last W tokens over
the stored rows from a zero state) was measured and refuted as implemented: 0.87 / 0.73 / 0.67 relative error at
W = 64 / 256 / 512, because the deeper layers' inputs are wrong while the lower layers' states converge; only a
replay from a coarse stored state remains open (session record 11.10). Low-rank storage is weaker: rank 8 loses
20 % of a head's state, and 99 % of energy needs rank 16 for half the heads but 50 to 80 for the tail [C].

Kronecker factorization is closed (session record 11.18, 2026-11-02, two keeps of 8k and 20k tokens on the CPU): a
nearest Kronecker product captures 0.63 of a head's energy where the SVD captures 0.98 at the same 4x budget, and a
third to two thirds of the SVD's energy on the latent rows at matched cost; predicting the rows from bf16 factors and
coding the exact residual saves 0.2 to 0.3 bits a value against 4.1 bits of factors at rank 128. The floor is the
mantissa: 10.6 of a bf16 row value's 16 bits and 27.3 of an fp32 state value's 32 are order-0 entropy, and no
structural model pays for its own factors. The lossless lever the same keeps expose is the dead half of the
double-buffered KDA state: zero (and free) only in a keep taken after a single prefill chunk since the slot's reset,
since every prefill chunk and every decode window flips the halves; stale in every other keep and carried on by
restores: 71 MB raw and 51 MB modeled a rank a keep, 16 % of a 20k-token entry and 24 % of an 8k one. The
snapshot-plan change that writes the live half only (`TF_GLM_KV_LIVE_HALF=1`, `cuda_snapshot_layout.zig`, commit
94d789153) was served on the ring the same day (session record 11.19, receipt `docs/receipts/glm-kv-live-half-94d78915/`):
replies identical to the whole-head run in all nine turns, every keep exactly 71,303,168 bytes smaller a rank with
every other byte identical, the prefix class modeled at 71.5 to 71.8 MB in place of 134.6 to 135.0 on stale keeps,
the codec's stored bytes 26.6 % lower over the run and the pages held after compaction 26 % fewer. Off by default
until the maintainer's gate; drafts-on coverage is still open. Rounding the live state to bf16 is a
relative error of 0.17 % at the median (2x; the fp8 head's 2.6 % is 4x).

Closed by the same captures: cross-prefix reuse of attention state (latent rows deviate 0.38 to 0.71 at the deep
plateau by content type, a third of a token's experts change in the middle layers); cross-session residual coding of
rows (XOR against the other context's row gains only in the first four attention layers); routed-expert prefetch on
GB10 (a row's experts are in the previous 16 rows' sets 47 to 75 % of the time, the cheap predictors reach recall 0.26
to 0.51 at k=8, three to six bytes fetched per useful byte; the prefetch and ring kernels are bit-exact and within noise
of stock). The exact page prune of the DSA index scan has 0 % prunable pages below about 16k tokens, about a third at 64k to
128k and half to three quarters past 128k (a 204,800-token capture, session record 11.9): its device path is worth
building for long contexts. Demoting rows by selection counts is closed: only 1 to 7 % of a long prompt's pools are
never selected by the sampled queries.

## Phase 0 — measure (gate M)

1. **Retained-entry byte ledger.** A dump tool (new `tools/zig/kv_dump.py` driver + a narrow engine
   hook under `GLM_CAPTURE`-style test builds) records, per request: every retained byte class
   (KDA recurrent H, conv history, MLA latent rows, DSA index rows, page-table/scale scratch),
   aliasing, and snapshot-vs-live duplication. Atlas's biggest accounting trap was that 89.26 % of
   the entry was the snapshot; TensorFold must know its own shares first.
   Acceptance: a reconciled formula `covered = f(blocks, shape)` matching every captured manifest
   with delta 0 (the two-shape lesson: complete-match and grow captures legitimately differ).
2. **Bit-pattern checks.** Test whether conv history and KDA H are widened low-precision in
   TensorFold's producers (zero low bits ⇒ exact field-split packing is available for free);
   verify on both backends, never assume the CUDA producer matches Metal.
3. **Distributions.** Exponent/magnitude histograms per stream and per head spread, from the same
   dumps; codec choice (plain fields vs row-centered vs shared-exponent) is made from these, not
   from Atlas's ordering.
4. **Warm-path boundary probes.** Extend `tools/bench_cache.py` cells with prompts straddling
   TensorFold's page/block boundary on both sides (the Atlas complete-match quirk: `cached_tokens`
   can report a full hit while the request recomputes at seed cost). Acceptance: `--require-reuse`
   and `--require-cold` cells demonstrate which side of the boundary resumes and which recomputes.
5. **Restore/encode/decode cost budget.** Time-to-restore a compressed record vs re-prefill of the
   same tokens, on both backends. This number decides rung admissibility (Atlas never measured it).
6. **FP8-gate interaction matrix.** Documented combinations of `TF_KV_PAGES` × `--kv-cache fp8` ×
   a compressed format; anything incompatible keeps today's fail-loud refusal pattern.

Measurement discipline (ported from the Atlas trap catalog): same-binary control legs, byte-identity
hashes before latency claims, n≥3 with rep-1 warmup discarded, shape-diff-first when bench and serve
disagree, no L2-inflated microbench absolutes, and store raw receipts alongside summaries.

## Phase 1 — lossless codec experiments (offline only)

Candidate transforms (re-ranked from Phase 0 distributions): field split (sign/mantissa vs exponent),
row-centered exponent coding, shared-exponent frame-of-reference, exact low-precision packing where
Phase 0 proves the producer invariant, and a lossless container with per-stream raw fallback.
Skip (measured negatives in Atlas): Gorilla, integer delta, double-difference, sign-XOR prediction,
dictionaries, low-rank SVD, per-head mixed precision.

Container contract (the KVEC1 lesson, reimplemented in Zig): modeled bytes == serialized bytes by
construction; bounded refusing decode (forged lengths, truncation, checksums refused before
allocation); explicit raw fallback when a producer invariant fails; versioned records with code and
capture hashes.

Oracle tests (the Atlas B0 defect register as a checklist): known-value numeric oracles for every
helper (sign-bit layout, entropy sign), consistent byte pricing (metadata, scales and codebooks
charged), full round trip proven before any size is reported (store reconstruction anchors), and
complete candidate enumeration with resumable, non-repeating search.

Acceptance: per-stream and aggregate modeled + **actual serialized** sizes on ≥3 captured shapes;
bit-exact round trip on every stream; encode/decode cost within the Phase 0 budget.

## Phase 2 — engine integration (gated)

Plugs into `zig/src/core/kv_pool.zig` (`Layout.token_bytes`, page sizing), the GLM format switch
(`zig/src/families/glm/cuda_latent.zig` `cacheBytes`/`latentWriteFormat`, `cuda_state.zig`
`stateSizes`), and the write/read glue kernels (`tf_glm_lat_write`, `tf_glm_chunks_*`).
Rules:
- Page-granular formats must respect the pool's page table; window formats stay behind the existing
  refusal gates (`error.Fp8OnPages` is the pattern for every incompatible pairing).
- Lossless restore must reproduce exact words **before** any lossy quantization; the resumed == fresh
  contract (`CONTRIBUTING.md` exactness list) is the hard gate.
- Every new or changed kernel registers with the kernel contract (`docs/kernel-contract.md`, five
  gates with planted controls) and posts each backend's `SUMMARY` line; modules stay ≤600 lines.
- Cross-rank agreement fields (`zig/src/families/glm/cuda_ranks.zig`) extend to the format, as they
  already do for `kv8`/`pages`.

Acceptance: allocator-visible savings measured on a served run (not modeled); a receipt in the
`docs/receipts/glm-kv-pages-*` style (Scope → Results → Reproduction → coverage limits →
redaction report, CI logs and planted mutations, `SHA256SUMS`); `tools/zig/reuse_served.py`
record/replay/compare shows token-digest equality with compression on and off; admission math
(`Layout.need`) reflects the new `token_bytes`.

## Phase 3 — lossy rungs (optional, gated)

Only after Phase 2 lands. Freeze margins before any evaluation (the Atlas registration practice):
non-inferiority on a fixed quality probe set with an A–A repeat floor, and a restore-latency
allowance; score with `tensorfold kl MODEL --dump/--against` plus served token-digest probes.
Candidate rungs are simple truncation/rounding of H and latent rows (clear-mantissa / f16-class),
not paper quantizers. Kill criteria: any quality probe outside the frozen margin, any
resumed ≠ fresh mismatch, or restore cost above re-prefill of the covered tokens.

## Decision gates

| Gate | Question | Kill criterion |
|---|---|---|
| M0 | Does a retained entry hold compressible slack (sizing + distributions)? | savings ceiling < 10 % modeled |
| M1 | Do producer invariants unlock exact packing? | no zero-bit/alias structure (generic codecs only; go/no-go on C1) |
| C1 | Does a lossless codec beat plain storage inside the cost budget? | no candidate ≥ 10 % smaller within budget |
| I1 | Does lossless compression survive the exactness contracts end-to-end? | any resumed ≠ fresh mismatch |
| I2 | Are allocator-visible savings real and restore cheaper than re-prefill? | modeled savings not visible, or restore loses to prefill |
| Q1 | Do lossy rungs hold the frozen quality margins? | any probe outside margin or below the A–A floor |

## Coordination and references

Coordinate via ATC before edits land (`docs/atc/SKILL.md`); related open issues: #1511 (KV pool
host-RAM tier), #1510 (pool vs prompt cache), #1312 (GLM conversation state eviction), #1293 (Metal
KV pages from the core pool), #1156 (Qwen attention caches on the core pool). This plan touches
GLM CUDA latent/paging first and should not collide with those, but `atc who` on `kv_pool.zig` and
`cuda_state.zig` is required before opening a PR.

In-repo references: `docs/atlas-kv-cache-compression.md` (findings + trust ledger),
`docs/atlas-performance-review.md` (perf context, kill criteria), `docs/cache-benchmarks.md`
(receipt semantics), `docs/receipts/glm-kv-pages-*/README.md` (receipt structure),
`docs/kernel-contract.md`, `CONTRIBUTING.md` (exactness contracts, module limits).
External references: TurboQuant arXiv:2504.19874; KVComp arXiv:2509.00579; SplitZip arXiv:2605.01708;
ZSTD KV study arXiv:2503.18869; KV compression survey arXiv:2508.06297; Gorilla TSDB (VLDB 2015);
ClickHouse CODEC clause; QJL add-back caution (dejan.ai 2026); vLLM decode context parallelism
(Apache-2.0) as token-shard prior art; Codec evaluation shortlist (dietgpu, pcodec, DeepZip, zipnn,
GPU-lossless-compression, ICS23-GPULZ, zipserv_kvcache_framework).
