# Decision log

## 2026-10-29 — Atlas salvage: port deliverable scope (major)

Reference: docs/atlas-performance-review.md, docs/atlas-kv-cache-compression.md,
docs/kv-cache-compression-plan.md (new).
Choices: (a) review report only; (b) report + research dossier + adapted plan in TensorFold docs;
(c) (b) plus experimental engine code.
Decision: **(b)**.
Why: the user asked to bring research forward "to be applied", and the Atlas program's own evidence
shows its measured-cost gaps (B2/B3 never ran) make implementation premature; the plan defers code
to gated phases.

## 2026-10-29 — Review breadth: all performance research (minor)

Choices: KV-only; KV + GLM-relevant; all research (kernels, decode, MTP, MoE, NVMe, concurrency).
Decision: **all**, assessed with HIGH/MED/LOW TensorFold relevance ratings.
Why: the user asked for "any of the performance research or learnings"; relevance ratings keep the carry set small.

## 2026-10-29 — License handling for AGPLv3 Atlas material (major)

Atlas is AGPLv3; TensorFold is Apache-2.0. Choices: knowledge/numbers only; adapted code allowed; outside the repos.
Decision: **TensorFold docs with adapted code allowed** — user holds relicense rights to the Atlas
work.
Why: user authorization ("code allowed"); docs still cite Atlas artifact paths as provenance and
copy no wholesale modules.

## 2026-10-29 — ATC coordination for this work (minor)

`atc board --json` (87 open issues) has no issue for a docs/research port. CONTRIBUTING.md requires
an issue for new model family, backend or arithmetic changes — none of which this docs work is.
Decision: work locally without a claim; run `atc who`/`atc overlap` before any PR; open an issue
when Phase 2 implementation starts (it adds kernels/arithmetic).
Why: avoids inventing external side effects (issue creation) while honoring the coordination rules
where they apply.

## 2026-10-29 — Docs layout for the port (minor)

Choices: flat docs/*.md (repo convention); new docs/research/ tree.
Decision: **flat** — docs/atlas-performance-review.md, docs/atlas-kv-cache-compression.md,
docs/kv-cache-compression-plan.md.
Why: every existing doc is a flat kebab-case file; three files do not justify a new tree.

## 2026-10-29 — Trust-tag convention carried from Atlas (minor)

Decision: every carried number keeps a trust tag (verified-modeled-lossless / repaired-in-B0 /
defective-do-not-use / unverified) and an Atlas artifact path.
Why: the Atlas program's B0 campaign proved self-reported verdicts wrong in both directions;
TensorFold docs must not launder defective numbers.

## 2026-10-29 — Plan architecture: measure-first, lossless-first (major)

Choices: port Atlas's B2/B3-style integration directly; measure-first with gated phases; skip to
lossy.
Decision: **measure-first, GLM CUDA first, lossless rungs only until allocator-visible savings
exist; lossy behind frozen margins; no hot-path decode compression.**
Why: Atlas measured fp8-KV as a capacity lever and a decode loss (same-binary control), and never
measured encode/decode/restore cost; TensorFold's exactness contracts make unmeasured integration
riskier than measurement tooling.

## 2026-10-29 — EXL3 kernel comparison scope (major)

Reference: docs/atlas-exl3-kernel-review.md (new).
Choices: (a) report + ranked port candidates; (b) report + implement top wins; (c) chat only;
scope strict matched pairs / matched + GLM-side / everything in exl3/;
evidence static + Atlas's own measurements / include local microbenchmarks.
Decision: **(a), everything in exl3/, static + Atlas measurements.**
Why: the ask is "might save a small amount of time" — a ranked candidate list with estimated
savings and risks lets TensorFold pick winners before any kernel touches the contract gate;
no local GPU qualification was requested.

## 2026-10-29 — Reconcile perf-review shortlist item 1 (minor)

Decision: annotate docs/atlas-performance-review.md shortlist item 1 — the EXL3 decode rework
items are Atlas kernel gaps, superseded for TensorFold by docs/atlas-exl3-kernel-review.md.
Why: the kernel comparison showed TensorFold already implements the prescribed decode shape
(funnel-shift windows, dp4a byte-sum, split-K ticket fold); leaving the item unqualified would
send future work after wins that already exist in-tree.

## 2026-10-31 — KV compression first-pass implementation: integration point (major)

Reference: docs/kv-cache-compression-plan.md; zig/src/families/glm/cuda_snapshot_codec.zig.
Choices: (a) a page format in kv_pool/cuda_latent (the plan's Phase 2 file list); (b) compress GLM kept states
(prompt-cache snapshots) in their TF_GLM_HOST_CACHE host pages.
Decision: **(b)**.
Why: (a) puts decompression on the decode path, the plan's own non-goal; kept states are off the hot path, already
charged through `snap_charged`, so a smaller lease is an allocator-visible saving with no kernel change.

## 2026-10-31 — Cross-rank page agreement for compacted states (major)

Ranks compress to different sizes but only rank 0's store decides evictions. Decision: compaction never trims at
keep time on TP; the gate after an announced keep all-reduces (max) each rank's page count and every rank trims to
it, so page use stays equal across ranks. The leader charges its own compacted pages (an understatement only when a
peer compresses worse; a later save then fails gracefully on the leader rather than diverging).

## 2026-10-31 — Atlas column codecs as candidates, not defaults (minor)

Reference: ~/Workspace/RustroverProjects/atlas/docs/plans/atlas-kv-column-codecs-plan.md.
Decision: Gorilla XOR, Delta, DoubleDelta (element or token-row lag) and T64 (frame-of-reference form) are
implemented as per-block candidates (TF_GLM_KV_CODEC_CANDIDATES=atlas|atlas-exhaustive); the default stays plain
byte planes + order-0 rANS. LZ4/ZSTD backends are replaced by order-0 rANS (no compressor in Zig std for either);
std deflate levels 1/6 are the benchmark's general-purpose baseline (plan B1).
Why: TensorFold's Atlas review recorded Gorilla/delta as measured negatives; they must win on TensorFold data first.

## 2026-10-31 — Dedup from the Atlas cache-server plan: measured only (minor)

Decision: blocks are cut from each part's start so equal prefixes give equal blocks; the codec counts held block
identities and logs `duplicate=` bytes per keep. No shared-page dedup storage.
Why: real dedup needs refcounted cross-state pages; the measurement decides whether that is worth building.

## 2026-10-31 — Block hash: Wyhash, not XxHash3 (minor)

Measured: Zig 0.17 std XxHash3 0.17 GB/s vs Wyhash 15 GB/s (x86, ReleaseFast, 1 MiB buffers). The block hash runs
on every keep and restore, so it set the raw-block path's speed. Container format v1 is unshipped, so no version bump.

## 2026-11-01 — Plain codec is the compaction format; Atlas candidates stay off (minor)

Measured on 16 real GLM-5.3-Flash kept states on a GB10 (`docs/kv-cross-context-probe-2026-11-01.md` 11.3): plain 0.718
of raw, atlas 0.717 at 3x the encode cost, atlas-exhaustive the same at 11x, deflate-6 0.819, ClickHouse ALP-RD 0.83,
Gorilla/DoubleDelta/FPC 0.94. Decision: plain byte planes + rANS is the format; the candidates remain selectable for
experiments and are not scheduled for a default.

## 2026-11-01 — Compaction is lazy, off the prompt pass, on a pinned worker pool (major)

Reference: `tools/kvprobe/STATUS-lazy.md`, `todo.md` codec items. Choices: (a) eager compaction inside the prompt pass
(today); (b) lazy on a persistent worker pool with `TF_GLM_KV_CODEC_CPUS` affinity, entries raw or compacted
intermixed, restore transparent to the form. Decision: **(b)**, eager kept as a setting. Why: the user asked for it;
encode at 1.7 GB/s on 8 cores is 0.6 s a GiB, too long inside a prefill; the engine is pinned to the efficiency
cores on GB10 and the operator must choose where compaction runs.

## 2026-11-01 — Rebuild the KDA state by replay for cold entries (major, gated)

Reference: `docs/kv-cross-context-probe-2026-11-01.md` 11.3 and 11.8, `tools/kvprobe/STATUS-replay.md`. Evidence: the
prefix class is 55 % of a kept entry and compresses worst (0.83); the KDA heads' decay horizons have a median of 1.4
to 8.6 tokens and a p90 of 40, the state has an effective rank of 2 to 12 and deviates 0.01 to 0.4 across prefixes.
Decision: implement `TF_GLM_KDA_REPLAY=W` (restore the attention rows, zero the arena head, replay the last W tokens)
with a check mode that measures the replayed state against the stored one, off by default; adopt only behind the
Phase 3 quality gate for the heads the window does not cover. Low-rank storage of the state is not pursued (rank 8
loses 20 %; 99 % of energy needs rank 50 to 80 for the tail heads).

## 2026-11-01 — Closed directions (minor)

Cross-prefix reuse of attention state and per-token difference caching (hidden taps 0.6, latent rows 0.38 to 0.71 at
the deep plateau, a third of experts change); cross-session residual coding of rows (gains only in the first four
attention layers); routed-expert prefetch and a deeper decode expert ring on GB10 (weak locality, predictor recall
0.26 to 0.51 at k=8, kernels bit-exact but within noise). The exact DSA page prune is deferred to long-context
evidence (0 % prunable below 8k tokens).

## 2026-11-01 — Exact DSA page prune: build the device path for long contexts; selection-count demotion closed (minor)

Reference: session record 11.9, `zig/src/families/glm/cuda_index_prune.zig` (host consumer with an oracle),
`tools/kvprobe/HINTS.md`. Measured on a 204,800-token prompt: the exact bound on prunable index pages is 0 below 16k
tokens, about a third at 64k to 128k and 0.47 to 0.75 past 128k; only 1 to 7 % of pools are never selected by any
sampled query. Decision: the prune's device path and the per-page key bounds in the keep path are worth building,
gated on long-context receipts; demotion of latent rows by selection counts is not pursued.

## 2026-11-01 — KDA replay restore: not adopted; hypothesis refuted as implemented (major)

Reference: session record 11.10, `tools/kvprobe/STATUS-replay.md`. Measured: replaying the last 64 / 256 / 512
tokens from a zero state rebuilds the stored KDA state to a relative L2 of 0.87 / 0.73 / 0.67 at the median over
heads (layer 0 converges, deeper layers compound their inputs' error); the replay form changed a greedy reply at
token 10 of 14. Decision: the replay form stays off, an experiment behind `TF_GLM_KDA_REPLAY*`; the earlier decision
to pursue it is superseded. The coarse-init variant (replay from a low-precision or low-rank stored state) is the
only follow-up worth a run, as a Phase 3 lossy rung.

## 2026-11-01 — Lazy compaction is the recommended codec mode (minor)

Served (session record 11.12): with `TF_GLM_KV_CODEC_LAZY=1` the prompt pass pays no encode (eager: 1.5 s a keep),
the workers run on the cores named by `TF_GLM_KV_CODEC_CPUS`, pages fall 532 to 387 at the same bytes as eager, and
restores from compacted entries are byte-identical. Decision: the recipe recommends lazy with the codec on; eager
stays the default of `TF_GLM_KV_CODEC=1` until the longer run in `tools/kvprobe/RUNBOOK-lazy.md` is a receipt.
Analyses on multi-GB dumps run under a memory cap (an uncapped `routing.py` on the 200k dump was OOM-killed at 23 GB
beside production).

## 2026-11-02 — KDA head in fp8 is the cold-tier candidate; replay is closed (major, gated)

Measured (session record 11.14): the fp8 copy of the arena head (e4m3, a power-of-two scale per row) restores within
2.6 % relative L2 at the median and 4.5 % at worst, uniformly across layers, with greedy replies byte-identical on the
one body tried; replaying tokens on top of the copy raises the error to 0.33 and more, and the zero-start replay
never converged. Decision: `TF_GLM_KDA_REPLAY_INIT=fp8` with `TF_GLM_KDA_REPLAY` unset (no replay) is the cold-tier
form to take through the Phase 3 quality gate; the replay windows stay an experiment. Bytes: 94 MiB a rank for an
8k-token entry against 272.

## 2026-11-02 — Exact page prune: device path exact and served-identical, off by default (minor)

Measured (session record 11.15): `TF_GLM_INDEX_PRUNE=1` on the GB10 passes 65 of 65 exactness cases with the
control caught, the contract's byte gates against the kernel it replaces, and six served replies (32k and 200k
documents, restores included) are byte-identical to the flag-off ring. It is slower on synthetic keys at every length
(x0.29 at one row: launch latency), and on this ring the whole index scan is under 1 % of a 124 ms decode step at
200k tokens and about 4 % of a long prefill's time a token. Decision: the flag stays off by default; the mechanism
(strips, bounds-first scan, oracle, emulation, contract) is the upstream candidate, the speed claim is not. The
non-finite line the contract prints for the planned kernel is the known relu finding of the kernel it replaces, not
a failure. Round 2 (three launches, `TF_GLM_INDEX_PRUNE_MIN_PAGES`, served skip counts, pending records in the
limits set) is merged and awaits its GB10 pass before any default changes.

## 2026-11-02 — The codec receipt: lazy stays recommended, the timer gap was pinning, the default waits (minor)

Measured (`docs/receipts/glm-kv-codec-lazy-2b61900a/`, session record 11.17): nine keeps a configuration, replies
identical across off, eager, eager-unpinned and lazy; pinned eager encodes 0.26 s a keep on the prompt pass, lazy
0.24 s a compaction off it; eager with unpinned workers is 6x slower in encode and decode, which explains the 5x of
2026-11-01 (the cores, not the timers). Lazy's time to first token is 0.6 to 1.5 s a turn above off, but off ran
without the ledger and the dump the codec configurations carried. The uninstrumented pair then ran (no ledger, no dump): lazy's
time to first token equals off's within noise in all nine turns, replies identical; the deltas were the dump's
writes. Decision: lazy is the recommended mode with the codec on and the workers pinned, and it costs the prompt
pass nothing measurable; moving `TF_GLM_KV_CODEC_LAZY`'s default is the maintainer's call at the gate.

## 2026-11-02 — Exact page prune, round 2: served regression and a bound that prunes nothing on real keys (major)

Measured (session record 11.15, `docs/receipts/.../gpu-2026-11-01/prune2/`): round 2 passes every kernel check and
prints its compile records, but served with the threshold at 0 all six replies differ from the flag-off ring where
round 1 was byte-identical, and its counters saw 0 skips in 229 million page visits on real keys at 32k and 200k
tokens (the bound, or the same bug: undecided). Decision: `TF_GLM_INDEX_PRUNE` stays off by default and must not be
served until round 3 restores exactness at the engine's shapes; no performance claim is made for the feature; the
bound-based skip share on real keys is to be measured offline on the kept pooled keys before any further device
work, and the feature is closed if no sound bound prunes them. The exactness machinery as of commit 2b61900ad
(strips, host oracle, emulation, contract records) remains the upstream candidate. Round 3 (f45dc1c02, merged as
e8227ae9e the same day, host-tested) found the regression in the counters' placement in the scratch (a select with
fewer rows or pages than its predecessor read the previous plan bytes as its counters; with the plan never written
no page could be skipped, so round 2's zero share was this bug); the fix pins the counters at the head of the
scratch and is unverified on a GPU, so the flag stays off until that pass.

## 2026-11-02 — Kronecker factorization of kept states: closed; the dead recurrent half is the lossless lever (minor)

Measured (session record 11.18, `docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/kron/`, two rank-0
keeps of 8k and 20k tokens on the CPU, no outage): the nearest Kronecker product of a KDA head's fp32 state captures
0.63 of its energy where the SVD captures 0.98 at the same 4x parameter budget (0.43 for the 32x32 (x) 4x4 shape),
and on the bf16 latent rows every Kronecker shape captures a third to two thirds of the SVD's energy at matched
cost. Lossless, a rank-128 prediction from bf16 factors with the exact residual coded saves 0.2 to 0.3 bits a value
and costs 4.1 bits of factors; the rows' 10.6 bits a value and the state's 27.3 are mantissa-dominated order-0
entropy that no structural model removes. Decision: no Kronecker work and no structured-prediction codec; the
precision rungs stay the lossy path (bf16 state 0.17 % relative error at 2x, fp8 2.6 % at 4x, record 11.14) and the
plain SVD the factorized one if a cold tier ever wants more. The same keeps show the codec's one unremoved prefix
cost: the dead half of the double-buffered recurrent state, zero only in a keep taken after a single prefill chunk
since the slot's reset (every prefill chunk and every decode window flips the halves) and stale in every other keep,
carried on by restores (71 MB raw, 51 MB modeled a rank a keep; 16 % of a 20k-token entry, 24 % of an 8k one), never
read after a restore by code reading.
Next: exclude it in the snapshot plan (write the half `cur` selects) and verify with the receipt harness of 11.17
(byte-identical replies); an outage, the user's call.
