# Atlas KV-cache compression — research dossier

This dossier carries forward the Atlas Inference Engine's KV-cache compression research
(AGPLv3 Rust/CUDA project, local clone reviewed 2026-10-29) for application to TensorFold.
Numbers are Atlas single-capture observations; the TensorFold application is
[kv-cache-compression-plan.md](kv-cache-compression-plan.md).
Broader performance research is in [atlas-performance-review.md](atlas-performance-review.md).

## Trust tags

Every number carries one of:
- **verified-modeled-lossless** — bit-exact modeled container bytes with round-trip evidence;
  "modeled" means the size model, never allocator or serving bytes.
- **repaired-in-B0** — produced after the research-evidence repair campaign (2026-10-21); accounting
  is fixed but scope caveats (proxies, single capture) remain.
- **defective-do-not-use** — produced by a known defect or invalidated protocol; never cite.
- **unverified** — proposal, estimate or proxy; arithmetic-checked only.

## Program status

Goal: shrink one retained KV-cache prefix entry of GLM-5.3-Flash (EXL3 checkpoint, TP=2, BF16 latent
cache) without losing resolution, then gate lossy rungs on real quality.
Gates: **B0 evidence repair — complete**; **B1 GLM baseline + corpus — executed, closed PARTIAL**;
B2 codec/access microbench, B3 exact restore in-engine, B4 precision calibration, B5 quality/latency
vs raw control, B6 sharing factorial — **all not run**.
Program verdict: the strongest supported result is an offline, single-capture lossless storage
reduction of ~22.75 %; no production profile is selected; no encode/decode/restore cost exists
anywhere in the program.

**TensorFold measurements (2026-11-01).** The plan's Phase 0 and Phase 1 ran on TensorFold's own GLM-5.3-Flash kept
states on two GB10s (`docs/kv-cross-context-probe-2026-11-01.md` section 11, `docs/kv-cache-compression-plan.md`
"Findings"): a kept entry is 55 % KDA state and conv history, 35 % MLA latent, 10 % index rows; the plain byte-plane
codec reaches 0.718 of raw with bit-exact round trips, the Atlas column codecs 0.717 at 3x the encode cost (not
adopted), deflate 0.82. The KDA state's low effective rank (2 to 12 of 128) and short decay horizons (median 1.4 to
8.6 tokens) point at rebuilding it by replay rather than storing it; low-rank storage of it is not pursued.

## Reference accounting (the one capture)

Anchor: 784 cached tokens of a 798-token prompt, `rank0/request-1`, 2026-10-20
(`atlas:kv-cache-findings.md`, `atlas:docs/research/kv-cache-review/review-lossless.json`).

| Component | Bytes | Share | Tag |
|---|---:|---:|---|
| Recurrent H (34 layers × 32 heads × 128 × 128 FP32) | 71,303,168 | 81.6 % | verified-modeled-lossless |
| Convolution history (1,671,168 FP32 words) | 6,684,672 | 7.7 % | verified-modeled-lossless |
| MLA latent unique (V aliases K in the producer) | 8,830,976 | 10.1 % | verified-modeled-lossless |
| Sparse index | 551,936 | 0.6 % | verified-modeled-lossless |
| Block map | 424 | ~0 % | verified-modeled-lossless |
| **Covered entry** | **87,371,176 (83.32 MiB)** | | verified-modeled-lossless |

H+conv (the one retained snapshot) is 89.26 % of the entry.
B1 later proved the entry has **two legitimate shapes**: a complete-match capture retains KV only
(9,383,336 B/rank) while a grow capture retains KV + the H/conv snapshot (delta exactly 77,987,840
B/rank); the reconciled formula `covered = blocks × 191,496 + 77,987,872 | + 32` matches 192/192
request manifests with delta 0. Ranks are byte-identical; sum rank-local bytes, never divide by TP.

## Lossless codec line (verified modeled bytes on the capture)

| Codec | Idea | Result |
|---|---|---|
| f32_fields | split sign/mantissa vs exponent streams per granule, ZSTD | H → 61,527,696 (−13.7 %) |
| f32_rowcenter | per-row median exponent center, deltas mod 256 | H → 59,281,312 whole-snapshot ZSTD19 |
| f32_minbase | per-row min exponent, bit-packed offsets | H → 60,405,008–60,574,992 (beats rowcenter at Z3, loses at Z19) |
| bf16_planes | sign/exponent/mantissa planes per layer | MLA latent → 5,616,016 |
| conv_bf16 | conv history is widened BF16 (zero low 16 bits in all words) | conv 6,684,672 → 2,327,408 |
| sparse_fp8 | index rows recover as E4M3 codes + scale | 551,936 → 269,440 (decode back to BF16) |
| block_map | delta + ZSTD of block ids | 424 → 144 |
| **Pass-4 aggregate** | all of the above composed | **67,494,320 B (64.37 MiB, −22.75 %)** |

Every row is verified-modeled-lossless except two scoped caveats: conv_bf16 is exact only for this
producer (re-check each producer), and sparse_fp8's scale `k` is an encoding choice, not the
producer's (exactness to the stored words verified; scale provenance unverified).
The aggregate is verified-modeled-lossless as modeled storage only.

Archive side-channel: `zstd -19 --long=27` shrinks the full 174 MB dump to 44.3 % (archive candidate).
Codec-choice lesson: minbase vs rowcenter ranking flips with ZSTD level, so pick per decode budget.

Negative results (measured, scoped — skip these families): Gorilla, integer delta, double-difference,
sign-XOR-delta, channel permutation, fixed-width dictionaries, low-rank SVD (rank-16 is 11.8 % of
raw H at rel L2 0.123), per-head mixed precision (0.0–1.4 % before selector metadata), exact outlier
residual (~56 KB, estimated). The blanket claim "transform families are exhausted" is itself
defective-do-not-use: temporal correlation remains unexplored.

## Lossy rungs (sizes verified; errors are unverified proxies)

Relative-L2 / sampled attention-proxy errors on one capture are **not** model quality.
H truncation ladder: clear 4 mantissa bits 55,603,856 B (1.0e-6), clear 8 46,665,792 (1.3e-5),
clear 12 36,870,512 (2.1e-4), f16 33,594,464 (2.1e-4), bf16 25,106,288 (3.4e-3).
MLA clear 1/2/4 bits: 5,947,760/5,633,968/4,553,040 B (attn-out 0.0053/0.016/0.073).
Composites: conservative (H clear-8 + MLA clear-1) −36.8 %; aggressive (f16 H + MLA clear-2) −52.1 %.
The older `bench2.py` ladder rounds differently (defective-do-not-use as a comparator).

## Paper quantizers (TurboQuant, KVComp) — headline claims did not survive

After the B0 accounting repair, the only admissible table is
`atlas:tools/kv-entropy/results/literature/20261021T012006Z-f8d4cea7.json` (repaired-in-B0):

| Method (MLA latent, raw 8,830,976 B) | Bytes | attn-out proxy | KL |
|---|---:|---:|---:|
| KVComp K-block ch-wise rel=0.02 | 3,317,968 | 0.01532 | 2.79e-5 |
| KVComp V token-wise rel=0.02 | 2,899,376 | 0.02407 | 6.89e-5 |
| TurboQuant mse b=4 | 2,225,344 | 0.06936 | 1.16e-3 |
| TurboQuant split (32 ch @3b + 480 @2b) | 1,173,648 (2.0625 index bits/value) | 0.26754 | |

Cross-table size-vs-error dominance is **not claimable**: the truncation ladder's errors are layer-0
proxy scope while the repaired table is all-layers causal, exactly the B0 proxy-scope defect.
The exclusion of both papers stands on trust and transfer grounds: their quality claims are model-evals
on other architectures (Llama2/Ministral/Gemma3), TurboQuant prod needs a custom kernel to be
reconstructable, and no model-quality measurement exists. Note for prioritization: at face value the
repaired KVComp row reaches 3,317,968 B at proxy 0.01532, so if Phase 3 ever needs more than
truncation offers, re-measure the paper methods in one matched TensorFold scope before dismissing them.
conv should stay exact (conv_bf16 packs it at 2,327,408 B, zero error; KVComp's conv row trades error
for size and must not replace it).
Historical "hybrid −54 % is actionable", "1.6–2× smaller than truncation", "lower KL" and all
pre-repair TurboQuant/KVComp tables are **defective-do-not-use** (framing double-count, underpriced
metadata, mismatched error scope, KL never recorded).
Both papers' quality claims are model-evals on other architectures (Llama2/Ministral/Gemma3) and do
not transfer to GLM-5.3-Flash's KDA/MLA state.

## B0 defect register — a ready-made test plan for any port

The research itself had six defect groups; each repaired with a fail-before/after test.
Carry this list as the oracle-test checklist for TensorFold's codec work:

1. E4M3 helper sign bit overlapped an exponent bit for E ≥ 8 — every analytic-FP8 number was wrong.
2. Conditional entropy returned −H(A|B) (sign error).
3. TurboQuant pricing: residual missing the original-vector norm; norms priced FP16 but computed
   FP32; the 32@3+480@2 split is 2.0625 index bits/value (2.1264 loaded), not 2.5; channel-map and
   format costs missing.
4. KVComp accounting: codebook bits omitted from totals; histograms sliced by max frequency instead
   of highest used symbol; inconsistent block-offset conventions; cross-layer codebook pooling.
5. Proxy scope: queries perturbed with K/V; distortion and bytes measured on different layer/head
   scopes; every cross-method quality ranking was invalid.
6. Search validity: 12 of 18 reshape candidates sampled; same-seed resume repeated draws;
   `delta_chan` reconstruction anchor not stored (round trip unproven before size was reported).

Persistence lessons: `len(serialized) == modeled_bytes` by construction (the KVEC1 container,
`atlas:tools/kv-entropy/kvrecord.py`); bounded refusing decode (forged lengths, truncation, checksums
refused before allocation); versioned results with code/env hashes; never overwrite results in place.

## B1 campaign findings (GLM-5.3-Flash EXL3 baseline, 52 cells / 192 manifests)

- **Two-shape retained state** (see the accounting table): byte-equality against a single reference
  anchor is the wrong gate; use a reconciled per-request formula (matched 192/192 manifests).
- **Complete-match full-recompute quirk**: a block-aligned exact full-prefix match can report
  `cached == prompt_tokens` yet recompute at seed cost (pilot observation, preregistration amendment 3:
  cached=1840, ~188 s; pilot cells are excluded campaign evidence) while partial matches return
  quickly (measured warm partial-match TTFT 0.56–3.46 s, two populations); 2047-token probes
  (straddling the block boundary) return in 3.4–4.0 s. Any warm-path benchmark must probe both sides
  of its own block boundary.
- **Graded coldness**: shared chat-template and base prefixes make "cold" a ratio, not a boolean;
  report `cached/prompt` and gate cold as `== 0 or ≤ 10 %`.
- **Campaign defects to design against**: a per-process dump quota silently served byte-identical
  stale dumps to later cells (verify freshness against the receipt, not the manifest against itself);
  off-target materialization silently returned "best" (~2990 tokens vs 784/2048 targets); stored
  `passed` flags were wrong in both directions — recompute verdicts from raw fields always.
- Raw control data usable for planning: warm partial-match TTFT 0.56–3.46 s (two populations,
  unexplained); decode ≈ 1.96 tok/s on Atlas hardware; p99 unqualified at n=3.

## Plans inventory (status audited)

| Plan | Design | Status |
|---|---|---|
| `atlas:docs/plans/atlas-kv-cache-server-plan.md` | remote KV payload pool (gRPC, NVMe spill, dedup) | partial code only in an agent worktree; bugs B1–B9; treat as design |
| `atlas:docs/plans/atlas-kv-column-codecs-plan.md` | ClickHouse-style codec pipelines, per-block selection | plan only; ancestor of the probes above |
| `atlas:docs/plans/atlas-kv-coord-plan.md` | metadata-only cluster coordinator (hash ring) | design only |
| `atlas:docs/glm-kv-shard.md` | token-sharded MLA latents on a TP2 pair, bitwise-equal | behind a flag; −44.6 % B/token modeled; decode −13 %/−5 % n=1 |

## What transfers to TensorFold (and what does not)

Transfers (methods, not numbers): the trust discipline (recompute verdicts from raw fields, hash
artifacts, separate modeled from allocator bytes); the two-shape ledger idea; the complete-match
boundary-probe test design; the KVEC1 container contract (modeled == serialized, bounded refusing
decode, explicit raw fallbacks); lossless transform concepts (field split, row-centered exponents,
shared-exponent frame-of-reference, widened-BF16 packing when the producer invariant holds); the B0
defect list as a test plan; frozen-margin quality registration.
Does not transfer: every byte total (EXL3/GLM geometry specific); MLA K/V aliasing accounting
(TensorFold's layout must be checked, not assumed — its fp8 rows use one fp32 scale per row);
sparse-index exact recovery (depends on the EXL3 quantizer's power-of-two scales); all TurboQuant/
KVComp results; glm-kv-shard's kernels and numbers; all B1 latencies.

TensorFold must measure first (all detailed in the plan): a per-family retained-entry byte ledger;
its conv/KDA bit patterns (is conv history widened BF16 here too?); exponent/magnitude distributions
before choosing a codec; its own complete-match/warm-path boundary behavior; encode/decode/restore
costs on Metal and CUDA (Atlas never measured them); and the interaction with the `--kv-cache fp8`
gate (TensorFold refuses FP8-on-pages today).

## External references

- TurboQuant: Zandieh et al., arXiv:2504.19874 (ICLR 2026).
- KVComp: arXiv:2509.00579 (Atlas sources disagree on authors: "Bo Jiang et al." vs "Chen et al.").
- SplitZip: arXiv:2605.01708 (lossless GPU-friendly KV encoding).
- ZSTD KV/weights study: arXiv:2503.18869 (~47 % KV reduction reported).
- KV cache compression survey: arXiv:2508.06297.
- Gorilla TSDB: VLDB 2015 (DoubleDelta/Gorilla); ClickHouse CODEC clause and codec blog.
- Float codecs: FPC, ALP, Chimp; `delta-compress-llm` (GitHub); XOR-float explainer (clemenswinter.com).
- QJL add-back caution: dejan.ai 2026 implementation study (adding the QJL term destroys output
  quality).
- vLLM decode context parallelism (`decode_context_parallel_size`, Apache-2.0) as token-shard prior art.
- Evaluation shortlist from `atlas:docs/plans/atlas-kv-column-codecs-plan.md` §7: dietgpu, pcodec,
  DeepZip, zipnn, GPU-lossless-compression, ICS23-GPULZ, zipserv_kvcache_framework.
