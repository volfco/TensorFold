# Atlas EXL3 kernels vs TensorFold — comparison and port candidates

A function-by-function comparison of the CUDA kernels Atlas added for EXL3 support
(`atlas:kernels/gb10/glm-5.3-flash/exl3/`, ~7.0k lines, through commit `09a6b4eb`) against
TensorFold's same-function kernels (`zig/kernels/cuda/exl3_*`, `exl3_block/`, GLM glue, ~6.4k
lines). Goal: improved implementations worth porting for **small time savings**. No kernel code
changed by this review. Broader context: [atlas-performance-review.md](atlas-performance-review.md).

Evidence grades: **[M]** measured (source path given), **[E]** estimated (reasoning or cited
source), **[U]** unknown. Atlas prior-art numbers recorded in `atlas:GB10_PRIOR_ART_CROSSREF_2026-10-10.md`
are marked [M-MR]/[M-SG] (other engines/boxes).

## Verdict

TensorFold's EXL3 kernels are already upstream-grade and mostly **ahead** of Atlas:
TensorFold's `exl3_decode.cuh` (funnel-shift windows, `__dp4a` byte-sum, two values per
`__hfma2`), split-K with ordered ticket fold, 16-row weight passes, grouped experts with a
device-side plan, and split mHC partials are exactly what Atlas's own optimization review
(§3.1–3.3) prescribes — Atlas's default tier is the slow one, and its promoted coop tier
(vendored exllamav3, MIT) mostly catches up rather than leads.
Atlas's measured wins are narrow: one transferable tile-size prior (N256 — from a different
kernel in another engine, not a TensorFold result), one fusion to
validate that TensorFold already ships default-off, and two mechanism ports (fused QKV launch,
on-device draft selector). Two measured Atlas results **confirm TensorFold choices** instead:
the KDA `[V,K]` state layout microbench (33.2→24.0 µs/call) matches TensorFold's shipped layout,
and the coefficient-decode fix list matches `exl3_decode.cuh`.
The per-engine arithmetic contracts (TensorFold's Triton/Python-family parity vs Atlas's
FLA/installed-reference parity) are not interchangeable: never port an mHC or KDA kernel body
across them, only structure.

## Pair 1 — linear/projection GEMM (decode M≤8 and prefill)

- TensorFold `tf_exl3_linear` (`zig/kernels/cuda/exl3_linear.cu`, `exl3_block/linear.cu`):
  host `plan()` sizes split-K to fill the machine at M=1, 16 rows decoded per weight pass,
  register double-buffer prefetch, `decode2` = `__dp4a` + `__hfma2`, fused `fwht128` output
  transform, ticket fold for splits. Prefill: `pfdense.cuh` (bit-exact unpack+GEMM epilogue),
  `dense3.cu`, `x3seg.cu` (many small projections in one launch).
- Atlas has two tiers. Default `exl3_linear_mma.cu` + `exl3_mma.cuh`: one CTA per 128 columns
  **per row** (weights re-decoded per row), serial slice loop, `tile_state` decode at
  ~50–70 instructions/coefficient with a runtime `%` [E] — grids fill 4–64 CTAs at M=1 [E]
  10–25 GB/s. Promoted coop tier (`exl3_coop_gemm.cu` + vendored `exllamav3_gemm_inner.cuh`):
  cp.async multi-stage ring (4–6 stages), ldmatrix fragments, slice-parallel split-K with a
  lock chain, 16-row chunks, fused `a_had`/`out_had`/`parts`/`a_act`.
- Verdict: parity once Atlas runs coop; TensorFold marginally ahead at low-M (SK fill at M=1,
  16-row passes, fused output transform). The one plausible Atlas edge: upstream's 3–6-stage
  smem ring vs TensorFold's one-k-step register prefetch at 7–8-bit widths.
- Hazards: Atlas's runtime-indexed `acc[4][4]` is the spill class of the N1 precedent
  ([M-SG] spill-free sibling +8–17%/layer); TensorFold's accumulators are compile-time indexed.
  TensorFold's ticket fold needs counters zeroed before launch (documented); Atlas's lock chain
  self-resets but carries the F1 under-alloc bug class (unmerged fix).

## Pair 2 — coefficient/codebook dequant

Every Atlas decode trick has a TensorFold equivalent: per-width aligned `dq8_*` specializations
↔ `step_shuffled<K2>`; `decode_3inst` `__dp4a`+`__hfma2` ↔ `decode2`; bfe64/funnel windows ↔
`window16`/`lane_states`. TensorFold decodes straight into MMA B fragments with no smem round
trip; Atlas's default tier still pays `tile_state`. Verdict: parity, TensorFold slightly ahead;
nothing to port. Keep the microbench lessons (Atlas `decisions.md` I-b/I-c): the I-c load-hoist
fetch-ahead measured flat, while the I-b `[V,K]` state-layout win (measured 33.2→24.0 µs/call)
belongs to TensorFold already.

## Pair 3 — experts/MoE (routed + shared)

- TensorFold routed: grouped by expert at decode (`exl3_experts.cu`/`experts_grouped.cuh`) and
  prefill (`x3gm*` persistent CTAs, claim-ahead, device-side plan `x3gm_plan.cu`), one fixed
  arithmetic (warp-order sums). GLM path: router → `tf_glm_topk` (warp per row, low-id ties) →
  plan → grouped kernels.
- Atlas routed: promoted stack still runs per-slot kernels (each expert re-streamed per slot,
  [E] 1.4–1.5× decode bytes, ~200× redundant re-decode at T=7168 prefill chunks — ~3.7× at a
  128-row chunk — crossref X5) outside its coop MoE; thin/fat
  numeric profiles chosen by per-call expert counts make rounding **occupancy-dependent**;
  measured [M]: coop-MoE-thin is 32% of the decode critical path; the FAT fallback is 91.9% of
  prefill GPU time (REVIEW-09, no token scope), and a 1024-token trace pinned 93.8% on
  `exl3_moe_mul1_matmul_precision_mma_split`.
- Shared experts: both engines keep shared experts BF16 at runtime; running them packed through
  the trellis kernel is an open item for both ([M-MR] ≈1.5 ms/step at M=8 over 42 layers).
- Verdict: TensorFold ahead (grouping, plan, deterministic arithmetic). Atlas's residue is narrow:
  gemm_inner's fused `a_act` (activation + down-input transform in the down-GEMM **prologue**) —
  TensorFold's nearest equivalent is `x3ld_epi`, which instead fuses the gateup epilogue and down
  combine into the GEMM **tails** (default-off, candidate 5) — and nothing else.

## Pair 4 — hyper-connection (mHC)

- TensorFold (`glm_glue_hc.cuh`): split partials (`tf_glm_hc_partial`/`partial8`) + `hc_finish`
  (256 threads/row, Sinkhorn on 32 lanes with shuffles, uint4 loads hoisted before Sinkhorn),
  1–2 barriers per kernel, Triton-reference arithmetic.
- Atlas: default `exl3_mhc_pre_norm` is a single-CTA 192-pass serial chain [E] ~2.7–20 ms/token;
  the promoted `SPLIT_MHC` (+52.79% in-model [M]) restructures it into partials + split fold —
  i.e. Atlas catching up to TensorFold's design. `PAR_SMALL` wide grids (+2.87% [M]) map to
  TensorFold's `rows8Launch`. The microbench sqsum vectorization (~0.05 ms/token [M]) targets an
  smem reduction pattern TensorFold's finish does not have.
- Verdict: TensorFold structurally ahead. Target bar to hold: [M-MR] vLLM tilelang mHC ≈1.9 ms/step
  at 6 rows. Hazards: Atlas's `pre_norm` returns silently on invalid splits (stale outputs);
  Atlas compiles mHC with `--fmad=true` while all its other EXL3 modules are `--fmad=false` —
  mixed contraction policy TensorFold does not have (explicit `__fadd_rn` chains).

## Pair 5 — planning / cooperative launch

TensorFold builds expert pass tables on-device in one CTA (`x3gm_plan.cu`: histogram, scans,
stable `__match_any_sync` scatter) and fills SMs via host `plan()`; Atlas uses cooperative launch
with per-column lock chains and pinned slice profiles. Verdict: TensorFold ahead (no
cooperative-launch constraint, no plan/launch gap, no lock-bug class).

## GLM-side kernels (bundled in Atlas's exl3/ dir)

- **KDA/FLA**: Atlas `kda_fla_*` is a ~1.2k-line bit-parity port of chunked FLA Triton kernels
  (KKT Gram, `(I+A)⁻¹`, w/u recompute, chunked state mma; PTX-pinned ops). TensorFold
  `glm_kda.cu` is the direct per-row gated delta rule (Python-family parity contract), serial in
  T for prompt chunks. The chunked algorithm is the one **structural** idea worth adapting
  (see candidate 2), but never port the bodies: the two arithmetic contracts are incompatible.
  The `[V,K]` layout microbench win (33.2→24.0 µs/call, [M], st sectors −50%) matches
  TensorFold's shipped layout — Atlas production is the laggard.
- **Indexer quantization**: Atlas `glm_indexer_quant.cu` implements EXL3 native index precision
  (FP8-e4m3 queries/pooled keys with explicit power-of-2 scales, folded f32 weights) — including
  the `exp2f(ceilf(log2f))` ulp hazard and a shared-memory race fixed in `52adbb16`. TensorFold's
  indexer is BF16/unquantized with plain ratio scales (the ulp hazard is absent because the
  power-of-2-exact contract is absent). All five mandatory red-flag hazard classes
  (silent split return, ulp scale, dead accumulator branches, return-before-`__syncthreads`,
  smem race) are **avoided** in TensorFold's counterparts (each verified by code inspection).
- **DFlash capture**: TensorFold `df_tap` already equals Atlas's `bf16mean` variant (per-stream
  rounding hoisted into the mHC contract). No gap.
- **Draft candidate selection** (naming correction: TensorFold `exl3_candidate.cu` is EXL3
  trellis projection ops, not selection): Atlas runs the full greedy bilinear chain on-device
  (`dflash2_candidate_selector.cu`, self-reported ~0.16→~0.03 ms [M]); TensorFold does GPU top-16
  then D2H readback + CPU chain (`dflash_select.zig`). This is a real mechanism gap.

## Ranked port candidates

| # | Candidate | Technique | Expected saving (grade) | Risk | TensorFold files |
|---|---|---|---|---|---|
| 1 | **N256 expert-tile screen** | CTA tile N128→N256 in grouped expert GEMMs; keep warp-order fold for bit-identity | routed-expert GEMM: N128 measured 14–17% slower than N256 [M-MR prior, different kernel — TF transfer [U]]; ≈5–8% of a decode step [E] | low for bits; **medium register pressure** (acc doubles — the N1 spill class; screen `ptxas -v` first) | `zig/kernels/cuda/exl3_experts.cu`, `exl3_block/experts_grouped.cuh`, `zig/src/cuda/exl3_experts.zig` |
| 2 | **Chunked FLA/WY KDA prompt path** | Adapt the chunked algorithm (KKT/solve/state-mma) to TensorFold's arithmetic contract; replaces the serial per-row prompt recurrence | long-prefill KDA [E] 5–20× (serial-T vs T/64-parallel); no TF measurement | **high** — changes output bits; needs A/B gate vs current chain | `zig/kernels/cuda/glm_kda.cu`, `zig/src/families/glm/cuda_kda.zig` |
| 3 | **Fused multi-component QKV GEMM** | One launch over fused Q/K/V width (exllamav3 `parts`, Atlas `FUSED_QKV`), per-component fold preserved | removes 2 launches + scratch round trips per set [E]; 1–3 ms/token-class in Atlas's launch census [E] | low bits; medium layout; **first verify** TF's EXL3 kits launch q/k/v separately (GLM does not) | `zig/kernels/cuda/exl3_linear.cu`, `zig/src/cuda/exl3.zig`, EXL3 kit loaders |
| 4 | **On-device DFlash candidate selector** | Port the mechanism: device-side total-order top-k + bilinear chain + graph-safe tickets; keep TensorFold's tie/chance semantics | removes one D2H + CPU chain per verify step [U]; Atlas's own on-device rewrite sped its selector 0.16→0.03 ms [M] (internal to Atlas, not vs TF's chain) | medium (contract: tie order, "chance kept") | `zig/kernels/cuda/glm_dflash.cu`, `zig/src/families/glm/dflash_select.zig`, `dflash_candidates.zig` |
| 5 | **A/B `x3ld_epi` fused epilogues** | Fuse gateup epilogue + down combine into the GEMM **tails** (`x3ld_epi`'s design; Atlas's `a_act` instead fuses into the down-GEMM prologue) | one saved pass + 2–3 launches/MoE layer at decode [E]; already in-tree (`TF_DSV41_X3LD_EPI`), validation cost only | header claims same bits; verify vs byte gates before flipping | `zig/kernels/cuda/exl3_block/x3ld_epi.cu` + dispatcher |
| 6 | **Deeper weight pipeline in `tf_exl3_linear`** | upstream SH_STAGES 4–6-class multi-stage prefetch behind the 7–8-bit k-walk (TF currently decodes with zero lead at PF=0) | [U]; same idea as `x3ld.cu`'s PD ring + PDL, proven in-family | loads only (bits safe); medium complexity | `zig/kernels/cuda/exl3_linear.cu` |
| 7 | **Packed shared experts** | run shared experts through the trellis GEMM instead of BF16 | [M-MR] ≈1.5 ms/step at M=8 (gate_up 77.7→54.1 µs, down 38.3→26.1 µs); ~0.7 GB/token | arithmetic-changing; re-qualify vs reference | `exl3_linear.cu` reuse, `zig/src/families/glm/cuda_moe.zig` |
| 8 | **EXL3 index precision** | FP8-e4m3 indexer keys with power-of-2 scales (Atlas `glm_indexer_quant.cu` pattern) | EXL3-checkpoint score parity; enables indexer-key cache 256→128 B/row [E] | medium-high: reproducing exact parity **imports the ulp hazard** TensorFold currently avoids | `zig/kernels/cuda/glm_glue_index.cuh`, `glm_glue_attn.cuh`, `glm_quant.cu`, `zig/src/families/glm/cuda_index.zig` |

Cheap validation actions (not ports): run the N1-style spill audit (`ptxas -v`, pass = 0 LDL/STL
in inner loops) on `tf_exl3_linear_*`, `tf_exl3_grouped_*`, `x3gm`; take a TensorFold launch
census before ranking #3; audit flashnext/Nemotron recurrent-state stores for the `[K,V]`
float4 sector double-write that Atlas's I-b measured (layout-generic pattern).

## Feature gaps (Atlas has, TensorFold lacks) — worth having?

Cooperative lock-buffer GEMM: no (TensorFold's ticket fold covers it, simpler). TF32 mHC
projection: only if mHC ever becomes ALU-bound past 16 rows (changes bits). Thin/fat numerics:
no (occupancy-dependent rounding). `exllamav3_kernel_map/ptx/util`: only if running Atlas kernels
verbatim. BF16 vocab-gather salvage: no for current checkpoints. `COOP_FILL_SMS` knob: no
(`plan()` already fills). EXL3 index precision: yes if EXL3-checkpoint score parity is required
(candidate 8). Chunked KDA prompt: yes (candidate 2).

## Do not port

Atlas's default MMA tier (`tile_state` decode, per-row grids, serial slices — slower on every
axis); `exl3_mma_split` (+0.08% under identical-config displacement; Atlas later retracted its
measurability — do not quote as measured); the I-a sqsum fetch (a landed +0.05 ms/token win in
Atlas, but its target smem pattern is absent in TensorFold) and the I-c load hoist (measured
flat); Atlas's top-k (TensorFold's is already
warp-parallel with the same tie-break); the lock-chain split-K fold; every mHC/KDA kernel body
(per-engine arithmetic oracles); thin/fat rounding profiles; `dense_gemv_batch*` (carries the
return-before-syncthreads hazard); `exl3_dflash_capture` (`df_tap` already matches).

## Reconciliation with the performance review

[atlas-performance-review.md](atlas-performance-review.md) shortlist item 1 ("EXL3 bit-identical
decode rework") described **Atlas's** kernel gaps, not TensorFold's: this comparison shows
TensorFold already implements the prescribed decode shape, so that item is not a TensorFold carry.
The residual TensorFold candidates are the ones ranked above; the ~45 tok/s byte floor (≈6 GB/token
at TP=2) applies to any engine and is a capacity/bytes question (see
[atlas-kv-cache-compression.md](atlas-kv-cache-compression.md)).

## References

In-repo: `zig/kernels/cuda/exl3_linear.cu`, `exl3_decode.cuh`, `exl3_experts.cu`,
`exl3_block/{linear,lin2,dense3,pfdense,x3gm*,x3ld*,x3seg,experts_grouped}`, `glm_glue_hc.cuh`,
`glm_kda.cu`, `glm_dflash.cu`, `glm_glue_index.cuh`, `glm_quant.cu`, `zig/src/cuda/exl3.zig`,
`zig/src/families/glm/{cuda_kda,cuda_moe,dflash_select,dflash_candidates}.zig`,
`CONTRIBUTING.md` (kernel contract), `docs/kernel-contract.md`.
Atlas (AGPLv3; knowledge and adapted snippets only): `kernels/gb10/glm-5.3-flash/exl3/` through
`09a6b4eb`; `decisions.md` 2026-10-16/19/20 entries (I-a/I-b/I-c microbenches, LOO attribution,
F1 bug); `GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md` §3.1–3.7, §7; `GB10_PRIOR_ART_CROSSREF_2026-10-10.md`
N1/N8/X1/X5/X6; `tests/exl3_validation/{microbench,coop_moe_bench,kda_fla}/`.
Upstream prior art: exllamav3 (MIT) packed-GEMM techniques (warp-per-tile, cp.async staging,
funnel-shift, `__dp4a`); vLLM tilelang mHC ≈1.9 ms/step bar; FLA Triton KDA reference kernels.
