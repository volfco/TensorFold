# Atlas performance research — port review

This is a salvage review of the Atlas Inference Engine's performance research (local clone of the
AGPLv3 Rust/CUDA project, review 2026-10-29), assessed for TensorFold.
Findings, measurements and design ideas transfer; Atlas code is not copied wholesale, and every
number below is a single-run Atlas observation until TensorFold re-measures it on its own binaries.
KV-cache compression research is reviewed separately in [atlas-kv-cache-compression.md](atlas-kv-cache-compression.md)
and applied in [kv-cache-compression-plan.md](kv-cache-compression-plan.md).

Evidence labels: **measured** (ran on GB10-class hardware), **modeled** (arithmetic, not run),
**speculative** (proposal), **anecdotal** (third-party or single-boot claim).
Relevance: **HIGH** (directly portable or blocks GLM-5.3-Flash serving), **MED** (transferable
pattern or caution), **LOW** (Atlas-internal or regime-mismatched).
Atlas source paths are relative to the Atlas repo root; `C-3`-style ids cross-reference the review
extraction kept outside this repo.

TensorFold already contributed to this space: Atlas derived its DFlash2 edge weighting, copy drafts,
drafter TP split, indexer row split and L2 prefetch work from TensorFold patches
(`atlas:docs/glm-prior-art.md`), so "back-port" below means an Atlas refinement of a TensorFold idea.

## Priority shortlist

1. **EXL3 bit-identical decode rework** (WQ-2, K-6): GLM-5.3-Flash's modeled decode floor is ~45 tok/s
   until packed decode is parallelized and `exl3_mhc_pre_norm`/coefficient decode are reworked.
   Upstream (exllamav3, MIT) spill-free precedent is +8–17 % per layer, measured.
   **Update (kernel review):** these are Atlas's kernel gaps, not TensorFold's — TensorFold already
   implements the prescribed decode shape; see [atlas-exl3-kernel-review.md](atlas-exl3-kernel-review.md).
2. **Batched speculative decoding, industry shape** (S-1, S-3): batched verify+propose, K-vs-batch
   ladder (`4:3,8:3,16:1` style), depth pruning; measured 1.09–1.90× at C=1..8 and 1.18× at C=16
   (see the trust caveats in the MTP section — these are pre-fix and config-debt figures).
   The fixed-K collapse at batch is the counter-case.
3. **Prefix-cache + snapshot tier with a matched-length gate and drafter state in the snapshot**
   (ME-1, C-2, S-5): 11 s → 1.5–1.9 s turn-2 starts measured on GLM with NVMe restore;
   the 256-token match gate and cold-drafter fix are each worth +7–9 % at low concurrency.
   TensorFold has its own tiers (`--learn-conversations`, `--spill-gib`); the gates and the
   drafter-state seam are the carry.
4. **Roofline and dispatch discipline** (D-2, D-3, K-4): use ~240 GB/s achievable (not 273 spec) as
   the GB10 denominator; audit every "choose kernel by M" site and every undispatched kernel handle;
   the dispatch family alone measured +30 %/+22 %/+9 %.
5. **Load what the checkpoint ships, and stack tuned kernels** (PF-5, PF-6): never double-quantize at
   load (measured −7 BFCL points historically); cuBLASLt FP8 GEMM is dead on sm_121 (NOT_SUPPORTED,
   or garbage behind a heuristic); tuned-kernel flags measured +41–75 % prefill — with fail-loud
   fallbacks, since silent fallback cost 40–50 % of prefill at concurrency.
6. **PDL/L2 weight-touch refinements** (K-3): in-kernel row prefetch in routed MoE kernels, PDL
   weight touch, side-stream prefetch during all-reduce waits — refinements of TensorFold-originated
   patches.
7. **Compute-for-bytes on GLM sparse MLA** (KV-4, KV-5): FP8 indexer keys (lossless 2×), V-pool
   aliasing (~39 % of MLA KV memory, lossless, modeled), GQA-grouped split-KV; the precision ladder
   is int8-block latent before fp8/4-bit.
8. **MoE prefill grouping with regime routing** (PF-4, C-3): grouped expert decode is worth up to
   ~200× redundant re-decode at T=7168 prefill and is a measured loss at decode batch sizes.
   Route by regime, never one kernel for both.
9. **Measurement-discipline kit** (K-4, S-7, D-4): same-binary control legs, byte-identity hashes
   before latency comparisons, warm-workload acceptance, warmup discard, A–A floor for
   nondeterministic baselines, shape-diff-first when bench and serve disagree.
10. **Box hygiene** (ME-3, ME-4): `vm.compaction_proactiveness=0` before any TPOT measurement
    (tails otherwise spike ~1.72× every ~32 s at ~94 % memory, one-boot evidence), weight-integrity
    hashes before blaming kernels for parity failures, and fail-loud on host-memory fallback
    (managed/host-registered pages measured ~35 % slower reads).

## KV cache precision (summary; full review in the KV dossier)

- **fp8-KV is a capacity lever, not a decode-speed lever** (measured): microbench warm TPOT −4.8 %
  but end-to-end TPOT +2.4 %, tps −2.7 % on a same-binary 1007/1007 control
  (`atlas:docs/campaigns/gb10-decode-fold-2026-07/raw/kv_ab/{bf16kv,fp8kv}.json`); at K=4 spec it
  freed 0 GB and cost +8.1 % p50 TPOT. Mechanism: bf16 TPOT is flat 3k→13k context, so decode is not
  KV-bandwidth-bound on that workload. HIGH as a kill criterion before touching KV precision for speed.
- **Asymmetric K/V precision** (measured): keep K at baseline precision, compress V
  (`atlas:docs/turboquant-plus.md`); capability win, throughput within ±2 %.
- **MLA latent ladder** (anecdotal + measured): block-int8 latent rel err ~2e-4, fp8 ~10× worse,
  4-bit ~60× worse; NVFP4 MLA projections measurably lossy (arithmetic 40/40→37/40, needle 12/12→10/12).
- **Lossless KV byte wins for GLM sparse MLA** (modeled): FP8 indexer keys 256→128 B/row lossless,
  V→K alias ~39 %, NVFP4 sparse-MLA KV 2.7× behind a quality gate.

## Prefill

- **Wall shape of agentic serving** (measured): decode 59.6 %, fixed per-turn TTFT 21.1 %, marginal
  prefill 18.8 %, median output 45 tokens (`atlas:.../LEVER_LEDGER_2026_07_25.md`). Per-request
  overhead competes with decode at short outputs. HIGH.
- **Warm replay/restore work is worth TTFT effort** (measured): a register-resident GDN warm-replay
  path cut wall −7.25 % and TTFT p50 −11.4 % with decode provably untouched. The pattern (optimize
  the state-restore path, prove decode untouched) transfers to KDA snapshots. MED.
- **Oversized-lever trap** (measured refutation): a claimed 16 %-of-wall replay waste was really
  4.6 % of replayed tokens (block-granularity rounding) ≈ 0.8 % of wall. Verify the mechanism in the
  target regime before sizing any lever. HIGH (method).
- **MoE prefill grouping** (measured prior art + modeled): grouped expert prefill 77–91 → 31 ms/layer
  at T=7168; the generic grouped GEMM is a measured loss at decode batch (140 ms vs 88 ms per-token).
  HIGH.
- **Load-what-the-checkpoint-ships** (measured): dequant-once-to-BF16 + BF16 GEMM recovered +15.5 %
  prefill and quality after the FP8 GEMM route died on sm_121; the earlier dequant→requant path cost
  −7 BFCL points. HIGH.
- **Dispatch-flag stacks dwarf precision work** (measured): baseline 507/740 tok/s → tuned stack
  717/1295 tok/s prefill (+41 %/+75 %); a silent fallback path alone cost 40–50 %. HIGH.
- **W4A4 prefill A/B dead on accuracy** (measured): median wall within noise, token match 70.6 %,
  mean KL 7.64. MED.
- **Packed-decode vs transient-BF16 rebuild crossover** (measured prior art): BF16 cuBLAS beats FP8
  Marlin 3.1–3.7× at M≥768; pick at load time by M. HIGH for EXL3 packs.
- **Prefill determinism: top-k order** (measured): above ~2051 tokens, indexer top-k selection order
  varies while the key set does not, and scatter-accumulation order changes FP32 sums; fix is a
  stable sort of selected ids before expansion. HIGH — this class caused repeated evidence-repair
  cycles in Atlas.
- **FlashKDA prefill accelerator** (measured standalone): 3.38–3.62× at 2048–4100 rows for
  H32/D128 eligibility; Moonshot's FlashKDA (MIT) is the reference. HIGH as a candidate.
- **Single-stream hybrid-MoE floor** (measured): decode ~85.9 tok/s near the architectural floor at
  C=1; flag wins are ~0 at C=1 and +8–12 % at C≥2. MED.

## Decode

**Measured on TensorFold 2026-11-01** (`docs/kv-cross-context-probe-2026-11-01.md` 11.2 and 11.8): a routing dump of
GLM-5.3-Flash on the GB10 ring shows weak expert locality (a row's 8 experts are in the previous 16 rows' sets 47 to
75 % of the time; 2048 rows touch 263 to 288 of 288 experts) and cheap predictors (token-keyed, layer-to-layer) with
recall 0.26 to 0.51 at k=8, three to six bytes fetched per useful byte; an L2 prefetch of picked experts and a deeper
register ring in the NVFP4 decode expert kernel are bit-exact and within noise of stock. Routed-expert prefetch is
closed on this platform; the L2 lever Atlas and the serving stack measured is the static next-kernel prefetch.

- **Roofline verdict** (measured): decode on GB10 is weight-bandwidth-bound; projections are 68 % of
  a step, attention 12.7 %, drafter 9 %; measured kernels run 74–88 % of achievable bandwidth.
  Every "decode kernel" lever tried was dead: access-pattern/prefetch 0.98–1.00×, W4A8 DP4A
  0.99–1.01×, NVFP4-GDN weights +4.6 % slower on one checkpoint. HIGH.
- **Achievable denominator** (measured): STREAM read 230 GB/s, copy 215; naive reads 234–241;
  treat ~240 GB/s as the ceiling and the 273 GB/s spec as marketing. HIGH.
- **Dispatch-by-M audit** (measured): routing to existing tensor-core tiles where scalar GEMV ran
  gave +30 %/+22 %/+9 %/+5.5 % across four sites; a mechanical sweep found 38 of 294 kernel handles
  with no dispatch site. HIGH — reuse the sweep method.
- **Alignment invariant** (measured): an odd vocab size (248077) made 16-byte `cp.async` loads fault
  on transposed lm_head rows; the standalone repro passed because it used an aligned checkpoint.
  Shape-diff first; pad strides and pass `ldb` explicitly. HIGH.
- **Register-residency wins depend on the whole function budget** (measured): GDN register retention
  regressed −11.6 % and −4.6 % before a static-index variant won +5.4 %; the prototype's 1.71× was
  irrelevant because it omitted the epilogue. HIGH.
- **Weight-load pipeline depth has an interior optimum** (measured): SoA loads +4.6 %, 3-deep
  pipeline +1.8–2.6 %, 4-deep −5 % (occupancy). HIGH.
- **Attention-kernel work at batched decode is worthless** (measured): paged attention = 0.4 % of GPU
  time at C=16; "wasted work behind a bandwidth wall is free". HIGH.
- **Batched GEMV bit-exactness proven; FP4-MMA verify dead at M≤8** (measured: 18.75 % lane
  utilization at M=3→16 padding). MED.

## Speculative / MTP drafting

- **A "K=4 regresses" verdict was a dispatch gap** (measured): the K=4 verify FFN fell through to a
  prefill GEMM on the attention layers; after the fix, K=4 measured −17.7 % TPOT vs K=3 and shipped
  as default. Re-check every historical "algorithm X regresses" verdict for dispatch gaps. HIGH.
- **Acceptance is workload- and warmth-dependent** (measured): cold p1≈0.72 vs warm 0.89–0.99;
  chains beat trees by 17 %; K≥5 flattens. Measure acceptance warm on the target workload before
  any drafter/depth decision. HIGH.
- **Fixed-K collapse at batch, and the cure** (measured): naive per-sequence MTP halved throughput
  at C=4; batched verify+propose, a K-vs-batch ladder (`4:3,8:3,16:1`), a cap and survival-score row
  pruning (D-Cut @ 0.75, +2.6 %) beat vLLM at 1.66/1.25/1.31/1.13/1.18/1.11/1.02/1.01× for C=1..128.
  Trust caveats: the K-ladder (1.09–1.90×) and D-Cut deltas are round-4 measurements under the
  sealed pre-fix defaults (same-settings A/B deltas, internally consistent, not comparable to
  post-fix runs); the 08-03 sweep is post-fix but ran an f16+fused-norm config whose numerics were
  never proven (explicit config, not made default). Re-validate before quoting externally.
  Break-even rule: a widened verify wins only while verify cost < ~1.8× plain decode. HIGH.
- **Batched fused verify killed with a pre-registered re-open key** (measured): byte-identity passed
  (and found a real cross-sequence corruption bug) but cost failed (3.37× plain); vLLM's inline
  per-sequence state slots are the known solution for GDN rollback if batched verify returns. MED.
- **Snapshots must carry drafter state** (measured): a snapshot restore skipped hidden-state capture,
  left the drafter cold, and cost −6.8/−9.2 % throughput; the fix gates restore on ≥256 matched
  tokens (crossover bracketed at 99 loss / 219 win). HIGH.
- **Spec is not output-neutral** (method): trajectory divergence invalidates TPOT comparisons across
  legs with different token counts; compare byte-identity first, and use probe/distribution gates
  with an A–A floor when the baseline is nondeterministic. HIGH.

## Weight quantization

- **Activation quantization is regime-dependent** (measured, blanket claim retracted): W4A8/FP4
  activations are dead at M≤3 (acts ~0.5 % of traffic) and dropping W4A4 at M=16 is a 10.4 %
  regression. Translate every literature "decode" claim into its M before importing. HIGH.
- **EXL3 GLM decode cost map** (modeled with measured precedents): per-token byte floor ~6 GB/token
  ≈ 45 tok/s at TP=2; current correctness-first kernels leave 3–10× on the table in projections and
  coefficient decode; upstream spill-free siblings measured +8–17 %/layer. HIGH.
- **FP8 weight-only on BF16 checkpoints** (measured prior art): ~10.6 ms/step saved vs ~10.6 modeled
  — step time tracks bytes on GB10. MED-HIGH.
- **Tensor-core W4A16 tiers trail vendored Marlin** (measured): 167–200 vs 184–240 GB/s at M≤8;
  adopt Marlin's structure (grid, prefetch, LUT handling) rather than tuning the current shape. MED.
- **Quantization frontiers are model-specific** (measured): W8A8 on KDA collapsed a top-token
  probability 0.778→0.0051; SSM h_state narrowing failed the token-match bar; recurrent-state-feeding
  projections stay BF16/FP32 ("exponential error amplification in the decay gate"). GLM KDA is the
  same class. HIGH.

## Memory / NVMe offload

- **NVMe prefix-cache tier for GLM** (measured sync path): evicted 25K-token conversations return in
  1.5–1.9 s vs 11.2–11.5 s re-prefill, byte-identical answers; restore of 1,582 blocks (168 MB/rank)
  in 534–824 ms; per-record `pwrite` 222–256 µs dominates; 8 concurrent writers into `fallocate`d
  extents hit 2.6 GB/s vs 0.36 GB/s growing files; the pitched-GPU-copy fast path is projected only
  (35–92 ms), never served a request. HIGH.
- **Two-hop GPU→host→NVMe ≈ 3 GB/s** on gen4 SSD, disk-dominated; GDS/cuFile unavailable on GB10.
  MED.
- **`kcompactd` destroys tails** (one-boot, free fix): `vm.compaction_proactiveness=0` removed
  ~1.72× step-time episodes correlated with compaction bursts over GPU-pinned memory. HIGH.
- **Managed/host-registered memory silently ~35 % slower** (measured): 153.7–157.9 vs 234–241 GB/s
  reads; a silent fallback can price any tensor at −35 %. Fail closed or loud. MED-HIGH.
- **Closed doors** (measured dead): lossless trellis weight compression (0.99998 best ratio),
  hardware decompression (unsupported silicon), CPU-side decompression (≤29 GB/s), KV content
  compression (~1.23×), page colouring. LOW — record to prevent churn.

## Concurrency and scheduling

- **The whole C≥2 deficit vs vLLM was batch-scaling efficiency** (measured): per-sequence parity at
  C=1 (14.1 vs 14.2 tok/s) but 6.8× vs 11.9× scaling; root causes were shadowed kernels, per-sequence
  launch fan-out, a dead host-sampling path, a chunk-cap bug and the drafter handoff — decompose the
  step and find structural dead paths before tuning kernels. HIGH.
- **Prefix caching needs a matched-length threshold** (measured): warm-hit cost ~0.5 s/request on
  26-token prompts; bracketed crossover 99 loss / 219 win; shipped 256-token gate +6.9 %/+9.5 % at
  C=1/C=2. HIGH.
- **Grouped GEMM at decode batch is a trap** (measured): revert to per-token loops at small N;
  fused k2/k3 paths win at N=2/3; CUDA graphs under EP neutral; the binding constraint is bytes and
  collective latency, not launches. HIGH.
- **CUDA graphs +2.8–3.2 % at batch** (measured) — launch overhead is a ~3 % class overall. MED.
- **Config-surface traps** (measured): presence flags treating `=0` as enabled, clap defaults sealed
  before the env was read (five knobs wrong-but-consistent for 55 waves), silently falling back
  kernels, self-interfering benchmark drivers. Typed, fail-loud config is the design lesson. HIGH.
- **Multi-adapter routing is free** (measured): routed LoRA decode bit-identical to active-served via
  an identical-copy oracle; concurrent co-decode needs scheduling. LOW-MED for TensorFold, but the
  oracle is reusable test design.

## Kernel-level

- **Shadowed kernel copies silently drop entry points** (measured): a whole-file fork predating new
  common kernels left 7 model dirs without them; `try_kernel` returned 0 and logged at debug. Fix is
  build-time entry-point diffing per target. HIGH — Metal+CUDA doubles this surface.
- **PDL coverage gaps** (modeled + audit): zero EXL3 kernels, most W4A16 tiers; prefetch of the next
  router+shared weights during all-reduce waits is nearly free on GB10's 24 MB L2. HIGH.
- **Measurement trap catalog** (measured): `ncu` freezes a serving GB10; isolated microbenches
  overstate ~1.5× from L2 reuse; a template-bounded kernel swept past its `MAX_M` measures a smaller
  problem; rep-1 warmup outliers fake 1 % effects; graph replay zeroes host timers. HIGH — adopt as
  TensorFold's checklist.
- **Correctness red flags to look for** (audit): silently returning kernels on invalid splits,
  ulp-sensitive `exp2f(ceilf(log2f))` scale paths (use `frexpf`), dead branches inflating accumulators,
  early returns before `__syncthreads`, tier-switching routers that change arithmetic with row count
  (breaks C1-vs-C4 reproducibility). HIGH for GLM kernels.
- **Sized bit-identical backlog** (measured): top-k on one thread ~0.4 ms/token, post/contract reload
  0.5–0.9 ms/token, 14 `__syncthreads` → shuffles, fused qkv launch +1.6 %, ~1600 launches/token at
  M=1. HIGH on the EXL3 path.

## Model-family specific

- **GLM KDA records-mode rollback** (measured in-tree): KV capacity 940K→1.26M tokens at −2.5 % step
  C1 / −5 % C4 (figures flagged unlocatable in source); snapshots must be pair-path-safe; do not
  narrow live recurrent state to BF16/FP16 at C≤4 (drift measured). HIGH — snapshot capacity is
  conversation capacity.
- **GLM sparse-MLA default decode kernel is the slow one** (modeled + audit): the default re-streams
  selected rows K then V (~270 MB L2 traffic/layer for 2.1 MB unique); a tensor-core split path
  exists behind a flag with a hard-coded split count. HIGH.
- **GLM MoE at small rows** (modeled): router ~90 µs warm at M=5 vs ~10 µs roofline; shared expert
  underfilled; estimated 5–8 % of a C1 verify step. HIGH.
- **GDN batched recurrent fallbacks fire silently** (measured): non-contiguous pool slots disable the
  fast path (28 % penalty when it fires). MED.
- **Architecture invariant trap** (measured): a model with `q_heads×head_dim != hidden` OOMed buffers
  sized by `hidden_size`; per-layer rope theta and layer-type variety were silently discarded.
  LOW for TensorFold models, MED as a loader checklist.
- **External repos worth knowing** (mixed): rvLLM `fp8_decode_v2.cu` (GQA-grouped split-KV),
  sm120_nvfp4_ops (FP4 sparse-MLA KV), FlashKDA (MIT), exllamav3 (MIT) EXL3 decode techniques,
  SWITCHLESS-NCCL measurements (2nd rail, `NCCL_CROSS_NIC`), gb10-attn L2-defeat measurement method.
  Not useful: FlashMoE, UltraEP, warpcourier, deepgemm-rs, optimized-CUDA-GB10 (contradictory peak),
  FlashAttentionV2-PLUS (no license).

## Do not carry

- Defective or retracted numbers: fp8-KV decode speedups (same-binary control refuted), the "16 % of
  wall" replay projection, the "31.39 ms vLLM TPOT" target, pre-fix KL-gate outputs, the "60–65 % of
  peak" GEMV figure, L2-inflated microbench absolutes, the 231 GB/s "ceiling", and all unlocatable
  `[R?]` figures. Numbers spanning the sealed-defaults fix must not be compared across it (same-
  settings A/B deltas remain internally consistent), and the 08-03 C=1..128 sweep carries
  undischarged f16+fused-norm numerics debt.
- Dead ends (measured dead; do not rebuild): W4A8/FP4 activation quant at small M, NVFP4 GDN weights,
  fp8-KV for speed, K≥5 chains and tree drafting with the current drafter, GDN register retention
  without an epilogue split, 4-stage weight pipelines, L2 persistence windows, split-K at M=1,
  attention rewrites at batched decode, FP16/BF16 SSM snapshots, lm_head+argmax fusion.
- Process: the per-lever adversarial fold ceremony (keep the gates: coherence smoke, byte-identity,
  same-binary control, n≥3 with warmup discard; run them as a checklist), single-boot prior-art as
  evidence, analytical "% of work wasted" as a speedup prediction (over-predicted 3× twice), and
  env-var presence semantics (`=0` meaning on).
