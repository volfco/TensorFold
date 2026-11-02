# Track 4 item 2: expert-tile prefetch prototypes (Zig engine)

Every kernel here is UNBUILT and UNTESTED: written on an x86 host with no nvcc and no GPU. No speed claim is made
anywhere in this file. Tags: [O] observed (read in source or a log), [C] calculated, [H] hypothesis, [U] untested.

## Contract (docs/recipes/SKILL.md form)

- Behaviour: two opt-in, loads-only changes to GLM-5.3-Flash's routed NVFP4 *decode* experts on CUDA.
  (a) `TF_GLM_EXPERT_PREFETCH=bulk|lines`: right after the top-k pick, a side stream asks L2 for the head of every
  column-block run of each picked expert's gate-up and down blocks. (b) `TF_NVFP4_DECODE_RING=2|3|4`: the decode
  expert kernel with D k32 groups of loads in flight in registers instead of the stock two. With both unset nothing new
  is loaded or launched and every existing symbol and launch is unchanged.
- Owning modules: `zig/src/families/glm/cuda_expert_prefetch.zig`, `zig/src/families/glm/cuda_expert_ring.zig`,
  `zig/kernels/cuda/glm_expert_prefetch.cu`, `zig/kernels/cuda/nvfp4_expert_ring.cu`; wiring in `cuda_moe.zig`
  (issue after `glue.select`, join before the next layer's router), `cuda_experts.zig` (ring dispatch on decode),
  `cuda_forward.zig` (load/unload in `Modules`), exports in `cuda.zig`; contract records
  `zig/src/core/contract/glm_expert_prefetch_pending.zig`; bench `zig/tests/glm_expert_prefetch_bench.zig`.
- Exactness: (a) writes nothing the model reads (prefetch instructions only; ids outside `[0, experts)` are skipped,
  so no address outside the expert tensors is formed). (b) computes groups 0..KG-1 in the same order into the same
  fp32 accumulators with the same `compute_stage`; only when loads issue changes. Both are "same bits by
  construction"; the GB10 bench checks it by hashing outputs.
- Cannot run here: compiling either .cu, register/spill numbers, loading, timing, the bit comparison.

## What the GLM NVFP4 routed experts run today [O]

- Path: `cuda_moe.body` (zig/src/families/glm/cuda_moe.zig:51-71) -> `experts.route` + `experts.run`
  (zig/src/families/glm/cuda_experts.zig) -> `cuda.experts.Grouped.decode` / `.prompt` -> `Nvfp4Experts.run`
  (decode) or `.runPrompt` (prompt) in zig/src/cuda/experts.zig:415-458. `exl3_*` is only taken when the checkpoint is
  EXL3 (`l.exl3 != null`, cuda_moe.zig). The NVFP4 recipe (docs/recipes/glm-5.3-flash-cuda.md:3-24) serves
  `nvidia/GLM-5.3-Flash-NVFP4`; the Python ring on the Sparks serves the EXL3 checkpoint, so the Python stack's numbers
  are for a different expert kernel.
- Decode symbols: `tf_nvfp4_experts::nvfp4_expert_kernel<M, EPI, 4>` (zig/src/cuda/experts.zig:346-349); GLM decode
  uses gate-up `<2, 2, 4>` and fp32 down `<1, 0, 4>` (`Which.down_f32` on non-prompt, cuda_moe.zig). Source:
  zig/kernels/cuda/nvfp4_experts.cu:133-143 (kernel), :98-131 (`unit_full`), :77-96 (`k_loop`).
- Decode load pipeline: per warp, a 32-column block of one item over all K. `k_loop` keeps `D = 2` register stages
  (nvfp4_experts.cu:80, `Stage<M> st[D]`): each stage is one k32 group, loaded with
  `ld.global.nc.L1::no_allocate.v4.u32` (nvfp4_experts.cu:14-20, :34-49: M uint4 of codes plus M uint4 of scales a
  lane, and the lane's bf16 inputs by `__ldg`). No shared memory, no cp.async: at most two groups (2 x M x 576 bytes a
  warp) in flight. Weights per expert: up `[NI/32][D/32][2][144]` int32, down `[D/32][NI/32][1][144]`
  (zig/src/cuda/experts.zig:24-25); expert e's column block cb starts at `(e * NB + cb) * KG * M * 36` uint4
  (nvfp4_experts.cu:107).
- Prompt path (for comparison): `Prompt4`/`tf_nvfp4_prompt_*`, cp.async ring of `STAGES` = 4/3/2 by shared bytes
  (nvfp4_experts.cu:309-314, :393-400; `promptSmem` in zig/src/cuda/experts.zig:362-366). Not touched here.

## What the Python stack measured, and what it implies [O]

`~/glm53-tensorfold-spark/patches/0460-glm-prefetch-comm.patch` (read on the rank-0 node): a tiny kernel on a side stream
forked at the site issues `cp.async.bulk.prefetch.L2.global` per chunk (or `prefetch.global.L2` per line). Static sites
(`a`, `f`, `o`: the next kernels' weights during the all-gather / small-kernel windows) plus a device-indexed site `e`
(first 64 KiB of each picked EXL3 expert right after the router). Its own summary
(`docs/CHANGES-SUMMARY.md` there, W12): 8 MiB a site, 1 stream +2.4-2.5 %, lone requests +2.8-3.2 %, 4 streams
+1.2 %; reply hashes equal. The same table lists **site `e` at 0 %** ("L2PF 4 MiB, site `e`: within bench noise;
+0.8 %; 0 %"). So the +2.4 % came from filling DRAM-idle windows with *static* next-kernel weights, not from
prefetching routed experts after the router. [H] The routed-expert prefetch here starts from a 0 % prior on a
different kernel (EXL3 there, NVFP4 here); the bench decides.

## First establish whether there is a stall to remove (GB10 unified memory) [U]

There is no host-to-GPU copy to hide: the expert read is LPDDR5X bandwidth. A prefetch can only help if (1) DRAM is idle
somewhere between the pick and the expert kernel's first loads, and (2) the expert kernel is latency-bound at its start
(few warps, each waiting on two register stages). Measure before believing either:

```
# one decode stream on the Zig engine, a few hundred tokens
nsys profile -t cuda --cuda-graph-trace=node -o glm-decode ./zig-out/bin/tensorfold serve <NVFP4 dir> ...
#   per MoE layer: gap from the select kernel's end to nvfp4_expert_kernel<2,2,4>'s start (route + plan time),
#   and the expert kernels' durations at 1, 4, 8 rows
ncu --kernel-name regex:nvfp4_expert_kernel --launch-count 50 \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,smsp__average_warp_latency_issue_stalled_long_scoreboard \
    ./zig-out/bin/tensorfold ...
```

Reading: if the expert kernels already sit near the DRAM roofline (dram throughput high, long-scoreboard stalls not
dominant), neither prototype can help and both should be dropped. If DRAM throughput is low with long-scoreboard
stalls dominant, (b) is the candidate; if the pick-to-kernel gap is long and DRAM idle in it, (a) is.

## Files (all new unless marked)

- `zig/src/families/glm/cuda_expert_prefetch.zig`: `TF_GLM_EXPERT_PREFETCH` (`0|1|bulk|lines`),
  `_GROUPS` (1-64, default 2), `_ROWS` (1-2048, default 64), `_IMAGE` (path to the hand-built fatbin); all refused by
  name when malformed, even with the mode off. Byte geometry from `cuda.experts.Layer`, `Prefetcher` (side stream,
  fork/join events, launch). Host tests.
- `zig/src/families/glm/cuda_expert_ring.zig`: `TF_NVFP4_DECODE_RING` (`2|3|4`), `_IMAGE`; refuses
  `TF_NVFP4_DECODE_WARPS` other than 4 (the ring is built at 4 warps only). `Ring.run` mirrors `Nvfp4Experts.run`'s
  arguments and grid. Host tests, including a model of the ring loop's slot discipline for D = 2..4, KG = 1..69.
- `zig/kernels/cuda/glm_expert_prefetch.cu`: `tf_glm_expert_prefetch_bulk`, `_lines`, and a bench-only spin kernel.
- `zig/kernels/cuda/nvfp4_expert_ring.cu`: includes `nvfp4_experts.cu` (unchanged), adds `k_loop_ring<M, D>`,
  `unit_ring`, and `extern "C"` entries `tf_nvfp4_ring_{gate_up,down_f32}_d{2,3,4}`. D = 2 is the control: it must
  equal the stock kernel's bits and time.
- `zig/src/core/contract/glm_expert_prefetch_pending.zig`: launch geometry of all 8 entries, usage pending (every
  applicable SM fails qualification until `tf-cuda-test kernel-contract-limits` supplies compiled usage), and the
  stock kernel each ring entry must equal. Imported by `registry.zig`'s test block (one line).
- `zig/tests/glm_expert_prefetch_bench.zig` (`tf-glm-expert-prefetch-bench`): synthetic NVFP4 weights (finite e4m3
  scales), random top-8 picks, host-built plan; variants stock, prefetch-lines, prefetch-bulk, ring2/3/4,
  ring3+prefetch-bulk; L2 flushed (256 MiB fill) before each rep; outputs hashed and compared to stock (exit 1 on any
  difference); n reps (>= 3), first discarded, median/min/max printed. The stock kernels are taken from the ring image
  (which carries `nvfp4_experts.cu`'s instances), so the bench needs no embedded fatbins.
- Changed: `zig/src/families/glm/cuda_moe.zig` (+8), `cuda_experts.zig` (+15), `cuda_forward.zig` (+5),
  `cuda.zig` (+3 exports), `zig/build/glm_family_host.zig` (+11: step `test-glm-expert-prefetch-host`, step
  `glm-expert-prefetch-bench`), `zig/src/core/contract/registry.zig` (+1 test import).

## Host checks run (Zig 0.17.0, x86, 2026-11-01)

- `zig build test-glm-expert-prefetch-host` -> 21/21 tests (19 module tests incl. a reference-everything test that
  forces analysis of `Modules.load`, `moe.body`, `experts.run`, `Ring.run`, `Prefetcher.issue`; verified to catch a
  planted arity error in `cuda_moe.zig`), 2 bench helper tests, and the bench executable compiles.
- `zig build test-kernel-contract compile-glm-host` -> pass (includes the new pending-records tests).
- `zig build test-cuda-host compile-glm-host test-kernel-contract test-glm-expert-prefetch-host` -> exit 0 (the
  FAIL lines in its log are the contract's planted controls failing as designed).

## GB10 commands (user runs; the production ring need not stop for the bench, but it shares the GPU: stop it for
## clean numbers, and restore it by hand afterwards)

```
cd <TensorFold checkout on a GB10>
NVCC=/usr/local/cuda/bin/nvcc
FL="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -Izig/kernels/cuda \
    -gencode=arch=compute_121,code=sm_121"
mkdir -p ~/kvprobe/img
$NVCC -fatbin $FL zig/kernels/cuda/glm_expert_prefetch.cu -o ~/kvprobe/img/glm_expert_prefetch.fatbin
$NVCC -fatbin $FL zig/kernels/cuda/nvfp4_expert_ring.cu   -o ~/kvprobe/img/nvfp4_expert_ring.fatbin
# registers and spills of every ring entry (D = 4 may spill: then drop it)
$NVCC -cubin $FL -Xptxas -v zig/kernels/cuda/nvfp4_expert_ring.cu -o /dev/null 2>&1 | grep -A1 tf_nvfp4_ring
# the bench (host-only cuda module: no -Dnvcc needed for it)
zig build glm-expert-prefetch-bench -Doptimize=ReleaseFast
for R in 1 4 8; do
  ./zig-out/bin/tf-glm-expert-prefetch-bench --ring-image ~/kvprobe/img/nvfp4_expert_ring.fatbin \
    --prefetch-image ~/kvprobe/img/glm_expert_prefetch.fatbin --rows $R --reps 7 \
    --dims <hidden_size> --width <moe_intermediate_size / TP> --experts <n_routed_experts>
done
# in the engine (decode only; prompt chunks wider than _ROWS skip the prefetch; the ring never runs on prompts)
TF_GLM_EXPERT_PREFETCH=bulk TF_GLM_EXPERT_PREFETCH_IMAGE=~/kvprobe/img/glm_expert_prefetch.fatbin ...
TF_NVFP4_DECODE_RING=3 TF_NVFP4_DECODE_RING_IMAGE=~/kvprobe/img/nvfp4_expert_ring.fatbin ...
```

Acceptance for either prototype: every bench line `bits=SAME` (any DIFFER is a bug, not noise); ring2 within noise of
stock (it is the same code); then a served A/B on the same binary with the flag off/on, n >= 3, reply hashes equal,
before any number is quoted. Also run `tf-cuda-test kernel-contract-limits` on the built images and replace the
pending records with the measured usage.

## Gate a maintainer must agree (not obtainable here)

CONTRIBUTING requires an issue and an agreed gate for kernel changes. Proposed gate: (1) bits: bench `SAME` on rows
1/4/8 and on real checkpoint weights, plus served token-digest equality flag on vs off; (2) contract: the 8 records
qualified on sm_121 with measured usage, no local-memory spills in the ring entries; (3) speed: a same-binary served
decode A/B at 1 and 4 streams with n >= 3 and the A-A floor reported; adopt only above it. Also needed: putting the two
.cu files into `zig/build/cuda_kernels.zig`'s list (and `zig/src/cuda/kernels.zig`) instead of hand-built images,
which this prototype deliberately avoided so an uncompiled file cannot break the default build.

## Open decisions

- Whether to pursue (a) at all given the Python stack's site `e` measured 0 %; the static next-kernel sites (`a`, `f`,
  `o`) that carried its +2.4 % have no Zig-engine port here (out of this track's scope; the same Prefetcher pattern
  would serve them).
- Overlap source for (a): in the Zig engine the shared expert runs after the routed experts (`moe.fill`), so only the
  route kernel separates the pick from gate-up. The Python stack moved its shared expert before the routed ones while
  prefetching (disjoint inputs/outputs, same bits); doing that here touches `moe.body`/`fill` ordering and is not done.
- Prompt rows: unchanged (cp.async ring already 2-4 stages).
