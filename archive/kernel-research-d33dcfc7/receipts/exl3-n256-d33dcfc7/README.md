# EXL3 N256 kernel research

## Scope

Local CUDA research based on `d33dcfc7`, on one GB10 (SM121), CUDA 13.0.88 and Zig 0.17.0.
The fixtures are synthetic: these receipts do not establish checkpoint fidelity or served throughput.
The original N256 kernel and both small-row revisions are recorded separately.

`TF_EXL3_N256` is off by default. The revised opt-in selects the small-row MCg4 down product only
for one to eight rows and whole 256-column tiles. Gate/up, larger windows, other codebooks and
unverified widths retain N128. Combining it with fused epilogues is refused.

## Results

The original N256 product passed its focused replacement gates: two kernels passed, one planted
row-alias control failed as intended, and neither inputs nor guard bytes changed. Each product
covered 13 shapes. The independent integer stress distinguished FP32 from FP16/BF16 sums.
GLM's synthetic family/common-owner integration exited zero: prompt/decode outputs met its existing
FP64 bounds, rows matched their solo calls, chunk boundaries held, and NaNs stayed in their row.

The one-expert timing fixture exaggerated the N256 underfill penalty, so the follow-up uses eight
distinct experts, distinct gate/up matrices, and two rotating weight banks. Each leg launches 160
products; six alternating repetitions retain the first as warmup and summarize the remaining five.
The banks cover 64 MiB for gate/up and 32 MiB for down, exceeding GB10's 24 MiB L2.

| Rows | Original N256 gate/up time change | Original N256 down time change | First small prototype down time change |
| --- | ---: | ---: | ---: |
| 1 | +1.24% | +6.98% | -2.98% |
| 4 | +4.26% | +7.56% | -2.45% |
| 8 | +2.93% | +6.07% | -3.23% |
| 16 | +5.07% | +10.10% | Not admitted |

These are operator time changes; positive means slower. The first small prototype retains only
the valid half of each MMA's accumulators. The production revision specializes its metadata and
dispatch for down products. Its first screening run overlapped CPU compilation and contains
outliers; `production-small/probe-device.log` preserves all samples. The separate compiler-idle
repeat passed all exact output comparisons and produced these median down-product times:

| Rows | N128 milliseconds | Small N256 milliseconds | Time change |
| --- | ---: | ---: | ---: |
| 1 | 0.090120 | 0.087399 | -3.02% |
| 4 | 0.088611 | 0.086521 | -2.36% |
| 8 | 0.091489 | 0.088430 | -3.34% |

`small-results.json` includes every retained sample and both production screening runs. The
original full N256 remained slower in every gate/up and down cell of the idle repeat.

The final focused command exited zero: three kernels passed, zero failed, and both planted
controls failed as intended. The new small kernel covered 14 shapes, five exact repeats,
unchanged inputs and guard bytes, and mixed finite/nonfinite witnesses. Its exact integer FP32
stress rejected FP16 and BF16 accumulation. Final opt-in GLM integration also exited zero, with
the worst difference using 0.186 of its existing FP64 tolerance. These are synthetic device
checks. The final runner hash is in `device-small/runner.sha256`.

The final small kernel compiled on SM80, 86, 89, 120 and 121 with 150 registers, 24,576 bytes
static shared memory, launch bound 128, one barrier and no stack, spills or local loads/stores.
Host checks and six-architecture semantic checks passed. Only SM121 received device tests.

The original full CUDA contract attempt timed out after 1,800 seconds, exit 124, before the new
research registrations or live limits phase. Its two Qwen attention nonfinite findings are already
declared `known` in the unchanged contract. The complete partial log is retained; there is no
full-contract `SUMMARY` or pass claim for that attempt.

## Reproduction

Build using the pinned Zig toolchain and CUDA 13.0:

```sh
zig build native install kernel-contract-fatbins -Doptimize=ReleaseFast \
  -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j2
TF_EXL3_N256=0 TF_EXL3_FUSED_EPILOGUES=0 TF_GLM_KDA_VECTOR=0 \
  zig-out/bin/tf-cuda-test exl3-n256
TF_EXL3_N256=1 TF_EXL3_FUSED_EPILOGUES=0 TF_GLM_KDA_VECTOR=0 \
  zig-out/bin/tf-cuda-test glm-exl3
nvcc -O3 --expt-relaxed-constexpr -arch=sm_121 -Xptxas=-v,-warn-spills \
  tools/zig/cuda_exl3_n256_probe.cu -o /tmp/exl3-n256-probe
/tmp/exl3-n256-probe
```

Use the existing GPU lock and one GPU job at a time. Original prototype sources and raw logs are
under `prototype-original/`; the maintained production probe is also copied under `production-small/`.
`environment.json` records the original tested runner's hash, and source hashes identify the CUDA
revision. Compiler/SASS artifacts cover SM80, 86, 89, 120 and 121; only SM121 ran on a device.

## Coverage limits

Eight shared routed experts are a controlled workload, not a full model's routing distribution.
No served speedup, whole-checkpoint comparison, concurrent serving or restart/resume result is
claimed here. Those belong to the separate paired-model campaign. The original negative timings
remain evidence even though the revised opt-in uses a different specialization.

## Redaction report

Artifacts are local research records. `privacy-report.json` records the text scan and any
redactions; `SHA256SUMS` covers every other artifact, including raw logs ignored by Git's global
patterns. Nothing in this directory has been uploaded.
