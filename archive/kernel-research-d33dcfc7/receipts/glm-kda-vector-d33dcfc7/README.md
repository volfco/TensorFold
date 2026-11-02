# GLM KDA vector staging research

Local CUDA research on baseline `d33dcfc7`, 2026-10-30 UTC. This receipt covers operator
screening, compilation and focused device/reference gates. The full backend contract,
full-model and served performance qualification remain pending; operator timings are not
served throughput claims.

## Scope

The proposed time-parallel WY algorithm changes the sequential FP32 recurrence and therefore
cannot replace TensorFold's existing prompt/decode arithmetic under its byte contract.
This implementation instead keeps every state update and reduction in the existing order,
vectorizes the Q/K/decay loads into shared memory, and unrolls two or four consecutive rows.
It does not claim the original review's speculative 5–20x KDA speedup.

`TF_GLM_KDA_VECTOR=1` enables the research path. Unset or `0` keeps the shipped path;
other values are refused. Both ranks agree on the setting. Rows below 256, unmeasured head
shares and Q/K/decay sources without 16-byte alignment keep the shipped step.
The measured dispatch is TR16/unroll4 for 32 heads and TR8/unroll2 for 21, 22 or 64 heads.

Both implementations retain FP32 recurrent state and accumulation, BF16 readouts, and
`--fmad=false`. The original step body and entry point remain the byte oracle.

## Hardware and method

One NVIDIA GB10, CUDA 13.0.88, driver 580.159.03, Ubuntu 24.04.4, Linux 6.17.0-1018-nvidia,
sm_121; the node had 121 GiB of physical RAM. The worker GPU
was leased exclusively, with `/tmp/atlas-bench.lock` held during each screen. Observed GPU
temperature during the two screens was 35–36 C. Input/output allocations were outside timing.
This operator-only run does not measure process peak footprint or checkpoint fit.

Prepared inputs contain deterministic FP32 Q/K, BF16 values, FP32 decay/beta and FP32 state.
Each candidate reads identical immutable inputs. Readouts and final state have separate
guarded allocations; every output byte is compared to W8/TR16 before timing, with unchanged
input and guard checks. Non-finite fixtures include NaN and both infinity signs.

Each timed cell uses five warmup launches and five pairs of 20-launch CUDA-event batches,
alternating baseline/candidate order. All observations, including negative results, remain
in `screen1/` and `screen2/`. Ratios are medians of paired baseline/candidate timings.

## Results

The first screen tested 12 W/TR combinations over 48 finite/non-finite shapes:
576 comparisons passed with zero readout/state mismatches and intact guards/inputs.
The original W8/TR16 was fastest at 21, 22 and 32 heads for every timed row count.
W8/TR32 was rejected at compile time: 53,376 bytes of static shared memory exceeds
the default 49,152-byte limit. Its compiler error remains in `screen1/rejected-w8t32.log`.

The second screen tested vector staging and bounded unrolling: 480 comparisons passed
with zero readout/state mismatches and intact guards/inputs. Selected prototype 2,048-row results:

| Heads | Baseline median, µs | Candidate median, µs | Median paired speedup |
| --- | ---: | ---: | ---: |
| 21 | 2,337.49 | 2,307.33 | 1.0132x |
| 22 | 2,332.76 | 2,305.79 | 1.0106x |
| 32 | 3,105.54 | 3,038.68 | 1.0224x |
| 64 | 6,213.76 | 5,867.86 | 1.0581x |

The TP2-sized operator gain is small. A whole-prefill gain depends on the measured share
of time spent in this recurrence and must be established separately with served requests.

The maintained probe was then rebuilt against the integrated production helper and rerun
after device/reference checks. All 768 comparisons (48 shapes, 16 variants) passed, with
zero state/readout differences and intact guards/inputs. It retained 1,280 alternating timing
pairs across the complete head/row/variant ladder. Selected 2,048-row results from this reprofile:

| Heads | Baseline median, µs | Candidate median, µs | Median paired speedup |
| --- | ---: | ---: | ---: |
| 21 | 2,333.48 | 2,304.15 | 1.0126x |
| 22 | 2,326.53 | 2,297.70 | 1.0128x |
| 32 | 3,109.22 | 3,038.50 | 1.0229x |
| 64 | 6,206.21 | 5,895.61 | 1.0581x |

The maintained probe exited 0. `reprofile/` retains every qualification/timing record,
its binary SHA256, exact start/end times and selected-cell summaries. GPU temperature
after completion was 32 C. These measurements remain operator evidence.

## Compilation and checks

The integrated CUDA source compiled for sm_80, sm_86, sm_89, sm_120 and sm_121 using the
repository's CUDA macros, C++20, `-O3` and `--fmad=false`. Both new entry points have
zero stack, spill stores and spill loads. Full SASS inspection found zero LDL/STL
instructions for either candidate on every compiled architecture.

| Kernel | Registers, SM80/86/89/120/121 | Driver-visible shared bytes | Threads |
| --- | --- | ---: | ---: |
| vector8 | 64 / 64 / 64 / 58 / 58 | 13,344 | 256 |
| vector16 | 48 / 80 / 80 / 80 / 80 | 26,688 | 256 |

`compile/resources.log` and `compile/records.json` retain raw ELF resource values.
`compile/driver-records.json` subtracts Blackwell's reserved 1,024 shared bytes, matching
the repository parser's driver-visible convention. Compilation does not qualify another GPU.
`compile/sass.log.gz` retains the full disassembly, `compile/sass-audit.json` its hash and
per-kernel local-memory instruction counts, and `compile/environment.txt` the tool/runtime versions.

The new backend-contract cases cover both candidates' complete state/readout bytes,
non-finite witnesses, guarded repeats, served shapes and tile edges. Separate one-row
update/readout fixtures have integer cancellation near 2^20, with FP16/BF16 simulations
required to miss the bound in sequential, pairwise and strided orders. A last-row omission
control must fail replacement equality. The integrated focused diagnostic exited 0:

```text
SUMMARY KDA diagnostic: 6 kernels, 6 pass, 0 fail; 1 controls, 1 red, 0 missed; limits not run
```

The update stress rejected BF16 accumulation by 171 times its bound, and the readout stress
by 256 times; FP16 missed by an infinite factor. Both candidates passed non-finite classes,
five identical runs, every registered shape/edge and untouched-input/guard checks. The planted
last-row omission failed replacement equality as intended. This focused run does not replace
the full backend summary or its launch-limit checks. Raw logs and exit statuses are in `device/`.

`glm-ref` additionally compares fused, original-wide and opt-in-vector chains, all six
output buffers and replayed state at 256, 70 and 3 rows, with its existing FP64 output oracle.
The full `glm-ref` exited 0. KDA's worst-column fractions of the unchanged FP64 tolerance
were 0.6728, 0.4828 and 0.3043 respectively; all saved buffers, outputs and replayed states
matched. The complete GLM reference suite's worst projection was 0.9820 of its tolerance.
The runner SHA256 is recorded in `device/runner.sha256`.

Host and all-image/control semantic compilation exited 0. The latter uses inert images only
to type-check every contract branch; it is not a device test.

## Reproduction

Build the maintained standalone probe from the repository root:

```sh
nvcc -O3 --fmad=false -std=c++20 -arch=sm_121 -Izig/kernels/cuda \
  tools/zig/cuda_kda_probe.cu -o /tmp/kda-probe
flock -n /tmp/atlas-bench.lock /tmp/kda-probe
```

`--qualify-only` stops after exact state/readout checks. The maintained probe includes
the original W/TR screen and vector variants; the archived screen sources preserve the
exact earlier experiment definitions. The CUDA source and common headers used by screen 1
are retained with its source snapshot; screen 2 reuses that baseline source and common headers.

Run the integrated backend contract and `glm-ref` using a native runner built with real
kernel images and controls. Keep operator and served receipts separate, retain cold prompt
IDs and bodies, and check tokens before interpreting latency differences.
