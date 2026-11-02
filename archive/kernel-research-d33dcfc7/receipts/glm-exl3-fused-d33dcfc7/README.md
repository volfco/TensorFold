# Common EXL3 fused decode epilogues: local research receipt

## Scope

Baseline source commit `d33dcfc7`, with the local kernel research changes identified in
`source.sha256`. GPU: NVIDIA GB10, CUDA architecture sm_121, driver 580.159.03, Linux aarch64.
The exact test-runner hash is in `runner.sha256`. The 2026-10-30 device campaign ran from
13:32:37 to 13:32:51 UTC. No checkpoint was loaded; these are operator checks, not served-model
qualification or an end-to-end speed measurement.

`TF_EXL3_FUSED_EPILOGUES=1` selects the common GLM decode slot path. Grouping initializes
scratch-owned counters; the gate/up projection's last arriving CTA runs its epilogue, and the
down projection runs its per-slot epilogue before returning. The implementation retains the
projection partials in global scratch and removes two standalone epilogue launches. The common
prompt path was already fused and is unchanged. The mode stays off by default. Selecting it
with `TF_EXL3_N256=1` is refused explicitly.

## Results

| Check | Result | Exit |
| --- | --- | --- |
| `tf-cuda-test exl3-fused` | 96 cells passed | 0 |
| `tf-cuda-test exl3-fused-contract` | Four registered kernels passed all arithmetic gates; two controls caught | 0 |
| CUDA source/include semantics | Six of six SM branches passed | 0 |
| `zig build test-exl3-cuda-host -j2` | Command completed; retain the attached warning/protocol log | 0 |

The 96 cells cover three codebooks, BF16 and FP32 activation arithmetic, clamped and unclamped
activation, rows 1/3/8/16/17/32, non-finite and signed-zero inputs, and pruned slots. Candidate
`Xd` FP16 and per-slot `y` FP32 bytes equal the unfused path after five eager runs and five graph
replays. Scratch/output guards survive, and the active arrival counters return to zero.

The focused contract registers both raw projection partials and the final FP16/FP32 tails.
Each positive entry passes non-finite, repeat/aliasing, shapes, and accumulation checks. Each
entry exercises 13 served/edge shapes. Independent FP64 projection stress proves FP32 sums;
final-tail stress also passes, while its registration retains an explicit limitation that
activation/output rounding can hide reduced-precision error. The deliberately broken row-alias
and tail-scale controls fail their intended shapes gate and count as caught controls.

```text
SUMMARY exl3-fused focused kernels=4 passed=4 failed=0 controls=2 caught=2 missed=0; launch limits belong to full runner
```

The cached synthetic timing diagnostic is a negative result:

| Codebook | Unfused ms | Fused ms | Candidate / baseline |
| --- | ---: | ---: | ---: |
| inst3 | 0.093981 | 0.096755 | 1.0295 |
| mcg | 0.092336 | 0.096432 | 1.0444 |
| mul1 | 0.090608 | 0.096557 | 1.0657 |

These timings use a small cache-resident fixture, alternating order, six timed samples per
path, ten calls per sample, after correctness warmups. They do not establish a production
regression or a served gain. The relevant mcg cell was approximately 4.4% slower. Keep the
experiment off until whole-model A/B evidence supports a different decision.

## Reproduction

With the candidate runner built from the recorded sources and CUDA runtime libraries available:

```sh
TF_EXL3_N256=0 TF_EXL3_FUSED_EPILOGUES=0 TF_GLM_KDA_VECTOR=0 \
  flock -n /tmp/atlas-bench.lock ./tf-cuda-test exl3-fused
TF_EXL3_N256=0 TF_EXL3_FUSED_EPILOGUES=0 TF_GLM_KDA_VECTOR=0 \
  flock -n /tmp/atlas-bench.lock ./tf-cuda-test exl3-fused-contract
```

The test commands select candidate kernels internally. `run.sh` records the actual locked
campaign, statuses, runner hash, times, and before/after GPU metadata. Device ownership was
clear before acquiring the lock; it was released after the campaign. The GB10 utilization
counter read 96% even with no clients and low idle power; it is retained in the raw metadata
and was not used as a readiness or performance measurement.

## Coverage limits and redaction

The focused summary excludes gate 5. Full backend contract/launch-limit qualification, served
before/after rates, model fidelity, drafted/plain equality, concurrency and cache/resume gates
belong to the parent campaign. This receipt establishes no Metal/HIP support or device results
on other CUDA architectures. All-SM compile/SASS records are retained separately by that campaign.

Local source/tool installation prefixes in copied host/source-check logs were replaced with
`$SOURCE_ROOT` and `$ZIG_ROOT`. Device logs contain no prompt or model data. Checksums describe
the retained receipt files after these substitutions; raw originals remain outside the source
checkout. No receipt was uploaded or published.
