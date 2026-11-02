# Local GLM CUDA kernel research

This is local research against `d33dcfc7`, not an upstream qualification or release receipt.
Device and served results are being collected; the implementation stays opt-in.

## Scope

- GLM-5.3-Flash, the existing Mia-AiLab EXL3 4-bit MCg checkpoint, revision
  `57edefd2f5d9b371c8345883304d5af68b52fa24`, at TP=2 on two GB10s with 128 GB RAM each.
- Checkpoint weights, BF16 activations and KV, and FP32 accumulation/state remain as the baseline stores them.
- Zig 0.17.0, CUDA 13.0.88, driver 580.159.03, NCCL 2.30.7, Ubuntu 24.04.4.
- Native serving: context 131072, four admitted streams, prompt passes up to 2048 rows, MTP plus copy drafts,
  thinking disabled, prompt cache disabled during cold and decode comparisons.
- The user authorized exclusive use of these two nodes and local work without ATC claims.
  No source, pull request or results are being published upstream.

## Implemented experiments

| Knob | Behavior | Unselected path |
| --- | --- | --- |
| `TF_GLM_KDA_VECTOR=1` | Vectorized staging and loop unrolling of the existing prompt recurrence | Original chain for decode, small rows, other head counts or unaligned inputs |
| `TF_EXL3_N256=1` | Small-row N256 down-projection tile for eligible uniform 4-bit MCg grouped experts | N128 for gate/up, larger rows, other formats or incompatible widths |
| `TF_EXL3_FUSED_EPILOGUES=1` | Gate/up and down tails in the common GLM EXL3 decode path | Existing unfused path when geometry is ineligible |

Each knob accepts only unset, `0` or `1`. N256 and fused tails are separate experiments and cannot both be selected:
the fused implementation uses an N128 tile and would otherwise hide the N256 selection. KDA can combine with either.
The ranks compare all three settings before loading weights.

WY-style chunking was not substituted for the existing KDA recurrence: it changes FP32 state bytes and cannot be
qualified as the required bit-preserving replacement. The implemented KDA experiment retains each update and
readout's arithmetic order. Its screening evidence, including the unsuccessful tile sweep, is in
[the KDA receipt](../glm-kda-vector-d33dcfc7/README.md).

## Correctness and build gates

- KDA: complete recurrent state and readout bytes, saved buffers, fused/wide/vector chain equality and replay;
  independent cancellation fixtures distinguish FP32 from FP16 and BF16 accumulation.
- N256: per-matrix projection bytes, independent FP64 projection reference, FP32 cancellation stress, finite and
  nonfinite outputs, row boundaries, and a row-alias control. Gate/up inputs are distinct.
- Fused tails: projection partials and final Xd/per-slot output bytes versus the existing epilogues, NaN behavior,
  repeated runs, guards, ticket initialization and graph replay. Tail-output accumulation limitations are explicit;
  the separate projection registration supplies the discriminating FP32 stress proof.
- All new operators are registered in the full CUDA contract runner. Focused diagnostics do not replace its summary.
- CUDA 13.0 compiles and SASS audits cover SM80/86/89/120/121. The records distinguish driver-visible shared memory
  from Blackwell's reserved ELF bytes and leave later compiler versions unqualified.
- The starting lean baseline is 425 findings; acceptance is zero new findings.

## Measurement protocol

Keep exact rendered inputs, request bodies and full token SHA256 values with every result. Reject a comparison that
changes output tokens. Cold cells use exact 2048/8192/32768/65536-token arrays, fresh state and cached-token count zero.
Four repetitions retain the first as warmup and use the remaining three for reported medians. Decode uses public
prose and code prompts, fixed seed 1729, 256 output tokens, greedy plain and drafted runs separately. Concurrent,
sampled, cancellation and resumed-path checks are separate correctness cells.

Compare the original binary with the candidate's experiments disabled before interpreting any enabled mode.
Alternate baseline/candidate runs for small differences. Operator timings and cache-resident launch diagnostics
are supplementary; they do not establish served speed. Keep peak process/device allocations and host memory
headroom with the run. Use the existing per-node GPU lock and one model process per node.

The first exploratory baseline omitted the RoCE adapter names, so native small gathers fell back to NCCL sockets.
It completed the cold cells with repeatable hashes and zero cache reuse, but decode spent roughly 80% of its time
in gathers. That run is retained as a socket control and was stopped deliberately. Subsequent comparisons select
the existing two-rail RoCE path explicitly and verify its startup message. This configuration change is not a kernel
speedup and must not be mixed into an arithmetic A/B.

Initial compilation overlapped part of that exploratory baseline. Final performance comparisons must run after
compilation has finished. The socket-control timings are not the final performance claim.

## Status

The final combined native CUDA build passed after the small-row addition. The final host checks completed 469/469 build
steps and 2167/2181 tests, with 14 skips. The lean checker reports 425 findings, the same count as the baseline.

On GB10, N256 passed two focused replacement contracts with its planted control caught, and the GLM EXL3
integration command passed, including row and chunk invariance and NaN isolation. Its worst error was 0.186 of
the existing FP64 tolerance. Across rows 1/4/8/16, N256 gate/up took 24–35% longer and down took 31–43% longer
than N128. A smaller-row prototype also preserved bytes but ran slower still. These operator timings used one
active expert, so grid occupancy differs from normal GLM routing; they do not establish a served regression.
With eight active experts, the full N256 tile was also slower. The revised small-row down-only tile passed final
focused and GLM integration tests. In the compiler-idle eight-expert probe it was 2.36–3.34% faster at rows 1, 4,
and 8. Its served comparison remains pending; see the [N256 receipt](../exl3-n256-d33dcfc7/README.md).

Fused epilogues passed 96 device cells including graph replay, and all four focused contract cases with both
planted controls caught. Cache-resident operator timings were about 3–7% slower; served performance is pending.

KDA's focused GPU diagnostic passed all six cases, with its planted control caught. The full `glm-ref` command
passed the fused, wide and vector chain comparisons and replay; the worst projection error was 0.982 of the
existing FP64 tolerance. The maintained probe passed 768 exact comparisons and retained 1,280 alternating timing
pairs. At 32 heads and 2,048 rows, the paired update medians were 3,109.22 µs baseline and 3,038.50 µs candidate.

The first full CUDA contract run timed out after 1,800 seconds before it reached these registrations or launch
limits. Its partial log is retained; it is not a full contract pass. The completed checks reproduced declared
Qwen non-finite findings and planted controls. A missing KDA image lookup and an understated fused-group launch
record were subsequently corrected, with boundary tests and compiler-image evidence in `limits-fix/`.

The first RoCE model campaign failed before readiness because the head rank could not register RDMA memory under
its 8 MiB locked-memory limit. A process-local limit adjustment was verified in a separate healthy startup, with
both ranks reporting the intended two-HCA RDMA path. No requests ran in the failed campaign. The corrected live
limits diagnostic reaches every registered image; all 17 research records pass. Its overall result remains a failure:
168 missing cross-SM usage records and 15 live record mismatches in other kernels. The affected records are unchanged
from the starting revision, and all eight affected image groups have identical baseline and candidate fatbin hashes.
The full CUDA contract summary and real-model comparisons remain pending.

The original, all-off, KDA vector, and second all-off model modes subsequently completed: each retained 46
audited responses, with identical token hashes for all 15 complete shared request bodies. A preliminary KDA
comparison bracketed by the two all-off modes does not establish a served speedup. Cold TTFT changed by
-0.10%, +1.12%, +0.28%, and +0.32% at 2K, 8K, 32K, and 64K respectively (positive is slower); decode
throughput changes ranged from -0.42% to +1.14%, with all four decode sample ranges overlapping controls.
Three measured repetitions per cell and the control drift limit the conclusion. Raw comparisons are in
`preliminary-kda/`; the remaining modes and final aggregate remain pending.

N256 and its trailing all-off control also completed, with all 15 complete request bodies token-identical across
the six completed modes. The preliminary N256 comparison against its neighboring controls shows no consistent
served gain: cold TTFT changed by -0.20% to +0.51% (positive is slower), and decode throughput changed by
-2.29% to +1.07%. Control drift reached 3.07%; three of four decode sample ranges overlapped the controls.
The remaining decode cell was slower with nonoverlapping observed ranges. Three measured repetitions per cell
limit the conclusion. Raw comparisons are in `preliminary-n256/`.

The complete eight-mode model matrix passed: 368 independently audited completed responses, 376 audited
requests including cancellations, and identical token hashes for all 15 complete shared request bodies. Fusion's
preliminary comparison against its neighboring controls showed decode throughput 0.05–1.47% lower and cold TTFT
between 0.69% faster and 0.34% slower; three of four decode sample ranges overlapped controls. No model speedup
was demonstrated. Raw comparisons and model dispatch evidence are in `preliminary-fusion/` and
`dispatch-evidence.json`. The full CUDA contract completed with 446 kernels passing numerical gates, all 321
planted controls caught, and six registered known findings. Its overall exit was 1 because of the 183 inherited
resource-record problems described above. The independent cache-resume check passed for all three experiments:
24 warm/fresh replies had matching inputs and output token hashes, nine warm requests showed positive cache reuse,
and all 12 fresh requests reported zero cached tokens. All six server phases cleaned up and both GPUs were released.
The [cache-resume receipt](cache-resume/README.md) contains the request-level audits and provenance.
The [post-contract receipt](../glm-kernel-post-contract-d33dcfc7/README.md) retains the full summary and
reproducible classification.
The [paired model receipt](../glm-kernel-paired-d33dcfc7/README.md) includes a reproducible, privacy-reviewed
derivation from the retained raw campaign archive.
