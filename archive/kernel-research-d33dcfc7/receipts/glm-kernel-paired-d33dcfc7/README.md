# GLM kernel experiments: paired served-model receipt

Local research on source `d33dcfc7` plus the recorded experimental changes. All three
experiments remain off by default. This receipt contains a privacy-reviewed projection of
completed model measurements; the unchanged raw archive remains outside the repository.

## Result

All eight modes completed successfully: original, off1, KDA vector, off2, N256, off3,
fused epilogues, off4. Each mode saved 46 replies and 46 independently captured complete
response hashes. All 15 complete request bodies produced matching token hashes across
all modes. The 376 audited requests include eight intentional cancellations; those bodies
are retained only by hash and descriptor here. Every saved reply reported zero cached tokens.
All model processes were stopped and owned cleanup was confirmed after each mode.

The timing pilot does not establish a consistent served-model speed gain. Decode throughput
changes against the mean of the surrounding all-off medians were:

| Experiment | Prompt 14, plain | Prompt 14, drafts | Prompt 17, plain | Prompt 17, drafts |
| --- | ---: | ---: | ---: | ---: |
| KDA vector | -0.18% | -0.42% | +0.42% | +1.14% |
| N256 small down | -0.40% | -2.00% | +1.07% | -2.29% |
| Fused epilogues | -1.34% | -1.47% | -1.24% | -0.05% |

There were three measured repeats per cell, after an excluded warmup. This is one
experiment pass between two control passes. The bracket reports retain every sample,
min/max, standard deviation, control drift and observed range overlap. Control drift reached
about 3%; overlapping observed ranges are not evidence of a resolved performance difference.
KDA's four decode ranges overlap controls; N256's final drafted cell and fusion's first plain
cell show slower, nonoverlapping observed ranges. Cold TTFT/wall results are retained separately.

## Setup and coverage

- Two NVIDIA GB10 GPUs, sm_121, driver 580.159.03, CUDA driver API 13.0, NCCL 2.30.7.
  The head ran Ubuntu 24.04.4 LTS on Linux/aarch64; its OS release was checked after the model run.
- Mia-AiLab GLM-5.3-Flash EXL3 4-bit routed-expert checkpoint, snapshot
  `57edefd2f5d9b371c8345883304d5af68b52fa24`, tensor parallel world 2.
- Context 131072, parallelism 4, prompt cache 0 GiB, thinking disabled, profiling disabled.
  Two RoCE HCAs per rank were verified before requests; both rails' counters increased.
  Thermal bands were 90/85 C. Launchers kept the original user and unlimited process-local memlock.
- Cold prompts: an excluded 1024-token warmup, then 2048/8192/32768/65536 tokens with
  four calls each, first excluded. Decode: two fixed generic prompts, 256 output tokens,
  drafts off/on, four calls per cell, first excluded. Verification includes seeded sampling,
  concurrency and a request following cancellation. Exact request parameters are in `requests.json`.
- `identities.json` pins both native binaries, final contract runner, launcher, wrapper,
  prompt corpus and matrix code. The same native binary hash was checked on both ranks.
  `dispatch-evidence.json` records source-hashed eligibility for the actual TP2 decode geometry.

These model token checks complement the [KDA operator receipt](../glm-kda-vector-d33dcfc7/README.md),
[N256 operator receipt](../exl3-n256-d33dcfc7/README.md) and
[fused epilogue operator receipt](../glm-exl3-fused-d33dcfc7/README.md). They do not compare every
hidden-state/logit byte. The overall live-contract command reported 183 unrelated registry
completeness/currency findings; all 17 research launch records passed. Full numerical-contract
and cache/resume results belong to their separate receipts, not this completed model snapshot.

## Reproduce the analysis

Python 3.11 or newer, standard library only; no GPU or network is used:

```sh
python3 reproduce.py
```

This verifies receipt/data hashes, all saved-to-independent response associations, all 15-body
cross-mode comparisons, zero fresh-cache counts, successful mode exits/cleanup, and exact
recalculation of all three bracket reports. `samples.jsonl` retains the original numeric
measurements; `audited-responses.jsonl` retains independent complete response hashes.

With access to the unchanged raw archive outside the repository:

```sh
python3 reproduce.py --raw /path/to/tensorfold-paired-r2-models.tgz
```

This additionally verifies its exact archive hash and all 1558 original manifest entries,
then reproduces every derived data file byte-for-byte through `derive.py` in a temporary folder.
The archive SHA256 is `fff265110c75e063a200fe54ab78b64f00eab9b8aff6b061b1b04ef9a4a2928e`.
Its final manifest SHA256 is `b04d8106b781666f271e5d7957e4bc7c8f6dbcac83e65c1e296fdfd4b4474e93`.

## Privacy and provenance

The raw archive contains private infrastructure addresses, home paths, operational
handoff authorizations and generated replies. It was not copied into the repository.
The derived data omits all prompt text/token arrays, generated text, process IDs, interfaces,
addresses, paths, operational scripts, authorization files and Python bytecode. Fixed benchmark
bodies are represented by complete byte hashes, lengths and non-content parameters.
No credential patterns were found in the raw snapshot. `privacy-review.json` records the review.

`provenance.json` pins every raw input used by the derivation, the derivation code and every
derived data file. Bracket JSON retains its original raw input hashes. `SHA256SUMS` describes
these retained receipt files. The raw archive and signed/raw campaign evidence were unchanged.
