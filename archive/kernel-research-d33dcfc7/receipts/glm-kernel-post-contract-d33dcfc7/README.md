# GLM kernel research: completed post-contract receipt

Both GPU commands finished with exit **1**. The numerical registrations pass under their
declared known-finding policies, but the overall contract remains red because of 183
unresolved resource-record problems. This receipt does not convert that red result into a pass.

```text
SUMMARY live CUDA limits: 183 problems; numerical gates not run
SUMMARY 446 kernels: 446 pass gates 1-4, 0 fail, 6 known findings; 321 controls: 321 red as planted, 0 missed; 183 limit problems; gates 2199248 ms, limits 2178184 ms
```

| Command | Exit | Result |
| --- | ---: | --- |
| `kernel-contract-live` | 1 | Complete live-resource audit; 183 problems |
| `kernel-contract` | 1 | Complete summary: 446 registrations pass, zero new gate failures, six known findings; all 321 controls caught; 183 resource problems |

The full command ran from 19:13:01 to 20:26:00 UTC on 2026-10-30, inside a 7200-second
limit; it did not time out. The runner's `limits` timing region also includes the later
page/learning component checks, so its 2178184 ms is not the duration of driver attribute
queries alone. Models were stopped before the commands; the owned head GPU lock was released
afterward. Subsequent cache/resume work is a separate campaign and is not qualified by this receipt.

## Research kernels

All 13 research numerical registrations passed every applicable nonfinite, repeat/aliasing,
shape and accumulation gate, and all five research negative controls were caught:

| Family | Positive registrations | Controls caught |
| --- | ---: | ---: |
| N256 generic products and small down | 3 | 2 |
| Fused EXL3 projection partials and FP16/FP32 tails | 4 | 2 |
| KDA vector8/vector16 bytes, state sums and readout sums | 6 | 1 |

The two KDA byte-comparison entries declare no accumulation test; their separate state/readout
entries pass FP32-sum stress. The 17 selected live-resource entries all pass: eleven experimental
symbols plus six comparison baselines. Exact gate records, controls and resource lines are in
`classification.json`; the complete reviewed logs are retained verbatim.

## Inherited findings remain unresolved

Six numerical findings are explicitly registered as known: four Qwen attention variants skip
a NaN chunk rather than propagating it, and two GLM score kernels use the documented ReLU/fmax
NaN behavior. They are preserved, not relabeled as fresh passing nonfinite checks.

The 183 resource problems comprise:

- 168 absent compiled-usage entries across the target-SM table: NVFP4 expert tiles 10,
  Qwen35 15, Qwen35 KV8 15, Qwen4 projection 20, GDN 20, HC 48 and PLE 40.
- 15 live usage mismatches on sm_121: eight missing records and seven stale Qwen register counts.
  Every live mismatch still fits the actual driver thread/shared-memory limits; the audit remains red.

The earlier independent source/image audit found the failing images byte-identical between
original and candidate and the failing records unchanged. Its hashes and limitations are
preserved in `inherited-resource-evidence.json`. No baseline full-contract runner was installed;
this is source/image inheritance evidence, not a claimed baseline command execution. The
findings need separate record collection/repair; no validation was relaxed for this receipt.

## Identities and scope

The final runner is pinned in `runner.json` and `source-identities.json`:
`b03d54ced7ce638706d5206724b8eefe6db557e59416a0fa49b276ea43fe8737`.
The model campaign used the same recorded runner identity. Hardware was the assigned NVIDIA
GB10, sm_121, driver 580.159.03; this establishes no device result on other GPU architectures.

Source base was `d33dcfc7` plus local research changes. `source-identities.json` records 20
selected kernel/owner/runner/gate source hashes observed after the completed run, with ten
matching earlier source cross-checks and final EXL3 artifact metadata. This is a selected
source observation of the frozen qualification tree, not a complete source tree embedded in
the runner. The [paired model receipt](../glm-kernel-paired-d33dcfc7/README.md) separately covers
token parity and performance. No campaign identity file was modified to create this receipt.

After cache qualification finished, the two-line comment in `contract_limits.zig` was shortened
to one line for the lean rule. `source-identities.json` retains the exact tested source hash;
`source-revisions.json` separately records the current hash and exact comment replacement.
No binary was rebuilt and no GPU test is claimed for the edited file bytes. The verifier
reverses only that recorded comment replacement and requires the result to hash to the tested
source, proving all bytes outside the comment span are unchanged. Any other source edit is rejected.

## Reproduction and privacy

Python 3.11 or newer, standard library only; these commands do not run GPU tests:

```sh
python3 reproduce.py
python3 reproduce.py --raw /path/to/tensorfold-paired-r2-post-contract.tgz
```

The first verifies retained hashes, complete summary parsing, every numerical red's declared
KNOWN/control classification, all research gates/controls, and the independent resource
classification. The second additionally checks all 14 raw manifest entries and reproduces the
derived files byte-for-byte. From the repository root, `--source-root .` also checks the 20
observed source files, reporting exact tested matches separately from the documented comment-only
revision. A successful receipt verifier does not change the actual GPU exits of 1.

Raw archive SHA256:
`c6f75fe174cb042aaf8c51f9e46bd18ef8f5297eb7b1d431ab3b5012080be9d4`.
Raw manifest SHA256:
`28ece4eef4b8cc4d73aa28a744e08ec7a07a3f50f196dbc969b96afe62c0d83f`.

The unchanged raw archive remains outside the repository. Reviewed logs contain generated
numerical fixtures, not model prompts/replies; private-address, home-path and credential scans
found no matches. Node/process/interface metadata and PIDs were excluded. `provenance.json`
pins raw inputs, derivation code and derived data. `privacy-review.json` records the review;
`SHA256SUMS` covers this receipt's retained files.
