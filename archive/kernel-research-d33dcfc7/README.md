# Archived: GLM/EXL3 kernel experiments (baseline d33dcfc7)

Archived 2026-10-31. Three opt-in CUDA experiments that passed their numerical gates and token
parity but showed no served speed gain on two GB10s at TP=2. Nothing here is built or imported.

| Experiment | Switch | Operator result | Served result |
| --- | --- | --- | --- |
| KDA vector staging (prompt rows >= 256) | `TF_GLM_KDA_VECTOR` | 1.0-2.2% faster at 21/22/32 heads, 5.8% at 64 | Cold TTFT -0.1% to +1.1% (8k slower, outside control range) |
| Small-row N256 expert down tile | `TF_EXL3_N256` | 2-3% faster; full N256 slower everywhere | Decode -2.3% to +1.1% |
| Fused EXL3 decode epilogues | `TF_EXL3_FUSED_EPILOGUES` | 3-7% slower | Decode -0.05% to -1.3% |

Upstream context: ashhart/TensorFold#1637 (KDA prefill tuning, closed not planned).
WY chunking was ruled out: it changes FP32 state bytes and needs its own arithmetic gate.

## Layout

- `tracked-changes.patch` — edits to tracked files, taken against origin/main after the
  2026-10-31 rebase (fingerprint layout 14, after upstream's `roce_multi` bump to 13). It also
  contains the `kernel-contract-live` subcommand hunk in `zig/tests/cuda/main.zig`, which was kept in-tree.
- `files/` — the new source, test and probe files at their original repo paths.
- `receipts/` — all six receipt directories, including the served paired run, the full
  contract run and cache/resume checks.

## Restoring

```sh
A=archive/kernel-research-d33dcfc7
cp -r $A/files/. .
git apply --3way $A/tracked-changes.patch   # main.zig's kernel-contract-live hunk will already be present
```

The receipts are pinned to d33dcfc7 binaries and do not qualify a rebased build.
