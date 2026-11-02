# Routing worker status (Track 1 item 3, Track 4 item 1)

Contract.
- Behaviour: serving-stack patch 0710 (next free number after the stack's 0700) extends the 0430 draft dump with,
  per committed prefill row: the router's top-8 expert ids and weights at every MoE layer, the DSA indexer's selected
  pools at chosen DSA layers and strided rows, and per-head mean log forget gates of the KDA layers. Off by default;
  with the knobs off the hooks are not called (one `DUMP is not None` test per chunk already present from 0430).
  Offline: `routedump.py` loads a dump, `routing.py` reports.
- Owning files: `tools/kvprobe/serving/0710-glm-route-dump.patch`, `serving/RUNBOOK-route-dump.md`,
  `serving/routetap_harness.py`, `tools/kvprobe/routedump.py`, `routing.py`, `test_routing.py`. Identical copies of
  the patch, runbook and harness are in the rank-0 node's `~/kvprobe/patches/` (patch sha256
  badb535035de15f7dd177b023cacdd111a7c647d3ebedcc54189cb3f784d9745).
- Cannot run here: the patch on a GPU (image rebuild plus a run with the production ring stopped, by the user).

## What was checked, with commands

| check | command | result |
|---|---|---|
| synthetic analyses and loader | `python3 tools/kvprobe/test_routing.py` | 10 tests OK |
| patch applies to the running image's tree | `git apply --check --verbose` and `patch -p1 --dry-run` on a copy | clean |
| patch applies after replaying the stack's patches 0005..0700 on its vendored TensorFold (Dockerfile order) | loop of `git apply \|\| patch -p1 --forward`, then `git apply --check` | clean |
| changed modules byte-compile | `python3 -m py_compile dump.py decode.py routetap.py` | ok |
| wrappers record what the buffers hold, skip MTP and lean chunks, follow the selection stride and sparse rule, refuse lean prefill by name | `CUDA_VISIBLE_DEVICES= python3 serving/routetap_harness.py <tree>/src/tensorfold/families/glm5_next/cuda/routetap.py` (torch 2.12 CPU, rank-0 node) | all checks passed |
| loader reads the real 2026-11-01 0430 dump (no 0710 arrays) | `routedump.load` + `routing.name_requests` on it | 18,432 rows, 4 requests, all four prompts matched |

## Findings while building it [O]

- The ring serves EXL3 4-bit routed experts (`quant_method exl3`, 4 bits), not NVFP4; per-rank expert bytes used:
  3 x 4096 x 2048 x 4 bit / 8 / 2 = 6,291,456 B (preset `exl3-4bpw-rank`), consistent with the stack's own
  0.264 GB per expert across 42 layers a rank [C]; `nvfp4-rank` = 7,077,888 B for the Zig engine's checkpoint [C].
- Production prefill runs lean + overlap + pipelined + multi-slot (`GLM53_TF_LEAN_PREFILL=1`, `_PREFILL_OVERLAP=1`,
  `_PREFILL_PP=1`, `_MULTI_PREFILL=1`), whose layer loops the patch does not hook; it refuses them by name, and the
  runbook's probe config turns them off (equal bits by their contracts, untested here).
- DSA selection in this stack is 512 pools of 4 tokens (2048 tokens) per sparse row; rows at positions up to 2050
  attend densely and have no selection.
- The fast KDA paths do not fill the chain's replay scratch, so the gate statistic is recomputed from the gate rows
  with `kda.cu`'s documented formula (not the kernel's bits).
- In the probe design the shared chunk repeats across requests, so a request-based train/test split leaks; the
  report trains outside every S and tests on A+S's S, and prints the share of held-out rows repeating a training
  8-gram.

## Untested / awaiting the GPU

- Everything in the patch on a real engine: hook order with `hc_cuda`, `dv2.route`'s buffers, the latent DSA path's
  selection call, byte-identical replies with knobs on vs off, `kda_lg <= 0`. The runbook's "first-run checks" list
  these.
- No real routing data exists yet, so sections A to E have numbers only on synthetic input.

## User commands

See `serving/RUNBOOK-route-dump.md` (build under a separate image tag from a copy, probe config, stop production,
start probe, send prompts, analyse detached, restore production by hand).

## Open decisions

- Whether to also hook the lean / pipelined / multi-slot prefill loops (needed only to record production traffic as
  served rather than a probe with those modes off).
- KDA gates cover rank 0's heads only (no gather added); the other half needs rank 1 to write or an all-gather.
- Decode-row routing (the quantity expert prefetch would act on in decode) is not recorded; prefill rows stand in for
  it [H]. Adding it means hooking `batch.py`'s round, which imports `moe_block` by name.
