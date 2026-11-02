# Runbook: routing, DSA-selection and KDA-gate dump on the serving stack (patch 0710)

Status: UNTESTED ON A GPU. What was checked off the GPU: the patch applies cleanly (`git apply --check` and
`patch -p1 --dry-run`) both to the tree inside the running image and to a fresh replay of the stack's patches
0005..0700 over its vendored TensorFold; the three changed modules byte-compile; `routetap.py`'s wrappers pass a CPU
harness (`routetap_harness.py`: fake `forward` / `sparse` / `lean` modules, real `routetap.py`; checks recorded ids
equal the buffers, the MTP layer and lean chunks are not recorded, selection rows follow the stride and sparse rule,
lean prefill is refused by name). Every step below that restarts or rebuilds is the user's to run by hand.

Files: `0710-glm-route-dump.patch` (adds `cuda/routetap.py`, hooks in `cuda/dump.py` and `cuda/decode.py`), this
runbook, `routetap_harness.py`. Analysis: `tools/kvprobe/routedump.py` (loader), `tools/kvprobe/routing.py`
(report), `tools/kvprobe/test_routing.py` (synthetic tests).

## What it records

With `GLM53_TF_DRAFT_DUMP` on and any 0710 knob, each prefill segment of the 0430 dump also carries (rank 0 writes):

| array | dtype, shape | knob |
|---|---|---|
| `route_ids`, `route_w` | int16 / float32 [n, 42, 8], record field `route_layers` | `GLM53_TF_DRAFT_DUMP_ROUTE=1` |
| `dsa_sel`, `dsa_pos`, `dsa_layer` | int32 [N, 512] (pool ids), int32 [N], int16 [N]; `dsa_stride` | `GLM53_TF_DRAFT_DUMP_DSA_SEL=1`, `_DSA_STRIDE` (16), `_DSA_LAYERS` (all) |
| `kda_lg` | float32 [n, 34, 32] (rank 0's heads), `kda_layers`, `kda_heads` | `GLM53_TF_DRAFT_DUMP_KDA_GATES=1` |

`kda_lg` is the per-head mean over 128 channels of log g = lower * sigmoid(exp(A_log) * (a + dt_bias)), recomputed
in torch from the gate rows the chain read (the formula in `kda.cu`'s header, not the kernel's bits). Selections are
recorded only for rows that attend sparsely (position past 2050) and whose position is a multiple of the stride.
Per prefill row: about 2 KB routing, 4.3 KB gates, 22.5 KB / stride selections (all 11 DSA layers).

Coverage and refusals: the exact and fast prefill chunks only (`forward.layer_forward`). At start the engine refuses,
by name, `GLM53_TF_LEAN_PREFILL=1` and `GLM53_TF_MULTI_PREFILL=1` with the 0710 knobs on (their layer loops are not
hooked; their bits equal the exact path's by their own contracts), a 0710 knob without `GLM53_TF_DRAFT_DUMP`, and
`GLM53_TF_DRAFT_DUMP_WHAT` without `prefill`. Both ranks must have the same knobs (the dump's start check includes
them). A recording failure prints one line and stops the 0710 records; serving continues.

## Steps (rank-0 node, in the serving project's checkout; run by the user)

1. Build an image with the patch under its own tag, from a copy of the project so the production tree is untouched:

   ```
   cp -a <serving project> ~/kvprobe/route/stack
   cp ~/kvprobe/patches/0710-glm-route-dump.patch ~/kvprobe/route/stack/patches/
   cd ~/kvprobe/route/stack && IMAGE=glm53-tensorfold:route scripts/serve.sh build
   ```

   (`build` also ships the image to the worker with `docker save | docker load`; `serve.sh` notes that page cache left
   by such a file operation can upset the engine's fit check, so let `MEM_GATE_GIB` or a wait settle it before step 3.)

2. Probe config: copy `config/kvprobe.env` to `config/kvprobe-route.env` in the copy and change:

   ```
   GLM53_TF_LEAN_PREFILL=0
   GLM53_TF_PREFILL_OVERLAP=0
   GLM53_TF_PREFILL_PP=0
   GLM53_TF_MULTI_PREFILL=0
   GLM53_TF_DRAFT_DUMP=/sessions/kvprobe-route-<date>
   GLM53_TF_DRAFT_DUMP_TAPS=0                 # taps are not needed for routing; keeps 8 KB a row
   GLM53_TF_DRAFT_DUMP_TAP_LAYERS=            # (unset: the drafter's; the probe config has no drafter)
   GLM53_TF_DRAFT_DUMP_WHAT=prefill
   GLM53_TF_DRAFT_DUMP_TOPK=1
   GLM53_TF_DRAFT_DUMP_ROUTE=1
   GLM53_TF_DRAFT_DUMP_DSA_SEL=1
   GLM53_TF_DRAFT_DUMP_DSA_STRIDE=16
   GLM53_TF_DRAFT_DUMP_KDA_GATES=1
   ```

   Lean off makes fast chunks use the main buffers (at most `GLM53_TF_PREFILL_ROWS_MAX` rows); if the start's fit
   check refuses, lower `GLM53_TF_PREFILL_ROWS_MAX` to 2048.

3. Stop production and start the probe (the production ring is down from here until step 6):

   ```
   cd ~/kvprobe/route/stack
   scripts/serve.sh stop
   IMAGE=glm53-tensorfold:route CONFIG=config/kvprobe-route.env scripts/serve.sh start
   scripts/serve.sh logs 0 | grep -m3 "draft dump"      # expect the 0430 line; no "patches/0710 ... failed"
   ```

4. Send the prompts (repository tools; any prompt set in `make_prompts.py`'s format works, including the content-type
   sets): `python3 tools/kvprobe/send_probe.py prompts.json /glm53-tf/sessions/kvprobe-route-<date>/<host>-<pid>-<t>/index.jsonl`.

5. Analyse on the CPU, detached (the dump directory is root-owned but world-readable):

   ```
   setsid nohup python3 -u tools/kvprobe/routing.py <dump dir> prompts.json --out ~/kvprobe/route/routing.md \
       > ~/kvprobe/route/routing.log 2>&1 &
   ```

   `--preset exl3-4bpw-rank` (default, the ring's checkpoint: 6,291,456 B an expert a rank) or `nvfp4-rank`
   (7,077,888 B), or `--expert-bytes N`.

6. Restore production by hand, from the production checkout (not the copy): `scripts/serve.sh restart` with no
   `CONFIG` or `IMAGE` set, then `scripts/serve.sh status`, and compare the container environment with the saved one.

## First-run checks before trusting a number

- `kda_lg` must be <= 0 everywhere (`routing.py` section E prints the share of positive values; expected 0). A
  positive share means the gate formula or `dt_bias`'s shape differs from what `routetap.py` assumes.
- `route_w` rows should sum to the routed scale (norm_topk on): a quick `np.allclose(w.sum(-1), w.sum(-1)[0, 0])`.
- Replies with the knobs on must be byte-identical to replies with them off (the hooks read only): send one prompt
  twice across a restart with the knobs off and on and compare the text and the 0430 `ids`.
- The rows per request in `index.jsonl` must equal the prompt lengths (`send_probe.py` checks this).
