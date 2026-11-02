# Stored-state probe on GB10: runbook

What this run measures: the quantities the Zig engine keeps (MLA latent rows per DSA layer, DSA index ring keys and
pooled keys, the KDA recurrent state per head, the short-conv history) for the same chunk S after no prefix, prefix
A, prefix B and a long prefix L, for four content types (prose, code, JSON tool output, chat-template boilerplate).
It is the follow-up of `docs/kv-cross-context-probe-2026-11-01.md` section 8 (H13). Nothing here has run on a GPU yet.

Placeholders: `R0` and `R1` are the two GB10 hosts (rank 0 serves HTTP), `R0_LINK` rank 0's address on the direct
link between them, `MODEL` the checkpoint directory (see "Checkpoint" below), `TF` the TensorFold checkout on each
host at the same commit and with the same working tree (the probe needs this tree's uncommitted codec and manifest
changes; copy the tree, do not `git pull`).

## 0. Stop the production ring (by hand)

The production ring holds about 114 of 121 GB on each host; the Zig engine cannot load beside it. Stop it and restore
it yourself; an agent must not (the serving stack's own commands, e.g. `scripts/serve.sh stop` and later
`scripts/serve.sh restart` with no `CONFIG`, then `scripts/serve.sh status` and the container environment compared
with the saved production one). Check `free -g` shows the memory back on both hosts before step 2.

## 1. Build (each host, aarch64, CUDA 13 nvcc at /usr/local/cuda)

```bash
# Zig 0.17.0 for aarch64 (the version in .zig-version), outside the checkout
curl -LO https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz   # check its sha256 against index.json
tar -xJf zig-aarch64-linux-0.17.0.tar.xz && export PATH=$PWD/zig-aarch64-linux-0.17.0:$PATH
cd TF
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121            # zig-out/native/bin/tensorfold-native
zig build kv-codec-bench                                             # zig-out/bin/tf-kv-codec-bench
zig build test-kv-codec test-glm-snapshots-host                      # the host checks this tree passed on x86
```

## 2. Serve with kept-state dumps on (both ranks, same arguments)

Keeps are written by the prompt cache at the rendered history's mark (the chunk start at or before it, GLM's
2048-row grid). `TF_GLM_KV_DUMP` only writes when the codec is on, and the codec needs the host cache:

```bash
mkdir -p /tmp/kvdump && export TF_GLM_HOST_CACHE=1 TF_GLM_KV_CODEC=1 TF_GLM_KV_LEDGER=1 TF_GLM_KV_DUMP=/tmp/kvdump
# rank 1 (on R1)
TF_GLM_RANK=1 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 zig-out/native/bin/tensorfold-native serve MODEL \
  --context 32768 --parallel 1 --prompt-cache-gib 24
# rank 0 (on R0)
TF_GLM_RANK=0 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 zig-out/native/bin/tensorfold-native serve MODEL \
  --context 32768 --parallel 1 --prompt-cache-gib 24
```

Set `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` for the direct link as for production. Each keep writes
`/tmp/kvdump/glm-rank<r>-keep<id>.{bin,parts,plan}` on its own host: rank 0 holds every latent and index row (they are
replicated) and its half of the KDA heads; rank 1's files hold the other half of the heads. A keep of the longest
probe prompt (about 12.5k tokens) is roughly 12.5k x 13 KB of rows plus about 0.2 GB of arena head; 16 prompts fit in
a few GB of disk. Check the rank-0 log for `KVLEDGER` lines (one per keep) and `KVCODEC` at exit.

## 3. Build the prompts and send them (from R0, the server on 127.0.0.1:8000)

```bash
cd TF/tools/kvprobe
python3 kv_probe_send.py build TF probe-prompts.json            # 16 chat bodies: 4 types x S, A+S, B+S, L+S
python3 kv_probe_send.py send probe-prompts.json /tmp/kvdump probe.json --rank1-dir /mnt/r1-kvdump
```

`build` tokenizes through the server's `/tokenize` and trims the prefixes until S starts at the same offset modulo
2048 in every context, so the keeps sit the same distance past S. `send` posts each body (`max_tokens` 1, greedy),
waits for its keep files and refuses with a named reason when rank 0 kept nothing. `--rank1-dir` is optional: mount
or copy R1's `/tmp/kvdump` there afterwards and rerun `send`'s bookkeeping by hand (edit `probe.json`'s `keeps` with
the rank-1 stems in the same order), or analyze rank 0 alone. A repeated run must restart the server (or change the
nonce, which `build` does): a prompt the cache already holds is not prefilled or kept again.

## 4. Analyze (CPU; long runs detached)

```bash
setsid nohup python3 -u kvstate.py probe.json stored-r0 --rank 0 --bench TF/zig-out/bin/tf-kv-codec-bench > kvstate-r0.log 2>&1 &
setsid nohup python3 -u kvstate.py probe.json stored-r1 --rank 1 > kvstate-r1.log 2>&1 &
```

Outputs `stored-r<r>.json` and `.md`: per content type and pair (A+S vs B+S, S vs A+S, L+S vs A+S) the latent rows'
relative L2, cosine, byte-plane entropy (raw, XOR, int16 delta) per DSA layer, the ring and pooled index keys' relative
L2, and per KDA layer the state's per-head deviation, effective rank, rank for 90 % energy, and the difference's
effective rank and energy in rank 8 and 32; `--bench` adds each keep's `tf-kv-codec-bench --file` lines (plan Phase 1
acceptance: at least three captured shapes). Do not pipe the logs through `grep | tee`.

## 5. Restore production (by hand)

Stop both Zig servers, then restore the production ring as in step 0 and compare its environment with the saved one.

## Checkpoint

The recipe (`docs/recipes/glm-5.3-flash-cuda.md`) serves NVIDIA's NVFP4 checkpoint or the EXL3/TR3 checkpoint. On the
hosts' shared hub only EXL3 checkpoints were present when this was written (observed with `ls`): the ring's own
(the hidden-tap probe's checkpoint), a TensorFold-converted EXL3 derivative, and a third-party EXL3 conversion; the
NVFP4 checkpoint was not. Which one to use is the user's decision: the ring's checkpoint matches the hidden-tap numbers
if the Zig loader accepts it (untested); the NVFP4 checkpoint is the recipe's, needs `tensorfold pull` and disk.

## Serving-stack variant (no Zig engine)

The routing, DSA-selection and KDA-gate dump for the Python serving stack is a separate patch with its own runbook:
`tools/kvprobe/serving/` (copies of the files placed under `~/kvprobe/patches/` on rank 0's host).
