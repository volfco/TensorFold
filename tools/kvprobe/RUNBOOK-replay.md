# KDA state replay on GB10: runbook

What this run measures: whether GLM-5.3-Flash's KDA recurrent state and short-conv history (the arena head, 55 % of a
kept entry on this checkpoint, session record section 11.3) can be rebuilt at restore by replaying the last W prompt
tokens over the stored attention rows instead of being kept. The engine side is `TF_GLM_KDA_REPLAY*`
(`zig/src/families/glm/cuda_kda_replay.zig`, status in `STATUS-replay.md`). Steps 2 to 4 ran on the ring on
2026-11-01 (session record section 11.10: the zero start does not rebuild the state). Steps 5 and 6 are the open
variant, the coarse start (`TF_GLM_KDA_REPLAY_INIT=fp8`, `zig/src/families/glm/cuda_kda_coarse.zig`), not yet run.

Placeholders as in `RUNBOOK.md`: `R0`/`R1` the GB10 hosts (rank 0 serves HTTP), `R0_LINK` rank 0's address on the
direct link, `MODEL` the checkpoint directory, `TF` this checkout copied with its working tree to both hosts (the
feature is uncommitted; copy the tree, do not `git pull`), `ROUTING` the receipts' routing report
(`docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/routing.md`).

The knobs (every rank the same; the ranks' settings fingerprint refuses a mismatch by name before the weights load):

| knob | values | effect |
|---|---|---|
| `TF_GLM_KDA_REPLAY` | 0 (off, default) to 2048 | W: the window in tokens; every keep records its last min(W, pos) prompt tokens |
| `TF_GLM_KDA_REPLAY_CHECK` | 0 or 1, needs W | a restore of a full entry also replays into the live head, compares, then puts the stored head back |
| `TF_GLM_KDA_REPLAY_KEEP` | 0 or 1, needs W | every keep whose tokens are known is kept without its arena head; its restore replays |
| `TF_GLM_KDA_REPLAY_INIT` | `zero` (default) or `fp8`; fp8 needs KEEP or CHECK and allows W = 0 | what a replay starts from: zero, or the entry's e4m3 copy of its KDA states and conv history (one power-of-two scale a row of 128) |
| `TF_GLM_KV_DUMP` | a directory | with CHECK: both heads dumped as `kda-replay-rank<r>-keep<id>-w<W>-{stored,replayed}.{bin,parts,plan}`; with fp8 three, `...-w<W>-fp8-{stored,coarse,replayed}` (no `replayed` at W = 0) |

## 0. Stop the production ring (by hand)

As `RUNBOOK.md` step 0. The operator stops and restores production; an agent must not.

## 1. Build (each host)

```bash
cd TF
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121          # zig-out/native/bin/tensorfold-native
zig build test-glm-snapshots-host compile-glm-host                  # the host checks STATUS-replay.md lists
```

No kernel changed: the replay pass is the stock prompt pass with the DSA latent and index writes skipped (two launches
not made). Optional regression with the knobs unset: the GPU check `RUNBOOK.md`'s build ran (`tf-cuda-test glm-exl3`)
and `kernel-contract-live`, expecting the October baseline's results.

## 2. Check mode at W = 64, 256, 512 (replies must not change)

One server pair per W (W is read at start and fingerprinted, so each W is a restart with identical flags on both
ranks). Dumps take about 0.15 GB per head per rank per restore (two heads a restore); give the dump directory room.

```bash
W=64   # then 256, then 512
mkdir -p /tmp/kvreplay/w$W
export TF_GLM_KDA_REPLAY=$W TF_GLM_KDA_REPLAY_CHECK=1 TF_GLM_KV_DUMP=/tmp/kvreplay/w$W TF_GLM_HOST_CACHE=1
# rank 1 (on R1), then rank 0 (on R0), as RUNBOOK.md step 2, logs kept:
TF_GLM_RANK=1 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 zig-out/native/bin/tensorfold-native serve MODEL \
  --context 32768 --parallel 1 --prompt-cache-gib 24 > /tmp/kvreplay/w$W/rank1.log 2>&1 &
TF_GLM_RANK=0 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 zig-out/native/bin/tensorfold-native serve MODEL \
  --context 32768 --parallel 1 --prompt-cache-gib 24 > /tmp/kvreplay/w$W/rank0.log 2>&1 &
```

`TF_GLM_KV_DUMP` alone writes the check dumps (the codec's own keep dumps additionally need `TF_GLM_KV_CODEC=1`, not
wanted here). Then send a chat body twice: the first prefills cold and keeps a state at the 2048-row mark at or before
the rendered history's end, the messages before the final user turn (a single long user turn has no history mark and
may keep nothing); the second finds it in the prompt cache, restores it, and so runs the check. The history must
exceed 2048 tokens (4096 to 8192 is a good size). Known-good bodies: the earlier probe's set (`probe-prompts.json`
from `kv_probe_send.py build`, e.g. a prose `A+S`), built to land a mark; greedy:

```bash
B=body.json   # {"model": ..., "messages": [...], "max_tokens": 64, "temperature": 0}
curl -s 127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d @$B > /tmp/kvreplay/w$W/first.json
curl -s 127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d @$B > /tmp/kvreplay/w$W/second.json
grep -h 'KVREPLAY total' /tmp/kvreplay/w$W/rank*.log
```

Expected per restore and rank: one `KVREPLAY total` line (heads, bit-identical heads, rec rel p50/p90/max, max abs,
conv identical heads and rel max, the replay's ms), one `KVREPLAY layer` line per KDA layer and one `KVREPLAY head`
line per head. Pass condition for the check mode itself: `second.json`'s reply equals `first.json`'s and equals a
server started without the knobs (same body; the check puts the stored head back before serving, so any difference is
a bug, report it). Repeat the second send two or three times for more restores (each writes its own pair of dumps,
keyed by the keep id; a pair from the same keep is overwritten).

Check mode allocates a device copy of the arena head (about 0.2 GB a rank) at each such restore: an allocation error
on the second send means the device had no room for it (lower `--context` or `--prompt-cache-gib`).

If no `KVREPLAY` line appears: first, the body's history held no 2048-row mark (see above); else the request did not
restore (the server's prompt cache line shows the cached
tokens; a body under 2048 tokens keeps nothing), or the keep's tokens were not all known (a keep after a decode commit
is kept whole without a window, by design; a fresh request's prompt pass always records them).

## 3. Offline comparison (CPU, either host)

```bash
cd TF/tools/kvprobe
for W in 64 256 512; do
  python3 replay_check.py --dir /tmp/kvreplay/w$W --horizons ROUTING --json /tmp/kvreplay/replay-w$W.json \
    > /tmp/kvreplay/replay-w$W.md
done
```

Each pair's table: per KDA layer (global layer id) bit-identical heads, the recurrent state's relative L2 over heads
(p50, p90, max) and max abs error, the conv history's identical heads and relative error, the layer's decay horizon
(p50/p90 from section E) and the direct-term bound (1 - 1/h)^W at those horizons. Rank 1's dumps hold the other half
of the heads (copy them over or run the tool on R1). What to bring back: the three `.md` and `.json` files, the
`KVREPLAY total` and `layer` lines of both ranks, and the reply comparison.

## 4. Replay form (the cold-tier trade), at the W step 3 favours

```bash
export TF_GLM_KDA_REPLAY=$W TF_GLM_KDA_REPLAY_KEEP=1 TF_GLM_HOST_CACHE=1   # no CHECK, no dump needed
```

Restart both ranks, send the same body twice. The keep now holds only the attention classes (the prompt cache's
charge for it drops by the arena head, about 0.2 GB a rank on this checkpoint; with `TF_GLM_KV_LEDGER=1
TF_GLM_KV_CODEC=1` its `KVLEDGER` line shows no prefix class), and the restore logs `KVREPLAY ... form=replay ms=`.
Compare the second reply against the knob-free server's (greedy, 64 or more tokens): the replay form is approximate by
construction, so the reply may differ; record the first differing token index and the share of equal tokens. That is
the quality evidence a cold-tier policy would need; it is not a default.

## 5. Coarse start, check mode at W = 0, 64, 256 (replies must not change)

Build as step 1 (the host checks now include `glm_snapshot_host_coarse.zig`; still no kernel changed). One server
pair per W, as step 2, with the fp8 init added; W = 0 is `TF_GLM_KDA_REPLAY=0` (the coarse copy alone, no replay):

```bash
W=0    # then 64, then 256
mkdir -p /tmp/kvreplay/fp8-w$W
export TF_GLM_KDA_REPLAY=$W TF_GLM_KDA_REPLAY_INIT=fp8 TF_GLM_KDA_REPLAY_CHECK=1 \
  TF_GLM_KV_DUMP=/tmp/kvreplay/fp8-w$W TF_GLM_HOST_CACHE=1
# rank 1 (on R1), then rank 0 (on R0), exactly as step 2 with logs in /tmp/kvreplay/fp8-w$W/rank{0,1}.log
```

The same body as step 2 (the prose A+S body of section 11.10, 8528 tokens, keep at 8192), sent twice, greedy:

```bash
curl -s 127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d @$B > /tmp/kvreplay/fp8-w$W/first.json
curl -s 127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d @$B > /tmp/kvreplay/fp8-w$W/second.json
grep -h 'KVREPLAY total' /tmp/kvreplay/fp8-w$W/rank*.log
```

Expected per restore and rank: two `KVREPLAY total` lines, `window=0 ... init=fp8` (the coarse copy alone) and, for
W > 0, `window=W ... init=fp8` (the copy plus the replay), each with its `layer` and `head` lines; three dumps (two at
W = 0), about 0.2 GB each a rank. Pass condition, as step 2: `second.json`'s reply equals `first.json`'s and the
knob-free server's (sha `95db6f07` for the 11.10 body); the check puts the stored head back before serving, so any
difference is a bug. The fp8 check also needs the pinned stage (about 22 MB a rank, made at the first fp8 restore).

Offline, as step 3:

```bash
cd TF/tools/kvprobe
for W in 0 64 256; do
  python3 replay_check.py --dir /tmp/kvreplay/fp8-w$W --horizons ROUTING --json /tmp/kvreplay/replay-fp8-w$W.json \
    > /tmp/kvreplay/replay-fp8-w$W.md
done
```

Each directory gives a `coarse copy alone` table (window 0) and, for W > 0, an `init fp8, window W` table. What to
bring back: the three `.md` and `.json` files, the `KVREPLAY total` and `layer` lines of both ranks, the reply
comparison. What they decide: the coarse copy's own error per layer on real states (the host bound is 1/16 a value),
and whether replaying W tokens on top of it lowers or raises that error against 11.10's zero start (0.87 / 0.73 /
0.67 median at W = 64 / 256 / 512).

## 6. Coarse start, replay form, at the W step 5 favours (W = 0 allowed)

```bash
export TF_GLM_KDA_REPLAY=$W TF_GLM_KDA_REPLAY_INIT=fp8 TF_GLM_KDA_REPLAY_KEEP=1 TF_GLM_HOST_CACHE=1   # no CHECK
```

Restart both ranks, send the same body twice. The keep holds the attention classes and the coarse copy (about 97 MiB
a rank against 272 MiB whole and 78 MiB with the zero start; with `TF_GLM_KV_LEDGER=1 TF_GLM_KV_CODEC=1` the
`KVLEDGER` line shows a prefix class of 19,675,424 bytes); the restore logs `KVREPLAY ... form=replay init=fp8 ms=`.
Record the restore time, the keep's added time (compare the prompt's keep timing with and without the knob), and the
second reply against the knob-free server's (greedy, 64 or more tokens): the first differing token index and the share
of equal tokens. Same pass condition as step 4: evidence for a cold-tier policy, not a default.

## 7. Restore production (by hand)

Stop both servers and restore the production ring as `RUNBOOK.md` step 5.
