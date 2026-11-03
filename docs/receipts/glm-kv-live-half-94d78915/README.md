# GLM-5.3-Flash on CUDA: the live-half keep (`TF_GLM_KV_LIVE_HALF=1`), served, receipt

Tested implementation: the binaries built from `94d78915…` (commit.txt has the commit and its tree). The receipt's
commit adds this folder and the session record's section 11.19 and changes no source.
Test date: 2026-11-02 (04:38:57 to 04:59:15 UTC, production stopped and restored). Platform: two GB10 (sm_121, 128 GB
shared memory) at TP=2 over a direct RoCE link, ranks pinned to CPUs 0-4,10-14, the codec's workers on 5-9.
Toolchain: Zig 0.17.0, nvcc 13.0 for sm_121; binaries.txt has the binary's sha256 (equal on both ranks).
Checkpoint: the GLM-5.3-Flash checkpoint of `tools/kvprobe/RUNBOOK.md` (`MODEL`).

## Scope

A kept GLM state carries both halves of the double-buffered KDA recurrent state (`rec`, fp32 [2][layers][heads][128][128],
142.6 MB a rank). The dead half is the state before the last prefill chunk or decode window: the forward reads
`recAt(cur)` and writes `recAt(1 - cur)`, every chunk and every committed window flips the bit, and a keep waits for
an empty window, so nothing reads the dead half after a restore (session record 11.18). With `TF_GLM_KV_LIVE_HALF=1`
the snapshot plan carries the live half alone (the arena head in two prefix parts around it), a restore builds its
plan from the kept state's own parity and leaves the slot's dead half as it was, the dump's manifest records the hole
(`half <h> <offset> <bytes>`), the settings fingerprint carries the flag, and the replay forms are refused with it.

This qualifies that change served at TP=2 with the lazy codec, the ledger and the dump on: the whole head (`both`,
the flag at 0) against the live half (`live`), the same binary, one growing greedy conversation of nine turns each
(the codec receipt's: six main turns, a raw-keep turn, a raw restore and a part-compacted restore), replies byte for
byte, the keeps byte for byte outside the dead half, the ledger's prefix class, the codec's totals and pages, time to
first token. It does not qualify the change with drafts on (commit's partial-accept replay path did not run), at
other TP sizes or under concurrent streams.

## Results

Replies: identical in every turn across the earlier plain off run (`docs/receipts/glm-kv-codec-lazy-2b61900a/records/off`,
another binary), `both` and `live` (records/compare.json).

| turn | kind | prompt | cached | TTFT s off / both / live | reply sha256 |
|---|---|---|---|---|---|
| 1 | main | 3102 | 0 | 5.77 / 6.40 / 6.16 | `ea582398230c` |
| 2 | main | 5322 | 2048 | 5.54 / 6.31 / 6.08 | `25d3cf4c723b` |
| 3 | main | 7544 | 4096 | 5.42 / 6.30 / 6.14 | `4bbd2f8df0b6` |
| 4 | main | 9775 | 6144 | 6.04 / 6.92 / 6.72 | `0dbcac0b9d7a` |
| 5 | main | 11999 | 8192 | 6.16 / 7.20 / 6.88 | `73e33772a797` |
| 6 | main | 14218 | 10240 | 6.15 / 7.35 / 6.96 | `1c313dc54dee` |
| 7 | raw-keep | 16440 | 12288 | 7.02 / 8.42 / 7.93 | `450d3bec5a69` |
| 8 | raw | 18661 | 16384 | 4.33 / 5.70 / 5.44 | `6d615174db62` |
| 9 | partial | 20881 | 18432 | 4.37 / 5.86 / 5.55 | `921fc4b0e3f2` |

`off` ran without the ledger and the dump (another day, another binary) and is the reply reference only; `both` and
`live` carry the instrumentation, whose cost scales with the bytes kept.

The keeps (records/compare-dumps-r0.json and -r1.json, `compare_half.py` over the two runs' dumps, nine keeps a rank):
every `live` keep is exactly 71,303,168 bytes smaller than its `both` twin (34 layers x 32 heads x 128 x 128 x 4),
every class outside the arena head is byte-identical, the head's bytes before and after the hole are identical, and
the live KDA states and the conv history read back identical through `kvdump.py`. In the `both` run the dead half
was all zero at keep 1 (position 2,048, one prefill chunk) and 0.4 % zero at keeps 2 to 9; the parity alternates
with the chunk count (2,048-row chunks).

The ledger's prefix class a keep, bytes and modeled bytes (rank 0; rank 1 within 0.02 MB):

| keep | position | both | live |
|---|---|---|---|
| 1 | 2048 | 163,130,624 / 83,482,124 | 91,827,456 / 71,506,955 |
| 2 to 9 | 4096 to 20480 | 163,130,624 / 134,608,157 to 135,013,936 | 91,827,456 / 71,580,266 to 71,761,842 |

The codec's totals (rank 0, lazy, after the nine turns; records/*/server-r0.log `KVCODEC ... keeps=`):

| config | keeps | restores (raw / partial) | raw bytes | stored bytes | ratio | pages raw -> kept | freed | lazy_encode_ms | decode_ms | copy_ms | lazy_copy_ms | keep_io_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both | 9 | 8 (2 / 0) | 2,002,762,752 | 1,377,548,400 | 0.6878 | 958 -> 660 | 298 | 1689.2 | 402.7 | 72.3 | 196.1 | 4069.5 |
| live | 9 | 8 (2 / 0) | 1,503,640,576 | 1,010,740,121 | 0.6722 | 720 -> 486 | 234 | 1307.4 | 320.1 | 57.2 | 142.2 | 3129.9 |

The raw bytes differ by 7 x 71,303,168 exactly (the seven compacted keeps); the stored bytes by 26.6 %; the pages
held after compaction by 26 % (held MiB at the end 2,228 against 1,734).

Reading: the dead half was the one prefix cost the codec could not remove (session record 11.18); leaving it out of
the plan removes 71 MB a rank from every keep's raw pages and about 63 MB from every stale keep's modeled bytes, with
the replies and every other kept byte unchanged. Time to first token is 0.2 to 0.5 s a turn lower in the
instrumented form, which the timers account for (the dump's write, the ledger's read and the restores' decode all
shrink with the bytes); without the instrumentation the prompt pass saves the copy and the decode only, tens of
milliseconds a turn. The flag stays off by default until the maintainer's gate.

## Reproduction

```
# on node0, production stopped by the script; both ranks run the same binary (binaries.txt)
bash validate-livehalf.sh            # records/both then records/live, compare.json, production restored
python3 -P compare_half.py DUMP_BOTH DUMP_LIVE OUT.json   # one rank's dump dirs (TF_GLM_KV_DUMP) keep by keep
```

Each configuration: `start-ranks2.sh "PROBE_CONTEXT=65536 TF_GLM_HOST_CACHE=1 TF_GLM_KV_LEDGER=1 TF_GLM_KV_DUMP=...
TF_GLM_KV_CODEC=1 TF_GLM_KV_CODEC_LAZY=1 TF_GLM_KV_CODEC_LAZY_MS=20000 TF_GLM_KV_CODEC_CPUS=5-9
TF_GLM_KV_LIVE_HALF=0|1" --no-drafts`, then `tools/kvprobe/lazy_receipt.py run --config <label> --lazy-ms 20000`,
`collect`, and `compare OFF BOTH LIVE`. The dumps (9 keeps a rank a configuration, 2.0 and 1.5 GB a rank) stay on
the nodes.

## Remaining coverage limits

- Drafts off: commit's partial-accept replay (`keep < pending`: reads `recAt(cur)` and the window scratch, writes
  `recAt(1 - cur)`) did not run; by code reading it needs no dead half either.
- One conversation, TP=2, no concurrent streams, the Zig keep format only (the serving stack's session store is not
  this code).
- The host test step `test-cuda-host` on the build node fails three unrelated steps whose generated inputs that
  node lacks (the grammar pin options, the CUDA contract image, the h3 fatbin); every test that compiled passed
  (4,152 run, 9 skipped; the log names no tests, so the ranks' fingerprint test is not identified in it), and a
  missing switch case in `cuda_ranks.zig` surfaced only in the native build, which compiles the file whole.

## Records

- `records/both/`, `records/live/`: `turns.json` (per-turn timings and reply hashes), `log-summary.json`, the two
  ranks' server logs.
- `compare.json`: `lazy_receipt.py compare` over off, both and live. `compare-dumps-r0.json`, `compare-dumps-r1.json`:
  the keep-by-keep dump comparison. `validate-livehalf.log`: the run's console. `validate-livehalf.sh`,
  `compare_half.py`: the scripts as run.
- Server logs are redacted of the home directory (redaction-report.json). Token ids and texts are not in the
  records; the dumps stay on the nodes.

## Publication handling

Private test on the team's two nodes; the records hold timings, hashes and byte counts only.
