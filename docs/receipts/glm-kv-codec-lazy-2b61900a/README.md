# GLM-5.3-Flash on CUDA: the KV codec's lazy compaction, served, receipt

Tested implementation: the binaries built from `2b61900a…` (commit.txt has the commit and its tree). The receipt's
commit adds this folder and changes no source.
Test date: 2026-11-02. Platform: two GB10 (sm_121, 128 GB shared memory) at TP=2 over a direct RoCE link, ranks pinned to CPUs 0-4,10-14.
Toolchain: Zig 0.17.0, nvcc for sm_121; binaries.txt has every binary's sha256.
Checkpoint: the GLM-5.3-Flash checkpoint of `tools/kvprobe/RUNBOOK.md` (`MODEL`).

## Scope

This qualifies the lossless KV codec on GLM's kept prompt states with the host cache on, served at TP=2: the codec
off, eager (compaction inside the keep, on the prompt pass) and lazy (`TF_GLM_KV_CODEC_LAZY=1`: the keep returns raw
and the workers compact it after the delay), one growing greedy conversation each: replies byte for byte across the
three, time to first token per turn, where the encode runs and what it costs, pages given back, and restores from
compacted, raw and part-compacted entries.
It does not qualify the codec at other TP sizes, under concurrent streams or with drafts on.

## What changed for these runs

- The codec's timers mean the same in both modes (`tools/kvprobe/STATUS-lazy.md` "Timers"): `encode_ms` and
  `lazy_encode_ms` count the workers' encode waves only, `decode_ms` the decode waves only; the ledger's read and the
  dump's writes are `keep_io_ms`, the engine thread's page copies and staging `copy_ms`, the compactor's write-back
  `lazy_copy_ms`, and the engine thread's waits for the workers or a compaction `wait_ms`. The eager and lazy totals
  of session record 11.11 (5x apart both ways for the same bytes) had timed the same work; the eager run's workers
  were unpinned and shared the engine's five CPUs, the lazy run's had five of their own. Here eager and lazy both
  pin the workers to the same CPUs (`TF_GLM_KV_CODEC_CPUS`).
- `tools/kvprobe/lazy_receipt.py` (the conversation, its raw and part-compacted cases, `collect`, `compare`),
  `recall_ab.py` (the Recall A/B) and `receipt.py` (this folder).

## Results

| Check | Result |
| --- | --- |
| Replies, off / eager / lazy (`records/compare.json`) | identical in all 9 turns (token sha256) |
| Codec timers, off (`records/off/log-summary.json`) | no KVCODEC totals (off: the codec is off, or the ranks were not stopped with SIGINT) |
| Codec timers, eager (`records/eager/log-summary.json`) | rank 0: `encode_ms=2353.4` on the prompt pass, `lazy_encode_ms=0.0` on the workers, `decode_ms=636.5`, `copy_ms=650.6`, `lazy_copy_ms=0.0`, `keep_io_ms=4170.9`, `wait_ms=0.0`; keeps 9, compacted 9, ratio 0.692, pages 1358 -> 942, threads 5; rank 1: no totals line |
| Codec timers, lazy (`records/lazy/log-summary.json`) | rank 0: `encode_ms=0.0` on the prompt pass, `lazy_encode_ms=1696.0` on the workers, `decode_ms=417.3`, `copy_ms=78.1`, `lazy_copy_ms=201.5`, `keep_io_ms=4482.5`, `wait_ms=0.0`; keeps 9, compacted 7, ratio 0.6878, pages 958 -> 660, threads 5; rank 1: no totals line |
| Pages, off | rank 0: 0 trims, 0 pages freed, last `free=None`; the store's last line 2716 MiB held, 12684 MiB spare; rank 1: 0 trims, 0 pages freed, last `free=None`; the store's last line None MiB held, None MiB spare |
| Pages, eager | rank 0: 0 trims, 0 pages freed, last `free=None`; the store's last line 1884 MiB held, 13524 MiB spare; rank 1: 0 trims, 0 pages freed, last `free=None`; the store's last line None MiB held, None MiB spare |
| Pages, lazy | rank 0: 7 trims, 298 pages freed, last `free=6660`; the store's last line 2228 MiB held, 13212 MiB spare; rank 1: 7 trims, 298 pages freed, last `free=6660`; the store's last line None MiB held, None MiB spare |
| Restore forms, lazy | totals: restores 8, raw 2, part compacted 0; the extra cases as the log shows them before each restore: raw case raw-or-partial, partial case raw-or-partial |
| `KVCODEC refused` or `BadSnapshotControl` | 0 lines |
| Codec timers, eager-unpinned (`records/eager-unpinned/log-summary.json`) | rank 0: `encode_ms=14280.1` on the prompt pass, `lazy_encode_ms=0.0` on the workers, `decode_ms=3485.4`, `copy_ms=680.7`, `lazy_copy_ms=0.0`, `keep_io_ms=4152.8`, `wait_ms=0.0`; keeps 9, compacted 9, ratio 0.692, pages 1358 -> 942, threads 5; rank 1: no totals line |

Time to first token per turn (the first streamed chunk carrying text; `records/<config>/turns.json`):

| Turn | Kind | Prompt | Cached off / eager / lazy | TTFT s off / eager / lazy | Total s off / eager / lazy | Same |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | main | 3102 | 0 / 0 / 0 | 5.77 / 6.70 / 6.41 | 11.61 / 12.70 / 12.19 | yes |
| 2 | main | 5322 | 2048 / 2048 / 2048 | 5.54 / 6.75 / 6.29 | 11.37 / 12.61 / 12.02 | yes |
| 3 | main | 7544 | 4096 / 4096 / 4096 | 5.42 / 6.90 / 6.54 | 11.31 / 12.84 / 12.46 | yes |
| 4 | main | 9775 | 6144 / 6144 / 6144 | 6.04 / 7.54 / 6.96 | 11.82 / 13.53 / 12.78 | yes |
| 5 | main | 11999 | 8192 / 8192 / 8192 | 6.16 / 7.86 / 7.14 | 12.11 / 13.81 / 13.07 | yes |
| 6 | main | 14218 | 10240 / 10240 / 10240 | 6.15 / 7.95 / 7.22 | 12.04 / 13.99 / 13.14 | yes |
| 7 | raw-keep | 16440 | 12288 / 12288 / 12288 | 7.02 / 9.07 / 8.38 | 12.91 / 14.95 / 14.07 | yes |
| 8 | raw | 18661 | 16384 / 16384 / 16384 | 4.33 / 6.65 / 5.85 | 10.27 / 12.61 / 11.48 | yes |
| 9 | partial | 20881 | 18432 / 18432 / 18432 | 4.37 / 6.82 / 5.87 | 10.30 / 12.75 / 11.63 | yes |

Further runs (`records/<label>/turns.json`), TTFT a turn in seconds:

| Turn | eager-unpinned |
| --- | --- |
| 1 | 7.96 |
| 2 | 8.85 |
| 3 | 9.33 |
| 4 | 10.26 |
| 5 | 10.72 |
| 6 | 11.26 |
| 7 | 12.91 |
| 8 | 10.70 |
| 9 | 11.09 |

## Notes on these runs (the lead, 2026-11-02)

- The four configurations were served in two script runs on the same binaries and settings: `eager-unpinned` and
  `lazy` first, then a production restore, then `off` and `eager`, each configuration a fresh server; the runner's
  first version asked `/tokenize` for an empty history on turn 1, which the server refuses, and its fix
  (`tools/kvprobe` commit 6e81bafeb) is later than the tested binaries' commit and changes no engine source.
- `records/eager-unpinned/` is eager with `TF_GLM_KV_CODEC_CPUS` unset (the workers float over the engine's cores):
  `encode_ms=14280.1` and `decode_ms=3485.4` against the pinned eager run's 2353.4 and 636.5 for the same bytes, a
  6x gap in both directions. That is the explanation of session record 11.11's "5x gap": where the workers run, not
  what the timers count. Pinned eager and lazy encode at the same rate (0.26 s a keep, 0.24 s a compaction).
- The time to first token of `eager` and `lazy` above `off` (0.6 to 2.5 s a turn, growing with the kept state) is
  not the codec alone: `off` wrote no ledger and no dump (its `OUT/dump-off` is empty; `TF_GLM_KV_LEDGER` and
  `TF_GLM_KV_DUMP` act only with the codec on), while the codec runs wrote 2.7 GB of dumps each and the ledger's
  per-class statistics (`keep_io_ms` 4170.9 and 4482.5 over nine keeps, about 0.5 s a keep). The codec's own time on
  the prompt pass in `lazy` is the restore's decode (417 ms over eight restores) and copies (78 + 201 ms). The
  remainder, about half of the delta, is outside the codec's timers; a rerun of `off` and `lazy` without ledger or
  dump (the recipe's production form) is the measurement that isolates it, and is queued as a further run.
- Rank 1 printed no `KVCODEC` totals in any configuration: the ranks were stopped with SIGTERM (`stop-ranks.sh`),
  which drains and tears rank 0 down but leaves rank 1's totals unprinted; rank 0's totals are the ones compared.

## Reproduction

Placeholders as in `tools/kvprobe/RUNBOOK.md`: `R0` and `R1` the two hosts (rank 0 serves HTTP), `R0_LINK` rank 0's
address on the direct link, `MODEL` the checkpoint, `TF` this commit's tree on both hosts, `OUT` a results directory
on R0. Each configuration is served alone (`tools/kvprobe/RUNBOOK-lazy.md` steps 1 and 2), both ranks under
`taskset -c 0-4,10-14`, `serve MODEL --context 32768 --parallel 1 --prompt-cache-gib 16 --no-drafts`,
`TF_GLM_HOST_CACHE=1 TF_GLM_KV_LEDGER=1 TF_GLM_KV_DUMP=OUT/dump-<cfg>` on both ranks, and:

| config | extra environment (both ranks) |
| --- | --- |
| `off` | `TF_GLM_KV_CODEC=0` |
| `eager` | `TF_GLM_KV_CODEC=1 TF_GLM_KV_CODEC_CPUS=WORKERS` |
| `lazy` | `TF_GLM_KV_CODEC=1 TF_GLM_KV_CODEC_LAZY=1 TF_GLM_KV_CODEC_LAZY_MS=20000 TF_GLM_KV_CODEC_CPUS=WORKERS` |

Per configuration, from R0 (the server on 127.0.0.1:18080):

```bash
python3 tools/kvprobe/lazy_receipt.py run --base http://127.0.0.1:18080 --config <cfg> --out OUT/<cfg> --log OUT/<cfg>-r0.log --lazy-ms 20000 --pause 30.0 --max-tokens 48 --first-tokens 3072 --piece-tokens 2200 --partial-frac 0.4 --source docs
# stop rank 0 with SIGINT (both ranks tear down and print their KVCODEC totals), then
python3 tools/kvprobe/lazy_receipt.py collect OUT/<cfg> --log OUT/<cfg>-r0.log --log OUT/<cfg>-r1.log
```

Then `lazy_receipt.py compare OUT/off OUT/eager OUT/lazy --json OUT/compare.json`. The Recall A/B: two serves of the
same build, `--recall` on a fresh record and `--recall-read-only` on an empty one, each replayed with
`recall_ab.py run --base http://127.0.0.1:18080 --label <recall|read-only> --store SESSIONS --out OUT/<label>
--log OUT/<label>-r0.log`, compared with `recall_ab.py compare OUT/recall OUT/read-only`. This folder:
`receipt.py --off OUT/off --eager OUT/eager --lazy OUT/lazy [--recall OUT/recall --read-only OUT/read-only]
--commit <sha> --binaries binaries.txt`.

## Remaining coverage limits

- No part-compacted restore was served: the partial case's restore fell before or after the compaction (the host tests in `zig/glm_snapshot_host_lazy.zig` cover that form).
- One conversation of 9 turns a configuration, served once each; TTFT differences of a few percent are within one serve's variation (CONTRIBUTING.md asks for alternating runs to separate them).
- Two ranks only: the page agreement under a rank-skewed schedule and the extra gate exchange's cost are not measured.
- The keep's raw copy into host pages (`pool.move` and the stream's synchronize in `cuda_snapshots.keepAs`) is outside the codec and in none of its timers; it is the same in every configuration with the host cache on.
- The served Recall A/B was not run with this receipt.

## Records

`records/<config>/turns.json` (each turn's timings, tokens, hashes and the log lines it caused),
`records/<config>/log-summary.json` and `server-r<rank>.log` (the server logs), `records/compare.json` and
`compare.md`;
`commit.txt`, `binaries.txt`, `redaction-report.json`, `SHA256SUMS`.

## Publication handling

Machine names, addresses, network interfaces and home and local paths replaced with placeholders
(`redaction-report.json` counts them a file); result values and token hashes kept. The conversation's text is not
included (`lazy_receipt.py run` rebuilds it from the tree's `docs/*.md` or its fixed synthetic prose), and the
Recall records hold counts and hashes only. SHA256SUMS pins every published file except itself.
