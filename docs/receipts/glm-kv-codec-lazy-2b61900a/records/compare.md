| turn | kind | prompt | cached off/eager/lazy | TTFT s off / eager / lazy | reply sha256 | same |
|---|---|---|---|---|---|---|
| 1 | main | 3102 | 0 / 0 / 0 | 5.77 / 6.70 / 6.41 | `ea582398230c` | yes |
| 2 | main | 5322 | 2048 / 2048 / 2048 | 5.54 / 6.75 / 6.29 | `25d3cf4c723b` | yes |
| 3 | main | 7544 | 4096 / 4096 / 4096 | 5.42 / 6.90 / 6.54 | `4bbd2f8df0b6` | yes |
| 4 | main | 9775 | 6144 / 6144 / 6144 | 6.04 / 7.54 / 6.96 | `0dbcac0b9d7a` | yes |
| 5 | main | 11999 | 8192 / 8192 / 8192 | 6.16 / 7.86 / 7.14 | `73e33772a797` | yes |
| 6 | main | 14218 | 10240 / 10240 / 10240 | 6.15 / 7.95 / 7.22 | `1c313dc54dee` | yes |
| 7 | raw-keep | 16440 | 12288 / 12288 / 12288 | 7.02 / 9.07 / 8.38 | `450d3bec5a69` | yes |
| 8 | raw | 18661 | 16384 / 16384 / 16384 | 4.33 / 6.65 / 5.85 | `6d615174db62` | yes |
| 9 | partial | 20881 | 18432 / 18432 / 18432 | 4.37 / 6.82 / 5.87 | `921fc4b0e3f2` | yes |

replies identical in every turn

| config | rank | keeps | restores (raw / partial) | ratio | pages | encode_ms | lazy_encode_ms | decode_ms | copy_ms | lazy_copy_ms | keep_io_ms | wait_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 0 | (no totals) | | | | | | | | | | |
| off | 1 | (no totals) | | | | | | | | | | |
| eager | 0 | 9 | 8 (0 / 0) | 0.692 | 1358 -> 942 | 2353.4 | 0.0 | 636.5 | 650.6 | 0.0 | 4170.9 | 0.0 |
| eager | 1 | (no totals) | | | | | | | | | | |
| lazy | 0 | 9 | 8 (2 / 0) | 0.6878 | 958 -> 660 | 0.0 | 1696.0 | 417.3 | 78.1 | 201.5 | 4482.5 | 0.0 |
| lazy | 1 | (no totals) | | | | | | | | | | |

| config | rank | trimmed | pages freed | free= last | held / spare MiB last |
|---|---|---|---|---|---|
| off | 0 | 0 | 0 | None | 2716 / 12684 |
| off | 1 | 0 | 0 | None | None / None |
| eager | 0 | 0 | 0 | None | 1884 / 13524 |
| eager | 1 | 0 | 0 | None | None / None |
| lazy | 0 | 7 | 298 | 6660 | 2228 / 13212 |
| lazy | 1 | 7 | 298 | 6660 | None / None |

restore forms (log-inferred, extra cases): off: {'raw': 'uncompacted', 'partial': 'uncompacted'}; eager: {'raw': 'compacted', 'partial': 'compacted'}; lazy: {'raw': 'raw-or-partial', 'partial': 'raw-or-partial'}
