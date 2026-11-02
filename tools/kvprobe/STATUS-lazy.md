# Lazy KV codec: status

## GPU result, 2026-11-01 late [O]

Served on the native ring (session record 11.12, receipts `gpu-2026-11-01/lazy/`): `mode=lazy delay_ms=2000
threads=5 cpus=5-9 (outside the engine's 0-4)`; four keeps compacted 2.3 s after each keep on the workers
(`encode_ms` 0.0 on the prompt pass, `lazy_encode_ms` 1173), pages 532 to 387 (ratio 0.727, the eager run's bytes),
charges trimmed to the compacted size, four restores from compacted entries with replies byte-identical to the first
sends and to the eager run. A CPU list outside the session's cpuset (15-19) was refused by name at start. Not
exercised served: raw and part-compacted restores, rank-skewed TP agreement (host tests cover them).


## Timers (2026-11-01, after the served runs) [O for the code, H for the cause]

The served totals of session record 11.11 / 11.12 were 5x apart both ways for the same 4 keeps and 4 restores of
1.11 GB (eager `encode_ms` 6089, `decode_ms` 1770; lazy `lazy_encode_ms` 1173, `decode_ms` 343), with
`TF_GLM_KV_LEDGER=1` and `TF_GLM_KV_DUMP` on in both. What each total included, from the code at `4ed7bc2b8`:

| total | before | after |
|---|---|---|
| `encode_ms` (eager) | `Codec.keep`: from `const t = now()` (codec.zig:164, after `observe`) to the keep's end (:170): `Shared.add`, then `compactNow`'s waves: the `work` lock, the workers' `pool.run`, the engine thread's serial write-back of the payloads (`View.write`, lazy.zig:281), `finish` (the block-identity map) | the workers' `pool.run` waves only (`step`'s `encode`) |
| `lazy_encode_ms` | `step`'s `ns` (lazy.zig:261 to :284): after taking `work`, the workers' `pool.run` and the compactor's write-back | the workers' `pool.run` waves only |
| `decode_ms` | `Codec.restore` from `t0` (codec.zig:211, after `pin`) to its end: the `work` lock, each wave's payload gathers, `pool.run`, `copyRange` issues, the `synchronize` on the slot two waves back (:248), and a raw rest's `moveFrom` (:213); not the last wave's copies landing (the caller's `synchronize`, cuda_snapshots.zig:83) | the workers' `pool.run` waves only |
| `copy_ms` (new) | - | engine thread: an eager keep's payload write-back; a restore's gathers, staging copies (issued and waited for, including a final `synchronize` now in `restore`) and raw rest |
| `lazy_copy_ms` (new) | - | the compactor's payload write-back |
| `keep_io_ms` (new) | in no total: `observe` (the ledger's read of the whole raw state, the dump's `.bin`/`.parts` writes) runs before `t`, and `writePlan` (the `.plan`) in `keepAs` | `observe` and `writePlan`, less their wait for `work` |
| `wait_ms` (new) | partly in `decode_ms` (the `work` lock) | engine thread: waiting for `work` (the compactor's wave) in a keep, a restore or `observe`, and a restore's `pin` waiting for a compaction under way |

So the old eager and lazy timers covered the same operations: the ledger and the dump were in neither, the raw copy
into host pages (`pool.move` and its `synchronize`, cuda_snapshots.zig:62-63) is in none (outside the codec, and
still not timed: that file is not this work's). The 5x is where the work ran [H]: the eager run's start line has
`threads=5` and no `cpus=` (unpinned: the workers inherit the engine's `taskset -c 0-4,10-14`, five CPUs in the
session's cpuset, shared with the engine's own threads), the lazy run's `cpus=5-9 (outside the engine's 0-4)`; the
eager encode also ran right after the dump's 284 MB writes of the same keep. A wave ends with its slowest worker,
so a worker that loses its CPU stretches every wave. The test of it: eager with `TF_GLM_KV_CODEC_CPUS=5-9` (the
receipt's `eager` configuration) against eager unpinned.

Line format (teardown, one per rank): the fields up to `threads=` unchanged, then `keep_io_ms= copy_ms=
lazy_copy_ms= wait_ms=`. `Codec.totalsLine` writes it (`log` prints it); bookkeeping under `Shared.m` (`finish`,
the block-identity map) is in no timer. Engine-thread time inside the codec is now `encode + decode + copy +
keep_io + wait` (eager; `encode` 0 in lazy); the compactor's is `lazy_encode + lazy_copy`.

Passed (x86, Zig 0.17.0): `zig build test-glm-snapshots-host test-kv-codec test-cuda-host compile-glm-host` 286/286
steps, 542/543 tests (1 skipped; 541/542 before, plus the timers test), `test-glm-snapshots-host` 33/33, `zig fmt
--check` on the three changed Zig files.

Check: `zig/glm_snapshot_host_lazy.zig` "an eager keep with the ledger and the dump on times the codec's encode apart
from its I/O and copies", twice (ledger alone, then ledger and dump): the `.bin`/`.parts`/`.plan` exist only with
the dump; after the keep `keep_io`, `encode` and `copy` > 0, `lazy_encode`, `lazy_copy` and `decode` 0, and their sum
with `wait` within the keep's wall time; after a restore `decode` > 0, `copy` grew, the keep's timers unchanged; the
totals line carries every field. Mutation: `observe`'s time counted on `encode_ns` fails it (the ledger-alone case).

## Runners (2026-11-01; nothing has run on a GPU with them yet)

- `lazy_receipt.py run|collect|compare|selftest`: the conversation of `RUNBOOK-lazy.md` step 3 made exact: 6 main
  turns (history past 2048 tokens at turn 2, a 2200-token piece a turn, so every turn from 3 resumes the last turn's
  keep and keeps a new one), then a raw case (turn 8 at once after turn 7) and a part-compacted case (turn 9 timed
  from turn 8's keep: its estimated time + `--lazy-ms` + `--partial-frac` of the compaction's estimated length, from
  this run's earlier `compacted` lines). Per turn TTFT (streamed), totals, tokens, the token sha256, `cached_tokens`,
  the log lines it caused and the restore form they imply. `collect` (after SIGINT) copies and parses the logs;
  `compare OFF EAGER LAZY` prints the tables. The lazy server's `TF_GLM_KV_CODEC_LAZY_MS` must equal `--lazy-ms`
  (default 20000: the keep falls before the last user message's prefill, so turn 7's tail and decode must end
  inside it for a raw restore). If `partial_restores` stays 0, move `--partial-frac` or widen the compaction with
  `TF_GLM_KV_CODEC_THREADS=1` (and say so).
- `recall_ab.py run|compare|selftest`: the served Recall A/B of `docs/recall.md` from session-store entries
  (`recall_yield.py`'s format): an entry that is a prefix of another is the same conversation, user (and tool-result) spans detokenized by
  the server and replayed greedily with the model's own replies (`max_tokens` 256); per reply rounds, accepted /
  drafted, tok/s, sha256; `GET /v1/recall` after. Output holds counts and hashes only. A store with no user turn
  followed by an assistant span is refused by name. The sample under `~/kvprobe/sessions-sample/` (2026-11-02) has
  none: 8 entries are one synthetic warmup prompt (a single cut user span, 29 distinct ids) and 4 carry no role
  tokens (completions); the A/B needs entries of real multi-turn conversations.
- `receipt.py`: `docs/receipts/glm-kv-codec-lazy-<sha>/` from the three runs (and the A/B): README in the receipts'
  style, records, `commit.txt` (read-only `git`), `binaries.txt`, `redaction-report.json`, `SHA256SUMS`.
- `served.py` (HTTP routes, log parsing) and `fake_server.py` (the self-tests' stand-in server and synthetic store).
  `python3 lazy_receipt.py selftest`, `recall_ab.py selftest` and `receipt.py selftest` pass (no server, no GPU).

The lossless KV codec (`zig/src/core/kv_codec.zig`) as a lazy compaction format for GLM's kept prompt states.
Tags as in `STATUS.md`: [O] observed, [U] untested. The GPU result above supersedes the "nothing has run on a GPU" notes below.

## Contract

- Behaviour (all behind flags; defaults unchanged: codec off, and with `TF_GLM_KV_CODEC=1` eager as before):
  - `TF_GLM_KV_CODEC_LAZY=1` (needs `TF_GLM_KV_CODEC=1`, which needs `TF_GLM_HOST_CACHE=1`): a keep copies the raw
    state into its host pages and returns; a persistent compactor thread compacts it in place later, on the codec's
    workers. Policy: an entry is due once neither kept nor resumed for `TF_GLM_KV_CODEC_LAZY_MS` (default 2000,
    0 to 3600000, else refused by name); least recently touched first; never an entry pinned by a restore or drop.
  - Forms: raw (no block compacted), part compacted (blocks `[0, done)` as payloads at the stream's start, the raw
    rest where the keep put it), compacted. A restore serves any form byte for byte; a restore or drop of an entry
    under compaction stops it at the end of the current wave (pinned first, so the compactor cannot take it back), and
    a later compaction resumes from `done`.
  - Pages and charges: the lease holds every raw page until the entry is compacted. Alone (one rank), the pages past
    the payloads go back when the store next asks (`snap_recharged`, polled from `prompt_cache_lifetime.collect` at
    every admission), and the store's charge drops to the compacted pages at that moment. With peers, the pages go
    back only through the gate's agreement (below); until the store's next charge, pages given back are held out of
    `snap_spare` (`unclaimed`), so `held + spare <= budget` holds at every step. `snap_reuses` asks for the raw pages
    a keep takes (it needs them until its compaction).
  - Two ranks or more: the fingerprint's `kv_codec` field is 0 off, 1 eager, 2 lazy, so the ranks agree on the mode.
    Lazy skips the eager `agreePages` (every copy is raw at its keep) and runs `cuda_snapshot_lazy.agree` at every
    gate: the leader names one compacted, unagreed entry (least tried, oldest; 0 for none: one extra word per gate);
    every rank answers its copy's compacted pages, or `not_ready` (2^32-1) while raw or part compacted; all ready, every
    rank trims to the most pages any copy needs and marks it agreed; otherwise the leader asks again later.
  - Workers: made once at start (`TF_GLM_KV_CODEC_THREADS`, default up to 8 of the CPUs they may use), shared by the
    eager keep, the restore's decode and the compactor (one wave at a time: `Shared.work`). `TF_GLM_KV_CODEC_CPUS`
    (`taskset -c` syntax) pins the workers and the compactor on Linux; a malformed list or a CPU the process may not
    run on (probed on a short-lived thread by setting and reading back the mask: offline, absent or outside the
    cpuset) is refused by name at start; elsewhere the list is ignored with a logged note. A thread may leave the CPUs
    it inherited from `taskset`; the start line says whether the list is inside or outside the engine's.
  - Logs: start `KVCODEC rank=<r> mode=<eager|lazy> delay_ms=<n> threads=<n> [cpus=<list> (inside|outside the
    engine's <list>)]`; lazy `KVCODEC rank=<r> compacted keep=<id> raw= stored= pages a -> b after_ms=`, `trimmed
    keep=<id> pages a -> b free=<n>`; totals at teardown with `encode_ms` (inside keeps, on the prompt pass) and
    `lazy_encode_ms` (on the compactor), `raw_restores`, `partial_restores`, `cancels`. `duplicate` counts blocks of
    compacted states only (a raw state has no blocks yet), as before when every held state was compacted.
- Owning modules: `zig/src/core/kv_codec_pool.zig` (CPU lists, affinity, persistent pool; re-exported as
  `kv_codec.worker_pool`), `zig/src/families/glm/cuda_snapshot_lazy.zig` (entries, forms, compactor, page agreement),
  `cuda_snapshot_codec.zig` (keep, restore of every form, settings, logs), `cuda_snapshot_pages.zig` (`moveFrom`),
  `cuda_snapshots.zig` (dispatch: `keepAs`, `restore`, `charged`, `recharged`, `spare`, `free`, `gate`). Outside the
  listed files, additive and minimal: `zig/src/core/prompt_cache.zig` (one vtable field, `recharged`),
  `zig/src/core/prompt_cache_lifetime.zig` (`recharge` at the end of `collect`, lowers charges only),
  `zig/src/native/cuda_snaps.zig` (one vtable line), `cuda_native.zig` (`snap_spare` via `snapshots.spare`, new
  `snap_recharged`), `cuda_ranks.zig` (`kv_codec_lazy` setting, refusal, fingerprint value 2, explain text),
  `cuda_ranks_test.zig` (one test), `zig/build/kv_codec.zig` (pool tests in `test-kv-codec`, linked with libc).
- Checks (each fails when the behaviour is wrong; mutations tried are listed under "What passed"):
  - `zig/src/core/kv_codec_pool_test.zig`: list parsing and printing; every refusal names the item (empty, syntax,
    reversed, too large, repeated); own CPUs allowed, an absent CPU refused by number alone and mixed with allowed ones;
    every index of a run exactly once over 20 runs; pinned workers read back their list; two callers at once.
  - `zig/glm_snapshot_host_lazy.zig` (in `test-glm-snapshots-host`): raw restore, compaction between two restores,
    compacted restore, charges and spare at each step, `encode_ms` 0 in lazy; a restore racing a compaction (seam:
    the compactor pauses after one block) served part compacted, `cancels` 1, resumed from block 1 then completed,
    plus 12 unscripted keep/restore/drop rounds a block at a time and a drop during a compaction; the real store
    (`prompt_cache.Store` over GLM's vtable) charged raw, then compacted at the next `collectExpired`, `held + spare ==
    budget` throughout, and more states than raw ones fit admitted without an eviction; two ranks over threads with an
    all-reduce-max board: nothing proposed, one copy compacted and the other raw (no trim, both restore byte for
    byte), both compacted (both trimmed to the max, agreed, unclaimed until recharged), nothing left to propose;
    refusals of `TF_GLM_KV_CODEC_CPUS` (syntax, a CPU not allowed), `TF_GLM_KV_CODEC_LAZY_MS`, lazy without the codec;
    pinned workers compact and restore.
  - `cuda_ranks_test.zig`: lazy without the codec refused, `0 or 1`, eager vs lazy ranks disagree on `kv_codec`.
- Cannot run here: any served measurement (no GPU), real NCCL agreement, GB10 core topology.

## What passed (x86, 24 CPUs, no GPU, Zig 0.17.0), 2026-11-01 [O]

| command | result |
|---|---|
| `zig build test-kv-codec` | 5/5 steps, 28/28 tests (22 codec + 6 pool) |
| `zig build test-glm-snapshots-host` | 4/4 steps, 32/32 tests (6 new lazy tests; includes the KDA replay agent's tests) |
| `zig build test-kv-codec test-kv-pool test-glm-snapshots-host test-cuda-host compile-glm-host` | 291/291 steps, 541/542 tests (1 skipped) |
| GLM module test binary from `compile-glm-host`, run directly | 168/168 (includes the new `cuda_ranks_test` case) |
| `zig build test` | 452/452 steps (tests counted only for steps not cached: 184/186, 2 skipped; baseline 1872/1890) |
| `zig build compile-kv-codec kv-codec-bench` | builds (the bench root re-exports the pool) |
| the snapshot host test binary rerun 15 times | 15/15 all pass (the threaded tests are stable) |

Mutations, each caught by the tests above (then reverted): the raw rest of a part compacted entry not copied
(3 lazy tests fail); `recharge` removed from `collect` (the store test fails); a raw rank answering 0 instead of
`not_ready` (the two-rank test fails); pinning after waiting for the compactor instead of before (the race test
fails: the compactor retakes the entry each wave and finishes it under the restore).

## GPU results, 2026-11-02: the receipt [O]

`docs/receipts/glm-kv-codec-lazy-2b61900a/` (session record 11.17): off, eager, eager-unpinned and lazy, nine
turns each on the two-rank ring, replies identical in every turn; eager pinned to CPUs 5-9 encodes 0.26 s a keep
and lazy compacts 0.24 s a keep off the prompt pass (ratio 0.692 and 0.688, lazy freed 298 pages over 7 trims);
eager unpinned is 6x slower in both encode and decode (the "Timers" hypothesis below, confirmed); lazy's time to
first token is 0.6 to 1.5 s a turn above off, but off wrote neither ledger nor dump, so the uninstrumented pair is
the next run (`validate-final.sh` on R0) before the cost is quoted.

## Open decisions

- Idle trigger: compaction is delay-based only. An engine-idle signal (the lane host's idle step, `cuda_lanes.zig` /
  `cuda_follow.zig`) could start it sooner; those files are not this work's.
- With peers, pages go back only at a gate (one per prompt chunk), so a ring that stays idle after its states compact
  frees nothing until its next prompt pass; an agreement on the idle step would need `cuda_follow.zig`.
- The extra all-reduce per gate in lazy TP mode (one word; two when a state is proposed) is unmeasured on its own; it is inside the uninstrumented off-versus-lazy delta the next run measures.
- `prompt_cache.zig` and `prompt_cache_lifetime.zig` are outside the listed files: the lead should confirm the
  `recharged` hook (it only ever lowers a charge; families without it are unchanged). `prompt_cache.zig` is now at
  600 lines.
- `TF_GLM_KV_LEDGER` / `TF_GLM_KV_DUMP` still read the raw state inside the keep (on the prompt pass) in lazy mode,
  and such a keep first waits for the compactor's current wave (they share `scratch` under `work`); both are now
  timed (`keep_io_ms`, `wait_ms`).
- Alone, a compacted state's pages go back at the store's next admission (`recharged`), not when its compaction ends:
  equivalent for admission, but the `prompt cache:` spare MiB lags until a request arrives (the `trimmed` lines show
  when).
- `zig fmt` (0.17) rewrote `@intFromEnum`/`@enumFromInt` in `cuda_snapshot_codec.zig` to `@backingInt`/
  `@fromBackingInt`; the rest of the tree still uses the old spellings.
- The glm_snapshot_host.zig Control harness is at 595 lines, so the two-rank test lives in
  `zig/glm_snapshot_host_lazy.zig` with real fixtures and a threaded all-reduce instead.
- Fingerprint layout stays 14: `kv_codec` gained the value 2 (lazy); an older build with the codec on reports 1 and is
  refused against a lazy rank by name.
