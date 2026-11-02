# Lazy KV codec: status

## GPU result, 2026-11-01 late [O]

Served on the native ring (session record 11.12, receipts `gpu-2026-11-01/lazy/`): `mode=lazy delay_ms=2000
threads=5 cpus=5-9 (outside the engine's 0-4)`; four keeps compacted 2.3 s after each keep on the workers
(`encode_ms` 0.0 on the prompt pass, `lazy_encode_ms` 1173), pages 532 to 387 (ratio 0.727, the eager run's bytes),
charges trimmed to the compacted size, four restores from compacted entries with replies byte-identical to the first
sends and to the eager run. A CPU list outside the session's cpuset (15-19) was refused by name at start. Not
exercised served: raw and part-compacted restores, rank-skewed TP agreement (host tests cover them).


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

## Awaits the GPU

`RUNBOOK-lazy.md`: off / eager / lazy on the two-rank GB10 ring: replies byte-identical, `encode_ms` off the prompt
pass, pages freed over time, restores from raw, part compacted and compacted entries, time to first token.

## Open decisions

- Idle trigger: compaction is delay-based only. An engine-idle signal (the lane host's idle step, `cuda_lanes.zig` /
  `cuda_follow.zig`) could start it sooner; those files are not this work's.
- With peers, pages go back only at a gate (one per prompt chunk), so a ring that stays idle after its states compact
  frees nothing until its next prompt pass; an agreement on the idle step would need `cuda_follow.zig`.
- The extra all-reduce per gate in lazy TP mode (one word; two when a state is proposed) is unmeasured.
- `prompt_cache.zig` and `prompt_cache_lifetime.zig` are outside the listed files: the lead should confirm the
  `recharged` hook (it only ever lowers a charge; families without it are unchanged). `prompt_cache.zig` is now at
  600 lines.
- `TF_GLM_KV_LEDGER` / `TF_GLM_KV_DUMP` still read the raw state inside the keep (on the prompt pass) in lazy mode,
  and such a keep first waits for the compactor's current wave (they share `scratch` under `work`).
- Alone, a compacted state's pages go back at the store's next admission (`recharged`), not when its compaction ends:
  equivalent for admission, but the `prompt cache:` spare MiB lags until a request arrives (the `trimmed` lines show
  when).
- `zig fmt` (0.17) rewrote `@intFromEnum`/`@enumFromInt` in `cuda_snapshot_codec.zig` to `@backingInt`/
  `@fromBackingInt`; the rest of the tree still uses the old spellings.
- The glm_snapshot_host.zig Control harness is at 595 lines, so the two-rank test lives in
  `zig/glm_snapshot_host_lazy.zig` with real fixtures and a threaded all-reduce instead.
- Fingerprint layout stays 14: `kv_codec` gained the value 2 (lazy); an older build with the codec on reports 1 and is
  refused against a lazy rank by name.
