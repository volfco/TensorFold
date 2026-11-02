# KDA state replay (TF_GLM_KDA_REPLAY): status

Tags: [O] observed, [C] calculated, [H] hypothesis, [U] untested. Evidence behind the feature: session record
`docs/kv-cross-context-probe-2026-11-01.md` sections 11.3 (the arena head is 55 % of a kept entry; the KDA state after
a keep deviates 0.01 to 0.4 across prefixes, effective rank 2 to 12 of 128) and 11.8 (decay horizon median 1.4 to 8.6
tokens per layer, p90 4 to 40). The zero-start replay ran on the ring (section 11.10: not adopted); the coarse start
(`TF_GLM_KDA_REPLAY_INIT=fp8`, below) has not run on a GPU.

## Contract

- Behaviour. Off by default; with every knob unset no default path changes a byte (the new fields default off, the
  keep and restore hooks return the plan unchanged, the forward's new branch is a skipped launch).
  - `TF_GLM_KDA_REPLAY=W` (1 to 2048): every keep records the last min(W, pos) prompt tokens before its position,
    when the bound slot's prompt passes committed all of them (prefill chunks and prefill_many note them; a decode
    commit, a fresh slot or a gap leaves them unknown, and that keep stays whole without a window).
  - `TF_GLM_KDA_REPLAY_KEEP=1` (needs W): such a keep is taken without its arena head (the plan's prefix part: pos_dev,
    KDA conv history, both halves of the recurrent states, the window scratch proj/save_*), and the plan manifest
    (`layout.manifest`) carries a `replay W` line. Its restore copies the attention classes back, zeroes the head,
    sets the position to at - W and runs the W tokens as one prompt pass on the prompt buffers with the DSA latent and
    index writes skipped (attention, dense and sparse, reads the stored rows for every position, the window's own
    included), then commits to `at`. MTP caches and `mtp_len` are restored as kept; no drafter, image, logprob, draw
    or MTP work runs in the pass. With images on (`--vision`) keeps stay whole (the splice is by position).
  - `TF_GLM_KDA_REPLAY_CHECK=1` (needs W): a restore of a full entry with a window backs up the head, replays into the
    live head, reads both states, optionally dumps both (`TF_GLM_KV_DUMP`), copies the stored head back and reapplies
    the kept host fields (parity, position, bases), then logs `KVREPLAY head|layer|total` lines: per KDA layer and head
    the recurrent state's relative L2 against the stored one, max abs error and bit identity, the same for the conv
    history (a head's q, k and v channel blocks over the three rows). Serving reads the stored state.
  - Refusals by name: `TF_GLM_KDA_REPLAY: 0 (off) to 2048 tokens`, `..._KEEP: 0 or 1`, `..._CHECK: 0 or 1`,
    `..._KEEP=1 requires TF_GLM_KDA_REPLAY`, `..._CHECK=1 requires ...` (settings, before load); a rank whose
    `kda_replay` fingerprint field differs (`TF_GLM_KDA_REPLAY, _KEEP or _CHECK`); restoring an entry kept without
    its head when no replay is configured: `error.KdaReplayNeedsWindow`; ranks that would restore differently:
    `error.KdaReplayRanksDisagree` (one decision word all-reduced per restore, two ranks or more); a pass that does not
    land at the kept position: `error.KdaReplayPosition`. The fp8 init's own refusals are listed in its section.
  - Exactness: CONTRIBUTING's resumed == fresh holds only with the knobs off or in check mode. The replay form is
    approximate by construction: the replayed state lacks the stored state's contribution from tokens before the
    window, which the recurrence scales by the product of the window's gates, about (1 - 1/h)^W for a head of decay
    horizon h [C]: at the p90 horizons of section 11.8 that is at most 0.2 at W = 64, 1.6e-3 at 256, 2.4e-6 at 512;
    at the median horizons under 4e-4 at W = 64. That bounds the recurrence's own term only: layers above the first
    KDA layer also see inputs changed by the replayed layers below them (and the replay's prompt pass is a different
    chunking than the fresh prefill's), so their error can exceed it [H]. A cold-tier trade, not a default.
- Owning modules. `zig/src/families/glm/cuda_kda_replay.zig` (window bookkeeping, keep/restore hooks, the rebuild,
  check, report and dump, the engine's runner, the knobs' parser) and `cuda_kda_replay_test.zig`. Plumbing:
  `cuda_snapshots.zig` (the plan through `keepPlan` in `keepAs`, through `restorePlan` in `restore`, and
  `afterRestore` at its end: three call lines), `cuda_snapshot_layout.zig` (`Plan.replay` and the manifest line),
  `cuda_dsa_block.zig` (`Kernels.fixed`: the two cache writes skipped) and `cuda_forward.zig` (`Forward.fixed_rows`
  into it), `cuda_engine.zig` (field, open/destroy, forget at `begin`, note after a prompt chunk's commit),
  `cuda_prompts.zig` (note after prefill_many's commits), `cuda_ranks.zig` (three Settings fields, the parse call, one
  Fingerprint field with its Field, `of`, mismatch and explain entries). Host harness: `zig/glm_snapshot_host_ram.zig`
  (last section), `zig/test_support/glm_snap_fixture.zig` (sets `kda_replay`, `slot`, `images`),
  `zig/test_support/glm_snap_driver.zig` (async memset and device-to-host copy). Tools: `replay_check.py`,
  `test_replay_check.py`, `RUNBOOK-replay.md`.
- Checks (host). `cuda_kda_replay_test.zig`: the token window (shorter than W, longer, a gap), per-head rel L2 / max abs /
  bit identity on planted states and conv blocks (zero stored heads), every knob refusal and the fingerprint mismatch,
  the stripped plan. `glm_snapshot_host_ram.zig` with a recording runner: a replay-form keep holds the plan minus the
  head exactly and fewer pages than a full one; its restore runs the pass once from at - W over exactly the W tokens
  before the kept position and lands at it, the head is the pass's alone (zeroed before it), every other byte is the
  kept one; the same entry with the knob removed is refused by name; a keep after a gap stays whole without a window;
  check mode serves the stored state byte for byte (parity, position, nothing pending), reports every head but the
  one with a planted element bit-identical and the conv history identical, frees its scratch, and dumps both heads as
  head-only keeps (the stored one equal to the kept bytes); a replay-form keep through `TF_GLM_KV_CODEC` dumps a
  manifest with `replay 2048` and no prefix part, and restores through the codec. `test_replay_check.py`: planted
  per-head differences, the bound, stem pairing, refusals; with `KVPROBE_REPLAY_FIXTURE` the Zig host test's own dumps.
- Cannot run here: any replay on real weights (the write skip's effect on real attention, the pass, its cost, the
  errors), the reply comparison, check mode's device allocation on a full card, multi-rank agreement on hardware.

## Coarse init (`TF_GLM_KDA_REPLAY_INIT=fp8`), 2026-11-02

The one variant section 11.10 left open: start the replay from a coarse stored copy of the head instead of zero.

- Knob. `TF_GLM_KDA_REPLAY_INIT=zero|fp8`, default `zero` (everything above, unchanged byte for byte). `fp8` needs
  `TF_GLM_KDA_REPLAY_KEEP=1` or `TF_GLM_KDA_REPLAY_CHECK=1`, and allows `TF_GLM_KDA_REPLAY=0` (unset): the coarse copy
  alone, no replay, no tokens needed (a keep after a decode commit still takes the form). With W > 0 a keep whose tokens
  are unknown stays whole, as before. The settings fingerprint packs it at bit 18 of the existing `kda_replay` word, so
  ranks that differ are refused by name at startup (`TF_GLM_KDA_REPLAY, _KEEP, _CHECK or _INIT`); the field count and
  the fingerprint's layout number (14) are unchanged.
- What is kept (`TF_GLM_KDA_REPLAY_KEEP=1`). The plan's arena head is replaced, as part 0 (class `prefix`), by a coarse
  copy of the live state only: each KDA layer's current recurrent half ([heads][128][128] fp32) and the conv history
  ([3][q|k|v of heads x 128] bf16). Not kept: the other recurrent half, `pos_dev` and the window scratch
  (proj, save_*), which a restore zeroes as the zero form does. Layout (`cuda_kda_coarse.zig`): a 32-byte header
  (`tfkdafp8`, layers, heads, conv_c, version 1), one fp32 scale per row (recurrent rows, then conv rows), then 128
  e4m3 codes per row. A row is 128 values: one key channel of one head's state, or one head's 128 channels of one
  conv block in one history row. Per row, not per head: a row much smaller than its head's largest keeps its own
  range, at 4 bytes per 128 codes (3 %).
- Arithmetic (host, deterministic, documented in the module head). amax = max |x| over the row's finite values; scale
  = 2^k, the smallest power of two with amax / 2^k <= 448 (k >= -126; 1 for a zero row); code =
  `e4m3FromF32(x * 2^-k)` (`core.exlr8.codec`, OCP E4M3, nearest even); value = `e4m3Value(code) * 2^k`. The scaling
  and the decode are exact in fp32 (and in bf16 for the conv history), so the only error is e4m3's rounding:
  |value - x| <= max(|x| / 16, scale / 1024) a value, a row's max lands in (224, 448] and never saturates, e4m3-grid
  values times the scale and zero rows come back bit-identical [O, host]. Infinities saturate to +-448 x scale, NaN
  stays NaN.
- Restore. The attention classes come back from the pages as before and the coarse copy lands in a mapped pinned
  staging buffer (`HostBuffer.allocMapped`, as the host page pool; made at first use, about 22 MB a rank on this
  checkpoint, freed with the replay object); the arena head is zeroed and the parity set to 0; the copy is decoded on
  the host into the stage's tail and copied (`cuMemcpyDtoDAsync`, one layer at a time, a sync after each) into the
  conv history and each layer's half 0; then the position is set to at - W and the existing replay runs the W tokens
  (W = 0: nothing runs, the position is `at`). The copy is the state at the kept position: the replay re-runs W tokens
  that are already in it. That is the contract, mechanically; whether coarse plus W beats coarse alone is what the
  check measures, not a claim. For W >= 3 the pass rewrites the conv history entirely, so only W = 0 exercises the
  coarse conv history.
- Check mode (`TF_GLM_KDA_REPLAY_CHECK=1` with fp8, on a full entry): the stored head is backed up and read, its coarse
  copy made on the host, then (a) the copy alone is decoded into the live head and compared (`window=0 init=fp8`),
  (b) the head is rebuilt again from the copy plus the replay of W tokens and compared (`window=W init=fp8`), then the
  stored head is put back and served. Every `KVREPLAY head|layer|total` line now carries `init=` (and the head line
  `window=`); the replay form's line carries `init=` too. Dumps (`TF_GLM_KV_DUMP`): zero init unchanged
  (`kda-replay-rank<r>-keep<id>-w<W>-{stored,replayed}`); fp8 adds a tag, `...-w<W>-fp8-{stored,coarse,replayed}`
  (no `replayed` at W = 0), all full-size head-only keeps `kvdump.py` reads. `replay_check.py --dir` pairs (stored,
  coarse) as window 0 and (stored, replayed) as window W, marks each `init` and `variant`, and leaves out the
  zero-start bound for fp8 pairs.
- Refusals by name: `TF_GLM_KDA_REPLAY_INIT: zero or fp8`; `TF_GLM_KDA_REPLAY_INIT=fp8 requires
  TF_GLM_KDA_REPLAY_KEEP=1 or TF_GLM_KDA_REPLAY_CHECK=1`; `..._KEEP=1 requires TF_GLM_KDA_REPLAY` (now only with the
  zero init); restoring an fp8 entry with no replay configured: `error.KdaReplayNeedsCoarseInit` (recognised by its
  length, plan minus head plus the copy); restoring an entry kept under the other init:
  `error.KdaReplayInitDiffers`; a coarse copy whose header or shape differs from the state: `error.KdaReplayCoarseShape`;
  W = 0 with the zero init: `error.KdaReplayWindow`.
- Manifest. A replay-form plan prints `replay <W>` as before for zero; fp8 prints `replay <W> init fp8 row 128` and the
  copy as `part 0 prefix <bytes> 0 -`. `kvdump.py` still refuses either (an unknown line; for fp8 the prefix part is
  not an arena head).

Bytes a kept entry costs, one rank of the two (GLM-5.3-Flash under TP2: 34 KDA layers, 32 KDA heads a rank, conv_c
12288) [C from the layout; the 272 / 78 MiB are 11.10's measurement at position 8192]:

| form | pages a rank | arena head share | host memory beside it |
|---|---|---|---|
| full | 272 MiB (11.10) | the whole head, 194 MiB measured (272 - 78); its fields by the layout are both recurrent halves 136 MiB, conv 2.4 MiB and the window scratch 17 MiB, the rest unexplained here | none |
| replay, `init=zero` | 78 MiB (11.10) | none | W x 4 bytes of token ids |
| replay, `init=fp8`, any W | 78 + 18.76 = about 97 MiB | the coarse copy, 19,675,424 bytes: 32 + 149,056 rows x (128 + 4) (139,264 recurrent rows, 9,792 conv rows) | W x 4 bytes; the pinned stage, about 22 MB a rank, once per process |

The coarse copy is one recurrent half and the conv history at a quarter width plus 3 % scales: 0.267 of the 73.8 MB
it encodes and about a tenth of the arena head (which holds both halves and the scratch). With `TF_GLM_KV_CODEC` the
copy is compacted with the rest of the entry like any prefix bytes; the store's admission estimate
(`snapshots.bytes`) still prices the full plan (open decision below).

Owning modules: `zig/src/families/glm/cuda_kda_coarse.zig` (layout, arithmetic, the stage, capture and apply) with
`cuda_kda_coarse_test.zig`; `cuda_kda_replay.zig` (the init on entries and the replay object, `stripAs`, the keep and
restore hooks, `rebuild`'s start, check mode's two reports, dump names, the knob); `cuda_snapshot_layout.zig`
(`ReplayInit`, `Plan.replay_init`, the manifest line); `cuda_ranks.zig` (the setting and its fingerprint bit). No
kernel changed; `cuda_engine.zig` (597 lines) and `cuda_snapshots.zig` are untouched.

Checks (host):
- `cuda_kda_coarse_test.zig`: rows from 1e-30 to 6e20 in magnitude: the scale a power of two putting the max in
  (224, 448], every value within max(|x|/16, scale/1024), relative L2 under 1/16, the same codes on a second encode;
  a row of e4m3 grid values times 2^-20 and a zero row bit-identical; infinities and NaN as documented; a whole copy
  of two layers round-trips within the bound; a copy of another shape or a damaged header is refused.
- `cuda_kda_replay_test.zig`: the knob's refusals, W = 0 allowed only with fp8, a fingerprint mismatch on the init
  alone, a W = 0 slot window, an fp8 plan with the copy as part 0 in the head's place.
- `glm_snapshot_host_coarse.zig` (actual owner, recording driver): an fp8 W = 0 keep holds the plan minus the head
  plus the copy exactly; its restore runs no pass, lands at the kept position with parity 0, and the head is zero but
  the position, the decoded conv history and half 0, which equal a decode computed apart from the restore bit for bit
  (the planted grid row equal to the stored one), every other byte the kept one, twice; the same entry without the
  replay object is refused (`KdaReplayNeedsCoarseInit`) and under the zero init (`KdaReplayInitDiffers`). An fp8
  W = 64 restore runs the pass once from at - 64 over the 64 tokens, and the pass starts from exactly the decoded copy;
  a keep after a gap stays whole. Through `TF_GLM_KV_CODEC` the entry is compacted, its manifest carries
  `replay 64 init fp8 row 128` and the copy as the only prefix part, and the restore decodes the same copy. Check mode
  with fp8 at W = 256 and W = 0 serves the stored state byte for byte, reports the copy alone within 1/16 on every
  head, and (W = 256) the copy plus the pass with every head but the perturbed one identical; it dumps the fp8
  triple (no `replayed` at W = 0), the stored dump equal to the kept head.
- `test_replay_check.py`: an fp8 triple beside the zero pair (three pairs, told apart; the coarse pair at window 0 and
  without the zero-start bound); with `KVPROBE_REPLAY_FIXTURE` the engine's own zero and fp8 dumps.

Awaits the GPU: everything measured (`RUNBOOK-replay.md` step 5): the copy's error on real states, coarse plus W
against coarse alone and against the zero start (11.10's 0.87 / 0.73 / 0.67), the reply, the keep's added cost (a
synchronous download of 73.8 MB a rank and a single-threaded host encode of 18.4 M values: expect tens to a hundred
milliseconds, to be measured), the restore's (host decode and 35 copies with a sync each), and the pinned-to-pinned
`cuMemcpyDtoDAsync` between the stage and the host page pool under UVA (device-to-pinned is the host pool's existing
path; pinned-to-pinned is that assumption one step further).

Not done: `rank16` (a per-head rank-16 factorization computed on the host at keep time). The knob refuses it
(`zero or fp8`). It would need an SVD of 1088 128x128 matrices a rank a keep on the host (or a device kernel) and a
second copy layout; section 11.3's effective ranks (2 to 12, rank90 at most 14) make it a candidate after the fp8
numbers are in.

## What passed (x86 host, Zig 0.17.0, no GPU), 2026-11-02 (coarse init)

| command | result |
|---|---|
| `zig build test-glm-snapshots-host` | 36/36 tests (4 new in `glm_snapshot_host_coarse.zig`; the 3 zero-init replay tests unchanged and passing) |
| `zig build test-cuda-host compile-glm-host compile-cuda-tests --summary line` | 287/287 steps, 515/516 tests (1 skipped); the coarse and replay unit tests run in `test-cuda-host` (checked with a planted failing test, removed) |
| `zig build test --summary line` | FULLTEST |
| `TF_KVPROBE_REPLAY_DUMP=<dir> <snapshot host test binary>` then `KVPROBE_REPLAY_FIXTURE=<dir> python3 -W ignore tools/kvprobe/test_replay_check.py` | 4 tests OK (engine dumps: zero pair 63/64 heads identical; fp8 coarse pairs rel max 0.024; fp8 replayed 63/64) |
| `python3 -W ignore tools/kvprobe/test_replay_check.py` | 4 tests OK (1 skipped without the fixture) |

## What passed (x86 host, Zig 0.17.0, no GPU), 2026-11-01

| command | result |
|---|---|
| `zig build test-glm-snapshots-host` (test binary run directly) | 26/26 tests (3 new) |
| `zig build test-glm-snapshots-host test-cuda-host compile-glm-host --summary line` | 279/281 steps, 3387/3396 tests (8 skipped, 1 failed): the failure was in an unrelated server test binary that passed 893/893 when rerun alone (a timing flake under the parallel build) |
| `zig build test --summary line` | 452/452 steps, 3116/3128 tests (12 skipped); the `failed command` lines are test binaries' stderr, as in the baseline |
| `zig build compile-cuda-tests` | 10/10 steps; a planted `@compileError` in the engine's replay pass fails it, so the pass, `open` and `Forward.fixed_rows` are compiled there |
| `TF_KVPROBE_REPLAY_DUMP=<dir> <snapshot host test binary>` then `KVPROBE_REPLAY_FIXTURE=<dir> python3 -W ignore tools/kvprobe/test_replay_check.py` | 3 tests OK (the engine-format check included: 63/64 heads identical, max abs 0.5, as planted) |
| `python3 -W ignore tools/kvprobe/test_replay_check.py` | 3 tests OK (1 skipped without the fixture) |
| `replay_check.read_horizons` on the receipts' `routing.md` | 34 KDA layers parsed; bound maxima as in the contract [C] |

## Awaits the GPU (`RUNBOOK-replay.md`)

1. Check mode at W = 64, 256, 512 (zero init): done on the ring, section 11.10.
2. Replay form at the chosen W (zero init): done at W = 256, section 11.10 (reply changed at token 10).
3. Coarse init check mode at W = 0, 64, 256 (`RUNBOOK-replay.md` step 5): the two reports per restore, the fp8
   triples through `replay_check.py`, the reply equality.
4. Coarse init replay form at the W step 5 favours (step 6): restore time, the pages charged (about 97 MiB a rank
   against 272 full and 78 zero-init), the keep's added time, and the reply against a knob-free server.

## Open decisions

- Coarse init: the copy is the state at the kept position, so a replay re-applies its window; capturing the state at
  at - W instead would need a state saved mid-prefill (not available at a keep). Whether to adopt fp8 alone (W = 0)
  as the cold form if step 5 shows its reply holds, and at what quality gate.
- Coarse init: scale per row (chosen; 3 % overhead) or per head (0.02 %, coarser); the measurement can decide.
- Coarse init: the keep's synchronous download and host encode sit on the prompt pass's keep; moving the encode to a
  worker (as the lazy codec does) is the obvious next step if the measured time matters.
- Coarse init: the pinned stage is held per process once made (about 22 MB a rank); it is not charged to the cache.

- Which entries take the replay form: today `TF_GLM_KDA_REPLAY_KEEP=1` takes every keep with known tokens; the cold
  tier policy belongs to the lazy-compaction work (the hook is `keepPlan`'s `replay` decision).
- A quality gate for the replay form (plan Phase 3): what reply agreement or KL at which W admits it.
- `kvdump.py` refuses a replay-form keep's dump (an unknown `replay` manifest line, no prefix part first); teaching it
  to read one is a small additive change in a file this work does not own.
- The settings fingerprint's layout number stays 14 (bumped from 13 in the same uncommitted tree for
  `host_cache`/`kv_codec`); if those land separately, the later change bumps it again.
- `cuda_dsa_block.zig` (not in the original ownership list) carries the write skip: `Kernels.fixed` and two `if`s.
- `snapshots.bytes` (the store's admission estimate, `Plan.size`) still prices a replay-form keep at the full plan, so
  the store may refuse a keep that would fit by the head's bytes (about 0.2 GB a rank); the policy work should price it.
- `agree()` refuses a planned disagreement by name, but a rank whose check fails mid-way (the head backup's device
  allocation, a download error) returns while its peers enter the pass's collectives: that path hangs, not refuses.
  Allocating the backup once at `open` (the head size is fixed) would remove the likeliest cause.
- Coupling with the codec work: the plan handed to `codec.keep`, `writePlan` and `codec.restore` is the stripped one
  for a replay-form entry, and `Snap.len` records the stripped length.
- `cuda_kda.zig` and `cuda_kda_block.zig` are unchanged: the replay runs the stock prompt chain.
- Window tokens live with the replay object in host memory (W x 4 bytes an entry and a slot); a persisted cache tier
  would have to carry them with the entry.
