# KDA state replay (TF_GLM_KDA_REPLAY): status

Tags: [O] observed, [C] calculated, [H] hypothesis, [U] untested. Evidence behind the feature: session record
`docs/kv-cross-context-probe-2026-11-01.md` sections 11.3 (the arena head is 55 % of a kept entry; the KDA state after
a keep deviates 0.01 to 0.4 across prefixes, effective rank 2 to 12 of 128) and 11.8 (decay horizon median 1.4 to 8.6
tokens per layer, p90 4 to 40). Nothing here has run on a GPU.

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
    land at the kept position: `error.KdaReplayPosition`.
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

1. Check mode at W = 64, 256, 512: `KVREPLAY` lines on both ranks, both dumps, `replay_check.py` tables against the
   horizon table, and the reply equality (check mode must not change a reply).
2. Replay form at the chosen W: restore time (`form=replay ms=`), the page charge saved, and the reply's agreement
   with a knob-free server.

## Open decisions

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
