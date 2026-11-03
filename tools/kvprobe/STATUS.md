# kvprobe follow-up tooling: status

## Upstream candidates (successful results, 2026-11-01)

These four are tagged successful on the GB10 ring and are candidates for upstreaming; the rest of this directory's
work is experiments (not adopted) or measurement tooling.

1. **Lossless codec on real kept states** (`zig/src/core/kv_codec.zig`, `kv_rans.zig`, `kv_ledger.zig`,
   `zig/src/families/glm/cuda_snapshot_codec.zig`): 0.718 of raw on 16 kept GLM-5.3-Flash states, bit-exact round
   trips, 1.7 GB/s encode and 6.4 GB/s decode on 8 Grace cores; served, restores byte-identical.
2. **Lazy compaction with intermixed raw and compacted entries and pinned workers** (`cuda_snapshot_lazy.zig`,
   `kv_codec_pool.zig`, `TF_GLM_KV_CODEC_LAZY`, `TF_GLM_KV_CODEC_CPUS`): served, the prompt pass pays no encode, the
   workers run on the operator's cores, pages 532 to 387, restores from compacted entries byte-identical. The
   receipt (`docs/receipts/glm-kv-codec-lazy-2b61900a/`, record 11.17): nine keeps a configuration, replies identical
   across off, eager, eager-unpinned and lazy; 0.24 s a compaction on five pinned cores; the 5x timer gap was
   unpinned workers (6x slower); without ledger and dump, lazy's time to first token equals off's within noise in all nine turns (the earlier deltas were the dump), so the codec in lazy mode is free on the prompt pass.
3. **Exact page prune of the DSA index scan** (`cuda_index_prune.zig`, the `HINTS.md` sidecar, `kv_hints.zig`):
   the host consumer's selection equals the full scan on 280 oracle cases; the 204,800-token capture bounds the
   skippable pages at 0.47 to 0.75 past 128k tokens. The device path (`TF_GLM_INDEX_PRUNE=1`: per-page key strips,
   a bounds-first scan, `cuda_index_prune_dev.zig`, `glm_index_prune.cuh`; `STATUS-prune.md`) ran on the GB10
   (session record 11.15): 65 of 65 exactness cases, the contract's byte gates against the kernel it replaces, six
   served replies byte-identical off and on at 32k and 200k tokens with restores. No speed gain on this ring: the
   scan is under 1 % of a 124 ms decode step at 200k and the single-row path still pays four extra launches (x0.29
   on synthetic keys), so the flag stays off by default. The candidate is the exactness machinery as of commit 2b61900ad; round 2 on
   main (three launches, a page threshold, served skip counts, compile records; 2026-11-02) passes every kernel
   check but its served replies differ from the flag-off ring (a regression, round 3 open), and its counters saw
   zero skips on real keys at 32k and 200k, which round 3 (f45dc1c02, merged, host-tested only) traced to the
   counters' placement in the scratch, not the bound: the fix and its one-scratch checks await a GB10 pass, and the
   bound's real-key share awaits a capture of the index queries (`prune_share.py`). No performance claim.
4. **Recall, cross-session token-record drafting** (`docs/recall.md`, already in the engine): on the ring a repeated
   reply is byte-identical and decodes 2.3 to 2.9x faster with 811 of 811 copies accepted. First production sample
   (36 minutes of decode rows, 2026-11-02, record 11.16): replies to a re-sent prompt are copied whole (46 % of the
   window's rows, 15.5 of 16 accepted), ordinary agent replies have copyable spans at about 1 % of positions; the
   week's dump decides the share of re-sent prompts.

## GPU results, 2026-11-01 evening [O]

Later the same night (session record 11.10 to 11.12): the KDA replay restore ran and is not adopted (zero-state replay
rebuilds the state only to 0.67 relative error at W = 512; the replay form changed a reply); lazy compaction ran
served (encode off the prompt pass, workers on CPUs 5-9, pages 532 to 387, byte-identical restores from compacted
entries); the long-context capture gave the prunable-page bound (0.47 to 0.75 past 128k tokens).

Run on the two GB10 nodes (production ring stopped and restored by the operator agent with the user's authorization);
full account in `docs/kv-cross-context-probe-2026-11-01.md` section 11, artifacts in
`docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/`.

- Build: this tree built on a GB10 (ReleaseFast, nvcc 13.0, sm_121); `tf-cuda-test glm-exl3` PASS;
  `kernel-contract-live` 168 problems, identical to the October baseline binary's (none new).
- Track 1: 16 keeps a rank with `.bin`/`.parts`/`.plan` and `KVLEDGER` lines (the manifest change works in the
  engine). Rank 1 analysis done (`stored-r1.md`): MLA latent rows deviate 0.09 (layer 3) to 0.38 (prose) / 0.46
  (code) / 0.71 (json, boilerplate) across prefixes; the KDA state per head deviates 0.01 early to 0.2-0.4 deep and
  has an effective rank of 2 to 12 of 128 (rank90 <= 14) everywhere; conv history 0.004 to 0.35. Rank 0's analysis
  with `--bench` landed (see the codec line below and the session record).
- Track 3: Recall proxy on the Zig engine: the repeated reply is byte-identical and 2.3 to 2.9x faster (rounds
  148/119/155 -> 41/42/42; 811 of 811 copied tokens accepted).
- Track 4: `tf-glm-expert-prefetch-bench` rows 1/4/8: every variant `bits=SAME`; times within noise. Fatbins compiled
  with zero spills.
- Track 1 codec bench on the 16 real keeps (4.74 GB): plain 0.718 median ratio (8 threads: encode 1.74 GB/s, decode
  6.4 GB/s), atlas 0.717 at 3x the encode cost, deflate-6 0.819; Phase 1 acceptance met. Rank 0's KDA heads agree
  with rank 1's (effective rank median 4 to 5, rank90 <= 14).
- Routing dump (patch 0710): first attempt exhausted memory (lean prefill off at 131k context, batch 4) and spark0
  hung; both nodes were then reset at 21:46 (external) and spark1 once more by a firmware reset the operator issued;
  the direct link needed a firmware-level recovery. Second attempt on a memory-safe config (32k context and pool,
  1024 prefill rows, batch 1, no vision, 6 GiB memory guards) completed: 18,432 rows with routing, selections and
  gates. `routing.py` results: top-8 routing Jaccard across prefixes 0.83 (layer 3) to 0.44-0.55 (layers 25-39);
  DSA selections inside S 0.90-0.94 Jaccard across prefixes of equal length; expert locality weak (2048 rows touch
  263-288 experts); best cheap predictor recall@8 0.26-0.51 (one useful byte per four fetched); KDA decay horizon
  median 1.4-8.6 tokens per head; prunable pages 0 % at <= 10k tokens.
- Long-context routing dump (204,800 tokens, capture 3): prunable pages (exact bound) 0.47 to 0.75 of the index
  scan's pages for query rows past 128k, 0.18 to 0.61 at 64k to 128k, 0 below 16k; selections reach across the whole
  context (59 to 85 % of a row's picks older than 8k); only 1 to 7 % of pools are never selected by any sampled row
  (layer 7: 50 %), so selection-count demotion is closed; stride-64 stability is low (Jaccard 0.06 to 0.27), decode
  stride-1 reuse still unmeasured. Session record 11.9.


Follow-up work on `docs/kv-cross-context-probe-2026-11-01.md` section 8 (the session record itself is not edited).
Tags: [O] observed, [C] calculated, [H] hypothesis, [U] untested. No speed or quality claim here rests on anything
but a cited measurement. The GPU results above supersede the per-track "nothing has run on a GPU" notes below. Per-track detail: `STATUS-routing.md` (Track 1
item 3 and Track 4 item 1), `STATUS-track3.md`, `STATUS-track4.md`; formats and design notes: `HINTS.md`; the GB10
procedure: `RUNBOOK.md` and `serving/RUNBOOK-route-dump.md`.

Toolchain for host checks: Zig 0.17.0 (official x86_64 tarball, sha256 checked against ziglang.org's index), unpacked
outside the repository; python3 with numpy 2.5 locally, numpy 2.4 on the serving node.

## Host checks (x86, 24 CPUs, no GPU), 2026-11-01

| command | before any change | after |
|---|---|---|
| `zig build test-kv-codec test-kv-pool test-glm-snapshots-host test-glm-index-host test-glm-index-emulate test-glm-pages-host` | 21/21 steps, 95/95 tests | passes (snapshots 23 tests, index 15) |
| `zig build test-kv-codec test-kv-pool test-kv-hints test-glm-snapshots-host test-glm-index-host test-glm-index-emulate test-glm-pages-host test-cuda-host compile-glm-host compile-kv-codec test-glm-expert-prefetch-host` | | 301/301 steps, 515/516 tests (1 skipped) |
| `zig build test` (the full step) | | 452/452 steps, 1872/1890 tests (18 skipped) |
| `python3 -W ignore tools/kvprobe/test_kvstate.py` (with `KVPROBE_FIXTURE=<dir>` for the engine-format check) | | 8 tests OK |
| `python3 -W ignore tools/kvprobe/test_hints.py` | | 5 tests OK |
| `python3 -W ignore tools/kvprobe/test_routing.py` | | 10 tests OK |
| `python3 -W ignore tools/kvprobe/test_draft_eval.py` | | 9 tests OK |

The `failed command: ./.zig-cache/.../test` lines in these logs are test binaries' stderr (the baseline run printed
them too), and the `FAIL qwen4 binary64 ...` lines are planted contract controls failing as designed; the summaries
above are the results.

The engine-format fixture: `TF_KVPROBE_FIXTURE_DUMP=<dir>` makes the snapshot host test keep its dump there (a zig
build step caches its run, so run the test binary itself, or change an input, to refill it); `kvdump.py` read it, and
`tf-kv-codec-bench --file` ran end to end on it (fixture bytes are a pattern, so its ratios mean nothing).

## Track 1: open dimensions (stored quantities across contexts)

Contract.
- Behaviour: a `TF_GLM_KV_DUMP` keep writes beside `.bin` a `.parts` file that now carries each cut's row bytes
  (`class length row`, the form `tf-kv-codec-bench --file` reads) and a `.plan` manifest (the plan's parts before
  cutting with their DSA cache index, the arena head's field offsets, the KDA parity bits, ring bases, position, shape),
  so an offline reader reassembles per-layer latent rows, ring and pooled index keys, the live KDA state per head and
  the conv history without guessing.
- Owning modules: `zig/src/families/glm/cuda_snapshot_layout.zig` (`Part.cache`, `manifest`),
  `cuda_snapshot_codec.zig` (`writePlan`, rows in the cuts), `cuda_snapshots.zig` (one call in `keepAs`),
  `zig/src/core/kv_ledger.zig` (`Dump.span` takes the row); `tools/kvprobe/kvdump.py` (reader, synthetic writer),
  `kvstate.py` (analyzer), `kv_probe_send.py` (prompt builder and driver), `test_kvstate.py`.
- Checks: `zig/glm_snapshot_host_ram.zig` "TF_GLM_KV_DUMP writes a keep's bytes, rowed cuts and a manifest that covers
  its plan" (parts sum to the plan with delta 0, offsets are the State's own, every cache part carries rows and a
  cache index); `test_kvstate.py` (planted rank-3 KDA states with a rank-1 difference, identical rows at different
  starts give exactly 0 so any misalignment shows, refusals of a damaged manifest or cuts, the live half of the
  double-buffered state is the one read, the driver's grid alignment against a fake per-message template).
- Cannot run here: any real capture.

Status: done on the host. Deliverables:
1. Probe driver and analyzer: `kv_probe_send.py build|send`, `kvstate.py` (per DSA layer: latent rel L2, cosine,
   head/body, other-token baseline, raw/XOR/delta byte-plane entropy; ring and pooled index keys; per KDA layer: per
   head deviation, effective rank, rank for 90 %, the difference's effective rank and energy in rank 8 and 32; conv
   history; `--bench` runs `tf-kv-codec-bench --file` on every keep, the plan's Phase 1 acceptance).
2. Prompt sets: prose, code, JSON tool output, chat-template boilerplate (tool-calling turns), built through the
   server's `/tokenize` with `messages` (docs/api.md); S is located by rendering with and without its messages, and
   the prefixes are trimmed so S starts at the same offset modulo the 2048-row grid in every context, which makes the
   keeps sit the same distance past S (the KDA states are compared only then).
3. Serving-stack patch 0710 (routing ids and weights, DSA selections, per-head KDA gate means) with its runbook and a
   CPU harness: `serving/`, detail in `STATUS-routing.md`. Untested on GPU.
4. `RUNBOOK.md`: build on GB10, the env (`TF_GLM_HOST_CACHE=1 TF_GLM_KV_CODEC=1 TF_GLM_KV_LEDGER=1
   TF_GLM_KV_DUMP=dir`), serve, send, analyze, and stopping and restoring the production ring by hand.

Findings while building [O]: keeps happen only at prompt-cache marks (the rendered history's chunk start on GLM's
2048-row grid), so raw `/v1/completions` id lists like the hidden-tap probe's keep nothing; the driver uses chat
bodies for that reason. The dump is written only with `TF_GLM_KV_CODEC=1`, which needs `TF_GLM_HOST_CACHE=1`. Latent
and index rows are replicated on both ranks; KDA heads are split, so rank 1's dump holds the other half.

## Track 2: cache-entry hints

Contract.
- Behaviour: a sidecar format (`HINTS.md`) with a reader that refuses by name on any identity, layout, bound or
  section fault; an external post-processor producing per-page pooled-key bounds, per-pool selection counts, a
  hot-page restore order and a token-keyed expert prior; the exact page prune as a host consumer with an oracle, its
  flag refused on a device run.
- Owning modules: `zig/src/core/kv_hints.zig` + `kv_hints_test.zig` (step `test-kv-hints` in
  `zig/build/kv_codec.zig`), `zig/src/families/glm/cuda_index_prune.zig` + `_test.zig` (in `test-glm-index-host`), one
  refusal line in `cuda_forward.zig`; `tools/kvprobe/hints.py`, `test_hints.py`.
- Checks: a golden sidecar built by the Zig writer and by `hints.py` hashes to the same sha256 (pinned in both tests);
  Zig's Wyhash test vectors in Python; every refusal planted alone (checksum refreshed so only the named check
  can fire); bounds tight and sound for every pool; pruned selection equal to the full scan over 280 cases including
  negative head weights, ties and k at the edges, with pages actually skipped; the unsound variant (negative weights
  bounded from above) caught by the same oracle.
- Cannot run here: the prunable fraction on real indexer queries (no dump records the queries); any device path.

Status: done on the host; the device prune, demotion, selection reuse and restore ordering are design notes in
`HINTS.md`, each with the measurement that would justify it. `hints.py --estimate` gives an upper bound on the
prunable page fraction from a 0710 routing dump's selections.

## Track 3: drafter caching (detail: `STATUS-track3.md`)

Status: done. `draft_eval.py` scores Recall-mirroring, own-context, prior-completion and token-keyed sources against
a 0430 dump's greedy picks; `recall_yield.py` reads the request log and the session store. Observed on the existing
probe dump [O]: Recall drafts land 1.19 tokens a call on the three repeats of S (strict rule, 8-token drafts). From
the session store [O on that proxy]: beyond the stream's own-context copy, Recall would land 12 to 15 % of new
assistant tokens. Expected yield [H]: at most about 7 % of decode rounds, probably less, because DFlash2 already does
best on repeated text. The ring's Python stack has no Recall; the decisive test is a served A/B on the Zig engine.

## Track 4: expert-tile prefetch (detail: `STATUS-track4.md`, `STATUS-routing.md`)

Status: written; kernels unbuilt. The NVFP4 decode experts run `nvfp4_expert_kernel` with a two-stage register
double buffer (no shared memory, no cp.async); the prompt path already has a 2 to 4 stage cp.async ring. Prototypes
behind flags: `TF_GLM_EXPERT_PREFETCH=bulk|lines` (L2 prefetch of picked experts' column-block heads on a side
stream) and `TF_NVFP4_DECODE_RING=2|3|4` (deeper register ring, same group order, same bits by construction), with
pending kernel-contract records and a GB10 bench that hashes outputs against the stock path. The serving stack's
patch 0460 measured its routed-expert site at 0 % (its +2.4 % came from static next-kernel prefetch), so the routed
prefetch starts from a 0 % prior; `STATUS-track4.md` gives the nsys/ncu steps to find a stall first. `routing.py`
section D computes the oracle headroom (layer-to-layer, token-keyed and previous-row predictors, recall@8/16/32, bytes
in flight) once a 0710 dump exists.

## GB10 commands for the user (in order)

1. Stored-state probe on the Zig engine: `RUNBOOK.md` steps 0 to 5 (stop production by hand first).
2. Routing dump on the serving stack: `serving/RUNBOOK-route-dump.md` (image build and restarts by hand), then
   `python3 routing.py <dump> prompts.json --preset exl3-4bpw-rank` and `hints.py --routing <dump> --estimate`.
3. Expert prefetch and ring: `STATUS-track4.md` "GB10 commands" (fatbins, `ptxas -v`, the bench at rows 1, 4, 8;
   every line must say `bits=SAME`).
4. Recall A/B on the Zig engine: `STATUS-track3.md`.

## Blocked / decisions for the user

- Checkpoint for the Zig probe: the hosts' hub holds only EXL3 checkpoints (the ring's, a TensorFold-converted EXL3
  derivative, a third-party EXL3); the NVFP4 checkpoint the recipe names is absent. The ring's checkpoint matches the
  hidden-tap numbers if the Zig loader accepts it (untested); NVFP4 needs a pull. The expert kernels Track 4 touches
  are the NVFP4 path; an EXL3 run exercises `exl3_*` instead.
- Whether to record decode-row routing and selections (needed for demotion and selection reuse), and whether to hook
  the serving stack's lean/pipelined/multi-slot prefill loops so production traffic can be recorded as served.
- The exact prunable fraction needs indexer query rows and head weights captured for a few hundred rows a layer.
- Arithmetic or kernel changes (the device prune, the prefetch and ring kernels) need an issue and a maintainer-agreed
  gate (CONTRIBUTING); none was opened. ATC is not installed here, so `atc who`/`atc overlap` were not run.
- The 0710 KDA gate statistic covers rank 0's heads only.
- The dead half of the double-buffered KDA state (session record 11.18): zero in a conversation's first keep, stale
  in every keep after the first reply's decode and carried on by restores, 71 MB raw and 51 MB modeled a rank a keep
  (16 % of a 20k-token entry, 24 % of an 8k one); by code reading it is never read after a restore. Excluding it is a snapshot-plan change verified with the receipt harness
  (byte-identical replies): an outage, your call. Kronecker factorization of kept states is closed by the same
  measurement (`docs/receipts/kv-cross-context-probe-2026-11-01/gpu-2026-11-01/kron/`).
