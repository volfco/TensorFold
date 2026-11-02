# Follow-ups

- Execute Phase 0 of docs/kv-cache-compression-plan.md (byte ledger, bit patterns, distributions,
  boundary probes, restore-cost budget). Gate M0/M1 decide everything downstream.
- EXL3 kernel candidates are ranked in docs/atlas-exl3-kernel-review.md. Cheap validations first:
  `ptxas -v` spill audit (0 LDL/STL in inner loops) on tf_exl3_linear/grouped/x3gm; a TensorFold
  launch census before judging the fused-QKV candidate; audit flashnext/Nemotron recurrent-state
  stores for the [K,V] float4 sector double-write (Atlas I-b mechanism, layout-generic).
- EXL3 port candidates to pick from (see the review for savings/risk): N256 expert tile,
  chunked FLA KDA prompt path, fused QKV GEMM, on-device DFlash selector, x3ld_epi A/B,
  deeper linear pipeline, packed shared experts, EXL3 index precision.
- Open a TensorFold issue before Phase 2 kernel/arithmetic work (CONTRIBUTING requirement; ATC
  claim then applies). Adjacent open issues to read first: #1511, #1510, #1312, #1293, #1156.
- Decide whether to archive the full extraction reports (~/Workspace/atlas-salvage/reports/
  kv-cache-review.md 447 l, perf-review.md 763 l, tensorfold-kv-state.md 89 l) into docs/; the
  three in-repo docs currently condense them. Reference: continue.md.
- Unverified scout gaps to close when convenient: which GLM Mac paths page vs window (receipts
  exclude Metal); which shipped Qwen serve entry points enable `kv8` (`@hasField(Engine, "kv8")`).
- Atlas items recorded as not-to-pursue (see docs/atlas-performance-review.md "Do not carry"):
  KV cache server/cluster-coord plans (unbuilt), glm-kv-shard (Atlas kernel ABI, stale numbers),
  TurboQuant/KVComp quantizers (dominated by truncation on the capture).

## KV codec (run on the GB10 ring 2026-11-01; see docs/kv-cross-context-probe-2026-11-01.md section 11)

- Done 2026-11-01: host tests, the codec bench on 16 real keeps (plain 0.718), the Phase 0 ledger on real keeps, the
  `.plan` manifest beside every dump; `tools/kvprobe/` holds the probe drivers and analyzers.
- Not done: plan Phase 0 item 4 (tools/bench_cache.py page-boundary probes), item 6 (FP8 x pages x codec matrix
  doc; the codec composes with --kv-cache fp8 since it is byte-level, untested), Phase 3 entirely, served receipt.
- No test drives the TP gate's page agreement (agreePages) through two ranks; add one to glm_snapshot_host.zig's
  Control harness.
- `snap_reuses`/admission still count raw bytes (a keep needs the raw lease before compaction): conservative.
- Keep compaction runs inside the prompt pass on rank 0: its encode milliseconds delay prefill directly. Measure
  KVCODEC encode_ms on a served run before enabling by default; a persistent worker pool would remove per-wave
  thread spawns.
- Restore Job.failed is written from several threads without an atomic (benign: any failure fails the restore).
- Real dedup storage (shared refcounted blocks) only if KVCODEC duplicate= shows material shared bytes.
- Done 2026-11-01: 16 shapes captured on GB10 (4 content types x 4 contexts), bench on each (Phase 1 acceptance met).
- Codec speed (x86, one thread, BF16 latents): encode 0.16 GB/s, decode 0.32 GB/s. Next: reciprocal-multiply
  encode (ryg_rans rcp_freq) instead of the per-symbol division; fold Counts, plane split and hash into fewer passes;
  then re-measure on GB10 Grace cores, which run the real keeps.
- Atlas candidates (model mode) cost about 5x plain encode for at most 2.6 % on synthetic rows; keep them off by
  default unless TF_GLM_KV_DUMP captures show more.
- Try byte planes + ZSTD(1) as a fast-encode backend (ZSTD alone is ~4x faster to encode than our rANS at
  8-11 % more bytes; planes may close that gap). Re-run the ClickHouse suite on real TF_GLM_KV_DUMP captures.

## After the 2026-11-01 GB10 runs

- Codec receipt: the lazy mode served at four keeps (encode off the prompt pass, 27 % of pages back, byte-identical
  restores); the repository-style receipt needs `tools/kvprobe/RUNBOOK-lazy.md`'s longer run, plus served raw and
  part-compacted restores and the TP gate exchange's cost.
- KDA replay restore: ran (0.87 / 0.73 / 0.67 error at W = 64 / 256 / 512), not adopted; the coarse-init variant
  (replay from an fp8 or low-rank stored state) is the one follow-up worth a run.
- Exact page prune: the 204,800-token capture gives 0.47 to 0.75 prunable pages past 128k; build the device path
  and the per-page key bounds in the keep path (`cuda_index_prune.zig` is host-only), gated on long-context receipts.
- Recall yield on agent traffic: a week of decode-row dumps (`GLM53_TF_DRAFT_DUMP_WHAT` with decode) on the production
  ring, scored with `tools/kvprobe/draft_eval.py`; or a served A/B on the Zig engine with real transcripts.
- Drafter refit from the 0430 dumps on served traffic (the serving project's DRAFTER-TRAINING.md).
- Operations: install a MemAvailable guard (`~/kvprobe/memguard.sh`) as a service on both nodes; the CX7 link recovery
  sequence is in the session record and the memory notes; never run the serving stack with lean prefill off at a
  large context.
- Replace the `exec`-of-`analyze.py` shortcut in `tools/kvprobe/lowrank.py`, `extra.py`, `recompute.py` with a module.
