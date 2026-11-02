# Continue

Atlas (Rust/CUDA, AGPLv3) was retired as a project; its research was salvaged into TensorFold docs
on 2026-10-29. Read these in order:

1. docs/atlas-performance-review.md — all Atlas performance research with TensorFold relevance
   ratings, priority shortlist, and the do-not-carry list (defective numbers, dead ends, traps).
2. docs/atlas-kv-cache-compression.md — the KV-cache compression research dossier with trust tags
   (what is safe to build on vs defective), the B0 defect register, and B1 campaign lessons.
3. docs/kv-cache-compression-plan.md — the applied plan for TensorFold: measure-first phases,
   decision gates with kill criteria, integration points, and receipts requirements.
4. docs/atlas-exl3-kernel-review.md — Atlas EXL3 CUDA kernels vs TensorFold's same-function
   kernels: verdict (TensorFold mostly ahead), ranked port candidates with estimated savings,
   feature gaps, and the do-not-port list.

Current state (2026-11-01 evening): the KV codec and the probe tooling ran on the two GB10 nodes; results, incidents
and the open GPU steps are in docs/kv-cross-context-probe-2026-11-01.md section 11 and tools/kvprobe/STATUS*.md. Two
features were written and run on the ring: lazy compaction with intermixed entries and core affinity (served, works;
tools/kvprobe/STATUS-lazy.md) and the KDA replay restore (ran, not adopted; tools/kvprobe/STATUS-replay.md).
Earlier (2026-10-31): local main rebased onto origin/main; the failed GLM/EXL3 kernel experiments are
archived in archive/kernel-research-d33dcfc7/. A first-pass KV compression implementation is written and
**compiles but has never been run** (see "KV codec first pass" below). Full extraction reports (with per-number source paths) live outside the repo at
~/Workspace/atlas-salvage/reports/ pending an archive decision (see todo.md).

Next items:

- Decide whether Phase 0 (measurement tooling: retained-entry byte ledger, bit-pattern checks,
  boundary probes) starts; it needs no engine arithmetic changes and is the gate for everything else.
- Pick EXL3 kernel candidates from docs/atlas-exl3-kernel-review.md (N256 expert tile is the
  best-evidenced small win); run the cheap validations listed there first.
- Before any Phase 2 kernel work: open a TensorFold issue (CONTRIBUTING requires one for
  arithmetic/kernel changes) and run `atc who`/`atc overlap` on zig/src/core/kv_pool.zig and
  zig/src/families/glm/cuda_state.zig.
- When this change is pushed: run `atc overlap --json` and link whatever issue is opened.

## KV codec first pass (2026-10-31; host tests pass, benchmarked on x86 only)

Ran 2026-10-31 on the x86 dev box (24 CPUs): test-kv-codec 17/17, test-glm-snapshots-host 22/22, test-cuda-host
all pass. Fixes from the run: T64 group counts typed u7 by @min (overflow); an over-tight rANS ratio assertion;
the codec host test's device-usage check ran before fixture teardown; a fingerprint padding assertion that never
held upstream; XxHash3 (0.17 GB/s in Zig 0.17 std) replaced by Wyhash (15 GB/s) as the block hash; rANS hot loops
unrolled across the four states with a packed slot table (+35-50 % decode).

Bench (32 MiB synthetic streams, 8 threads; raw logs in the session scratchpad, not in the repo):
state_mix plain ratio 0.720 (deflate-6 0.834), encode 0.91 GB/s, decode 1.77 GB/s, i.e. about 1.1 s to keep and
0.6 s to restore a 1 GiB state. Atlas candidates win only on row-structured BF16 (delta along token rows:
0.700 vs 0.719; state_mix 0.707 vs 0.720) at about 5x the encode cost; XOR and T64 never win; exhaustive search
always agreed with the model's pick. These are synthetic streams; real dumps decide (todo.md).


Compile-only checks passed: `zig build compile-kv-codec compile-glm-snapshots-host compile-glm-host`. Not compiled by
any step yet: core/root.zig's test block (`_ = kv_codec;`), which `test-cuda-host` reaches.
Run first, in this order:

1. `zig build test-kv-codec` — rANS edge lengths/tables, block and container round trips, refusals, Atlas candidates,
   ledger. The riskiest code is zig/src/core/kv_rans.zig (interleaved state order) and kv_xform.zig's T64 packing.
   The "within 1 % of order-0 entropy" check is a bet on 12-bit frequency quantization: if only that assertion
   fails, loosen its margin; the byte-identical round trip beside it is the real check.
2. `zig build test-glm-snapshots-host` — TF_GLM_KV_CODEC keep/restore byte equality, page charge and trim.
3. `zig build compile-glm-host` then the GLM module tests (cuda_ranks_test: settings and fingerprint layout 14).
4. `zig build kv-codec-bench-run` (or `kv-codec-bench`, then zig-out/bin/tf-kv-codec-bench --help) — overhead.

Pieces: core kv_rans/kv_codec/kv_xform/kv_ledger/kv_synth (+ kv.zig module root); GLM cuda_snapshot_codec.zig
(keep/restore/ledger/dump/dedup count), Plan.Part class/row tags, gate page agreement, kv_pool.Pool.truncate.
Env: TF_GLM_KV_CODEC=1 (needs TF_GLM_HOST_CACHE=1), TF_GLM_KV_CODEC_CANDIDATES=plain|atlas|atlas-exhaustive,
TF_GLM_KV_CODEC_THREADS, TF_GLM_KV_LEDGER=1, TF_GLM_KV_DUMP=dir.

## ClickHouse codec comparison (2026-10-31, x86, synthetic, one thread)

`zig build kv-codec-bench -Dkv-bench-libs=true` then `zig-out/bin/tf-kv-codec-bench --suite clickhouse`.
Reference codecs in zig/src/core/kv_ch_int.zig (Delta, DoubleDelta, T64, GCD) and kv_ch_float.zig (Gorilla, FPC,
ALP RD); LZ4/LZ4HC/ZSTD through the system libraries (bench only). Size ratios (plain = our planes + rANS):

| stream | plain | best ClickHouse | ZSTD(3) |
|---|---|---|---|
| f32_from_bf16 | 0.332 | T64>ZSTD 0.384 | 0.454 |
| f32_gauss | 0.832 | ALP RD 0.878 | 0.925 |
| bf16_gauss | 0.664 | ALP RD 0.752 | 0.777 |
| bf16_rows | 0.719 | T64>ZSTD 0.775 | 0.825 |
| e4m3 | 0.808 | ZSTD 0.813 | 0.813 |
| state_mix | 0.720 | T64>ZSTD 0.798 | 0.825 |

Gorilla, DoubleDelta and FPC store every block raw on every KV-like stream (Gorilla 0.56 on widened FP32 only);
LZ4 and Delta>LZ4 never compress. ZSTD(1) encodes about 4x faster than plain (0.66 vs 0.17 GB/s) at 8-11 % more
bytes. Not tested: ALP STD (needs decimal-origin floats), PCO (Rust library), ZXC and GDeflate (experimental),
SZ3 and Quantized (lossy), ZSTD_QAT/DEFLATE_QPL (obsolete, Intel hardware), AES (encryption only).
The reference unT64 is bit-at-a-time (0.02 GB/s decode): its speed is not ClickHouse's.
