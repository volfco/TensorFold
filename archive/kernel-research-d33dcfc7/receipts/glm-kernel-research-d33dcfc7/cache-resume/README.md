# GLM research: cache resume versus fresh execution

Cache correctness passed for all three optional modes on the two assigned GB10 nodes.
The test completed on 2026-10-30 at 20:56:33 UTC with cache exit **0**. All six model
phases cleaned up, both nodes had no model/GPU-test processes left, and both GPU locks
were independently checked free.

| Mode | Warm cached tokens | Fresh cached tokens | Full output hashes / rendered IDs |
| --- | --- | --- | --- |
| KDA vector `(1,0,0)` | 0, 2048, 2048, 4096 | 0, 0, 0, 0 | 4/4 corresponding pairs equal |
| N256 `(0,1,0)` | 0, 2048, 2048, 4096 | 0, 0, 0, 0 | 4/4 corresponding pairs equal |
| Fused epilogues `(0,0,1)` | 0, 2048, 2048, 4096 | 0, 0, 0, 0 | 4/4 corresponding pairs equal |

The four frozen requests are two turns, each drafted and plain. They use a public synthetic
system guide and synthetic tool results, temperature 0, seed 1729, thinking disabled, and
at most 32 output tokens. Rendered prompts contain 3792 and 5038 tokens; completions contain
30 and 32 tokens. Canonical request bodies were generated once and reused in every mode.
Across warm/fresh servers this gives **24 replies**, **9 positive reuse witnesses**,
**12 strict integer zero fresh counters**, and **54 saved client HTTP responses** including
tokenization. Every full 64-character output-token digest, completion length, request-body
hash and rendered token array matches. Drafted/plain output also matches within each turn.
This is bounded correctness coverage, with no performance conclusion.

Each warm server used a 1 GiB cache; each fresh server used exactly zero. Context remained
131072 and parallelism 4. Actual native-process user, unlimited soft/hard memlock, effective
feature/cache settings, two-HCA RoCE transport and model identity were checked before requests.
The memlock guard raises only its owned launcher shell's limit and leaves native execution
under the same unprivileged user. All 12 rank readiness/cleanup proofs are retained in
`readiness-cleanup.json`; full raw process records remain outside Git.

## Overall contract remains red

The independent cache handoff was explicitly authorized after all model audits and focused
research checks passed. It retained the completed post-contract outcomes unchanged:

- `kernel-contract-live`: exit **1**, 183 resource problems.
- `kernel-contract`: exit **1**; 446 numerical registrations pass, zero new gate failures,
  six declared known findings; all 321 controls caught; 183 resource problems.

`summary.json` and `qualification.json` preserve `overall_contract_passed: false`, exact
summaries and original log hashes. All 17 selected research resource records and all focused
research positives/controls remained mandatory. The [post-contract receipt](../../glm-kernel-post-contract-d33dcfc7/README.md)
details the unresolved findings. Receipt-verifier success describes evidence consistency.

## Provenance and privacy

`provenance.json` pins the tested native binary, runner, original tools, reviewed research
sources, checkpoint revision and separate raw archive. The tested `contract_limits.zig` hash
remains `49d3db427b3bb04e1ec23ef0c727a91670cb5634e1496b0b1cd79393b6f700eb`.
A later comment-only revision is recorded separately in the post receipt; no binary was
rebuilt or GPU retest claimed for that comment.

Raw request/response payloads, token arrays, process IDs, node/interface identifiers and
machine paths are excluded from Git. `privacy-review.json` records the review. Complete raw
evidence, original executable tools and tested source snapshots are retained separately as
`paired-20261030-r2-cache-resume-raw.tar.gz` (SHA256 in `provenance.json`).

## Reproduce without a GPU

From this directory, using Python 3.11 or newer:

```sh
python3 -B reproduce.py --selftest
python3 -B reproduce.py --raw-archive /path/to/paired-20261030-r2-cache-resume-raw.tar.gz
python3 -B tools/cache-reuse.py selftest
python3 -B tools/cache-resume-controller.py --selftest
python3 -B tools/handoff-selftest.py
python3 -B tools/startup-selftest.py
python3 -B tools/offline-e2e.py --source ../../../../tools/zig/reuse_served.py
```

The raw-archive command checks its hash, safely extracts it into a temporary directory,
verifies every original manifest entry, rederives all seven reviewed JSON artifacts, and
reruns the independent raw HTTP audit. These commands start no GPU job or network request.
`audits.json` contains all per-request output/request digests and typed counters;
`comparisons.json` contains the three original comparison results.

## Repeat the live protocol

The `tools` copies retain the tested algorithm, checks and cleanup policy. Their limited
configuration/privacy revisions and original/new hashes are in `tool-revisions.json`.
Configure SSH alias `worker`, stage the tools on both nodes under the task root, and provide
node-local `cache-node.env` with `NIGHT_MODEL` (the pinned checkpoint), `NIGHT_NCCL_COMM_ID`,
`NIGHT_SOCKET_IFNAME`, and `NIGHT_ROCE_HCA`. The portable launcher reads this private config;
its original tested form remains in the raw archive. Use the paired-campaign layout and
qualification receipts described by `cache-handoff.py --help` and the parent receipt.

`cache-handoff.py --prepare` creates a **disabled** authorization template and validates the
actual focused/preflight evidence. Its default mode only prints a plan. An explicitly reviewed
and enabled scoped authorization plus `--execute` is required to wait for complete model/post
receipts, idle nodes and both free locks, then run the six cache phases. Red/timeout post outcomes
remain attached to the cache result. `cache-wire-audit.py` and `cache-final-collect.py` provide
the independent final HTTP/cleanup audit; the collector requires the same recorded run layout.
