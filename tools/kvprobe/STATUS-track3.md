# Track 3: drafter caching (status, 2026-11-01)

Tags: [O] observed, [C] calculated from observed values, [H] hypothesis, [U] untested. One capture, one log, one
session store; no replication.

## Contract

- Behaviour: score any draft source offline against the greedy picks a patches/0430 dump records, and estimate what
  Recall would land on the ring's agent traffic from the token ids the session store already keeps.
- Modules: `draft_eval.py` (Recall record mirror, sources, evaluator, CLI over a 0430 dump), `test_draft_eval.py`
  (synthetic 0430 dumps; numpy index equivalence; decode-walk accounting), `recall_yield.py` (request-log shape and a
  decode walk over session-store entries).
- Checks: `python3 -W ignore tools/kvprobe/test_draft_eval.py` -> 9 tests OK (host, and again on the serving node).
  The off-by-one test proves draft[0] is compared with row i's own pick: a source shifted one row either way scores 0.
- Cannot run here: the dump, the log and the store live on the serving node; runs there are CPU jobs (detached,
  `nice -n 19`, `ulimit -v 8000000`: the node had 15 GB available beside the serving ring).

## What mirrors what

`draft_eval.Record`/`RecallSource` follow `zig/src/core/lanes/recall_record.zig` (8-id keys, newest first, at most 64
candidates, extend to 64, copies stop at a segment mark), `recall.zig` (`Params.min_match` 16, copies of fewer than 2
ids dropped, `Rates` priors/probes behind `--gate`) and `recall_store.zig` (a request adds its prompt from the resume
point less a 64-id overlap, then its reply, then a mark). `ContextCopy` is `lanes/windows.zig` `copyProposal`'s own
context copy (`enter_match` 8), asked first. Copy width 7 is the GLM Zig engine's (`cuda_lanes.zig`: window 8 rows,
`max_copy` = width - 1). Difference: exact keys instead of hash buckets (the Zig walk can lose candidates to bucket
collisions; this one cannot), so Recall coverage here is an upper bound on the lookup side.

## Results

### Probe dump (4 requests, the shared 2048-token prose chunk S under 4 prefixes; 18,432 prefill rows) [O]

`draft_eval.py <dump> --k 4,8,16 --json probe.json` (and `--sources recall,engine --gate`). Rows are teacher-forced
prompt rows; "strict" counts a draft token only if the prompt also continued as the draft did.

| source | k | coverage | accepted / proposal (strict) | accepted / position | depth 1..4 acceptance |
|---|---|---|---|---|---|
| Recall (record of earlier requests) | 8 | 0.331 | 1.19 | 0.392 | 0.49 0.27 0.16 0.10 |
| own context copy | 8 | 0.027 | 1.57 | 0.042 | 0.63 0.36 0.23 0.14 |
| engine order (context, then Recall) | 8 | 0.356 | 1.21 | 0.430 | 0.50 0.27 0.16 0.10 |
| engine order, Recall priced (`--gate`) | 8 | 0.032 | 1.59 | 0.051 | 0.62 0.36 0.24 0.15 |
| token prior, last token | 8 | 0.839 | 0.42 | 0.351 | 0.28 0.08 0.04 0.02 |
| token prior, last 2 tokens | 8 | 0.577 | 0.81 | 0.466 | 0.42 0.19 0.10 0.05 |
| prior completion | any | 0 | - | - | (the probe decoded 1 token a request: nothing to draft from) |

Reading [C]: Recall finds S in the record on all 3 repeats (coverage 0.331 = 6,144 of 18,432 rows), but a copy of
human prose lands only where the model's greedy pick equals the text: about half a token per depth (the record's 48 to
50 % top-1-equals-next-token). Copying text the model did not write is the weak case; the record's 77 % cross-context
greedy agreement does not apply to it, because a copy proposes the text, not the other context's picks. Recall's
pricing gate correctly backs off here (coverage 0.002 for its own copies).

### Request log (4,499 requests) [O]

- Prompt tokens 242.1 M, of which 97.7 % resumed from a cache (lcp share 97.8 %; same-conversation lcp 96.7 %,
  other-conversation lcp 20.0 %): 5.55 M new prompt tokens, 1.04 M decode tokens (0.19 decode tokens per new prompt
  token). New prompt tokens per request p50 396, p90 1,594; decode p50 94, p90 558.
- Finish: tool_calls 2,388, stop 1,796, length 263, cancelled 52; 2,481 requests carry tools; thinking on for 4,230.
- The current drafter (DFlash2) lands 3.58 tokens a round, decode-weighted (p10 2.30, p50 3.35, p90 5.50 per request;
  3.64 on tool-call replies, 3.17 on stops).
- 1,017 conversations, turns p50 1, p90 4.4, max 371; 263 system prompts, 923 conversations share one with another.
- The log holds counts and hashes only: no ids or text, so repetition cannot be measured from it.

### Session-store decode walk (1,341 kept entries, 50.5 M ids; 1.60 M new ids past each entry's longest common
prefix with an earlier entry; 313,111 new assistant ids, 61,955 of them inside `<think>`) [O on this proxy]

`recall_yield.py --store <session store> --k 7` (and `--k 15`):

| k | source | copies | landed / copy | share of assistant ids landed by copies | copies landing nothing |
|---|---|---|---|---|---|
| 7 | own context | 21,262 | 4.44 | 30.2 % | 13.6 % |
| 7 | Recall record (only where own context had no copy) | 5,817 | 6.56 | 12.2 % | 1.7 % |
| 15 | own context | 14,671 | 6.49 | 30.4 % | 15.8 % |
| 15 | Recall record | 3,543 | 13.10 | 14.8 % | 2.3 % |

Caveats: the spans are the history as the template re-renders it, so reasoning from earlier turns is mostly absent
(62 k think ids against an estimated majority of the logged 1.04 M decode tokens being reasoning [H]); the walk
compares copies with the tokens actually written (sampled, not greedy); the store holds 1.6 M of the log's 6.6 M new
tokens (eviction and minimum entry size); an exact-prefix dedup cannot see a conversation re-sent with an edited
first 256 tokens, so a re-sent conversation counts as repetition, which is the regenerate case Recall is for.

## Recall's expected yield on this workload [H]

- Where it can land: Recall adds 12 to 15 % of the model-written answer and tool-call ids beyond what the stream's own
  context copy already lands (30 %), at 6.6 of 7 ids a copy, 98 % of copies landing something [O, proxy above]. At
  those landing rates Recall's own price (expected tokens > the stream's head, 3.6 here) is met in every band, so the
  gate should not suppress these copies [C from `recall.expected`: 0.94 an id over 7 ids -> 6.5 tokens a round].
- Round savings, bounded [C]: 5,817 copy rounds carry about 44 k ids (6.56 + 1 each); at DFlash2's 3.58 a round the
  same ids take about 12.3 k rounds, so Recall saves at most about 6.5 k of the about 87 k rounds the 313 k assistant
  ids would take on DFlash2 alone: up to 7 % fewer rounds on the visible answer and tool-call text. Repeated text is
  also DFlash2's easiest text, so the true saving is lower; round costs differ by window width.
- First production sample [O] (session record 11.16): 36 minutes of decode rows (39,560 rows, 32 requests) scored
  offline with `draft_eval.py`'s decode-only loader: replies to a re-sent prompt are copied whole (46 % of the
  window's rows, 15.5 of 16 accepted a proposal), ordinary agent replies have copyable spans at about 1 % of
  positions (replies only in the record, so a lower bound); the pricing gate withheld nothing.
- Session-store walk on the node [O] (06:41, 100 entries): own-context copies land 20.7 % of the 9,678 new assistant
  ids (3.84 a copy), Recall across entries lands 0 of them; a thin sample that agrees with the decode-row dump.
- Reasoning text: not measured (stripped from history). If reasoning repeats less than answers do [H], the decode-wide
  yield is below the visible-text figure.
- Precondition: Recall runs only in the Zig lanes engine; the ring serves the Python stack, which has no Recall. On the
  Zig GLM CUDA engine (`cuda_lanes.zig`: copies on, width 8), the gain needs a served A/B with `--recall` against
  `--recall-read-only` on an empty record (docs/recall.md "Benchmarks and gates") [U].

## What awaits a GPU or a new capture

- A 0430 dump with decode rows (`GLM53_TF_DRAFT_DUMP_WHAT` covering decode) on real agent traffic: the prior-completion
  source and the strict acceptance of copies against greedy picks (not sampled tokens) need it.
- The served A/B above on the Zig engine, tokens/s and rounds per reply, n >= 3, warm-up discarded.

## Commands

```
python3 -W ignore tools/kvprobe/test_draft_eval.py
python3 -u tools/kvprobe/draft_eval.py <0430 dump dir> --k 4,8,16 --json probe.json
python3 -u tools/kvprobe/draft_eval.py <0430 dump dir> --k 4,8,16 --sources recall,engine --gate --json probe-gate.json
python3 -u tools/kvprobe/recall_yield.py --log <requests.jsonl> --json log.json
python3 -u tools/kvprobe/recall_yield.py --store <session store dir> --k 7 --json full-k7.json
```

Outputs of this run are in the serving node's `~/kvprobe/track3/` (probe.json, probe-gate.json, log.json,
full-k7.json, full-k15.json and their .txt logs).

## Open decisions

- Whether to capture decode rows (0430 `WHAT` including decode) on production traffic for a week to score drafts
  against greedy picks; it costs dump disk and writer bandwidth on the ring.
- Whether the Zig engine's Recall A/B is worth a ring outage now, given the bounded 7 % round saving on visible text.
