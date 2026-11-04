# Exploration: branching DFlash2 drafters (a sub-drafter for the rows the main drafter misses)

Date 2026-11-04. Read-only exploration; no code changed, the engine untouched, nothing committed. Tags: [O] observed,
[C] calculated from observed values, [H] hypothesis, [U] untested. Scripts and outputs: `~/kvprobe/explore-branching/`
on spark0 (`stats.py`, `hidden.py`, `stats-1793687905.json`, `hidden-1793687905.json`, their logs; both run on the same
snapshot, `EXPLORE_RECORDS=6057`); the scripts are also in this session's scratchpad `explore-branching/`.

The idea (the user's words): the drafter gets the cheap rows right and misses the complex rows; is there signal in the
first rows it misses that could train a sub-drafter activated only for trajectories the main drafter cannot hit? The
refinement: draft the easy rows, decode the middle rows serially, and use what those rows turned out to be to inform
the drafter for the rest.

## 0. Summary

- The dump records no drafts, so DFlash2's misses are not visible; the proxy throughout is the model's own next-token
  distribution: a row is "hard" when its top-1 probability is below 0.9 (29.5 % of 1.02 M rows) [O].
- Hard rows cluster: P(hard | previous row hard) 0.57 against 0.18 after an easy row, runs of 2.3 rows, and the 8 rows
  after a hard row are harder than average (43 to 52 % easy against 70 %) [O]. The refinement's premise, that a decoded
  hard row opens an easy stretch, is the opposite of what the replies show.
- Hardness is a phase and content property: 54 % of reasoning rows against 15 % of answer rows; pending punctuation
  43 %, words 35 %, digits 17 %, special tokens 8 %. Between-request variance is 11 % of the total: a gate must act
  within a reply, which mindreader's admission-only design rules out [O].
- A causal gate signal exists: predicting that the pending row is hard from what the engine holds at draft time
  reaches AUC 0.82 from the previous row's distribution, token class, position and recent hard rate, and 0.904 with
  the previous row's final-normed hidden row (precision 0.74 at recall 0.69 on a 32 % base rate); the lift holds
  within the reasoning phase (0.69 to 0.81) and within the answer phase (0.80 to 0.93) [O, later requests held out].
- The gate buys little unless the sub-drafter is a much better drafter. Over the actual row sequences, with a baseline
  calibrated to the log's 3.58 tokens a round: a second candidate verified at every hard row adds 0.11 tokens a round
  with a perfect gate and 0.03 with a realistic one; suspending or shortening drafts at a hard frontier lowers tokens
  a round (2.2 to 3.2) and only pays through verify rows, which 0071 already prices per draft; a mixture head that
  landed half of the flagged hard rows would add 0.4 to 1.1 tokens a round [C].
- The dump is the wrong instrument: 97 % of its rows are sampled requests, which `auto` drafts with the MTP arm, not
  DFlash2 (patches/0010 docstring; RESULTS.md:1196), and it holds no outcomes. The measurement that decides is a
  per-round miss log at the 0430 decode hook (window, kept count, arm, per-draft probabilities; tens of bytes a round).
- Recommendation: closed as a project on this evidence; one cheap capture (the miss log, a day) and the offline
  comparison in section 7. If the misses are the reasoning clusters, the lever is T2 (the stack's planned re-fit of
  the block drafter) on reasoning rows, not a second drafter.

## 1. The sample and its caveats

| item | value | source |
|---|---|---|
| launch | `gx10-eb0e-1-1793687905`, index snapshot of 6,057 records at 2026-11-04 18:45 UTC | index.jsonl |
| rows / requests / hours | 1,023,011 decode rows, 3,454 requests, 36.1 h (record 11.20 scored 914,673 of them) | `stats.py` [O] |
| a row holds | the token fed, the final-normed hidden row (bf16, 4096), the top-32 log-probabilities and ids of the next-token distribution at temperature 1; no taps (`TAPS=0`), no drafts, no outcome | meta.json, patches/0430 |
| chaining | rows chained into requests by position (`draft_eval.decode_only`); the prompt is not in the dump, so position means position in the reply | |
| sampling | 345 requests (2.9 % of rows) are greedy end to end (3.2 % hard: canaries and probes); the rest are sampled at temperature 1, where the next token equals the top-1 on 88.3 % of rows (62 % of hard rows, 40 % where p1 < 0.5) | `greedy_vs_sampled_requests`, `greedy_equals_next` [O] |
| which arm drafted them | `auto` with `GLM53_TF_AUTO_FDRAFTS=7`, `MAX_DRAFT_ROWS=16`, lookup on (meta.json env); `auto` drafts sampled requests with the MTP arm (a:0.6:0.85) and DFlash2 only on greedy requests (0010 docstring; RESULTS.md:1196 "DFlash2 greedy rounds, MTP sampled"), so these rows were mostly MTP and lookup rounds | [O] |
| phases | the template opens `<think>` in the prompt; rows before a reply's first `</think>` are reasoning: 37.0 % of rows; answer 62.4 %; 0.7 % in replies without a close tag | `phase` [O] |
| hard | p(top-1) < 0.9: 29.5 % of rows (p1 < 0.5: 9.3 %); reproduces peaked.py's 70 / 60 / 9 % | `peaked` [O] |

Caveats. (1) "Hard for the model" is not "missed by the drafter": both drafters condition on more than the pending
token and land rows the proxy calls hard. (2) Under the keyed-noise sampler a draft is kept only when it equals the
serial sample, so acceptance at a hard row needs the drafter to match the distribution, not the top-1. (3) The 3.58
tokens a round is the request log's decode-weighted figure across arms (STATUS-track3), a different population from
the dump's rows. (4) One launch, one node, 36 hours; no replication. (5) The classifier's held-out set is the later
30 % of requests by dump order (time), not a random split.

## 2. Mindreader's angle (Q1)

| finding | tag | file:line |
|---|---|---|
| Routing between proposers deferred: "Routing between multiple proposer types is a later extension because switching can require new draft KV state" | [H] | RESEARCH.md:132 |
| The timing rule: "States produced while verifying a candidate cannot retroactively help draft that candidate" | rule | RESEARCH.md:154 |
| R1: "fixing an early recurring mismatch can be more valuable than improving an equally frequent mistake at the end of a draft" | [C] | FURTHER_RESEARCH.md:45 |
| R2, learn where to branch: trees at "an uncertain identifier or tool argument"; "Treat unsupported tree execution as an implementation prerequisite" | [H], never built | FURTHER_RESEARCH.md:55-74 |
| The difficulty confound: a prompt-only predictor "may explain much of a pooled result"; a within-task control is required | [H] | FURTHER_RESEARCH.md:290-300 |
| Content class is the largest measured axis: category means from 1.70 (creative) to 3.38 (math), "Max/min = 1.996", F(7, 8) = 1023.75, pre-registered RULE IN | [O] | docs/research-reports/58-prompt-mix-acceptance-result.md:58-73 |
| Deep positions collapse on hard categories: a_1 0.69 to 0.90, a_5 0.10 to 0.49 | [O] | 58:101-104 |
| Reasoning against answer: D = -0.048, CI [-0.137, +0.041], INCONCLUSIVE; "Prompt content varies acceptance more than phase does" | [O] | 56-answer-vs-reasoning-acceptance-result.md:66-86 |
| The first draft row is the hardest: marginal acceptance 0.78 to 0.37 falls with depth, survival-conditional 0.78 to 0.83 rises | [O], one boot | 83-mindreader-x0-x2.md:105-108 |
| Content-class DFlash2 design: classify at admission, train a prose specialist; end state "Class-conditioned single drafter ... a class embedding"; resident multi-drafters rejected, the draft phase "roughly doubles"; rule: "The current request's realized acceptance is an outcome, never a feature" | [H], never run | archive/docs/design-content-class-dflash2.md:101, 505-506 |
| X1, the one decision-time test of hidden features, matched a constant on one boot; "the scientific stop condition is withdrawn"; tree lanes killed (D81) then "geometry benefit/cost remains untested" (D94) | decisions | decisions.md:59, 290, 313, 405 |

How it relates. Mindreader asked the admission-side question: which drafter or width to assign a request from its
content class, with the request's own acceptance barred as a feature. It measured that content moves acceptance
two-fold and that misses concentrate at the first draft position, but it never measured a per-row miss signal, never
trained a specialist, and never ran its own confound control. This exploration is the within-reply version: whether
the row-level signal the engine holds at draft time can gate a second proposal. Its two warnings carry over: the
timing rule (section 4.3) and the difficulty confound (section 3.4, where the gate is scored inside each phase).

## 3. Is there signal at a miss? (Q2)

### 3.1 Where the hard rows are

| cut | rows | hard share | source |
|---|---|---|---|
| all | 1,023,011 | 0.295 | [O] |
| reasoning / answer phase | 378,221 / 638,027 | 0.539 / 0.148 (mean p1 0.77 / 0.94) | `phase` [O] |
| position 0-16 / 16-64 / 64-256 / 256-1k / 1k-4k / 4k+ | 55k / 154k / 310k / 319k / 164k / 20k | 0.21 / 0.24 / 0.26 / 0.31 / 0.38 / 0.46 | `by_position` [O] |
| first / second / last third of replies of 30+ rows | 342k / 340k / 339k | 0.344 / 0.315 / 0.228 | `by_third_of_reply` [O] |
| per request (2,804 requests of 64+ rows): hard share p10 / p50 / p90 | | 0.04 / 0.16 / 0.45; between-request variance 11 % of the total | `per_request_hard_share` [O] |

Pending-token class (the token just committed, known at draft time) against the hardness of the row it feeds, and the
predicted token's class (descriptive) [O]:

| class of the pending token | share of rows | hard | class of the token predicted | hard |
|---|---|---|---|---|
| word (alphabetic) | 0.515 | 0.348 | word | 0.380 |
| mixed (word pieces with digits or symbols) | 0.152 | 0.207 | mixed | 0.163 |
| code symbol (brackets, operators, `_`) | 0.138 | 0.285 | code symbol | 0.260 |
| digit | 0.064 | 0.168 | digit | 0.136 |
| punctuation (`. , ! ? ' "`) | 0.062 | 0.431 | punctuation | 0.388 |
| special (`<|...|>`, think / tool-call tags) | 0.042 | 0.080 | special | 0.042 |
| space-only / newline | 0.028 | 0.16 / 0.21 | | 0.15 / 0.29 |

The class table is a weak gate on its own (AUC 0.61, section 3.4). The phase split is the strong one, and it is the
confound: a gate that only learned "reasoning is hard" would score well pooled and do nothing inside a phase.

### 3.2 Hard rows cluster

| statistic | value | source |
|---|---|---|
| P(next row hard | this row hard) / (| this row easy) / base | 0.573 / 0.179 / 0.295 | `transition` [O] |
| hard runs: count, mean, median | 129,005, 2.34, 2 | `hard_runs` [O] |
| share of hard rows in runs of 1 / 2-3 / 4-7 / 8+ | 0.21 / 0.32 / 0.31 / 0.15 | [O] |
| easy runs: count, mean, median | 131,560, 5.48, 2 | `easy_runs` [O] |
| share of all rows in easy runs of 4+ / 8+ | 0.56 / 0.46 | [O] |
| next row hard after a hard row whose greedy pick was taken / where another token was sampled | 0.552 / 0.608 | `next_after_hard_by_pick` [O] |

Reconciliation with the brief's "easy runs average 2.39 rows": peaked.py counts one run per hard row (zero-length runs
between consecutive hard rows), so its 2.39 is easy rows per hard row, (1 - 0.295) / 0.295 = 2.39 [C]; maximal easy
runs within a request average 5.48. Its "47 % of rows in easy runs of 8+" matches (46 %).

### 3.3 What a hard row's distribution looks like (what a branch could recover)

| rows | n | next token is top-1 / in top-2 / top-3 / top-4 / top-8 | mean p1 / p2 |
|---|---|---|---|
| easy (p1 >= 0.9) | 717,688 | 0.997 / 1.000 / 1.000 / 1.000 / 1.000 | 0.993 / 0.005 |
| hard (p1 < 0.9) | 301,869 | 0.622 / 0.844 / 0.918 / 0.952 / 0.992 | 0.601 / 0.214 |
| 0.5 <= p1 < 0.9 | 206,458 | 0.725 / 0.926 / 0.971 / 0.987 / 0.999 | 0.704 / 0.195 |
| p1 < 0.5 | 95,411 | 0.400 / 0.666 / 0.805 / 0.878 / 0.977 | 0.378 / 0.255 |

`next_in_topk` [O]. At a hard row the second candidate is the actual token 22 % of the time (0.844 - 0.622); a 2-way
branch there recovers at most that, a 4-way branch 33 %.

### 3.4 A gate from the hidden row (the classifier)

Setting. At draft time the engine holds the hidden rows and distributions of every verified row up to j-1 and the
pending token j; row j's hidden and distribution are what the next verify forward computes (the timing rule). So the
gate question is: predict hard(row j) from row j-1 and token j. The same-row classifier (hidden[j] predicting its own
p1) is run only as a sanity check, since the logits are a linear function of that row.

Data: 160,100 rows from 942 whole flush segments drawn at random, 159,158 (j-1, j) pairs inside a segment; training on
requests with index < 2,475 (the earlier 70 % by dump order), test on the later 48,503 pairs, base rate 0.323. Hidden
rows projected by PCA fitted on 24,000 training rows (256 components keep 55 % of the variance, 64 keep 33 %).
Logistic regression on standardised features; a gradient-boosted model as a nonlinear check. `hidden.py` [O].

| features available at draft time | AUC | AP | precision at recall 0.5 | flag the top 30 %: precision / recall |
|---|---|---|---|---|
| position in the reply only | 0.563 | 0.38 | 0.37 | 0.39 / 0.36 |
| pending token class only | 0.605 | 0.38 | 0.38 | 0.38 / 0.72 |
| row j-1's distribution (p1, entropy, tail mass, hard, p2) | 0.801 | 0.63 | 0.65 | 0.62 / 0.58 |
| + recent hard rate (8 rows) | 0.821 | 0.64 | 0.66 | 0.64 / 0.59 |
| + token class + position | 0.823 | 0.64 | 0.66 | 0.64 / 0.60 |
| + hidden[j-1] PCA-256 | **0.904** | 0.77 | 0.78 | 0.74 / 0.69 |
| hidden[j-1] PCA-256 alone | 0.903 | 0.77 | 0.78 | 0.74 / 0.69 |
| GBM, all causal features (PCA-64) | 0.894 | 0.75 | 0.77 | 0.73 / 0.68 |
| GBM, causal scalars without the hidden row | 0.837 | 0.66 | 0.68 | 0.65 / 0.61 |
| SANITY same-row hidden[j] PCA-256 | 0.950 | 0.88 | 0.90 | 0.83 / 0.77 |

Within a phase (the confound control; logistic, test rows):

| features | reasoning (21,190 rows, 54 % hard): AUC / AP | answer (26,634 rows, 14.5 % hard): AUC / AP |
|---|---|---|
| causal scalars (distribution, recent, class, position) | 0.690 / 0.69 | 0.803 / 0.42 |
| + hidden[j-1] PCA-256 | 0.814 / 0.80 | 0.925 / 0.63 |

Geometry. The norm carries nothing (easy 87.1 +- 5.0, hard 87.5 +- 6.2: the row is RMS-normed). A 32-centroid k-means
on PCA-64 of hidden[j-1] gives clusters whose hard rate runs from 0.01 to 0.68 (std 0.22), and the cluster rate alone
scores AUC 0.79 on the test rows (0.86 for same-row clusters): a set of directions, not a few outlier rows [O].

Reading [C]. The previous row's hidden state predicts that the next decision is uncertain well beyond the previous
distribution, the recent miss rate and the token class, pooled and within each phase. That is enough to train a gate.
It is not evidence that the gate's positives are rows the drafter misses (section 5), nor that the best real gate
feature, the drafter's own candidate margin that 0071 already calibrates per position, would not carry the same
information; that comparison needs the miss log.

## 4. Trajectory conditioning: decode the hard rows, then draft the rest better (the refinement)

### 4.1 Phase structure of real replies

| tenth of the reply (replies of 30+ rows) | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 |
|---|---|---|---|---|---|---|---|---|---|---|
| hard share | 0.34 | 0.35 | 0.35 | 0.34 | 0.32 | 0.31 | 0.29 | 0.26 | 0.23 | 0.17 |

`hard_by_tenth_of_reply` [O]. Hard rows are front-loaded and spread, falling steadily to the end: the reasoning-then-
answer shape (54 % against 15 %), not an easy front, a hard middle and an easy rest. Inside a reply they come in
clusters of 2.3 (section 3.2): 21 % of hard rows are isolated, 15 % sit in clusters of 8 or more. The "40 / 20 / 40"
ordering by confidence does not map onto positions; it maps onto phases and clusters.

### 4.2 Does knowing the hard row's token reduce uncertainty downstream?

Rows i+1 .. i+8 after a row of each kind, within the request; the model's own next distribution is the oracle for what
a re-anchored drafter could know [O, `rows_after`]:

| after a row that is ... | +1 mean p1 / easy | +2 | +4 | +8 | unconditional |
|---|---|---|---|---|---|
| hard (301,869 rows) | 0.753 / 0.427 | 0.774 / 0.471 | 0.784 / 0.494 | 0.794 / 0.518 | 0.877 / 0.705 |
| hard, p1 < 0.5 (95,411) | 0.703 / 0.338 | 0.737 / 0.401 | 0.754 / 0.435 | 0.769 / 0.467 | |
| hard and a non-top-1 token was sampled (113,987) | 0.735 / 0.392 | 0.759 / 0.444 | 0.772 / 0.472 | 0.784 / 0.498 | |
| easy (717,688) | 0.930 / 0.821 | 0.921 / 0.803 | 0.916 / 0.793 | 0.912 / 0.782 | |
| at the end of an easy run of 8+ (328,446) | 0.981 / 0.947 | 0.978 / 0.939 | 0.974 / 0.929 | 0.969 / 0.918 | |

Reading [C]. After a hard row the following rows are harder than average, not easier: 43 % easy at +1, recovering only
to 52 % by +8, against 70 % unconditional and 82 % after an easy row. The harder the row and the more surprising its
token, the harder what follows. A hard row's decoded token therefore mostly carries "this stretch is uncertain", which
the previous-row features already encode (section 3.4); it does not open an easy stretch that a re-anchored drafter
could exploit. The reliably easy rows ahead are those after long easy runs (95 % easy at +1), where the drafter
already lands its block.

### 4.3 What the drafter sees today, and what a "sub-drafter informed by the middle rows" would add

Both drafters are re-anchored on every committed row each round: the MTP head reads the committed token and the
final-normed row (`decode.absorb`); DFlash2 reads every committed position's taps (`Drafter.add_taps` after each
verify: the mean of the four residual streams after layers 5, 14, 24, 33, 42) and the pending token, and proposes a
block of 7. A hard row's decoded token and its taps enter the next round's draft by construction. What neither
conditions on is the outcome: nothing knows that the previous block missed, except 0071's per-position calibration
of the drafter's own probabilities (`depth.py`, `cal.record`), which shortens the chain. The candidate additions:

| addition | what it adds beyond re-anchoring | bound from the dump |
|---|---|---|
| (a) a head specialised on post-hard-row continuations, trained on the rows after misses | a proposal distribution fitted to the uncertain stretches (the rows after a hard row are themselves 48 to 57 % hard) | section 5.1: +0.4 to +1.1 tokens a round only if it lands half of the flagged hard rows, which no measured drafter does on prose (W11 a_1 0.68) |
| (b) a wider draft at the row after a hard one | a second candidate where the actual token is the model's second choice (22 % of hard rows) | +0.11 tokens a round with a perfect gate, +0.03 realistic, section 5.2 |
| (c) a policy that suspends drafting through a hard cluster and resumes with a longer window | fewer verify rows spent on misses; tokens a round fall | section 4.4: 2.2 to 3.2 tokens a round against 3.58, rows a token 1.29 to 1.58 against 2.22 |

### 4.4 The policy simulation

A decode walk over the actual row sequences of every request. Drafter model: at pending row i it proposes up to 7
drafts; draft d lands with probability h_easy if row i+d-1 is easy and h_hard if it is hard; the chain stops at the
first miss; the round commits hits + 1. The baseline is calibrated so that fixed 7-draft rounds make the log's 3.58
tokens a round: h_easy = 1 with h_hard = 0.259 (0.97 / 0.309; 0.95 / 0.345). Policies decide the depth from whether
the row before the pending row was hard (its distribution produced the pending token, so the engine knows it); P6
cheats with the pending row's own hardness. The time estimate uses round cost = 33 ms + 5 ms a verify row, fitted
to W11's production rows (prose 46.2 tok/s at 2.39 tokens a round and about 3.4 rows; agent 83.6 tok/s at 6.34 and
about 7.5 rows) [C, rough; the real per-row cost is what 0071's calibration carries].

| policy | tokens a round | rows a round | rows a token | rounds | est. ms a token |
|---|---|---|---|---|---|
| P0 fixed 7 (the calibrated baseline) | 3.580 | 7.95 | 2.22 | 285,748 | 20.3 |
| P1 serial round when the previous row was hard, else 7 | 2.372 | 3.68 | 1.55 | 431,308 | 21.7 |
| P2 1 draft when the previous row was hard, else 7 | 2.778 | 4.04 | 1.45 | 368,284 | 19.1 |
| P3 3 drafts when the previous row was hard, else 7 | 3.214 | 5.07 | 1.58 | 318,279 | 18.2 |
| P4 serial at a hard frontier, 3 on the first easy row after, then 7 | 2.234 | 2.87 | 1.29 | 457,978 | 21.2 |
| P5 fixed 3 | 2.713 | 3.99 | 1.47 | 377,051 | 19.5 |
| P6 oracle: serial when the pending row itself is hard, else 7 | 3.137 | 4.54 | 1.45 | 326,080 | 17.8 |

`policies` [C over O]. Every causal gating policy lowers tokens a round, because the row after a hard row is still
easy 43 % of the time and a suspended round forgoes those; the policies pay only through verify rows, and that saving
(P3: about 10 % of the rough time model) is the kind 0071 already extracts per draft from the drafter's own
probabilities (W11: mean DFlash2 depth 2.8 on prose under the depth cut). "Resume with a longer window" has nothing to
resume with: the block is 7 wide, and DEEP-VERIFY's 16-row simulation found -0.1 to +1.5 % outside copy cells.

### 4.5 What the simulation assumes, and what replaces the assumption

Assumed: hits depend only on the row's hardness, independently per row, with no benefit from re-anchoring beyond
what the real drafters already get (it is inside the 3.58 calibration). Not modelled: the depth cut, the lookup arm,
the MTP arm on sampled requests, shared batch rounds, keyed-noise acceptance at sampled rows. To replace the
assumption, production has to log per round what the 0430 decode hook already receives in `batch.Batcher._verify`
(`(state, off, keep, win[:keep], pos)`): the whole proposed window `win` (not only the kept prefix), `keep`, the arm
(m / f / l) and 0071's per-draft probabilities (`DepthOptimizer.used`). That is the per-row accept/reject outcome with
the drafter's confidence, about 40 bytes a round, 7 MB a day at this launch's rate [C].

## 5. What a sub-drafter would be, and the bound on each (Q3)

Simulated over the actual row sequence with the calibrated baseline (h_easy = 1, h_hard = 0.259; the other two
calibrations in brackets). `simulations` [C over O].

### 5.1 A mixture of drafters gated by the hidden row (a second head trained on hard-row trajectories)

| hard-row hit rate after the sub-drafter | tokens a round (h_easy 1 / 0.97 / 0.95) | rounds saved against 3.58 |
|---|---|---|
| 0.26 (no change) | 3.58 / 3.58 / 3.58 | 0 |
| 0.5 | 4.57 / 4.32 / 4.17 | 22 / 17 / 14 % |
| it lands half of what the main drafter missed | 5.26 / 5.09 / 4.98 | 32 / 30 / 28 % |
| 0.75 | 6.02 / 5.64 / 5.40 | 40 / 37 / 34 % |
| 1.0 (every hard row) | 7.91 / 7.34 / 6.99 | 55 / 51 / 49 % |
| gate recall 0.5, precision 0.5; it lands half of the flagged hard rows | 4.31 | 17 % |
| gate recall 0.7, precision 0.45; same | 4.65 | 23 % |
| gate recall 0.3, precision 0.6; same | 3.99 | 10 % |

Two readings of the brief's "DFlash2 is at the cheap-drafter ceiling" [C]. An easy-rows-only drafter (hit iff
p1 > 0.9) makes 2.87 tokens a round in the 7-draft walk (the brief's 3.4 is the uncapped 1 + 2.39), below the log's
3.58, so under this proxy the production drafters already land about a quarter of the hard rows; but the populations
differ (the dump's rows are sampled, MTP-arm traffic; W11's sampled baselines are prose 2.28, code 3.32, agent 5.30
tokens a round), so the comparison is indicative only. The headroom is real (every hard row landed would double the
rate) but it is "a better drafter on uncertain stretches", which is T2 (DRAFTER-TRAINING.md section 5, expected +0.4
to 0.7 tokens on prose rounds) with the gate choosing the head. Costs not in the table: a second resident drafter
(incoai's layout is 1.1 B parameters at 4 bits) and a second block forward on gated rounds; mindreader rejected
resident multi-drafters on the draft phase "roughly doubling" (design-content-class-dflash2.md:506). The class-
embedding form (one drafter conditioned on a gate vector) avoids both and is the form to test first.

### 5.2 A wider draft at the uncertain row (a 2-way tree at the first draft position), and a top-2 head

DFlash2 already computes 16 candidates a position (`Drafter.candidates`, `top_k`) and chains one; the second candidate
at the first position is free to produce and costs one verify row (block 6 behind it, 8 rows in all), which the
Python path's chain verifier cannot take (`tokens = [out[-1]] + drafts`). TensorFold's Zig core has tree machinery
(`zig/src/cuda/tree_round.zig`, `lanes/config.zig` `head_trees`, DeepSeek's `draft/branches.zig`); whether the GLM
CUDA lanes path would verify a non-chain tree is [U] (the brief says it refuses trees today).

| gate | tokens a round | gain |
|---|---|---|
| perfect: a second candidate at every hard row | 3.694 | +0.11 (3.2 %) |
| perfect, only where p1 < 0.5 | 3.634 | +0.05 |
| recall 0.5 / precision 0.5 | 3.610 | +0.03 |
| recall 0.7 / precision 0.45 | 3.611 | +0.03 |
| recall 0.3 / precision 0.6 | 3.603 | +0.02 |

The bound is low because the branch adds a token only on rounds whose first draft misses with the actual token as the
second choice (22 % of hard rows), and the shorter block behind it costs on rounds that would have run long. A 4-way
branch recovers 33 % for three rows. A head that predicts the model's top-2 for the verify is the same arithmetic
with the candidate from the MTP head: at most +0.11, and the top-2 is only 84 % of hard rows' actual tokens under
temperature-1 sampling. Both closed by the row statistics [C].

## 6. Training data (Q4)

| item | state |
|---|---|
| in the dump now | 1.28 M decode rows over 55 h in the two long launches (1.02 M in this one): token, final-normed hidden row, top-32 log-probabilities and ids; a distillation set for the MTP head (T1, DRAFTER-TRAINING section 4) and for a gate (section 3.4 trains on 111 k pairs in 30 s) |
| missing: the drafts and misses | not recorded; the 0430 hook sees the window, `keep` and the arm and writes only the kept rows. Without it "hard row" is a proxy and a sub-drafter has no miss-labelled trajectories |
| missing: the taps | `TAPS=0` in every launch (meta.json); DFlash2 training or re-fit needs them: bf16 taps raise a record from 8,388 to 49,348 B (fp8 28,888 B); at this launch's 28 k rows an hour that is 1.2 GB an hour (0.6 GB fp8) [C from the 0430 docstring] |
| missing: the prompt rows | `WHAT=decode`; the block drafter's context window at training time needs prefill rows for the first 2,048 positions (DRAFTER-TRAINING 5.1) |
| cheapest capture to add | the per-round miss log at the 0430 decode site (section 4.5): `win`, `keep`, arm, `DepthOptimizer.used`; about 40 B a round, no new GPU work, replies unchanged; a day gives about 175 k rounds with each arm's per-draft probabilities and outcomes [C] |

## 7. Recommendation (Q5)

Closed as a project on this evidence; one cheap capture decides whether to reopen it.

- The gate signal exists (AUC 0.90 causal, within-phase 0.81 / 0.93) and is cheap to serve (a 256 x 4096 projection
  and a logistic head on the previous row). But every use of it that does not come with a better proposal
  distribution is small: the 2-way branch tops out at +3 %, the suspend / shorten policies trade tokens a round for
  rows and duplicate 0071, and the trajectory premise (hard rows open easy stretches) is the opposite of what the
  replies show.
- What would pay is a sub-drafter that lands hard rows, which is a drafter-quality question on reasoning and prose
  rows; the stack has a plan and a trainer for it (T2), and the dump cannot score it without the drafter's misses.
- The one measurement: log the rounds (window, keep, arm, per-draft probabilities) for a day at the 0430 decode hook,
  then score offline (1) the share of first-position misses the hidden-row gate flags against the share 0071's own
  probability cut flags at equal recall, per arm, and (2) how clustered the misses are (P(miss | previous miss)) and
  how they split by phase. If the gate adds recall over the drafter's confidence on isolated, in-answer misses, a
  conditioned head is worth a T2-sized experiment; if the misses are the reasoning clusters, the lever is T2 on
  reasoning rows and this idea folds into it.

## Appendix: commands (spark0, read-only on the dump, outputs under `~/kvprobe/explore-branching/`)

```
export EXPLORE_RECORDS=6057   # the index snapshot both scripts used
systemd-run --user --scope -p MemoryMax=4G -q nice -n 19 taskset -c 5-9 python3 -P -u stats.py <launch dir> <tokenizer.json> stats-1793687905.json   # 10 s
systemd-run --user --scope -p MemoryMax=4G -q nice -n 19 taskset -c 5-9 python3 -P -u hidden.py <launch dir> <tokenizer.json> stats-1793687905.json hidden-1793687905.json 160000   # 33 s
```

Tokenizer: the HF-cached `RedHatAI/GLM-5.3-Flash-NVFP4` tokenizer.json on the node (154,856 ids; the dump's vocabulary
is 154,880); `</think>` is id 154842.
