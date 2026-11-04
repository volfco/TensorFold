# Explore: can a cache entry store the drafter's view of a row instead of the row?

2026-11-04. Research exploration only: no code changed, no engine run, nothing committed. Tags: [O] observed,
[C] calculated from observations, [H] hypothesis, [U] untested. Scripts and their outputs are on spark0 under
`~/kvprobe/explore-drafter-cache/` (`reqlog.py`, `queue.py`, `fit_hidden.py`, `fit-sweep.json`, `*.txt`, `*.log`);
everything ran read-only on the CPU beside production (numpy, 4 GB cap, cores 5-9, about five minutes in all).

The idea (the user's words): "if the drafter is able to accurately predict the easy rows, can the cached entry only
store the drafted token information? Assuming the cached data from the drafter is smaller than the full cache entry
for the row. Maybe the drafter cache entry plus a small correction value to precisely recompute the token if needed."

## Verdict in one paragraph

The literal form is closed by structure, not by measurement. A kept entry's rows are a deterministic function of the
token sequence, so the token itself (4 bytes, already implied by the conversation) is the smallest entry that
"recomputes the row if needed"; that entry is the no-cache baseline, re-prefill, which costs about 0.85 to 1.8 ms of
GPU time a token on this ring against the 12 KB a rank the rows cost [O, section 1]. Whether a drafter would have
predicted the token adds no information about the row, and the drafter (DFlash2) does not compute anything from
tokens alone: its inputs are the target's own residual-stream taps, which exist only after the forward pass the entry
was meant to avoid [O, section 2]. A "small correction" cannot be small for bf16 rows: the best predictor measured here
(a linear map from the target's final hidden row, one of the drafter's own inputs) leaves 29 to 90 % of a layer
input's energy on unseen text [O], and session record 11.18 already showed that even a rank-128 prediction from the
rows themselves (21 % energy left) saves 0.2 to 0.3 of a value's 10.6 bits because the mantissa noise remains. A
lossy correction is quantization of the row, the sibling report's subject. The recurrent-state variant (tokens plus a
coarse state, replay the rest) was measured and refuted in 11.10 and 11.14. The one reading that survives is
"easy stretches at lower fidelity": it is bounded at about 4 to 7 % of an entry (a quarter of uniform fp8 rows) and
one block-perturbation experiment, run as a rider on the fp8 quality gate, decides it (section 4).

## 1. Information content: what a token buys and what its rows cost

What a kept GLM-5.3-Flash entry holds, rank 0 of the native ring, from the dump manifests (`kvdump.py` classes;
`~/kvprobe/native/dump/glm-rank0-keep*.plan`, 2026-11-02; bf16 latent, TP=2, 11 DSA caches, `mtp_len 0`) [O]:

| kept position (tokens) | prefix (KDA states, conv, scratch) | latent rows | index rings | pooled keys | entry a rank | bytes a token |
|---|---|---|---|---|---|---|
| 2,048 (keep1) | 163,130,624 | 23,068,672 | 11,534,336 | 1,447,424 | 199,189,504 | 97,260 |
| 8,192 (keep4) | 163,130,624 | 92,274,688 | 23,068,672 | 5,772,800 | 284,255,232 | 34,699 |
| 20,480 (keep9) | 163,130,624 | 230,686,720 | 23,068,672 | 14,423,552 | 431,318,016 | 21,061 |

Reading [C]: the per-token part is 11,968 B a rank (11 caches x 1,024 B of bf16 latent + 11 x 64 B of pooled keys;
the plan's 13,056 B/token layout figure is the same sum over 12 caches, the MTP cache included). The prefix class is
fixed (163 MB whole-head, 92 MB in the live-half form of 11.19) and the rings are capacity-sized (4,096 rows a cache),
so below about 14k tokens the entry is mostly the recurrent state, not rows. Rank 1's keeps hold the same latent
bytes (`~/kvprobe/receipt-half/dump-both-r1/glm-rank1-keep{1..4}.plan`: latent = pos x 11 x 1,024 at 2,048 to 8,192
tokens, pooled and ring equal to rank 0's) [O], so the rows are held on both ranks: 23.9 KB a token on the ring [C].
A token id is 4 bytes; the rows are about 3,000 times larger a rank.

What re-prefill costs. Production's request log (`/glm53-tf/sessions/requests.jsonl`, 9,100 requests without error,
2026-10-31 to 11-04, the Python stack with fp8 latent KV; `reqlog.py`) [O]:

| new prompt tokens a request | n | prefill_tps p10 / p50 / p90 | prefill_s p50 | cold rows (cached 0): n, tps p50 |
|---|---|---|---|---|
| 0 to 64 | 1,055 | 54 / 84 / 112 | 0.48 | 319, 81 (the fixed cost, about 0.5 s) |
| 256 to 1,024 | 4,644 | 324 / 405 / 487 | 1.19 | 2, 372 |
| 1,024 to 3,000 | 1,598 | 496 / 598 / 750 | 2.41 | 11, 557 |
| 3,000 to 8,000 | 294 | 369 / 845 / 1,072 | 5.37 | 36, 806 |
| 8,000 and above | 176 | 388 / 1,042 / 1,197 | 18.95 | 48, 1,172 |

The Zig native ring (receipt `docs/receipts/glm-kv-codec-lazy-2b61900a/README.md`, TTFT per turn) [O]: a 3,102-token
cold first turn reaches its first token in 5.75 to 5.80 s (about 540 tokens/s with the fixed cost inside); turns that
restore a kept state and add 2,200 tokens take 5.4 to 6.2 s; restores alone 4.3 to 4.6 s to first token with a
2,200-token continuation. The codec restores a 1 GiB entry in 0.16 s at 8 Grace threads (plan, 2026-11-01) [O].
Production's disk tier: the log's `queue_s` of disk restores does not grow with the cached length (Spearman 0.002
over 2,252 rows), and the 53 disk restores whose predecessor had finished before they started show queue_s p50 0.27 s
and p90 1.0 s at a median 83,776 cached tokens (`queue.py`) [O]; so the large queue_s medians (14 to 57 s) are
queueing behind other requests, not the restore [C].

The trade a token [C]: storing the rows costs 12 KB a rank and buys 0.85 ms (8k+ tokens at 1,172 tokens/s) to
1.8 ms (1 to 3k tokens at 557) of prefill a token, plus the 0.5 s fixed cost a turn; re-prefilling keep9's 20,480
tokens would take about 17.5 s on the Python stack and about 38 s on the Zig ring at turn 1's rate, against under
0.1 s to restore its 431 MB. The "token only" entry is therefore not a compression rung; it is the choice production
already makes when it evicts (the log: 98.9 % of prompt tokens resumed from a cache, 416 cold requests of 9,100).

## 2. Can a drafter's output stand in for the rows?

What is knowable from the data on hand. The keeps hold no token ids (the manifest carries positions and shapes
only), and their prompts are pieces of this repository's `docs/*.md` wrapped as "Part i of the text"
(`lazy_receipt.py`), while the 0430 dumps' prompts are the serving-stack project's `RESULTS.md`, `README.md`,
`serve.sh` and `PATCHES.md` (`make_prompts.py`); production's decode dump holds replies to private traffic. No row
of a keep has its token, hidden row or drafter view in any dump [O]. So the question was answered by proxy, on the
2026-11-01 prefill tap dump (`/glm53-tf/sessions/kvprobe-20261101/`, 18,432 prompt rows, every layer tapped).

What the drafter sees. The dump's `taps_are` line and `docs/WINDOWS.md` line 55: DFlash2's inputs are the mean of
the four residual streams after its tap layers, written by the target into a ring, plus the final-normed row; it
proposes token ids from them [O]. Two consequences [C]: (a) a drafter's view of a cached row exists only after the
target's forward over that row, the pass a kept entry exists to skip; (b) its output is a token id, which the entry
already implies. Its input is at least one 4,096-wide bf16 vector, 8 KB a row (GLM's tap set is not in anything read
here; Qwen's drafter reads five taps, `docs/WINDOWS.md`), the same order as the 11 KB of rows it would replace.

Who the "easy rows" are. Of the tokens the engine processed in the log, 10.66 M were new prompt tokens and 2.49 M
were decode tokens, so at most 19 % of an entry's positions were ever drafted; the rest is user and tool text the
drafter never saw [C]. On the prompt rows of the tap dump the target's own top-1 pick equals the next token on 48 to
53 % of rows, has probability above 0.9 on 21 to 26 % and above 0.99 on 11 to 14 % [O, `fit-sweep.json` `easy`];
DFlash2 lands 3.7 tokens a round at the median in production (`tokens_per_round` p50 3.735, p90 5.75) [O].

The measurement. DSA layer l's latent row is RMSNorm(W_kv_a . x_l) in 512 dimensions, x_l the layer's input, which
`hcPre` builds from the four streams by a Sinkhorn-mixed combination (`cuda_glue.zig` line 201, `cuda_dsa_block.zig`
line 114) [O, code]. The dump has the mean of the streams after layer l-1, used here as the proxy for x_l; the
projection and norm need the engine's weights, so the 4,096-wide input was predicted, not the 512-wide row [C: a
fixed projection preserves an energy fraction in expectation, a learned one may concentrate or dilute it; U].
Predictors: a ridge linear map from the final-normed row (the most informative single vector the drafter receives,
strictly more than its output); the per-token-id mean (what a drafter knowing only the token could reproduce); the
global mean. Trained on L+S (10,240 rows), tested on the A and B prefixes (README and shell script, 1,024 rows each,
text unseen in training; 62 to 71 % of their token ids seen) and on S inside A+S (text seen under another context, a
leak in the idea's favour). Ridge strength chosen per test set from {0.01, 0.1, 1, 10, 100} x the mean eigenvalue,
again in the idea's favour. Unexplained energy = sum of squared residual over sum of squared target [O]:

| DSA layer (tap) | A prefix: linear / token-id / mean | B prefix: linear / token-id / mean | S in A+S (leaked): linear / token-id / mean | residual rel L2 p50, A prefix linear |
|---|---|---|---|---|
| 3 (2) | 0.46 / 0.29 / 0.69 | 0.55 / 0.36 / 0.69 | 0.23 / 0.07 / 0.64 | 0.71 |
| 7 (6) | 0.68 / 0.51 / 0.85 | 0.81 / 0.70 / 0.90 | 0.38 / 0.11 / 0.80 | 0.81 |
| 11 (10) | 0.70 / 0.61 / 0.86 | 0.86 / 0.84 / 0.93 | 0.40 / 0.22 / 0.80 | 0.83 |
| 15 (14) | 0.73 / 0.68 / 0.88 | 0.82 / 0.89 / 0.95 | 0.44 / 0.26 / 0.83 | 0.84 |
| 19 (18) | 0.76 / 0.80 / 0.90 | 0.62 / 1.00 / 1.00 | 0.57 / 0.41 / 0.86 | 0.87 |
| 23 (22) | 0.87 / 0.95 / 0.94 | 0.70 / 1.00 / 0.99 | 0.63 / 0.63 / 0.88 | 0.90 |
| 27 (26) | 0.81 / 0.95 / 0.92 | 0.80 / 1.00 / 0.98 | 0.58 / 0.68 / 0.87 | 0.88 |
| 31 (30) | 0.72 / 0.92 / 0.89 | 0.90 / 1.01 / 0.97 | 0.47 / 0.65 / 0.84 | 0.84 |
| 35 (34) | 0.48 / 0.76 / 0.76 | 0.78 / 0.95 / 0.91 | 0.27 / 0.51 / 0.70 | 0.68 |
| 39 (38) | 0.45 / 0.78 / 0.79 | 0.76 / 0.96 / 0.92 | 0.24 / 0.50 / 0.72 | 0.65 |
| 43 (42) | 0.29 / 0.86 / 0.84 | 0.75 / 0.99 / 0.95 | 0.12 / 0.55 / 0.74 | 0.52 |

Reading [C]: the shallow layers' inputs are mostly token identity (the token-id mean beats the linear map at layers 3
to 11 on both prefixes and at 15 on the prose one, 0.29 against 0.46 at layer 3), the middle layers are mostly
context (nothing explains more than 38 % on unseen text at layers 19 to 31, and usually under 30 %), and only the
last DSA layer is well predicted by the final hidden, and only on the prose
prefix (0.29; 0.75 on the shell script). Neither finding helps the idea: the token is free and predicts the row no
better than 29 % at the cheap end, and the final hidden is the forward pass itself. The training fit (0.07 to 0.55
unexplained) against the test fit shows the map learns the training text, not a transferable projection. On the
leaked S rows the picture is the same with lower numbers (0.12 to 0.63 left).

Why a precise correction cannot be small [C, on 11.18's measurements]. A bf16 row value costs 10.6 bits order-0, 7 of
them its mantissa. A predictor whose residual has relative magnitude r can save at most about log2(1/r) exponent bits
a value: 1 bit at r = 0.5, 0.2 bits at r = 0.85 (the residuals above are 0.52 to 0.90 at the median on unseen text),
and the mantissa bits of the difference of two bf16 values are as random as the row's own. 11.18 measured exactly
this on the rows themselves: the rank-128 self-prediction (r = 0.46, 21 % of energy left, far better than anything
a drafter could carry) saves 0.2 to 0.3 bits a value as whole symbols and nothing as byte planes, against 4.1 bits a
value for the factors. An exact correction also needs the predictor to be bit-reproducible at restore, that is, run
again in the same engine. So "drafter view plus exact residual" is the row, stored less conveniently.

What a lossy correction means [C]: quantizing the residual to k bits around a predicted centre is quantizing the row
with log2(1/r) extra bits of range; at r of 0.5 to 0.9 that is under one bit better than quantizing the row directly,
with a predictor to store and rerun. The direct rung exists (`--kv-cache fp8`: 512 x e4m3 + one fp32 scale, 516 B a
row, dense windows only, refused on pages, plan "Ground state") and its error and quality gate are the sibling
report's subject (`tools/kvprobe/EXPLORE-selective-quantization.md`: e4m3 rounds every class, the latent rows
included, to 2.5 to 2.7 % relative L2 [O there]; the fp8 KDA head measured 2.6 % at the median, 11.14).

## 3. The partial form: tokens plus a coarse recurrent state

Measured and refuted on the ring, session record 11.10 and 11.14 (`TF_GLM_KDA_REPLAY`, `cuda_kda_replay.zig`,
`cuda_kda_coarse.zig`), the 8,528-token prose body, keep at 8,192 [O]:

| stored head | replay window W | state rel L2 p50 / p90 / max over 1,088 heads a rank | reply |
|---|---|---|---|
| none (zero start) | 64 / 256 / 512 | 0.87 / 0.98 / 1.45, 0.73 / 0.91 / 1.17, 0.67 / 0.88 / 1.15 | changed at token 10 (W = 256 keep form) |
| fp8 copy (19.7 MB a rank) | 0 | 0.026 / 0.028 / 0.040 | identical (one sample) |
| fp8 copy | 64 / 256 | 0.37 / 0.76 / 1.49, 0.33 / 0.62 / 1.31 | not served |

Reading [C, 11.10]: a replayed token's deeper-layer inputs are wrong while the lower layers' states converge, so the
error compounds with depth (layer 0 bit-exact at W = 512, layer 44 at 0.80) and falls only slowly with W; replaying
on top of a coarse copy makes it worse, because the copy already holds those tokens. The per-head effective rank of 2
to 12 (11.3) and decay horizons of 1.4 to 8.6 tokens at the median (11.8) describe the mean channel, not the slowest
one, and the input error dominates anyway. The coarse state without replay is the live rung (fp8 head, 11.14, at the
quality gate); the replay mechanism is closed and a drafter adds nothing to it (its inputs are the same taps).

## 4. The reading that survives: easy stretches at lower fidelity

The question becomes: is the model's sensitivity to a row's precision related to how predictable the token at that
row was? Existing evidence says the row's context sensitivity is not [O, section 6.5 of the session record, A+S vs
B+S, state rel L2 p50 by the greedy pick's confidence quartile]:

| layer | confident quartile | unsure quartile | frequent tokens | singleton tokens |
|---|---|---|---|---|
| 9 | 0.086 | 0.091 | 0.092 | 0.085 |
| 21 | 0.441 | 0.470 | 0.527 | 0.399 |
| 44 | 0.411 | 0.408 | 0.386 | 0.416 |

That is sensitivity to context, not to precision, but it is the same per-row quantity the idea assumes is special at
easy rows, and it is not. There is also a role mismatch [C]: a token's confidence is a query-side property (how
predictable the next token was), while a cached row's importance is key-side (how much later queries attend to it
and how large those weights are); nothing measured links the two, and the DSA indexer's selection counts
(11.9: only 1 to 7 % of a long prompt's pools are never selected) show no large unattended class to demote.

Bound on the saving [C]: rows with top-1 probability above 0.9 are 21 to 26 % of prompt rows [O above]; the latent
class is 53 % of keep9's raw entry (35 % of the 16-keep average, plan) and fp8 halves it. Selective fp8 on a quarter
of the rows saves 0.25 x 0.5 x 0.53 = 6.6 % of keep9 (4.4 % at the plan's share); uniform fp8 rows save 27 % (17.5 %).
The selective form only matters if uniform fp8 rows fail the quality gate and the selective rows pass it; with
drafts on, acceptance would mark at most the 19 % of positions that were decoded.

Experiment (needs an engine run, one outage, about an hour; the pattern of 11.10's check mode) [U]:
1. Capture: on the receipt conversation (`lazy_receipt.py`, 9 turns) with `TF_GLM_KV_DUMP`, the keep at each turn,
   and per row the target's top-1 log-probability (the 0430 dump's `lp` on the Python stack, or a `GLM_CAPTURE`-style
   hook on the Zig ring) and, with drafts on, DFlash2's acceptance per position from the round log.
2. Perturb: rewrite a keep's latent rows (`kvdump.write` produces the engine's format; whether a restore can be fed
   from a rewritten dump rather than host pages is unverified [U]) in three blocks of equal size at the same
   precision (e4m3 + scale, then 4 and 3 mantissa bits): the most confident quartile, a random quartile, the least
   confident quartile; and uniform fp8 as the reference.
3. Score: `tensorfold kl --dump/--against` on the continuation against the unperturbed restore, the reply's token
   digest, and the attention-output delta at the perturbed layers' later positions.
4. Decide: the one measurement is the correlation between per-row perturbation sensitivity and that row's
   confidence (or acceptance); if the confident block's KL is not below the random block's by more than the A-A
   repeat floor, the selective form is closed and only the uniform rung remains.

## 5. Recommendation

- The literal idea (drafter view plus correction in place of the rows): closed. By structure (the rows are a function
  of the tokens; the drafter reads the target's taps and outputs a token), by 11.18 (predict-then-code saves 0.2 to
  0.3 of 10.6 bits a value with a far better predictor than a drafter holds), and by the fit above (the final hidden
  leaves 29 to 90 % of a layer input's energy on unseen text, 12 to 63 % on leaked text). No engine run needed.
- The recurrent-state form (tokens plus a coarse state, replay the rest): closed by 11.10 and 11.14.
- The fidelity reading (confident rows at fp8): a cheap experiment, worth at most 4 to 7 % of an entry, a quarter of
  the uniform fp8-rows rung it depends on. Do not run it alone; attach the three-block perturbation of section 4 to
  the Phase 3 quality gate of the fp8 latent rung, which has to run anyway. The deciding measurement is the KL of the
  confident block against the random block at the same precision.

## Artifacts and reproduction

- `~/kvprobe/explore-drafter-cache/reqlog.py` -> `reqlog.txt` (prefill bins); `queue.py` -> `queue.txt` (disk restore
  queue_s by cached length); `fit_hidden.py fit-sweep.json 0.01,0.1,1,10,100` -> `fit-sweep.json`, `fit-sweep.log`
  (305 s on cores 5-9 under a 4 GB cap; the first run `fit.json` at 0.01 alone). All read-only on the node.
- Inputs: `/glm53-tf/sessions/requests.jsonl` (read), `/glm53-tf/sessions/kvprobe-20261101/gx10-eb0e-1-1793554398/`
  (the 2026-11-01 tap dump, memory-mapped), `~/kvprobe/native/dump/glm-rank0-keep{1..16}.plan` (manifests only).
- Not covered [U]: the latent row itself (needs W_kv_a and the norm from the checkpoint); the drafter's own
  acceptance per prompt row (drafts off in the tap dump); any served measurement.
