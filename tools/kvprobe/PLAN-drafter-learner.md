# Plan: an external learning unit that improves the GLM-5.3-Flash drafter from live traffic

Date 2026-11-04. A plan only: no code, no engine runs, nothing committed. Tags: [O] observed in a cited source, [C]
calculated from cited values, [H] hypothesis or assumption, [U] untested. Sources are cited as file:section; the
stack is `jayleaton/glm53-tensorfold-spark` (clone `glm53-stack` in the scratchpad, branch `draft-hints`), the evidence
is `tools/kvprobe/EXPLORE-branching-drafter.md` (EXPLORE), the probe record `docs/kv-cross-context-probe-2026-11-01.md`
(record 11.x), `tools/kvprobe/STATUS-track3.md` (STATUS) and `~/Workspace/mindreader` (MR).

The user's idea: an external unit (a Jetson Xavier class box) that continuously learns from live traffic and adds
"branches" to the drafter: sub-drafters activated for the trajectories the main drafter cannot hit.

## 0. Summary: what the sources change about the idea

1. **The branch that reaches traffic on this ring is the MTP head, not DFlash2.** 97 % of the dump's rows are sampled
   requests and `auto` drafts those with the MTP arm; DFlash2 drafts only greedy requests, 2.9 % of rows (canaries and
   probes) (EXPLORE §1 table, §0; patches/0010 docstring; RESULTS.md:1196). The stack already has the swap path for that
   arm (`GLM53_TF_MTP_WEIGHTS`, patches/0430 `mtpw.py`), a trainer (`train/mtp_distill.py`), an offline evaluator
   (`train/eval_accept.py`) and the export is licence-clean (DRAFTER-TRAINING §10). A DFlash2 branch serves the sampled
   97 % only if `auto:E:EVERY:MARGIN` is turned on for sampled requests, where DFlash2 "measured slower" (patches/0020
   header; patches/0010 docstring). So: MTP head first; DFlash2 forms second and conditional on that policy change.
2. **The metric the plan is scored by is not yet measurable.** "Tokens a round per arm" is in the request log once the
   `draft-hints` branch lands (0710: `draft_src` per source rounds / drafted / accepted); "per phase" (reasoning / answer)
   needs the first `</think>` position in the request line, which 0300 does not record (counts and hashes only; grep of
   0300 / 0710 / 0150 finds no think-end field) [O]. That field is Phase 0 engineering.
3. **The unit's binding constraint is memory, not FLOPs.** The two trainers are written for a Spark with the engine
   stopped: T1 (MTP head) about 40 GB, T2 (block drafter) 35 to 50 GB (DRAFTER-TRAINING §4, §5.2). A day of production
   decode rows is 4 to 13 PFLOP of training (§5 below), one to two hours even on a Xavier, but a 32 GB Xavier cannot run
   either trainer as written. Recommended: Jetson AGX Orin 64 GB [H]; Xavier only with a memory-lean trainer variant.

The expected gain if the hard rows are landed is real (EXPLORE §5.1: 10 to 23 % fewer rounds when a branch lands half of
the flagged hard rows), but the only measured drafters land about a quarter of them (EXPLORE §5.1 reading), so the
decision is made by data the ring does not record yet: the per-round miss log (EXPLORE §4.5, §7; patches/0710).

## 1. Objective, metric, safety

| item | definition | source |
|---|---|---|
| objective | more tokens a verify round on live traffic, at unchanged replies | |
| primary metric | tokens a round = 1 + accepted / rounds, per arm (m MTP, f DFlash2, l lookup, s none) from the request log's `draft_src` (0710), decode-weighted as STATUS reports it | patches/0710 header; STATUS "Request log" |
| per phase | the same split by rounds whose `pos` is before / after the reply's first `</think>` (id 154842); needs `think_end` in the request line [U] | EXPLORE §1 "phases", Appendix |
| baseline today | 3.58 tokens a round decode-weighted (p10 2.30, p50 3.35, p90 5.50 per request); W11 classes: prose 2.39 greedy / 2.28 sampled, code 4.10 / 3.32, agent 6.34 / 5.30 | STATUS "Request log"; RESULTS.md:1196-1203 |
| hardness proxy per round | the frontier p1 that 0720 writes to the round log at `GLM53_TF_FRONTIER_CONF=0.0001` (measures, never fires). 0720 is refused with `GLM53_TF_GPU_ROUND` sample / resident; `config/prod.env.example` sets no `GPU_ROUND`, the ring's own `prod.env` is unchecked [U]. If the ring runs the GPU round, hardness comes from the dump's top-32 through the join (§2) and only the outcome from the round log | patches/0720 header; prod.env.example |
| safety | drafts only propose; the verify window, keyed sampler and commit decide every token, so replies are byte-identical with any head ("drafted == serial holds with any head"); checked by exact 10/10 and batchexact 4/4 | PATCHES.md 0430 "Exactness"; DRAFTER-TRAINING §6 gate 2; 0720 header |
| how the unit uses it | ship a head, measure the metric, roll back by the metric: a bad head costs speed only, never a reply | [C] from the above |

## 2. Data pipeline

What exists: patches/0430 writes, per committed decode row, the token id, the final-normed hidden row (bf16, 4,096),
the DFlash2 taps (off in every launch so far, `TAPS=0`) and the target's top-32 log-probabilities and ids; rank 0 writes,
100 GiB cap, replies unchanged (record 11.16; EXPLORE §1, §6). The `draft-hints` branch adds per-round outcomes (0710
round log: req, slot, i, pos, src, depth, acc, conf; 0720 adds p1, hint); neither is on the ring yet (0720 header: "not
measured on the ring yet"; the TESTING.md the README cites is not in the clone) [O].

| capture | per unit | rate | a day | source |
|---|---|---|---|---|
| decode rows, no taps (today's dump) | 8,388 B a row | 23 to 28 k rows an hour (1.28 M rows / 55 h; 1.02 M / 36.1 h) | 0.56 to 0.67 M rows, 4.7 to 5.6 GB | 0430 docstring; record 11.20; EXPLORE §1 |
| decode rows, fp8 taps | 28,888 B | same | 16 to 19 GB | 0430 docstring [C] |
| decode rows, bf16 taps | 49,348 B | 1.2 to 1.4 GB an hour | 28 to 33 GB | EXPLORE §6 [C] |
| prefill rows (`WHAT=prefill,decode`) | the same formats | new prompt tokens are 5.3 x decode tokens (5.55 M against 1.04 M) | x 6.3 the rows: 30 to 35 GB without taps, 175 to 210 GB bf16 taps; prefill +4 to 6 % | STATUS "Request log"; DRAFTER-TRAINING §2.1 [C] |
| per-round outcome log (0710 + 0720) | about 40 B a round | about 175 k rounds a day | about 7 MB; no ids | EXPLORE §4.5, §6 [C] |
| request log (0300 + 0710) | one JSON line a request, hashes only | about 2,200 requests a day (3,162 / 35 h) | small | record 11.20 [C] |

Decisions the pipeline needs:

- **Context rows.** `mtp_distill` trains 1,024 anchors after 1,024 context rows, `dflash_distill` 128 anchors in 2,048
  rows after 2,048 context rows (train/*.py argparse; DRAFTER-TRAINING §4, §5.1); replies are p50 94 decode tokens
  (STATUS). A decode-only dump cannot supply those contexts: either dump prefill rows (the x 6.3 row above) or train with
  short contexts and measure the approximation with `eval_accept --sparse-docs` [U]. Phase 1 trains decode-only with
  short contexts first (cheapest); Phase 3 turns prefill rows on if the offline gate shows the context matters.
- **Taps.** Only a DFlash2 branch needs them (EXPLORE §6). Phase 0's one tap day (29 GB, bf16) is the T2 sample and the
  gate's second feature set; off after it until a DFlash2 candidate is chosen (§3).
- **Join.** The round log is keyed (req, slot, i, pos) per launch; the dump's decode segments carry slot and position.
  Phase 0 verifies the join lands 1:1 before any miss-labelled set is built: that join turns "hard" (the proxy, p1 < 0.9)
  into "missed" (the outcome) (EXPLORE §4.5, §7) [U].
- **Retention.** Node: the dump's 100 GiB cap (about 20 days without taps, 3 days with bf16 taps) [C]. Unit: 30 days of
  decode rows without taps is 140 to 170 GB; with bf16 taps 0.85 to 1 TB: a 2 TB NVMe on the unit [H].
- **Transfer.** Nightly rsync over the LAN: 5 GB in about a minute, 30 GB in five, 200 GB in half an hour at 1 GbE
  [C, 110 MB/s assumed]. The round and request logs ride along.
- **Privacy rule.** The round log and request log hold no ids and may leave the node. The dump holds token ids and
  hidden rows (text is decodable from the ids: DRAFTER-TRAINING §9 "Privacy"): it stays on the node or the unit, is
  deleted after each training window, and only aggregates enter the repo, as record 11.16 did. Trained exports hold no
  ids.

## 3. What a "branch" is in this stack

A branch is a gate plus a specialised proposal: the gate says "the pending row is one the main drafter misses", the
specialised head proposes there. The gate signal exists: hard(row j) from row j-1's hidden row and causal scalars reaches
AUC 0.904 pooled, 0.814 within reasoning and 0.925 within answer (precision 0.74 at recall 0.69 on a 32 % base rate);
it is a 4,096 x 256 projection and a logistic head, trained in 30 s on 111 k pairs (EXPLORE §3.4, §6, §7). The timing
rule holds: it reads row j-1, which the engine holds when it drafts row j (EXPLORE §3.4; MR RESEARCH.md:154).

| candidate | what it is | how it loads and how both ranks agree | engineering gap | source |
|---|---|---|---|---|
| A. MTP head trained on the hard / reasoning subset | T1 self-distillation (3 chained steps, KL to the dumped top-32, 185 M trainable parameters), rows weighted or filtered by phase and by the round log's misses; started from the previous export (`--init`) | `GLM53_TF_MTP_WEIGHTS=<export>` at load on both ranks (serve.sh passes every `GLM53_TF_*` to both containers: serve.sh:163); the prepared tree keys on the folder's files; the routed experts stay the checkpoint's | none to serve; one gap: no cross-rank digest of the export (0430 checks dump settings only; `_gather_ints` carries drafter / capacity / context / serial / f_most) [O]. Add the export's `files_ident` to the startup fingerprint (half a day) | DRAFTER-TRAINING §4; patches/0430 mtpw.py; patches/0010 engine.py |
| B. DFlash2 re-fit on reasoning-phase rows | T2 on-policy distillation of the block drafter (every weight but the selector, QAT to the 4-bit copy) | `DRAFTER=<export>` on both ranks (incoai layout loads today); prepared at ~0.4 GB a rank | serves only greedy requests under `auto` (summary 1); needs taps in the dump; licence (§7); DFlash2-G start needs 2 to 3 days of loader work | DRAFTER-TRAINING §5, §5.3; BOOT.md:52 |
| C. gate-conditioned single drafter (MR's class embedding) | one drafter with a gate vector added to its block query embeddings; the gate vector from the hidden-row cluster or the phase bit | one resident drafter, no second KV group: the form EXPLORE §5.1 and MR D3 both recommend | trainer and `dflash2.py` changes; per-round gate vector in the round plan from rank 0 [U] | MR design doc:505; EXPLORE §5.1 |
| D. mixture: two heads, chosen per round by the gate | the gate score picks arm m or f (or head 1 / head 2) each round | the choice must be rank 0's and travel in the round plan / request header, as every draft setting does (0020: "the settings travel in the request header from rank 0"); rank 1 never reads its own environment for them (0720 header) | a new per-round gate in `Stepper` / `auto_decode`; a second resident head: the MTP layer's trainable set is 185 M parameters (the routed experts and the target's head are shared), about 0.1 GB at 4 bits [C: 185e6 x 0.5 B], a second DFlash2 0.4 GB a rank plus caches; MR rejected two resident drafters when both run every round ("the draft phase roughly doubles") but a gated mixture runs one block a round | patches/0020, 0720; MR design doc:506 |

What the stack lacks for a swap without restart (all forms): both knobs are load-time; q4mse quantization of the export
happens at load (mtpw.py); the MTP CUDA graphs hold weight pointers (an in-place copy keeps them, a reallocation needs a
recapture); the batcher must drain (0630's idle doorbell gives the boundary); both ranks must replace at the same round
boundary on the NCCL control path; the fingerprint must carry the digest. Estimated 5 to 10 engineering days [H]; kept
out of Phases 0 to 2 (§6).

## 4. Training on the unit

Assumptions [H], to be checked against NVIDIA's data sheets before purchase: Jetson AGX Xavier 32 GB LPDDR4x, 137 GB/s,
Volta sm_72 (fp16 tensor cores, no bf16; the dump's bf16 rows go to fp16 / fp32 on load), about 11 TFLOPS fp16 dense
peak at 30 W, JetPack 5 era CUDA and PyTorch; Jetson AGX Orin 64 GB, 204.8 GB/s, Ampere sm_87 (bf16), tens of TFLOPS
fp16 dense, JetPack 6. Achieved training throughput assumed at 30 % of peak on either.

Compute per day of production data (0.6 M decode rows; x 6.3 with prefill rows):

| job | cost model | a day of decode rows | Xavier (3.3 TFLOPS achieved) | Orin (12 TFLOPS achieved [H]) | source |
|---|---|---|---|---|---|
| gate (PCA-256 + logistic, ~1 M parameters) | 30 s on 111 k pairs, CPU | minutes | CPU | CPU | EXPLORE §3.4, §6 |
| A. MTP head (T1) | 21 GFLOP an anchor (3 steps; 6 x N x T with N about 1 B through the head, active experts and 3 steps agrees) | 12.6 PFLOP | about 1.1 h | about 18 min | DRAFTER-TRAINING §4 [C] |
| B. DFlash2 (T2), 1.17 B parameters (the 0.7 GB prepared head at 4 bits) | 6 x 1.17e9 x T; the doc's 11 GFLOP a block row x 8 + 1.5 a context row at 1 anchor in 16 rows gives the same 7 GFLOP a row | 4.2 PFLOP (28 PFLOP with prefill rows) | 21 min (2.4 h) | 6 min (40 min) | DRAFTER-TRAINING §5.2; BOOT.md:52 [C] |

Memory and storage:

| job | as written | Xavier 32 GB | Orin 64 GB | source |
|---|---|---|---|---|
| T1 | about 40 GB: frozen experts dequantized 14.5 GB bf16, head and embedding 2.5 GB, optimizer 3 GB, activations 15 to 25 GB at 2 x 1,024 anchors | does not fit; a lean variant (experts kept quantized or dequantized a layer at a time, 256-anchor windows, 8-bit AdamW) is 3 to 5 trainer days [H] | fits | DRAFTER-TRAINING §4 |
| T2 incoai layout | 18 GB weights + master + AdamW, 2.5 GB head and embedding, activations: 25 to 35 GB | marginal; selector-only or LoRA scope [H] | fits | DRAFTER-TRAINING §5.2; MR design doc:442-451 |
| storage | 30 days of dumps (§2) | 2 TB NVMe | 2 TB NVMe | [H] |

Cadence: daily. Each night the unit pulls the day's dump and logs, rebuilds the miss-labelled set, trains from the
previous export for one epoch of the new rows plus a replay share of older rows (MR's 80 / 20 mix, design doc:413-415),
then scores the candidate offline on held-out rounds before anything is offered to the ring. Held-out means later
requests by time, by conversation (DRAFTER-TRAINING §6; EXPLORE §1 caveat 5), never a random row split.

Offline evaluation: `train/eval_accept.py` gives cumulative acceptance a_k per draft position three ways; the one that
predicts this ring is the coupled-sampled estimate at T = 1 (97 % of traffic is sampled; a draft is kept only when it
equals the serial sample: EXPLORE §1 caveat 2), reported on the hard subset and per phase. The gate for a candidate:
a2 and a3 up by at least 0.08, a1 not lower (DRAFTER-TRAINING §6 gate 2), plus a replay of the day's round log with the
candidate's per-row hits in place of the served head's (EXPLORE §4.4's walk, with real misses) giving predicted tokens
a round per arm and phase.

What the unit cannot do: run the target (164 GB checkpoint, 82 GB a rank: BOOT.md). What stands in for it: the dump's
hidden rows are exactly the MTP head's input and the taps exactly DFlash2's context, and the top-32 log-probabilities
(k = 32 holds over 0.99 of the mass on most rows: train/common.py) are the distillation target, so the unit never needs a
target forward: it trains the head against what the target already computed while serving (DRAFTER-TRAINING §2.1,
§3). The one thing it cannot reproduce is the engine's verify of the candidate's own chain under keyed-noise sampling,
which `eval_accept`'s Monte Carlo estimates and the served A/B measures.

## 5. Deployment loop

| step | mechanism | source |
|---|---|---|
| ship | copy the export, identical bytes, under the sessions mount on both nodes (`/sessions/mtp-export/<date>/`, mounted at `/sessions` in both ranks: serve.sh:149); `GLM53_TF_MTP_WEIGHTS=/sessions/mtp-export/<date>` in `config/prod.env` (serve.sh passes every `GLM53_TF_*` to both containers, only `ROCE_HCA` filtered: serve.sh:141-163); restart | serve.sh; DRAFTER-TRAINING §4, §8 step 2 |
| agreement | both ranks slice the same BF16 tensors by `split.rule` and quantize them as the checkpoint's; the prepared key includes the folder's files; add the export digest to `_gather_ints` so mismatched ranks refuse to start (Phase 2 prerequisite) | mtpw.py; patches/0010 engine.py [U] |
| exactness check | exact 10 / 10 and batchexact 4 / 4 with the new head | DRAFTER-TRAINING §6 |
| A/B | greedy requests: `auto`'s DrafterChoice already races m against f by committed tokens per millisecond inside each request (2 rounds of each, a 3 % margin, a probe every 8): a within-request control for a DFlash2 candidate. Sampled requests (97 %): MTP only, so the control is between launches: A-B-A-B by restart, at least 3 alternations (MR design doc:464-476), scored from the request log per arm, per phase and per prompt class | patches/0010 docstring; record 11.20 |
| sample size | per-request tokens a round spans 2.30 to 5.50 (p10 to p90), sd about 1.25 [C]; a 0.1-token difference at 95 % needs about 600 requests an arm, under a day of traffic each (2,200 requests a day); decode-weighted figures converge faster | STATUS; record 11.20 [C] |
| rollback | unset the knob and restart; rule: two consecutive days with tokens a round (decode-weighted, same phase) more than 3 % under the previous export's 7-day mean, or any exactness failure, reverts | [H] |
| restart cost | 303 to 313 s a start before 0140, 444 s observed on 2026-11-01, 40 to 70 s with prepared folders and cached calibration; the user runs restarts (`scripts/serve.sh restart`) | BOOT.md; record 11 line 578; memory spark-ring |
| live swap (later) | the §3 list; plus per-request head selection through `tf_knobs` with two heads resident for a within-request A/B on sampled traffic | [U] |

## 6. Phases

| phase | work | the one measurement that decides | exit | stop |
|---|---|---|---|---|
| 0. capture (3 to 4 days + 1 restart) | land `draft-hints` (0710 round log, 0720 at 0.0001) on the ring; add `think_end` to the request line; dump on without taps (as now), one day with bf16 taps (29 GB); verify the round-log to dump join; repeat EXPLORE's miss / phase / cluster analysis on real misses, per arm | the share of first-position misses that are stable clusters (P(miss given previous miss), by phase) and the gate's recall over 0071's own probability cut at equal recall, per arm (EXPLORE §7) | a measured share of misses in stable clusters and a gate recall figure, both from real outcomes | the gate adds no recall over the drafter's confidence and misses are isolated: close the branch idea, keep T1 as a plain re-fit. Misses are the reasoning clusters: the lever is a head trained on reasoning rows (candidate A), proceed |
| 1. one branch offline (8 to 12 days; the unit bought and set up) | unit set up; nightly pull; candidate A trained on the hard / reasoning subset from the dump (short contexts first); gate trained; `eval_accept` at T = 1 on held-out rounds; replay of the round log | predicted tokens a round over the served head on the hard subset, per phase, from the replay | a2, a3 up at least 0.08 with a1 not lower on the hard subset, and a predicted gain of at least 0.2 tokens a round on the affected rounds [H threshold] | no candidate clears the offline gate in three training variants (phase filter, phase weight, miss weight): stop; report what the held-out acceptance ceiling was |
| 2. served A/B (4 to 6 days + 4 restarts) | fingerprint digest; exact / batchexact; A-B-A-B over at least 4 days | tokens a round per arm and per phase from the request log, A against B, with its CI | a gain with CI excluding zero on the sampled MTP arm in the phase the head targets; replies byte-identical by construction; the request log lines are the receipt | no gain or a loss on any phase after 3 alternations: unset the knob, back to Phase 1 with the served misses as new labels |
| 3. the unit in the loop (8 to 15 days) | automate pull, train, gate, export, digest, the restart window (weekly, user-run), monitoring and the rollback rule; drift checks (§7) | week-over-week tokens a round per arm and phase against the previous export | two consecutive weekly swaps that hold or improve the metric without a rollback | two rollbacks in a row: freeze the loop at the last good export and review the data (drift, join, phase labels) |

Later, conditional: candidate C or D for the DFlash2 arm (needs taps, the sampled-side `auto` change and the licence
decision), and the live swap (§3), each its own plan.

## 7. Risks

| risk | what the sources say | mitigation |
|---|---|---|
| arm mismatch | sampled traffic (97 % of rows) is drafted by the MTP head; a DFlash2 branch serves 2.9 % of rows unless `auto` races DFlash2 on sampled requests, where it measured slower | train the MTP arm first (candidate A); test `auto:E:EVERY:MARGIN` on sampled traffic only with a DFlash2 that lands reasoning rows (EXPLORE §1; patches/0010, 0020) |
| drift between the unit's data and the serving checkpoint | the hidden rows and top-32 are the serving checkpoint's own (q4mse non-experts, EXL3 experts); any change of image, NONEXPERT or checkpoint invalidates the dump; the reference is near-exact, not bitwise (±0.02 noise; gate 1 ±0.08) | tag every dump segment with `GLM53_TF_IMAGE_ID` and the prepared key; retrain from scratch on a checkpoint change; keep gate 1 (parity) in the nightly run (DRAFTER-TRAINING §3, §9) |
| the DFlash2 licence | incoai/GLM-5.3-Flash-DFlash2 is CC BY-NC-ND 4.0. Two readings in the sources: the stack's, "a private re-fit is fine; sharing it is not" (DRAFTER-TRAINING §10, §5.3); mindreader's, research-only, "never deployed to a serving engine, never uploaded" (MR PROGRAM.md:404-407, design doc:519). ND bars sharing adapted material; NC bars commercial use, which if the ring is commercial already covers the stock drafter. Not legal advice | the MTP head's export holds MIT-chain tensors only (DRAFTER-TRAINING §10): it is the one arm whose trained export is clean to both deploy and share. A shareable block drafter starts from DFlash2-G (Apache-2.0) or DSpark (MIT), 2 to 3 loader days (§5.3) |
| ring memory | 114 of 121 GB used while serving (memory spark-ring; RUNBOOK.md:15); 7 GB headroom | candidate A replaces tensors: zero extra memory. A second resident head costs about 0.3 GB (MTP non-experts at 4 bits [C]) or 0.4 GB a rank plus caches (DFlash2, BOOT.md:52); prefer the single conditioned drafter (C) over a mixture (D) |
| swap without restart | every knob is load-time; the restart is 5 to 7 minutes of outage today (BOOT.md; record 11 line 578) | weekly swap windows run by the user until the loop has proven gains; then the §3 live-swap work |
| the proxy is not the outcome | "hard for the model" is not "missed by the drafter" (EXPLORE §1 caveat 1); the gate's positives may be rows the drafter lands | Phase 0 trains and scores on the round log's real misses before any head is trained |
| within-phase confound | a gate that only learned "reasoning is hard" scores well pooled and does nothing inside a phase (EXPLORE §3.1, §3.4; MR FURTHER_RESEARCH.md:291-302) | every gate and every A/B is scored within phase |
| data volume | published MTP recipes used 5 to 8 k sequences (Red Hat) to 389 k (FastMTP); block drafters 737 k to 1.7 M; a day of the ring is about 2,200 requests (0.6 M decode rows, replies p50 94 tokens): a quarter to a half of Red Hat's set, months short of FastMTP's (DRAFTER-TRAINING §0 compares rows to sequences; do not inherit that) | nightly training is an incremental step from the previous export with a replay share (§4), never a run from scratch on one day; a DFlash2 re-fit accumulates weeks and needs prefill rows (DRAFTER-TRAINING §2.4) |
| the unit's software | Xavier is Volta / JetPack 5: no bf16, older PyTorch [H] | Orin; or fp16 with fp32 master weights on Xavier |

## 8. Budget

| item | estimate | note |
|---|---|---|
| Phase 0 | 3 to 4 engineering days, 1 restart | 0710 / 0720 exist on `draft-hints`; `think_end` and the join check are new |
| Phase 1 | 8 to 12 days; 3 to 5 more for the lean trainer if Xavier | the trainer, evaluator and export exist (train/) |
| Phase 2 | 4 to 6 days, 4 restarts (about 30 min of outage in all) | fingerprint digest half a day |
| Phase 3 | 8 to 15 days | automation, monitoring, rollback |
| live swap (optional) | 5 to 10 days [H] | §3 |
| hardware | Jetson AGX Orin 64 GB developer kit plus a 2 TB NVMe [H, price to be checked]; Xavier 32 GB only with the lean trainer | the unit never holds the target |

Expected gain band, against the ceiling the data allows:

| case | tokens a round | rounds saved | source |
|---|---|---|---|
| a gate alone (a second candidate at hard rows), realistic gate | 3.61 (+0.03) | under 1 %; the perfect gate 3.69, 3.2 % | EXPLORE §5.2 |
| a branch that lands half of the flagged hard rows, gate recall 0.3 / 0.5 / 0.7 | 3.99 / 4.31 / 4.65 | 10 / 17 / 23 % | EXPLORE §5.1 |
| T1 as planned (MTP a 0.74 / 0.45 / 0.22 to 0.80 / 0.58 / 0.38) | MTP rounds 2.41 to 2.76 | sampled cells +8 to 12 % | DRAFTER-TRAINING §7 |
| T2 as planned, prose DFlash2 rounds | +0.4 to 0.7 | prose +10 to 20 % (greedy arm only here) | DRAFTER-TRAINING §7 |
| the proxy's ceiling: every hard row landed | 7.91 | 55 % | EXPLORE §5.1 |
| ADAPTIVE-DRAFT's oracle (drafts exactly what will be kept) | | +15 to 21 % | ADAPTIVE-DRAFT.md:30, 46 |
| Recall (independent, Zig engine only) | | about 5 % | record 11.20 |

The honest expectation for Phase 2 is the T1 band (+8 to 12 % on the sampled arm), with the branch band (10 to 23 %)
reachable only if the hard-row hit rate measured in Phase 0 can be doubled on the subset the gate flags, which no measured
drafter has done on prose (EXPLORE §4.3, W11 a_1 0.68). Phase 0 costs a few days and one restart and settles which band
applies before any hardware is bought.
