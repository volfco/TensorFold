# Exploration: selective quantization of kept cache entries (2026-11-04)

The question (the user's words): "can cache entries be post-processed to selectively quantize rows that are not
referenced frequently or don't need to be stored at full precision? e4m3 has a 2.6 % loss at 25 % of the entry size,
so I'm wondering if there is a way to intelligently quantize the cache entries in ways that does not affect the
accuracy in a meaningful way."

Tags: [O] observed, [C] calculated from observations, [H] hypothesis, [U] untested. No code changed, no engine run,
nothing committed. Measurements ran read-only on spark0 beside production (numpy under a 4 GB scope on cores 5 to 9,
about one minute in all); scripts and JSON results are on the node under `~/kvprobe/explore-quant/` (`quant.py`,
`sel.py`, `quant-keep{4,9,12}.json`, `sel-route{2,3}.json`), not in the tree.

## Short answer

1. A correction to the premise [C]: "2.6 % at a quarter of the size" is the fp32 KDA head's number (session record
   11.14). The latent rows, index keys, rings and conv history are already bf16, so e4m3 halves them (516 to 528 B per
   1,024 B row), it does not quarter them. The 4x lever exists only on the prefix class.
2. By error magnitude alone nothing is "safe" and nothing is "sensitive" [O]: every class lands on the same e4m3
   rounding floor, 2.5 to 2.7 % relative L2, uniform across the 11 DSA layers, the 34 KDA layers, three keeps and two
   conversations, with a worst column of 2.7 to 2.9 %. The one exception is the conv history (0.9 % of the entry),
   whose worst columns lose 9 to 29 %. The selection has to come from the consumer's sensitivity, not from the bytes.
3. "Rarely referenced rows" do not exist at the keeps' lengths [O]: at 10k tokens nine pools in ten are selected by
   at least 10 to 14 % of the sampled queries, under 0.2 % are never selected (Gini 0.28 to 0.45). At 200k tokens the
   distribution is concentrated (the top 10 % of pools take 31 to 58 % of selections) but 94 to 99 % of pools are
   still selected at least once. Section 11.9's closure of selection-count demotion stands.
4. Age is a real gradient per row but not per byte [O]: the chance a pool is selected falls 2 to 3x between the last 512
   tokens and 4k to 8k away (10k prompt), 7 to 18x by 16k to 32k away (200k prompt). Yet old rows carry most of the
   selection mass: 23 to 29 % of selections land on rows older than 4,096 tokens at 10k, 67 to 89 % at 200k. A
   two-tier scheme (recent rows bf16, old rows e4m3) saves 25 % of the latent at 8k and 40 % at 20k, and puts the
   quantization error on the rows the model reads most at long context.
5. So the decision is a quality gate, not a byte-level selection, and the cheapest experiment is to gate the extreme
   first: everything e4m3 (82 MB a rank at 8k, 158 MB at 20k, against 284 and 431 today; section 5). If that passes
   the frozen margins the tiers are moot; if it fails, the old-rows tier is the only frequency-shaped fallback the
   data supports, and the fp8 head (11.14) remains the one rung with a served sample.

## 1. Where the bytes are, and what each rung costs in error

Keeps read: `glm-rank0-keep4` (8,192 tokens) and `keep9` (20,480) are turns 4 and 9 of the 11.17 conversation (one
prompt at two lengths); `keep12` (14,336, 2026-11-01 set) is a different conversation. All three are whole-head keeps
(both KDA halves, `cur` all 0). Rank 0 only; 11.19 found rank 1 within 0.02 MB.

| class | 8k (keep4) | share | 20k (keep9) | share | contents |
|---|---|---|---|---|---|
| prefix | 163.13 MB | 57.4 % | 163.13 MB | 37.8 % | rec 142.61 (both halves; live 71.30), conv 2.51, scratch 18.02 (zeros) |
| latent | 92.27 MB | 32.5 % | 230.69 MB | 53.5 % | 11 caches x pos x 512 bf16 |
| ring | 23.07 MB | 8.1 % | 23.07 MB | 5.3 % | 11 x 2 x 4,096 rows x 128 bf16 (keys, gates), the last 4,096 positions |
| pooled | 5.77 MB | 2.0 % | 14.42 MB | 3.3 % | 11 x (pos/4 + 2) rows x 128 bf16 |
| total | 284.26 MB | | 431.32 MB | | keep12 at 14k: 357.79 MB, latent 45 % |

With the live-half keep (11.19) the prefix is 91.83 MB: 43 % of an 8k entry, 25.5 % of a 20k one [C]. The latent
class is the majority past about 10k tokens and the only class that grows per token besides the pooled keys.

Rungs measured (`quant.py`): e4m3 with a per-row amax/448 fp32 scale over 512 values (upstream's `--kv-cache fp8`
rule, `kv_fp8.cuh` `quant_row<512>`), e4m3 with a per-128 amax/448 fp32 scale, e4m3 with a per-128 power-of-two
scale (11.14's `cuda_kda_coarse.zig` rule, amax / 2^k in (224, 448]), and bf16 round-to-nearest-even for the fp32
state. The encoder rounds to the nearest e4m3 magnitude by midpoints (ties go up; a measure-zero event on scaled bf16
inputs). Relative L2 is ||Q - X|| / ||X||; the worst column is max over the row dimension of the column's relative L2
(CONTRIBUTING: an average must not hide a worse column); the column spread is max / median column RMS.

| class, rows | rung | rel L2 | per-row p50 / p90 / max | worst column | column spread |
|---|---|---|---|---|---|
| latent, 11 layers x 3 keeps | e4m3 row-512 (upstream) | 0.0263 every layer | 0.0263 / 0.0276 / 0.030 to 0.031 | 0.027 to 0.028 | 1.2 to 1.9 |
| latent | e4m3 per-128 fp32 | 0.0257 | | | |
| latent | e4m3 per-128 pow2 (11.14) | 0.0266 | | 0.027 to 0.028 | |
| pooled keys, 2k to 5k rows | e4m3 row-128 | 0.0254 to 0.0259 | max 0.031 to 0.034 | 0.027 to 0.028 | 2.1 to 4.1 |
| ring keys, 4,096 rows | e4m3 row-128 | 0.0254 to 0.0259 | max 0.031 to 0.034 | 0.027 to 0.028 | 2.0 to 3.7 |
| ring gates, 4,096 rows | e4m3 row-128 | 0.0238 to 0.0265 | max 0.031 to 0.034 | 0.027 to 0.029 | 1.9 to 6.4 |
| KDA state, 1,088 heads | bf16 | 0.0016 p50 / 0.0018 p90 / 0.0028 max | | 0.0033 | |
| KDA state | e4m3 per-128 fp32 | 0.0213 p50 / 0.0241 p90 / 0.0274 max | by layer 0.006 to 0.025 | 0.054 | |
| KDA state | e4m3 per-128 pow2 (11.14) | 0.0264 p50 / 0.0281 p90 / 0.044 max | by layer 0.020 to 0.033 | 0.047 to 0.054 | |
| conv history, 34 layers | e4m3 per-128 fp32 | 0.0247 median | row max 0.026 | 0.09 to 0.19 | |
| conv history | e4m3 per-128 pow2 | 0.0266 median | row max 0.028 | 0.16 to 0.29 | |

All [O]. The KDA rows reproduce 11.14's served fp8 head (0.0264 / 0.0281 to 0.0283 / 0.0395 to 0.045 over the same
1,088 heads a rank) and 11.18's bf16 figure (0.0017), so the offline encoder matches the engine's. Readings [C]:

- The 2.6 % is e4m3's three mantissa bits, not the data: a bf16 value carries seven, and rounding away four gives a
  uniform relative error of about 2.6 % RMS whatever the tensor. That is why the figure is identical to three decimals
  across layers, keeps and classes. No class is "already coarse", and no class has outlier channels that a per-row
  scale would crush (column spread under 2 on the latent, under 7 on the gates); per-128 scaling buys 0.0006.
- Sensitive by magnitude: the conv history's worst columns (a channel of small values in a row of 128 sharing one
  scale). It is 2.5 MB a rank, so quantizing it buys nothing; keep it bf16 even in the fp8 head (11.14 rounds it).
- The power-of-two scale costs 0.0003 more error than the fp32 scale but makes the restore exact: an e4m3 code (four
  significant bits) times 2^k is representable in bf16, so a quantized row restores to exactly its quantized value
  and a second keep re-encodes it bit-identically (no drift across keep/restore cycles; 11.14 chose it for this). The
  upstream amax/448 fp32 scale rounds again on dequantization and re-quantizes against a new amax each cycle [H].
- Sizes per row: latent 512 bf16 = 1,024 B; e4m3 + one fp32 scale = 516 B (0.504); e4m3 + four per-128 scales
  = 528 B (0.516). Index rows of 128: 256 B to 132 B (0.516).

## 2. Frequency: how concentrated are DSA selections across pools

Source: the patch-0710 routing dumps of the Python serving stack on the probe ring (11.8: four hidden-tap prompts,
selections every 16 prefill rows; 11.9: one 204,800-token prompt, every 64 rows), 11 DSA layers, 512 pools of 4 tokens
selected per query row. Both dumps were taken with the stack's own `GLM53_TF_KV_DTYPE=fp8` latent cache [O], so the
selections already come from an fp8 index path on that engine. Method (`sel.py`): a pool can only be selected by
queries after it, so each pool's count is divided by its eligible sampled queries (rows with pos > 4 pool + 3; pools
with fewer than 5 % of the rows eligible are dropped), giving a rate per pool; the flat-case rate is 512 / (pos / 4).

| prompt, queries, pools | layers | mean rate | top 10 % of pools' share | top 50 % share | Gini | never selected | at most one query | rate under half the mean | rate p10 / p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|
| A+S, B+S (3,072 tok), 63 queries, 752 pools | all 11 | 0.80 to 0.82 | 0.12 | 0.59 to 0.62 | 0.12 to 0.17 | 0 to 0.4 % | 0 to 1.1 % | 4 to 13 % | 0.33 to 0.56 / 0.86 to 0.95 / 1.0 |
| L+S (10,240 tok), 511 queries, 2,460 pools | 10 of 11 | 0.33 to 0.36 | 0.20 to 0.26 | 0.70 to 0.82 | 0.28 to 0.45 | 0 to 0.2 % | 0 to 0.8 % | 16 to 35 % | 0.10 to 0.14 / 0.28 to 0.32 / 0.59 to 0.83 |
| L+S, layer 7 | 7 | 0.32 | 0.31 | 0.90 | 0.55 | 4.4 % | 8.4 % | 46 % | 0.009 / 0.19 / 0.92 |
| 204,800 tok, 3,167 queries, 48,672 pools | 10 of 11 | 0.019 to 0.027 | 0.31 to 0.58 | 0.79 to 0.94 | 0.43 to 0.72 | 0.6 to 6.4 % | 2.2 to 12.6 % | 31 to 62 % | 0.001 to 0.005 / 0.008 to 0.016 / 0.034 to 0.068 |
| 204,800 tok, layer 7 | 7 | 0.020 | 0.88 | 1.00 | 0.91 | 48 % | 59 % | 82 % | 0.000 / 0.000 / 0.036 |
| 204,800 tok, queries at 128k to 200k only (1,152) | 10 of 11 | 0.014 to 0.020 | 0.40 to 0.77 | 0.88 to 1.00 | 0.56 to 0.87 | 5 to 51 % (layer 3: 51 %) | 13 to 61 % | 44 to 76 % | |

All [O]. Readings [C]:

- At the lengths the keeps have (8k to 20k) the distribution is flat: the top decile of pools holds 20 to 26 % of the
  selection mass against 10 % for a uniform draw, the bottom decile is still selected by one query in seven to ten,
  and under 0.2 % of pools are never selected (layer 7: 4 %). "Share of pools selected by under 1 % of queries" is a length artifact (the mean rate
  itself is 2 % at 200k), so the length-free figures are the top-decile share and the Gini. A row that "nobody
  references" does not exist at 10k; at 200k it is 1 to 6 % of pools (layer 3 at late positions 51 %, layer 7 48 %),
  which agrees with 11.9's 1 to 7 % and keeps that closure.
- Layer 7 is the odd one: the top 10 % of its pools take 88 % of its selections and half its pools are never chosen.
  It is one of 11 DSA layers (9 % of the latent bytes), and a frequency tier built for it alone saves about 4 % of
  the latent at the cost of a per-layer policy. (11.9 called it "the one layer whose selections stay local"; by the
  age table below its selections are the least recency-biased, so "local" there meant "concentrated on few pools".)

## 3. Age: does selection fall with distance from the query

Per query, the selected pools binned by distance (query position minus the pool's first position) against the pools
available in that bin, summed over queries: density = selected / available (the chance a pool at that distance is
selected by a query); share = that bin's fraction of all selections.

| distance bin | 0 to 512 | 512 to 2k | 2k to 4k | 4k to 8k | 8k to 16k | 16k to 32k | 32k to 64k | 64k to 128k | past 128k |
|---|---|---|---|---|---|---|---|---|---|
| density, L+S 10k, layers 3 to 43 | 0.46 to 0.61 | 0.39 to 0.46 | 0.31 to 0.33 | 0.21 to 0.26 | 0.15 to 0.26 | | | | |
| share, L+S 10k | 0.12 to 0.15 | 0.30 to 0.34 | 0.28 to 0.29 | 0.21 to 0.26 | 0.02 to 0.03 | | | | |
| density, 200k, all queries, layers 11 to 43 | 0.21 to 0.33 | 0.10 to 0.13 | 0.06 to 0.08 | 0.04 to 0.06 | 0.027 to 0.044 | 0.018 to 0.029 | 0.014 to 0.017 | 0.005 to 0.012 | 0.002 to 0.013 |
| density, 200k, layer 3 / layer 7 | 0.27 / 0.07 | 0.17 / 0.06 | 0.13 / 0.05 | 0.09 / 0.045 | 0.05 / 0.04 | 0.03 / 0.03 | 0.008 / 0.02 | 0.001 / 0.01 | 0.000 / 0.005 |
| share, 200k, all queries, layers 11 to 43 | 0.05 to 0.08 | 0.07 to 0.09 | 0.06 to 0.08 | 0.08 to 0.12 | 0.10 to 0.17 | 0.13 to 0.20 | 0.17 to 0.21 | 0.08 to 0.21 | 0.01 to 0.08 |
| share, 200k, queries at 128k to 200k, layers 15 to 43 | 0.07 to 0.09 | 0.07 to 0.10 | 0.04 to 0.07 | 0.04 to 0.07 | 0.03 to 0.06 | 0.04 to 0.11 | 0.14 to 0.22 | 0.22 to 0.33 | 0.08 to 0.23 |

All [O]. Readings [C]:

- Selection density falls monotonically with distance in every layer: 2 to 3x by 4k to 8k away on the 10k prompt,
  7 to 18x by 16k to 32k away on the 200k one, 20 to 60x past 64k. Per row, old rows are read less often; a two-tier
  scheme has a basis in that sense.
- The mass goes the other way, because there are more old rows than new ones: 23 to 29 % of the 10k prompt's
  selections, and 67 to 89 % of the 200k prompt's (73 to 92 % for queries past 128k), land on rows older than 4,096
  tokens (the engine's ring boundary, `base`); for queries past 128k, 58 to 74 % land on rows older than 16k (layers
  11 to 43). 11.9's "selection reach" says the same.
  So "old rows in e4m3" does not shield the model from the error; at long context it puts the error on most of what
  the attention reads. What it does do is bound the bytes: rows older than `base` are 50 % of the latent at 8k and
  80 % at 20k.
- Which old rows matter is not in these dumps: a selected pool's attention weight is what sets the model's error
  from rounding it, and the softmax over 2,048 selected tokens is usually dominated by a few of them [H]. The routing
  dump records the selection, not the weight; a weight-aware tier needs an engine-side per-row importance
  accumulated during prefill (the heavy-hitter idea) and a gate experiment to justify the policy [U].

## 4. Sensitivity rather than frequency: the gate experiment

What exists today [O]:

- `--kv-cache fp8` upstream: e4m3 latent rows with a per-row fp32 scale in the live cache, dense windows only, refused
  on pages (`error.Fp8OnPages`, `cuda_latent.zig`; the `chunks_*_fp8` kernels read it). It quantizes at write time
  and attends from e4m3; it is not a kept-entry format.
- The fp8 head (11.14, `TF_GLM_KDA_REPLAY_INIT=fp8`): the keep-time form, e4m3 per-128 power-of-two scale on the KDA
  state and conv history, dequantized to the slot's fp32 on restore; served once, three replies equal, one body.
- `tensorfold kl MODEL --dump / --against` (`cli/cuda_kl.zig`): one run writes the greedy rows' bf16 logits with the
  token fed next, the other teacher-forces the same tokens and scores KL, top-1 agreement and NLL per row.
- The nine-turn receipt harness (11.17, 11.19) for served replies and token digests, with an outage of about 20 min.
- The Python serving stack has run its own fp8 latent cache in production since before 2026-11-01 (the dumps' env)
  [O]; its quality evidence is not in this repository and is a different engine [U].

What a kept-entry gate needs [H]: a keep-time quantizer per class behind one knob (say `TF_GLM_KV_QUANT=head`,
`head,latent-old`, `head,latent`, `all`), a restore that dequantizes into the bf16 pages (11.14's form, so the decode
path is untouched and pages stay allowed), and a manifest flag so a quantized keep cannot be mistaken for an exact one.

The gate run, one outage: on the 11.17 conversation (2k to 20k tokens, nine turns) and three more bodies (prose,
code, long-document QA at 8k and 20k; both greedy and sampled per 11.14's reading), for each form: (1) fresh run,
`kl --dump` of the reply rows; (2) keep after the prompt with the form on, restore, teacher-force the same reply with
`kl --against`: KL per row, every top-1 difference with both margins, NLL; (3) the served reply's digest against the
fresh one. Prerequisite to verify [U]: `cuda_kl.zig` scores one engine run's rows against another's; whether the `kl`
command can start its teacher-forced leg from a restored kept state, or that leg must come from a served path that
returns logprobs, was not established here. The exact restore (lossless codec) is the A-A floor and is zero today (resumed equals fresh byte for byte,
11.17, 11.19).

Thresholds CONTRIBUTING implies [C]: it gives no numbers; it requires teacher-forced histories with all top-token
differences and margins recorded, worst per-column error beside the aggregate, and allows a top-1 difference only
under a documented one-step tie policy with both margins inside it. Read strictly, no lossy tier can satisfy
"a supported resumed path equals a fresh execution"; a lossy keep is a separate opt-in tier under the plan's Phase 3
(gate Q1): margins frozen before the run. Proposed margins to freeze [H]: served greedy replies identical on every
body (what 11.14 achieved on one); teacher-forced top-1 agreement at least 99.5 % per body with every flip listed and
its margin under 0.1 nats; mean KL under 0.005 nats a row and p99 under 0.05; worst-column error of the attention
output against an FP64 reference reported for every quantized class and no worse than the bf16 path's own. Kill on any
probe outside the margins or any restore slower than re-prefill of the covered tokens.

## 5. Tiering proposal with sizes and errors

Raw bytes a rank (TP=2; the other rank is within 0.02 MB), e4m3 with per-128 power-of-two scales (0.516 of bf16;
the fp8 head is 11.14's 19,675,424 B, which replaces the whole prefix class including the zero scratch). The codec
column is the lossless block codec's modeled size: row A from the 11.18 ledger [O]; row B from 11.19's measured live
prefix (71.5 to 71.8 MB) plus the plan's class ratios (latent 0.667, ring 0.671, pooled 0.665) [C] (11.18 had
estimated 164 / 262 from the stale ledger before that run); the rows below with e4m3 bytes assumed at 0.8 of raw and
a bf16 state at 0.75 (order-0 planes) [H], to be measured with `tf-kv-codec-bench --file` on a quantized keep [U].

| tier | 8k raw | 20k raw | 8k / 20k codec | error added | status |
|---|---|---|---|---|---|
| A. whole: bf16 rows, fp32 head, both halves | 284.3 MB | 431.3 MB | 216 / 314 [O] | none | served, default |
| B. A with the live half only | 213.0 | 360.0 | 152 / 251 [C] | none (byte-identical replies, 11.19) | served, off by default |
| C. B with the live state rounded to bf16 | 177.3 | 324.4 | 118 / 217 [H] | state 0.17 % median, 0.28 % max; worst column 0.33 % | [U] |
| D. fp8 head (11.14's served form, conv included): state and conv e4m3, scratch dropped | 140.8 | 287.9 | 97 / 195 [H] | state 2.6 % median, 4.4 % max, worst column 5.4 %; conv 2.7 %, worst column 16 to 29 % (a bf16-conv variant costs 1.2 MB more a rank and removes that column) | served once, identical (11.14) |
| E. D with e4m3 latent rows older than `base` (4,096) | 118.5 | 198.5 | 85 / 148 [H] | latent 2.7 % on 50 % / 80 % of rows, carrying 23 to 29 % of selections at 10k, 67 to 89 % at 200k | [U] |
| F. D with e4m3 latent on every row | 96.1 | 176.1 | 73 / 136 [H] | latent 2.7 % everywhere (worst column 2.8 %) | [U]; upstream's dense-window form serves this class live |
| G. everything e4m3: F plus rings and pooled keys | 82.2 | 158.0 | 66 / 126 [H] | index keys 2.6 %: changes which pools are selected, a discrete error the row rungs do not have | [U] |
| B with e4m3 latent on every row, head lossless | 168.3 | 248.3 | 129 / 192 [H] | latent 2.7 %, state exact | [U] |

[C] for the arithmetic. Keep the conv history bf16 in every tier below A (its worst columns are the only sensitive
numbers in section 1 and it is 2.5 MB). Tier G against A is 0.29 of the bytes at 8k and 0.37 at 20k raw; against the
lossless tier B it is 0.39 and 0.44.

The single deciding experiment [C]: run the gate of section 4 on tiers D, F and G in one outage (E is implied by F:
if every row at e4m3 passes, the age tier is unnecessary; if F fails and E is wanted, E needs its own run). The order
of outcomes decides the design without any per-row policy:

- G passes the frozen margins: adopt G as the cold-tier format; frequency and age tiers are moot, since the whole
  entry at e4m3 costs no more quality than any selective subset by the measured error floor.
- G fails and F passes: keep the index keys bf16 (they decide selection; 10 % of an 8k entry, 9 % of a 20k one).
- F fails and D passes: the latent rows are the sensitive class; E (old rows only) is the one frequency-shaped
  fallback the selection data supports, and its gate must show the 23 to 89 % of selections landing on e4m3 rows
  are tolerated. A weight-aware policy needs the per-row importance accumulator first [U].
- D fails: no lossy tier; the lossless ceiling is tier B (251 MB modeled at 20k, 0.58 of A's raw bytes) plus, if its
  0.17 % passes on its own, tier C.

## 6. Caveats

- Two GB10 nodes, one checkpoint (GLM-5.3-Flash EXL3 4bpw, TP=2), rank 0's keeps; three keeps from two conversations
  of one kind (the receipt's chat bodies and the 2026-11-01 set). The error figures are an e4m3 property and should
  transfer; the byte shares depend on prompt length only.
- Selections come from prefill rows of the Python serving stack (patch 0710) with its fp8 latent cache on, sampled
  every 16 or 64 rows, not from decode rows and not from this engine; 11.9 already notes that decode-time selections
  could be narrower. Four short prompts and one 200k document; the 200k prompt is the repository's own documentation.
- A selection count is not an attention weight; the sensitivity question (section 4) is untested here.
- The codec sizes for the lossy tiers are [H]; the restore cost of dequantizing a full e4m3 entry was not timed
  (11.14's head dequantization showed `ms=0.00` in the ledger, the latent would be 5 to 12x more bytes).
- The e4m3 encoder breaks ties upward rather than to even; the agreement with 11.14's engine numbers to three
  decimals bounds the effect.
