# Exact page prune of the DSA index scan on GB10: runbook

What this run establishes, for `TF_GLM_INDEX_PRUNE=1` (`zig/src/families/glm/cuda_index_prune_dev.zig`, kernels in
`zig/kernels/cuda/glm_index_prune.cuh` and `tf_glm_index_scores_prune` in `glm_index_scores.cuh`):

1. The kernels build and register (compile records for the contract's gate 5).
2. Synthetic exactness at 32k, 131k and 204k positions: pruned selections equal the full scan's, byte for byte, with
   the share of pages skipped, and the planted control (strips left unwritten) caught.
3. Timing: microseconds a query row for the full and the pruned scan at each length (a decode row, a decode window of
   8 and a prompt block of 512 rows).
4. Served byte identity: the same greedy prompts with the flag off and on give the same replies, at about 32k and
   200k tokens of context, including a resumed turn (a restore, whose strips are recomputed).

Nothing here has run on a GPU yet. Host checks and their results: `STATUS-prune.md`.

Placeholders as in `RUNBOOK.md`: `R0` and `R1` the two GB10 hosts (rank 0 serves HTTP), `R0_LINK` rank 0's address on
the direct link, `MODEL` the checkpoint, `TF` this checkout at the same tree on both hosts, `OUT` a results directory
on R0.

## 0. Stop the production ring (by hand)

As in `RUNBOOK.md` step 0: the operator stops the production ring and restores it afterwards; an agent must not.
Steps 1 to 3 need one GPU (R0 alone is enough); step 4 needs the ring.

## 1. Build, host checks, compile records (each host)

```bash
cd TF
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121                       # zig-out/native/bin/tensorfold-native
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast        # zig-out/bin/tf-cuda-test, real images
zig build test-glm-index-host test-glm-index-emulate test-glm-index-prune-emulate   # host oracle and the .cuh as host C++
zig build fatbins kernel-contract-fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121
zig-out/bin/tf-cuda-test kernel-contract-limits zig-out/fatbin 2>&1 | tee OUT/limits.log
grep -E 'tf_glm_index_scores_prune|tf_glm_prune_(bounds|plan|cut)' OUT/limits.log
```

The four launches are registered as pending (`zig/src/core/contract/glm_index_prune_pending.zig`, usage `none`); the
limits run prints their `RECORD` lines. Paste those lines into `glm_cuda_records.zig` (and remove the pending file's
entries), or send them to the lead. Expected: none above the SM limits (plan: about 32 KiB static shared; planned
scores: tf_glm_index_scores' 32 KiB plus 128 bytes). Also check that `tf_glm_index_scores`' own record did not move
(it is now `scores_body<false>`, the same arithmetic): its line must equal the one in `glm_cuda_records.zig`.

## 2. Exactness and skip share (R0, one GPU)

```bash
zig-out/bin/tf-cuda-test glm-index-check 2>&1 | tee OUT/index-check.log        # tf_glm_index_scores unchanged in bytes
zig-out/bin/tf-cuda-test glm-index-prune 2>&1 | tee OUT/prune.log
zig build kernel-contract -Dnvcc=/usr/local/cuda/bin/nvcc 2>&1 | tee OUT/contract.log
grep -E 'tf_glm_index_scores_prune|tf_glm_prune_bounds|strips use a swapped|SUMMARY' OUT/contract.log
```

`glm-index-prune` prints one line per (length, case, rows): `PASS|FAIL <tokens> tokens, <rows> rows, <case>: pages
skipped S of W (share)` (512-row lines also compare the scan under `TF_GLM_INDEX_CUDA=1`'s kernels, pruned and
not, with the glue's), then seven `PASS|FAIL one scratch, <shape>: <rows> rows at <pos>: pages skipped S of W, rows
cut N` lines (the engine's shapes in sequence on one scratch each way, as a forward's buffers serve every select: a
2048-row chunk, a 493-row last chunk, its 492 MTP rows, a 1024-row chunk, a decode row at 131k then one at 31k, a
window of 8; each must equal the plain scan and cut every row once; round 2's served divergence, 11.15, came from
exactly this reuse), then the control line (`PASS control: strips left unwritten change the selection`) and a final
`PASS glm-index-prune: ...`. Exit status 1 on any FAIL. A strip mismatch after the chunked write stops it with
`FAIL glm-index-prune: N strip values differ ...`. Record every line. The skip shares are on synthetic keys (16
topics, a page per topic); they show the mechanism prunes, not the share on real queries.

In the contract run: `cuda glm tf_glm_index_scores_prune, every page planned, on pages out of order` must pass as a
replacement of `tf_glm_scores` (bytes equal), `cuda glm tf_glm_prune_bounds on pages out of order` must pass gates 1 to
3, and `control: GLM page strips use a swapped table` must be red.

## 3. Timing (R0, one GPU, nothing else on it)

```bash
zig-out/bin/tf-cuda-test glm-index-prune timing 2>&1 | tee OUT/prune-timing.log
grep TIME OUT/prune-timing.log
```

`TIME <tokens> tokens, <rows> rows: wall full X pruned Y us a row (xR); GPU full X pruned Y us a row (xR); pages
skipped S`: the whole selectTokens (the plan, the planned scores with the cut and the second planned scores when
pruned; the select and the token lists), medians of five alternating runs after a warm-up, wall clock around a
synchronized stream and CUDA events recorded on it (GPU time, the launches' host latency excluded), and the pruned
runs' share of whole pages skipped. The check forces `min_pages` to 0, so every length prunes; pick
`TF_GLM_INDEX_PRUNE_MIN_PAGES` (default 256 pages, 131,072 tokens) as the shortest length whose 1-row `wall` ratio
is above 1. Run it twice; keep both. Optional, per kernel: `nsys profile -o OUT/prune zig-out/bin/tf-cuda-test
glm-index-prune timing` (`tf_glm_scores`, `tf_glm_prune_plan`, `tf_glm_index_scores_prune` twice a select).

## 4. Served byte identity (the ring)

Two configurations, one at a time, same arguments on both ranks; the flag must be set alike on both ranks (a rank
without it stops at the fingerprint check naming `TF_GLM_INDEX_PRUNE (its KV pages carry strips)`). Pages are on by
default with two ranks; with one rank add `TF_KV_PAGES=1` (without pages the engine refuses the flag by name:
`IndexPruneNeedsPages`).

```bash
ARGS="serve MODEL --context 245760 --parallel 1 --prompt-cache-gib 24"
# off: no extra environment.  prune: export TF_GLM_INDEX_PRUNE=1 on both ranks first.
# for the skip share at every length (the 32k document too), also TF_GLM_INDEX_PRUNE_MIN_PAGES=0 (default 256 pages)
TF_GLM_RANK=1 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 zig-out/native/bin/tensorfold-native $ARGS > OUT/<cfg>-r1.log 2>&1
TF_GLM_RANK=0 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 zig-out/native/bin/tensorfold-native $ARGS > OUT/<cfg>-r0.log 2>&1
```

The prompts, from R0 (server on 127.0.0.1:8000), the same files for both configurations: `DOC32.txt` (about 30k
tokens) and `DOC200.txt` (about 200k tokens; the repository's documentation concatenated, as the long-context dump
used). For each document: a first greedy turn, the same request again (served from the kept prompt state: a restore),
and a follow-up turn on the same history.

```bash
python3 - OUT/<cfg>-replies.json DOC32.txt DOC200.txt <<'EOF'
import hashlib, json, sys, time, urllib.request
base = "http://127.0.0.1:8000"
model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0]["id"]
out, docs = sys.argv[1], sys.argv[2:]
def ask(msgs):
    body = {"model": model, "messages": msgs, "max_tokens": 256, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["choices"][0]["message"]["content"], time.time() - t0, r.get("usage", {})
rows = []
for doc in docs:
    msgs = [{"role": "user", "content": open(doc).read() + "\n\nSummarize the text above in five sentences."}]
    for turn in ("first", "again", "follow"):
        if turn == "follow":
            msgs = msgs + [{"role": "assistant", "content": rows[-1]["text"]}, {"role": "user", "content": "Which section is the longest, and why?"}]
        text, secs, usage = ask(msgs)
        rows.append({"doc": doc, "turn": turn, "secs": secs, "usage": usage, "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
        print(doc, turn, f"{secs:.1f}", rows[-1]["sha256"][:16], usage, flush=True)
json.dump(rows, open(out, "w"), indent=1)
EOF
```

Stop rank 0 with SIGINT between configurations. Then:

```bash
cd OUT
python3 -c "import json; a,b=(json.load(open(c+'-replies.json')) for c in ('off','prune')); [print(x['doc'], x['turn'], x['sha256'][:16], y['sha256'][:16], 'SAME' if x['sha256']==y['sha256'] else 'DIFFERENT') for x,y in zip(a,b)]"
grep -i 'IndexPrune\|refused\|error' prune-r0.log prune-r1.log
grep -h 'INDEXPRUNE' prune-r0.log prune-r1.log   # per prompt pass and the totals at exit: the skip share on real keys
```

Expected (to confirm): every line `SAME`; the `again` turns report cached prompt tokens in `usage` (a restore ran);
no `IndexPrune*` error. Prompt seconds (the `secs` of `first` at 200k) off against prune give the served effect on
prefill; a decode-rate comparison needs a longer `max_tokens` (e.g. 1024 on the 200k document) and the server's
token rate lines.

## 5. The skip share on real keys, offline (no GPU)

`tools/kvprobe/prune_share.py KEEP_STEM QUERIES.npz [...] [--rows 64] [--json OUT/share.json]` reports per DSA cache
and position bin the share of whole pages the device bound excludes (with the device's seed cut and with the row's
own 512th best score), next to the oracle share and tighter bounds (margin 2^-12, per-head exact maxima, principal
directions). Inputs: a `TF_GLM_KV_DUMP` keep of the long prompt (`glm-rank0-keep<id>.plan/.bin/.parts`) and the
index queries and head weights of sampled rows of the same prompt (format in the tool's docstring); no dump records
the queries yet. `python3 tools/kvprobe/prune_share.py --synthetic` checks the tool itself.

## 6. Report

Write `OUT/summary.md`: the limits `RECORD` lines, the `glm-index-prune` lines (PASS/FAIL, skip shares), the `TIME`
lines, the served table (SAME/DIFFERENT per document and turn, seconds off/prune), and the exact commit and
`git status --short` of `TF`. Update `STATUS-prune.md` with what passed. Copy `OUT` into `docs/receipts/` only if the
lead asks.
