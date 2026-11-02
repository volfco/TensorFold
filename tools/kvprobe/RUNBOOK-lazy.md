# Lazy KV codec on GB10: served runbook

What this run measures, on the two-rank GB10 ring with `TF_GLM_HOST_CACHE=1`, for the same conversation served three
times (codec off, eager, lazy):

1. The encode milliseconds taken off the prompt pass: `KVCODEC ... encode_ms` (inside keeps) against
   `lazy_encode_ms` (on the workers), and each turn's time to first token.
2. Pages freed over time: the `KVCODEC ... compacted` and `trimmed` lines and the `prompt cache:` lines' spare MiB.
3. Restores from mixed entries: turns that resume a state before its compaction (raw), during it (part compacted) and
   after it (compacted); the totals' `raw_restores`, `partial_restores` and `restores`.
4. Replies byte-identical with the codec off, eager and lazy (greedy decoding, the same turns).

Nothing here has run on a GPU yet. Host checks: `STATUS-lazy.md`.

Placeholders as in `RUNBOOK.md`: `R0` and `R1` the two GB10 hosts (rank 0 serves HTTP), `R0_LINK` rank 0's address on
the direct link, `MODEL` the checkpoint, `TF` this checkout at the same tree on both hosts (copy the tree; it carries
uncommitted changes), `OUT` a results directory on R0.

## 0. Stop the production ring (by hand)

As in `RUNBOOK.md` step 0: the operator stops the production ring and restores it afterwards; an agent must not.
Check `free -g` on both hosts before step 2.

## 1. Build and host checks (each host)

```bash
cd TF
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build test-kv-codec test-kv-pool test-glm-snapshots-host     # the lazy, race, store and two-rank checks
lscpu -e=CPU,CORE,MAXMHZ                                         # which numbers are the efficiency cores
```

The convention here is the engine on the efficiency cores, `taskset -c 0-4,10-14`. Confirm with `lscpu` that those are
the lower-`MAXMHZ` cores and take the other ten (`5-9,15-19` if so) as `WORKERS`; if the numbering differs, use what
`lscpu` shows and write it in the results.

## 2. Serve (three configurations, one at a time, same arguments on both ranks)

Common to all three (both ranks):

```bash
export TF_GLM_HOST_CACHE=1
ARGS="serve MODEL --context 32768 --parallel 1 --prompt-cache-gib 24"
# rank 1 on R1, then rank 0 on R0, each under the engine's CPUs
TF_GLM_RANK=1 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 taskset -c 0-4,10-14 zig-out/native/bin/tensorfold-native $ARGS > OUT/<cfg>-r1.log 2>&1
TF_GLM_RANK=0 TF_GLM_WORLD=2 NCCL_COMM_ID=R0_LINK:29583 taskset -c 0-4,10-14 zig-out/native/bin/tensorfold-native $ARGS > OUT/<cfg>-r0.log 2>&1
```

(Set `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` for the direct link as for production; run each rank detached with
`setsid nohup ... &` if the shell may close.) The configurations, set on both ranks before starting them:

| `<cfg>` | extra environment |
|---|---|
| `off` | none |
| `eager` | `TF_GLM_KV_CODEC=1` |
| `lazy` | `TF_GLM_KV_CODEC=1 TF_GLM_KV_CODEC_LAZY=1 TF_GLM_KV_CODEC_LAZY_MS=2000 TF_GLM_KV_CODEC_CPUS=WORKERS` |

Optional fourth: `lazy-inside`, as `lazy` with `TF_GLM_KV_CODEC_CPUS=0-4,10-14` (the workers share the engine's cores).

Check at start, in each rank's log: `KVCODEC rank=<r> mode=lazy delay_ms=2000 threads=8 cpus=WORKERS (outside the
engine's 0-4,10-14)` for `lazy`; a refused setting stops the start with `KVCODEC refused: <setting>: <reason>`. Ranks
started with different codec settings stop at the fingerprint check naming `TF_GLM_KV_CODEC or
TF_GLM_KV_CODEC_LAZY`.

## 3. The conversation (from R0; the server on 127.0.0.1:8000)

Six greedy turns of one growing chat. The first user message is long (a document of about 16k tokens: any text file,
the same one for every configuration), so each turn's history is several 2048-token chunks and its kept state many
pages. Turns 2 and 3 follow at once (their states are resumed raw, or part compacted); turns 4 to 6 follow after a
pause longer than the delay (resumed compacted).

```bash
python3 - DOC.txt OUT/<cfg>-turns.json <<'EOF'
import hashlib, json, sys, time, urllib.request
base = "http://127.0.0.1:8000"
doc, out = open(sys.argv[1]).read(), sys.argv[2]
model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0]["id"]
msgs = [{"role": "user", "content": doc + "\n\nSummarize the section above in three sentences."}]
asks = ["Name the two most important points.", "Quote one sentence that supports the first point.",
        "Now in one sentence.", "What would you add?", "List three keywords."]
pauses = [0, 0, 10, 10, 10]  # seconds before turns 2..6: the delay is 2 s
turns = []
for i in range(6):
    if i > 0:
        time.sleep(pauses[i - 1])
        msgs.append({"role": "user", "content": asks[i - 1]})
    body = {"model": model, "messages": msgs, "max_tokens": 64, "temperature": 0, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0, first, text = time.time(), None, ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content") or ""
            if delta and first is None:
                first = time.time() - t0
            text += delta
    msgs.append({"role": "assistant", "content": text})
    turns.append({"turn": i + 1, "ttft_s": first, "total_s": time.time() - t0, "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
    print(turns[-1]["turn"], f"{first:.2f}", f"{turns[-1]['total_s']:.2f}", turns[-1]["sha256"][:16], flush=True)
json.dump(turns, open(out, "w"), indent=1)
EOF
```

If the server refuses `"stream": true`, drop it (read `text` from `choices[0].message.content`) and record `total_s`
only; time to first token is then not measured. Do not use `TF_GLM_KV_CODEC_LAZY_MS=0` for this measurement: it
compacts during the prompt pass, where the workers compete with the GPU for GB10's shared memory bandwidth.

Then wait 30 s (the last states compact), stop rank 0 with SIGINT so both ranks tear down and print their `KVCODEC`
totals (if a rank is killed instead, its per-compaction lines still stand; note the missing totals), and start the next configuration. Run the sequence twice per configuration (the first run warms the
page cache of the checkpoint); keep the second run's files.

## 4. Read the results

```bash
cd OUT
# 4: replies byte-identical across configurations (must print one line per turn, three identical hashes)
python3 -c "import json; r={c: json.load(open(c+'-turns.json')) for c in ('off','eager','lazy')}; [print(t, *(r[c][t-1]['sha256'][:16] for c in r)) for t in range(1,7)]"
# 1: encode time on and off the prompt pass, per rank (totals line at exit)
grep -h 'KVCODEC rank=. mode=' *-r0.log *-r1.log | grep keeps=
# 1: time to first token per turn (the turns files), off vs eager vs lazy
python3 -c "import json; [print(c, [round(t['ttft_s'],2) for t in json.load(open(c+'-turns.json'))]) for c in ('off','eager','lazy')]"
# 2: pages freed over time (lazy): when each state compacted and when its pages went back
grep -h 'KVCODEC rank=0 compacted\|KVCODEC rank=0 trimmed\|prompt cache:' lazy-r0.log
# 3: restores by form (lazy): raw_restores, partial_restores, restores in the totals line
grep -h 'mode=lazy keeps=' lazy-r0.log lazy-r1.log
```

Expected shape (to confirm, not assumed): hashes equal in all three; `eager` shows `encode_ms` > 0 and `lazy` shows
`encode_ms=0.0` with `lazy_encode_ms` > 0; `lazy` turns 2 and 3 resume raw or part compacted (`raw_restores` or
`partial_restores` > 0) and turns 4 to 6 compacted; `trimmed` lines follow `compacted` lines on both ranks for the same
keeps (with two ranks, at the next prompt chunk after both copies are compacted) and `free=` grows; `prompt cache:`
lines in `lazy` show more MiB spare than in `off` once states are compacted.

Also record: each rank's `pages <raw> -> <kept>` per compaction and `after_ms` (how long after its keep it
compacted), the `ratio=` and `duplicate=` of the totals, and whether any `KVCODEC refused` or `BadSnapshotControl`
appeared (neither should).

## 5. Report

Write `OUT/summary.md` with the table of turns (ttft off/eager/lazy, sha equal yes/no), the totals lines, the
`lscpu` core map used for `WORKERS`, and the exact commit and `git status --short` of `TF`. Copy `OUT` into
`docs/receipts/` under a dated directory only if the lead asks.
