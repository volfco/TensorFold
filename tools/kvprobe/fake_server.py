"""The runners' self-tests: an in-process stand-in for the Zig server (the routes and log lines served.py reads), a
synthetic session store, and the checks. No GPU, no model: replies are a fixed function of the messages.
"""
import base64, hashlib, json, os, re, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import served
from recall_yield import SYSTEM, USER, ASSISTANT, OBSERVATION

GRID = 2048


class Fake:
    def __init__(self, log, mode, lazy_ms=0):
        self.log, self.mode, self.lazy = log, mode, lazy_ms / 1000
        self.vocab, self.words = {}, {}
        self.kept = []  # [ids, mark, kept_at, keep id, compacted]
        self.keeps = self.restores = self.raw = self.copies = 0
        self.lock = threading.Lock()
        open(log, "w").write(f"info: KVCODEC rank=0 mode={mode} delay_ms={lazy_ms} threads=5\n" if mode != "off" else "start\n")

    def write(self, line):
        with open(self.log, "a") as f:
            f.write(line + "\n")

    def tok(self, text):
        out = []
        for w in text.split():
            if w.startswith("t") and w[1:].isdigit():
                out.append(int(w[1:]))
                continue
            if w not in self.vocab:
                self.vocab[w] = 1000 + len(self.vocab)
                self.words[self.vocab[w]] = w
            out.append(self.vocab[w])
        return out

    def render(self, msgs):
        out = []
        for m in msgs:
            out += [{"system": SYSTEM, "user": USER, "assistant": ASSISTANT}.get(m["role"], OBSERVATION)] + self.tok(m["content"])
        return out + [ASSISTANT]

    def chat(self, body):
        with self.lock:
            ids = self.render(body["messages"])
            now = time.time()
            for k in self.kept:  # a lazy compactor: due once untouched for the delay
                if self.mode == "lazy" and not k[4] and now - k[2] >= self.lazy:
                    k[4] = True
                    self.write(f"info: KVCODEC rank=0 compacted keep={k[3]} raw={k[1] * 1000} stored={k[1] * 700} pages {k[1]} -> {k[1] * 7 // 10} after_ms={(now - k[2]) * 1000:.1f}")
                    self.write(f"info: KVCODEC rank=0 trimmed keep={k[3]} pages {k[1]} -> {k[1] * 7 // 10} free={4000 + k[3]}")
            best = max([k for k in self.kept if ids[:k[1]] == k[0][:k[1]]], key=lambda k: k[1], default=None)
            cached = best[1] if best else 0
            if best:
                self.restores += 1
                self.raw += self.mode == "lazy" and not best[4]
                best[2] = now
            hist = len(self.render(body["messages"][:-1])) - 1
            mark, kept = hist // GRID * GRID, 0
            if mark > cached:
                self.keeps += 1
                kept = 1
                self.kept.append([ids[:mark], mark, now, self.keeps, self.mode == "eager"])
            h = hashlib.sha256(json.dumps(body["messages"]).encode()).digest()
            n = min(body.get("max_tokens", 16), 4 + h[0] % 5)
            reply = [2000 + b for b in h[1:1 + n]]
            self.copies += h[2] % 3
            self.write(f"info: prompt cache: {len(ids)} tokens, resumed at {cached}, kept {kept}; {len(self.kept)} states, 1 MiB and 2 MiB spare of 3 MiB (hits {self.restores}, misses 0, evicted 0, refused 0, failed 0)")
            sha = served.token_sha256(reply)
            self.write(f"[tensorfold] done chatcmpl-{sha[:8]} prompt={len(ids)} cached={cached} tokens={n} sha={sha[:12]} finish=length rounds={n - 1} accepted={n - 2}/{2 * n} ttft=0.010 decode=0.020 elapsed=0.030 queued=0.000 tok/s=50.0")
        tf = {"token_sha256": sha, "token_ids": reply, "time_to_first_token": 0.01, "tokens_per_second": 50.0,
              "seconds": 0.02, "prefill_seconds": 0.005}
        usage = {"prompt_tokens": len(ids), "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": cached}}
        return " ".join(f"r{t}" for t in reply), tf, usage, {"rounds": n - 1, "drafted": 2 * n, "accepted": n - 2}

    def stop(self):
        if self.mode == "off":
            return
        lazy = self.mode == "lazy"
        self.write(f"info: KVCODEC rank=0 mode={self.mode} keeps={self.keeps} compacted={self.keeps} restores={self.restores} raw_restores={self.raw} "
                   f"partial_restores=0 raw=1000 stored=700 ratio=0.7000 duplicate=0 pages 10 -> 7 encode_ms={0 if lazy else 5.0} "
                   f"lazy_encode_ms={5.0 if lazy else 0} decode_ms=1.0 cancels=0 threads=5 keep_io_ms=2.0 copy_ms=1.5 lazy_copy_ms={0.5 if lazy else 0} wait_ms=0.0")


def serve(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                return self.send({"data": [{"id": "fake"}]})
            if self.path == "/v1/recall":
                return self.send({"state": "on", "ids": 10 * fake.copies, "copies": {"proposals": fake.copies, "proposed": 3 * fake.copies, "accepted": 2 * fake.copies},
                                  "bands": [{"matched_from": 16, "landing_rate": 0.5}]})
            self.send({"error": "no"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/tokenize":
                return self.send({"tokens": fake.render(body["messages"]) if "messages" in body else fake.tok(body["prompt"])})
            if self.path == "/detokenize":
                return self.send({"prompt": " ".join(fake.words.get(i, f"t{i}") for i in body["tokens"])})
            text, tf, usage, spec = fake.chat(body)
            if not body.get("stream"):
                return self.send({"choices": [{"message": {"content": text}}], "usage": usage, "tensorfold": tf, "speculative": spec})
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for w in text.split(" "):
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': w + ' '}}]})}\n\n".encode())
            last = {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": usage, "tensorfold": tf, "speculative": spec}
            self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def check(cond, what):
    if not cond:
        raise SystemExit(f"selftest FAILED: {what}")
    print(f"ok: {what}")


def selftest_lazy(main):
    tmp = tempfile.mkdtemp(prefix="lazy-receipt-")
    dirs = {}
    for cfg in ("off", "eager", "lazy"):
        log = os.path.join(tmp, f"{cfg}.log")
        fake = Fake(log, cfg, lazy_ms=300)
        srv, base = serve(fake)
        d = dirs[cfg] = os.path.join(tmp, cfg)
        main(["run", "--base", base, "--config", cfg, "--out", d, "--log", log, "--lazy-ms", "300", "--pause", "0.4",
              "--source", "synthetic", "--first-tokens", "2100", "--piece-tokens", "2100", "--max-tokens", "8", "--log-settle", "0.05"])
        srv.shutdown()
        fake.stop()
        main(["collect", d, "--log", log])
    res = main(["compare", dirs["off"], dirs["eager"], dirs["lazy"], "--json", os.path.join(tmp, "compare.json")])
    check(len(res["turns"]) == 9 and res["first_difference"] is None, "9 turns, replies identical across off / eager / lazy")
    check(all(t["cached"][0] > 0 for t in res["turns"][2:]), "every turn from 3 resumes the last turn's kept state (the history's)")
    check(res["turns"][1]["prompt"] > GRID, "the history passes 2048 tokens by turn 2")
    lazy = json.load(open(os.path.join(dirs["lazy"], "turns.json")))["turns"]
    check(lazy[2]["form"] == "compacted" and lazy[7]["form"] == "raw-or-partial", "lazy: turn 3 resumes compacted, the raw case raw")
    check("partial_plan" in lazy[8], "the partial case was timed from the keep's estimate")
    t = res["codec"]["lazy"][0]
    check(t["encode_ms"] == 0 and t["lazy_encode_ms"] > 0 and "keep_io_ms" in t and "copy_ms" in t, "the totals' timers parsed")
    check(res["pages"]["lazy"][0]["freed"] > 0 and res["codec"]["off"] == [None], "pages freed (lazy); no totals (off)")
    out = os.path.join(tmp, "dry")
    main(["run", "--config", "lazy", "--out", out, "--dry-run", "--source", "synthetic"])
    check(len([f for f in os.listdir(out) if f.startswith("body-")]) == 9, "dry run: 9 bodies, no server")
    try:
        main(["run", "--config", "x", "--out", out, "--piece-tokens", "2000", "--dry-run"])
        check(False, "a piece of 2000 tokens is refused")
    except SystemExit as e:
        check("refused" in str(e), "a piece of 2000 tokens is refused by name")
    print(f"selftest passed ({tmp})")
    res["_dirs"] = dirs
    return res


def tfs(path, ids, when):
    h = json.dumps({"ids": base64.b64encode(np.asarray(ids, dtype="<i4").tobytes()).decode(), "n": len(ids)}).encode()
    with open(path, "wb") as f:
        f.write(b"TFSESS\x00\x01" + len(h).to_bytes(8, "little") + h)
    os.utime(path, (when, when))


def selftest_recall(main):
    import recall_ab
    tmp = tempfile.mkdtemp(prefix="recall-ab-")
    ent = os.path.join(tmp, "store", "abcd", "rank0", "entries")
    os.makedirs(ent)
    w = lambda k, n: list(range(1000 + k, 1000 + k + n))
    conv1 = [5, 6, SYSTEM] + w(0, 30) + [USER] + w(100, 20) + [ASSISTANT] + w(200, 40) + [USER] + w(300, 25) + [ASSISTANT] + w(400, 10)
    conv1b = conv1 + w(410, 30) + [USER] + w(500, 15) + [ASSISTANT] + w(600, 5)  # the same conversation, later
    conv2 = [5, 6, SYSTEM] + w(700, 30) + [USER] + w(800, 20) + [OBSERVATION] + w(900, 12) + [ASSISTANT] + w(950, 9) + [USER] + w(980, 7)
    for i, ids in enumerate((conv1, conv2, conv1b)):
        tfs(os.path.join(ent, f"e{i}.tfs"), ids, 1_700_000_000 + i)
    tfs(os.path.join(ent, "e3.tfs"), [5, 6, USER] + w(990, 40), 1_700_000_003)  # a single cut prompt (a warmup): no turn
    shared = [5, 6, SYSTEM] + w(3000, 200)  # two sessions with the same long system prompt and tools, then apart
    tfs(os.path.join(ent, "e5.tfs"), shared + [USER] + w(4000, 9) + [ASSISTANT] + w(4100, 3), 1_700_000_005)
    tfs(os.path.join(ent, "e6.tfs"), shared + [USER] + w(5000, 9) + [ASSISTANT] + w(5100, 3), 1_700_000_006)
    tfs(os.path.join(ent, "e4.tfs"), w(0, 50), 1_700_000_004)  # a completion: no role tokens
    convs, skipped = recall_ab.load_store([os.path.join(tmp, "store")], 0)
    check(len(convs) == 4 and convs[0]["entries"] == 2 and len(convs[0]["ids"]) == len(conv1b), "a conversation's entries grouped, the longest kept")
    check(convs[2]["entries"] == 1 and convs[3]["entries"] == 1, "two sessions sharing a 200-id system prompt stay apart")
    check(skipped["entries"] == 7 and skipped["no turn"] == 2, "a cut single prompt and a completion are not conversations")
    empty = os.path.join(tmp, "empty", "x", "rank0", "entries")
    os.makedirs(empty)
    tfs(os.path.join(empty, "e.tfs"), [5, 6, USER] + w(990, 40), 1_700_000_000)
    try:
        main(["run", "--label", "recall", "--store", os.path.join(tmp, "empty"), "--out", os.path.join(tmp, "e"), "--dry-run"])
        check(False, "a store without conversations is refused")
    except SystemExit as e:
        check("refused" in str(e), "a store without conversations is refused by name")
    s1, t1 = recall_ab.turns(convs[0]["ids"])
    check(len(s1) == 1 and len(t1) == 3, "conversation 1: a system span and 3 user turns")
    s2, t2 = recall_ab.turns(convs[1]["ids"])
    check(len(t2) == 1 and [p[0] for p in t2[0]] == ["user", "observation"], "conversation 2: user + tool result merged; its cut last user span dropped")
    check(recall_ab.turns(convs[1]["ids"], "skip") is None, "--tools skip drops a conversation with tool results")
    dirs = {}
    for label in ("recall", "read-only"):
        fake = Fake(os.path.join(tmp, f"{label}.log"), "off")
        srv, base = serve(fake)
        d = dirs[label] = os.path.join(tmp, label)
        main(["run", "--base", base, "--label", label, "--store", os.path.join(tmp, "store"), "--out", d, "--log", fake.log])
        srv.shutdown()
    res = main(["compare", dirs["recall"], dirs["read-only"], "--json", os.path.join(tmp, "ab.json"), "--md", os.path.join(tmp, "ab.md")])
    check(res["same"] == 6 and len(res["pairs"]) == 6, "6 replies, byte-identical in both configurations")
    sa = res["summary"][0]
    check(sa["replies"] == 6 and sa["rounds"] > 0 and sa["tokens_per_round"] > 1 and sa["decode_tps_weighted"] > 0, "summary: rounds, tokens a round, tok/s")
    check(res["recall"][0]["copies"]["accepted"] >= 0 and "bands" in res["recall"][0], "GET /v1/recall recorded")
    rows = json.load(open(os.path.join(dirs["recall"], "recall-ab.json")))
    check(not re.search(r"t1\d{3}|\b10\d\d\b", json.dumps(rows["replies"])), "no conversation text or ids in the output")
    main(["run", "--label", "recall", "--store", os.path.join(tmp, "store"), "--out", os.path.join(tmp, "dry"), "--dry-run"])
    check(os.path.exists(os.path.join(tmp, "dry", "plan.json")), "dry run: the plan without a server")
    print(f"selftest passed ({tmp})")
    res["_dirs"] = dirs
    return res
