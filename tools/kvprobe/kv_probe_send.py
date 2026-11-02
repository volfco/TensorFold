"""Drive the Zig engine's server through the stored-state probe: for each content type, the chunk S after no prefix,
prefix A, prefix B and the long prefix L, each request kept by the prompt cache with TF_GLM_KV_DUMP on, then a
PROBE_JSON naming each request's keep files for kvstate.py.

    python3 kv_probe_send.py build REPO_ROOT PROMPTS_JSON [--base URL] [--types prose,code,json,boilerplate]
    python3 kv_probe_send.py send PROMPTS_JSON DUMP_DIR PROBE_JSON [--base URL] [--rank1-dir DIR]

Each request is a chat body: [system: a probe nonce] + prefix messages (X) + S's messages + filler messages (F, the
same for every context) + a last user message. The prompt cache keeps the rendered history (everything before the
last user message) at the chunk start at or before its end (GLM's prompt grid, prefill_rows); F puts that mark past
S's end. S's place in the ids is found by rendering with and without S's messages (the template renders messages
one by one), so S's ids are identical in every context. The prefixes are trimmed so that S starts at the same
offset modulo the grid in every context: with the same F the keeps then sit the same distance past S, which makes the
KDA recurrent states comparable (kvstate.py skips them otherwise).
Content types: prose (repository docs), code (Zig sources), json (a tool result: a JSON listing of repository files),
boilerplate (S is a run of tool-calling turns, so its tokens are mostly the chat template's markup).
"""
import glob, json, os, secrets, sys, time, urllib.request

GRID = 2048          # GLM prompt chunk rows (zig/src/families/glm/cuda_state.zig prefill_rows): keeps sit on it
S_TOKENS = 2048
PREFIX = {"A+S": 2048, "B+S": 2048, "L+S": 8192}   # target lengths before trimming to the grid offset
FILLER = 2 * GRID + 256                             # enough that the history mark lands past S's end


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


class Server:
    def __init__(self, base):
        self.base = base

    def ids(self, text):
        return post(self.base, "/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]

    def text(self, ids):
        return post(self.base, "/detokenize", {"tokens": ids})["prompt"]

    def render(self, messages, tools=None):
        body = {"messages": messages, "add_generation_prompt": True}
        if tools:
            body["tools"] = tools
        return post(self.base, "/tokenize", body)["tokens"]

    def clip(self, text, n, skip=0):
        ids = self.ids(text)[skip:skip + n]
        if len(ids) < n:
            raise SystemExit(f"source text too short: {len(ids)} of {n} tokens")
        return self.text(ids)


def read_all(paths):
    return "\n\n".join(open(p, encoding="utf-8", errors="replace").read() for p in sorted(paths))


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def boilerplate(root, turns):
    """Tool-calling turns: an assistant call and its tool result, short, so most tokens are template markup."""
    files = sorted(glob.glob(os.path.join(root, "zig/src/core/*.zig")))[:turns]
    out = []
    for i, p in enumerate(files):
        rel = os.path.relpath(p, root)
        out.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{i}", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": rel})}}]})
        out.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps({"path": rel, "lines": sum(1 for _ in open(p))})})
    return out


def sources(root, srv):
    """Each piece is cut at a fixed fraction of its source's tokens (not a fixed offset), so a shorter tree still
    builds; a source too short for its pieces is refused before any request is sent."""
    docs = srv.ids(read_all(glob.glob(os.path.join(root, "docs/*.md"))))
    code = srv.ids(read_all(glob.glob(os.path.join(root, "zig/src/core/*.zig"))))
    readme = srv.ids(read_all([os.path.join(root, "README.md")]))
    shell = srv.ids(read_all(glob.glob(os.path.join(root, "scripts/*.sh")) + glob.glob(os.path.join(root, "tools/*.sh"))))
    listing = [{"path": os.path.relpath(p, root), "bytes": os.path.getsize(p)}
               for p in sorted(glob.glob(os.path.join(root, "zig/src/**/*.zig"), recursive=True))]

    def cut(ids, n, at, what):
        if len(ids) < n:
            raise SystemExit(f"{what}: {len(ids)} tokens, {n} needed")
        a = int(at * (len(ids) - n))
        return srv.text(ids[a:a + n])

    # docs hold S (prose), L and F at disjoint fractions: check they do not overlap
    need = S_TOKENS + PREFIX["L+S"] + 256 + FILLER
    if len(docs) < need:
        raise SystemExit(f"docs/*.md: {len(docs)} tokens, {need} needed for S, L and F without overlap")
    return {
        "S": {"prose": [{"role": "user", "content": srv.text(docs[:S_TOKENS])}],
              "code": [{"role": "user", "content": cut(code, S_TOKENS, 0.2, "zig/src/core")}],
              "json": [{"role": "user", "content": "List the sources."},
                       {"role": "assistant", "content": "", "tool_calls": [{"id": "call_ls", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \".\"}"}}]},
                       {"role": "tool", "tool_call_id": "call_ls", "content": cut(srv.ids(json.dumps(listing)), S_TOKENS, 0.0, "file listing")}],
              "boilerplate": boilerplate(root, 40)},
        "A": cut(readme, PREFIX["A+S"] + 256, 0.0, "README.md"),
        "B": cut(shell, PREFIX["B+S"] + 256, 0.0, "shell scripts") if len(shell) >= PREFIX["B+S"] + 256
             else cut(code, PREFIX["B+S"] + 256, 0.7, "zig/src/core"),
        "L": srv.text(docs[S_TOKENS:S_TOKENS + PREFIX["L+S"] + 256]),
        "F": srv.text(docs[len(docs) - FILLER:]),
    }


def body_of(nonce, prefix, s_msgs, filler, last="Reply with one word: done."):
    msgs = [{"role": "system", "content": f"probe {nonce}"}]
    if prefix is not None:
        msgs += [{"role": "user", "content": prefix}, {"role": "assistant", "content": "Noted."}]
    msgs += s_msgs
    if s_msgs and s_msgs[-1]["role"] != "assistant":
        msgs.append({"role": "assistant", "content": "Noted."})
    msgs += [{"role": "user", "content": filler}, {"role": "assistant", "content": "Noted."}, {"role": "user", "content": last}]
    return msgs


def locate(srv, with_s, without_s, tools):
    """S's [start, end) in the ids of `with_s`: the two renders share a prefix to start and a suffix from end."""
    a, b = srv.render(with_s, tools), srv.render(without_s, tools)
    start = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    end = len(a) - (len(b) - start)
    if end <= start or a[:start] != b[:start] or a[end:] != b[start:]:
        raise SystemExit("S's messages do not render as one contiguous run of ids")
    return a, start, end


def build(root, out, base, types):
    srv = Server(base)
    src = sources(root, srv)
    nonce = secrets.token_hex(4)
    prompts = []
    for t in types:
        s_msgs = src["S"][t]
        tools = TOOLS if t in ("json", "boilerplate") else None
        ref = None
        for ctx in ("S", "A+S", "B+S", "L+S"):
            prefix = None if ctx == "S" else src[ctx[0]]
            for _ in range(6):  # trim the prefix until S starts at the reference offset modulo the grid
                msgs = body_of(f"{nonce}-{t}-{ctx}", prefix, s_msgs, src["F"])
                ids, s0, s1 = locate(srv, msgs, body_of(f"{nonce}-{t}-{ctx}", prefix, [], src["F"]), tools)
                if ref is None:
                    ref = s0 % GRID
                over = (s0 - ref) % GRID
                if over == 0 or prefix is None:
                    break
                cut = srv.ids(prefix)
                prefix = srv.text(cut[:len(cut) - over])
            else:
                raise SystemExit(f"{t}/{ctx}: S's offset did not settle on the grid")
            if (s0 - ref) % GRID:
                raise SystemExit(f"{t}/{ctx}: S starts at {s0}, not {ref} modulo {GRID}")
            prompts.append({"type": t, "context": ctx, "messages": msgs, "tools": tools, "ids_len": len(ids),
                            "s_start": s0, "s_len": s1 - s0, "s_ids": ids[s0:s1], "suffix_len": len(ids) - s1})
            print(f"{t:12s} {ctx:4s} ids {len(ids):6d} S at {s0:6d}..{s1:6d}", flush=True)
    json.dump({"grid": GRID, "nonce": nonce, "prompts": prompts}, open(out, "w"))


def keep_files(d):
    return set(os.listdir(d)) if d and os.path.isdir(d) else set()


def send(prompts_path, dump_dir, out, base, rank1_dir=None):
    P = json.load(open(prompts_path))
    model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0]["id"]
    caps = []
    for p in P["prompts"]:
        before = {0: keep_files(dump_dir), 1: keep_files(rank1_dir)}
        t0 = time.time()
        body = {"model": model, "messages": p["messages"], "max_tokens": 1, "temperature": 0}
        if p["tools"]:
            body["tools"] = p["tools"]
        r = post(base, "/v1/chat/completions", body, timeout=3600)
        time.sleep(2)  # the keep's dump is written inside the prompt pass; give the filesystem a moment
        keeps = {}
        for rank, d in ((0, dump_dir), (1, rank1_dir)):
            new = sorted(f[:-5] for f in keep_files(d) - before[rank] if f.endswith(".plan"))
            if new:
                keeps[str(rank)] = [os.path.join(d, f) for f in new]
        usage = r.get("usage", {})
        print(f"{p['type']:12s} {p['context']:4s} {time.time() - t0:7.1f} s usage {usage} keeps {keeps}", flush=True)
        if "0" not in keeps:
            raise SystemExit("no keep on rank 0: is the prompt cache on (TF_GLM_KV_CODEC=1, TF_GLM_HOST_CACHE=1, "
                             "TF_GLM_KV_DUMP set) and the history long enough to reach a grid mark?")
        caps.append({k: p[k] for k in ("type", "context", "s_start", "s_len", "suffix_len", "ids_len")} | {"keeps": keeps, "usage": usage})
    json.dump({"grid": P["grid"], "captures": caps}, open(out, "w"), indent=1)
    print("wrote", out)


def main(argv):
    if len(argv) < 2 or argv[1] not in ("build", "send"):
        print(__doc__)
        return 2
    args, opts = [], {"--base": "http://127.0.0.1:8000", "--types": "prose,code,json,boilerplate", "--rank1-dir": None}
    it = iter(argv[2:])
    for a in it:
        if a in opts:
            opts[a] = next(it)
        else:
            args.append(a)
    if argv[1] == "build":
        build(args[0], args[1], opts["--base"], opts["--types"].split(","))
    else:
        send(args[0], args[1], args[2], opts["--base"], opts["--rank1-dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
