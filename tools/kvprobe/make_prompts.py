"""Build the probe prompts as token-id lists via the server's /tokenize: S (shared chunk) under four prefixes."""
import json, sys, urllib.request
BASE = "http://127.0.0.1:8000"
def tok(text):
    req = urllib.request.Request(BASE + "/tokenize", data=json.dumps({"prompt": text, "add_special_tokens": False}).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["tokens"]
def take(path, n, skip=0):
    text = open(path, encoding="utf-8", errors="replace").read()
    ids = tok(text)
    ids = ids[skip:skip + n]
    assert len(ids) == n, (path, len(ids), n)
    return ids
root = sys.argv[1]
S = take(f"{root}/docs/RESULTS.md", 2048, skip=4000)          # shared chunk: prose with numbers
A = take(f"{root}/README.md", 1024)                            # prefix A: README prose
B = take(f"{root}/scripts/serve.sh", 1024)                     # prefix B: shell code
L = take(f"{root}/docs/PATCHES.md", 8192, skip=1000)           # prefix L: long prose
prompts = {"S": S, "A+S": A + S, "B+S": B + S, "L+S": L + S}
# the four first tokens must differ so no prefix is shared between the requests
firsts = [tuple(v[:8]) for v in prompts.values()]
assert len(set(firsts)) == 4, firsts
meta = {k: {"len": len(v), "s_start": len(v) - len(S)} for k, v in prompts.items()}
json.dump({"S": S, "prompts": prompts, "meta": meta}, open(sys.argv[2], "w"))
print(json.dumps(meta))
