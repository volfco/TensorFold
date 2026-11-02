"""Send the four probe prompts one at a time (max_tokens=1, greedy) and check the dump's prefill rows."""
import json, sys, time, urllib.request
BASE = "http://127.0.0.1:8000"
P = json.load(open(sys.argv[1]))
dump_index = sys.argv[2]
model = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][0]["id"]
for name, ids in P["prompts"].items():
    t0 = time.time()
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    print(f"{name}: {len(ids)} tokens, usage={r.get('usage')}, {time.time() - t0:.1f} s, text={r['choices'][0]['text']!r}", flush=True)
    time.sleep(2)
time.sleep(5)
rows = sum(json.loads(l)["n"] for l in open(dump_index) if json.loads(l)["kind"] == "p")
want = sum(len(v) for v in P["prompts"].values())
print(f"dump prefill rows: {rows} (want {want})", "OK" if rows == want else "MISMATCH")
