"""Offline draft acceptance: score draft sources against the greedy picks a patches/0430 dump recorded.

A 0430 dump row i holds token t[i] and ids[i], the greedy (top-1) pick for the token after t[0..i]. A draft proposed
with context t[0..i] is d[0..k); d[m] is compared with ids[i + m] (row i's pick is the draft's FIRST token, not row
i+1's). Two acceptance rules:

- loose: the longest prefix with d[m] == ids[i + m]. This is what the session record calls teacher-forced: rows past
  i were computed on the prompt's tokens, not on the draft's, so it is exact only where the prompt continues as the
  draft does.
- strict: as loose, and every earlier draft token equals the prompt's token there (d[m-1] == t[i + m]), so each pick
  compared was computed on the draft's own prefix. Exact for what a verifier would accept on this text.

Sources (pluggable: an object with .propose(ctx, most) -> list[int], optional .start(req) / .finish(req)):
- RecallSource: the Zig engine's Recall (zig/src/core/lanes/recall.zig, recall_record.zig, recall_store.zig): 8-token
  keys, newest first, at most 64 candidates a lookup, a match extended back to 64 ids, a copy needs 16 matched ids
  (Params.min_match = bands[0]) and stops at a segment mark; requests add their prompt from the resume point less a
  64-id overlap, then a mark. Optional pricing gate: recall.Rates with its priors and probes.
- ContextCopy: the stream's own context copy (lanes/windows.zig copyProposal, enter_match 8), asked before Recall.
- PriorCompletion: only the continuations earlier requests generated (decode rows), entered by 16 matching ids.
- TokenPrior: the most frequent next token keyed by the last `order` tokens, fitted on the other requests.

Numpy only. CLI: draft_eval.py DUMP_DIR [--k 4,8,16] [--sources recall,context,engine,prior,completion]
                 [--gate] [--json out.json]
"""
import argparse, json, sys
from collections import defaultdict

MARK = -1
KEY_LEN = 8
MAX_STEPS = 64
MAX_MATCH = 64
MIN_MATCH = 16
OVERLAP = 64
BANDS = (16, 32, 64)


class Record:
    """recall_record.Record over Python lists: exact keys instead of hash buckets (the Zig walk also steps over
    bucket collisions, so it can see fewer than 64 matching keys; this one never does)."""

    def __init__(self):
        self.ids = []
        self.keys = defaultdict(list)  # key -> positions where the key ends, oldest first
        self.run = 0

    def push(self, t):
        p = len(self.ids)
        self.ids.append(t)
        if t == MARK:
            self.run = 0
            return
        self.run += 1
        if self.run >= KEY_LEN:
            self.keys[tuple(self.ids[p + 1 - KEY_LEN:p + 1])].append(p)

    def add(self, ids):
        for t in ids:
            self.push(int(t))
        self.push(MARK)

    def _extend(self, p, ctx):
        n = KEY_LEN
        while n < MAX_MATCH and n < len(ctx) and p >= n:
            t = self.ids[p - n]
            if t == MARK or t != ctx[len(ctx) - 1 - n]:
                break
            n += 1
        return n

    def longest(self, ctx):
        """(len, at): the longest match of ctx's tail with an id after it, the newest of equal length."""
        if len(ctx) < KEY_LEN:
            return 0, 0
        cands = self.keys.get(tuple(ctx[-KEY_LEN:]))
        if not cands:
            return 0, 0
        best = (0, 0)
        end = len(self.ids)
        for p in reversed(cands[-MAX_STEPS:]):
            if p + 1 >= end or self.ids[p + 1] == MARK:
                continue
            n = self._extend(p, ctx)
            if n <= best[0]:
                continue
            best = (n, p + 1)
            if n == MAX_MATCH:
                break
        return best

    def copy(self, at, most):
        out = []
        while len(out) < most and at + len(out) < len(self.ids):
            t = self.ids[at + len(out)]
            if t == MARK:
                break
            out.append(t)
        return out


class Rates:
    """recall.Rates: per band, what copies landed; the price a copy must pass, and the probes when it does not."""

    def __init__(self):
        self.landed = [0.75, 2.0, 4.5]
        self.missed = [0.5, 0.5, 0.5]
        self.probe_every, self.since = 32, 0

    @staticmethod
    def band(n):
        b = 0
        while b + 1 < len(BANDS) and n >= BANDS[b + 1]:
            b += 1
        return b

    def rate(self, b):
        return self.landed[b] / (self.landed[b] + self.missed[b])

    def observe(self, b, proposed, accepted, probing):
        self.landed[b] = 0.95 * self.landed[b] + accepted
        self.missed[b] = 0.95 * self.missed[b] + (1 if accepted < proposed else 0)
        if probing:
            self.probe_every = 32 if 2 * accepted >= proposed else min(1024, 2 * self.probe_every)

    def pays(self, n, head):
        return any(expected(self.rate(b), n) > head for b in range(len(BANDS)))

    def probe(self):
        self.since += 1
        if self.since < self.probe_every:
            return False
        self.since = 0
        return True


def expected(q, n):
    total, reach = 1.0, 1.0
    for _ in range(n):
        reach *= q
        total += reach
    return total


class RecallSource:
    """Drafts from a record of earlier requests (scope 'record'), as recall.Reader.propose does."""

    name = "recall"

    def __init__(self, min_match=MIN_MATCH, gate=False, head=2.0):
        self.record, self.min_match, self.gate, self.head = Record(), min_match, gate, head
        self.rates = Rates()
        self.last = None  # (band, proposed, probing) for observe

    def start(self, req):
        pass

    def propose(self, ctx, most):
        self.last = None
        if most < 2:
            return []
        probing = False
        if self.gate and not self.rates.pays(most, self.head):
            probing = self.rates.probe()
            if not probing:
                return []
        n, at = self.record.longest(ctx)
        if n < self.min_match:
            return []
        got = self.record.copy(at, most)
        if len(got) < 2:
            return []
        b = Rates.band(n)
        if self.gate and not probing and expected(self.rates.rate(b), len(got)) <= self.head:
            probing = self.rates.probe()
            if not probing:
                return []
        self.last = (b, len(got), probing)
        return got

    def observe(self, accepted):
        if self.gate and self.last:
            b, p, probing = self.last
            self.rates.observe(b, p, min(accepted, p), probing)

    def finish(self, req):
        resumed = req.get("resumed", 0)
        self.record.add(list(req["tokens"][max(0, resumed - OVERLAP):]) + list(req.get("completion", [])))


class ContextCopy:
    """The stream's own context: the longest earlier match of its tail, enter_match ids or more."""

    name = "context"

    def __init__(self, enter_match=8):
        self.enter_match = enter_match
        self.record = None
        self.fed = 0

    def start(self, req):
        self.record, self.fed = Record(), 0

    def propose(self, ctx, most):
        while self.fed < len(ctx) - 1:  # the context's ids before its last; a key needs an id after it
            self.record.push(int(ctx[self.fed]))
            self.fed += 1
        if self.fed < len(ctx):
            self.record.push(int(ctx[self.fed]))
            self.fed += 1
        n, at = self.record.longest(ctx)
        if n < self.enter_match:
            return []
        got = self.record.copy(at, most)
        return got if len(got) >= 2 else []

    def finish(self, req):
        pass


class EngineOrder:
    """What the lanes engine does: its own context first, Recall when the context has no copy."""

    name = "engine"

    def __init__(self, gate=False):
        self.ctx, self.rec = ContextCopy(), RecallSource(gate=gate)

    def start(self, req):
        self.ctx.start(req)
        self.rec.start(req)

    def propose(self, ctx, most):
        self.rec.last = None
        got = self.ctx.propose(ctx, most)
        return got if got else self.rec.propose(ctx, most)

    def observe(self, accepted):
        self.rec.observe(accepted)

    def finish(self, req):
        self.rec.finish(req)


class PriorCompletion:
    """Earlier requests' generated continuations only (their decode rows), each entered by its last 16 prompt ids."""

    name = "completion"

    def __init__(self, enter=MIN_MATCH):
        self.record, self.enter = Record(), enter

    def start(self, req):
        pass

    def propose(self, ctx, most):
        n, at = self.record.longest(ctx)
        if n < self.enter:
            return []
        got = self.record.copy(at, most)
        return got if len(got) >= 2 else []

    def finish(self, req):
        comp = list(req.get("completion", []))
        if comp:
            self.record.add(list(req["tokens"][-self.enter:]) + comp)


class TokenPrior:
    """Most frequent next token after the last `order` tokens, fitted on every other request (leave one out)."""

    name = "prior"

    def __init__(self, reqs, order=1):
        self.order = order
        self.per = []
        self.total = defaultdict(lambda: defaultdict(int))
        for r in reqs:
            c = defaultdict(lambda: defaultdict(int))
            seq = list(r["tokens"]) + list(r.get("completion", []))
            for i in range(order, len(seq)):
                c[tuple(seq[i - order:i])][seq[i]] += 1
            self.per.append(c)
            for k, v in c.items():
                for t, n in v.items():
                    self.total[k][t] += n
        self.cur = None

    def start(self, req):
        mine = self.per[req["index"]]
        self.cur = mine

    def _best(self, key):
        tot = self.total.get(key)
        if not tot:
            return None
        mine = self.cur.get(key, {})
        best, bn = None, 0
        for t, n in tot.items():
            n -= mine.get(t, 0)
            if n > bn or (n == bn and best is not None and t < best):
                best, bn = t, n
        return best if bn > 0 else None

    def propose(self, ctx, most):
        seq = list(ctx[-self.order:])
        if len(seq) < self.order:
            return []
        out = []
        while len(out) < most:
            t = self._best(tuple(seq[-self.order:]))
            if t is None:
                break
            out.append(t)
            seq.append(t)
        return out if len(out) >= 2 else []

    def finish(self, req):
        pass


def accepted(draft, toks, greedy, i, strict):
    """Accepted draft tokens with context toks[0..i]: draft[m] against greedy[i + m]."""
    m = 0
    n = len(greedy)
    while m < len(draft) and i + m < n and draft[m] == greedy[i + m]:
        if strict and m > 0 and (i + m >= len(toks) or draft[m - 1] != toks[i + m]):
            break
        m += 1
    return m


def evaluate(reqs, source, ks=(4, 8, 16), rows=None):
    """reqs: dicts with tokens, greedy (same length), index; optional completion, resumed. rows(req) -> positions to
    score (default all). Every request is visited in order; the source sees each request's start and finish."""
    kmax = max(ks)
    out = {k: {"positions": 0, "proposals": 0, "loose": 0, "strict": 0,
               "depth_loose": [0] * k, "depth_strict": [0] * k} for k in ks}
    for req in reqs:
        toks, greedy = list(req["tokens"]), list(req["greedy"])
        source.start(req)
        for i in (rows(req) if rows else range(len(toks))):
            d = source.propose(toks[: i + 1], kmax)
            a_obs = accepted(d, toks, greedy, i, True) if d else 0
            if hasattr(source, "observe") and d:
                source.observe(a_obs)
            for k in ks:
                s = out[k]
                s["positions"] += 1
                if not d:
                    continue
                dk = d[:k]
                s["proposals"] += 1
                lo, st = accepted(dk, toks, greedy, i, False), accepted(dk, toks, greedy, i, True)
                s["loose"] += lo
                s["strict"] += st
                for m in range(lo):
                    s["depth_loose"][m] += 1
                for m in range(st):
                    s["depth_strict"][m] += 1
        source.finish(req)
    for k, s in out.items():
        p = max(1, s["proposals"])
        s["coverage"] = s["proposals"] / max(1, s["positions"])
        s["mean_loose"] = s["loose"] / p
        s["mean_strict"] = s["strict"] / p
        s["per_position_loose"] = s["loose"] / max(1, s["positions"])
        s["per_position_strict"] = s["strict"] / max(1, s["positions"])
        s["rate_loose"] = [x / p for x in s["depth_loose"]]
        s["rate_strict"] = [x / p for x in s["depth_strict"]]
    return out


# ---- 0430 dump reading ---------------------------------------------------------------------------------------------
def load_dump(dump):
    """Requests in dump order: prefill rows (kind p, a request starts at start == 0) and, when the dump holds them,
    the decode rows after them as its completion (kind d; their tokens). Only tokens and top-1 ids are read."""
    import numpy as np
    recs = [json.loads(l) for l in open(f"{dump}/index.jsonl")]
    shards = {}

    def arr(r, name):
        off, dtype, shape = r["arrays"][name]
        f = shards.setdefault(r["shard"], np.memmap(f"{dump}/{r['shard']}", dtype=np.uint8, mode="r"))
        dt = {"bfloat16": np.uint16, "int32": np.int32, "<i4": np.int32, "float16": np.float16,
              "float32": np.float32}.get(dtype, dtype)
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        return f[off:off + n].view(dt).reshape(shape)

    reqs = []
    for r in recs:
        if r["kind"] == "p":
            if r["start"] == 0:
                reqs.append({"index": len(reqs), "tok": [], "ids": [], "completion": []})
            if not reqs:
                raise ValueError("dump: prefill rows before any request start")
            if r["start"] != sum(len(x) for x in reqs[-1]["tok"]):
                raise ValueError(f"dump: segment at {r['start']} does not continue its request")
            reqs[-1]["tok"].append(arr(r, "tokens"))
            reqs[-1]["ids"].append(arr(r, "ids")[:, 0])
        elif r["kind"] == "d" and reqs and "tokens" in r["arrays"]:
            reqs[-1]["completion"].extend(int(x) for x in arr(r, "tokens"))
    if not reqs and any(r["kind"] == "d" for r in recs):  # a decode-only dump (DRAFT_DUMP_WHAT=decode)
        reqs = decode_only(recs, arr)
    for q in reqs:
        q["tokens"] = [int(x) for x in np.concatenate(q.pop("tok"))]
        q["greedy"] = [int(x) for x in np.concatenate(q.pop("ids"))]
    return reqs


def decode_only(recs, arr):
    """Decode-only dumps hold no prompt rows: each record is a flush of one request's decode rows, chained by position
    (a record continues the open request whose end is its start, else opens one; concurrent requests at the same
    position are told apart by flush order, oldest first, which is the stack's round-robin). The request's tokens are
    the ids it was fed (the reply, after its last prompt token) and greedy their top-1 picks; the prompt is not in
    the dump, so context copies see the reply alone and Recall records replies only (a lower bound on both)."""
    reqs, open_ = [], defaultdict(list)  # end position -> requests that ended there, oldest first
    for r in recs:
        if r["kind"] != "d" or "tokens" not in r["arrays"]:
            continue
        q = open_[r["start"]].pop(0) if open_.get(r["start"]) else None
        if q is None:
            q = {"index": len(reqs), "tok": [], "ids": [], "completion": []}
            reqs.append(q)
        q["tok"].append(arr(r, "tokens"))
        q["ids"].append(arr(r, "ids")[:, 0])
        open_[r["start"] + r["n"]].append(q)
    return reqs


def make_source(name, reqs, gate):
    if name == "recall":
        return RecallSource(gate=gate)
    if name == "context":
        return ContextCopy()
    if name == "engine":
        return EngineOrder(gate=gate)
    if name == "completion":
        return PriorCompletion()
    if name == "prior":
        return TokenPrior(reqs, order=1)
    if name == "prior2":
        return TokenPrior(reqs, order=2)
    raise SystemExit(f"unknown source {name}")


def report(name, res, w=sys.stdout):
    for k, s in sorted(res.items()):
        rl = " ".join(f"{x:.2f}" for x in s["rate_strict"][: min(k, 8)])
        w.write(f"{name:10s} k={k:2d} positions={s['positions']} coverage={s['coverage']:.3f} "
                f"accepted/proposal loose={s['mean_loose']:.2f} strict={s['mean_strict']:.2f} "
                f"accepted/position strict={s['per_position_strict']:.3f} depth-rate strict[{rl}]\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--k", default="4,8,16")
    ap.add_argument("--sources", default="context,recall,engine,completion,prior,prior2")
    ap.add_argument("--gate", action="store_true", help="price Recall copies as recall.Rates does (head 2.0)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ks = tuple(int(x) for x in a.k.split(","))
    reqs = load_dump(a.dump)
    print(f"requests {len(reqs)}, prefill rows {sum(len(r['tokens']) for r in reqs)}, "
          f"completion tokens {sum(len(r['completion']) for r in reqs)}")
    allres = {}
    for name in a.sources.split(","):
        res = evaluate(reqs, make_source(name, reqs, a.gate), ks)
        report(name, res)
        allres[name] = res
    if a.json:
        json.dump(allres, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
