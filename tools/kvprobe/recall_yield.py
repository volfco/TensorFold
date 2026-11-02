"""Recall's expected yield on served agent traffic, from what the serving node already keeps. Aggregates only: no token
ids, text or hashes leave this script's output.

Two inputs, either may be omitted:
- the request log (requests.jsonl of the serving stack): per request prompt, cached, lcp, decode_tokens,
  tokens_per_round, finish, tools, conv. Gives the traffic's shape and the drafter's current landing rate.
- session-store entries (<store>/<compat>/rank0/entries/*.tfs): each header holds the full token ids of a kept
  conversation prefix. In time order (file mtime), each entry's ids past its longest common prefix with any earlier
  entry are its new tokens; the assistant spans among them (<|assistant|> to the next role token) are the text the
  model wrote. A decode walk over each new assistant span asks, at each round, (1) the conversation's own context
  (the lanes engine's context copy, enter_match 8) and then (2) a Recall record of every earlier entry's new tokens
  (draft_eval.Record: 8-id keys, 16 matched ids, 64 candidates, copies stop at segment ends), and advances by the
  copy's matched length plus one. The record is a numpy index (sorted 8-gram hashes), not the Python dict.

Caveats printed with the result: the spans are the history as the chat template re-renders it (earlier turns'
reasoning may be stripped), and the copy is compared with the tokens that were written, not with a greedy pick.

Usage: recall_yield.py [--log requests.jsonl] [--store DIR ...] [--k 7] [--limit-entries N] [--json out.json]
"""
import argparse, base64, glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np

SYSTEM, USER, ASSISTANT, OBSERVATION, EOT = 154826, 154827, 154828, 154829, 154820
THINK, UNTHINK = 154841, 154842
ROLES = (SYSTEM, USER, ASSISTANT, OBSERVATION, EOT)
KEY, MIN_MATCH, ENTER_MATCH, MAX_MATCH, STEPS, OVERLAP = 8, 16, 8, 64, 64, 64
MAGIC_LEN = 8


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if len(xs) else float("nan")


# ---- request log ---------------------------------------------------------------------------------------------------
def log_stats(path):
    rs = []
    for line in open(path):
        try:
            rs.append(json.loads(line))
        except ValueError:
            pass
    out = {"requests": len(rs)}
    prompt = np.array([r.get("prompt") or 0 for r in rs], dtype=np.float64)
    cached = np.array([r.get("cached") or 0 for r in rs], dtype=np.float64)
    lcp = np.array([r.get("lcp") or 0 for r in rs], dtype=np.float64)
    dec = np.array([r.get("decode_tokens") or 0 for r in rs], dtype=np.float64)
    tpr = np.array([r.get("tokens_per_round") or 0 for r in rs], dtype=np.float64)
    out["prompt_tokens"] = float(prompt.sum())
    out["cached_tokens"] = float(cached.sum())
    out["new_prompt_tokens"] = float((prompt - cached).clip(0).sum())
    out["cached_share"] = float(cached.sum() / max(1, prompt.sum()))
    out["lcp_share"] = float(np.minimum(lcp, prompt).sum() / max(1, prompt.sum()))
    out["decode_tokens"] = float(dec.sum())
    out["decode_per_new_prompt_token"] = float(dec.sum() / max(1, (prompt - cached).clip(0).sum()))
    out["prompt_p50_p90"] = [pct(prompt, 50), pct(prompt, 90)]
    out["new_prompt_p50_p90"] = [pct(prompt - cached, 50), pct(prompt - cached, 90)]
    out["decode_p50_p90"] = [pct(dec, 50), pct(dec, 90)]
    ok = tpr > 0
    out["tokens_per_round_weighted"] = float((tpr[ok] * dec[ok]).sum() / max(1, dec[ok].sum()))
    out["tokens_per_round_p10_p50_p90"] = [pct(tpr[ok], 10), pct(tpr[ok], 50), pct(tpr[ok], 90)]
    out["finish"] = dict(Counter(str(r.get("finish")) for r in rs))
    out["kind"] = dict(Counter(str(r.get("kind")) for r in rs))
    out["cache_src"] = dict(Counter(str(r.get("cache_src")) for r in rs))
    out["with_tools"] = int(sum(1 for r in rs if (r.get("tools") or 0) > 0))
    by_fin = defaultdict(lambda: [0.0, 0.0])
    for r in rs:
        t, d = r.get("tokens_per_round") or 0, r.get("decode_tokens") or 0
        if t > 0:
            by_fin[str(r.get("finish"))][0] += t * d
            by_fin[str(r.get("finish"))][1] += d
    out["tokens_per_round_by_finish"] = {k: v[0] / max(1, v[1]) for k, v in by_fin.items()}
    convs = Counter(r.get("conv") for r in rs if r.get("conv"))
    sizes = sorted(convs.values(), reverse=True)
    out["conversations"] = len(convs)
    out["turns_p50_p90_max"] = [pct(sizes, 50), pct(sizes, 90), max(sizes) if sizes else 0]
    out["requests_in_top10_conversations"] = int(sum(sizes[:10]))
    sys_convs = defaultdict(set)
    for r in rs:
        if r.get("sys_hash") and r.get("conv"):
            sys_convs[r["sys_hash"]].add(r["conv"])
    out["system_prompts"] = len(sys_convs)
    out["conversations_sharing_a_system_prompt"] = int(sum(len(v) for v in sys_convs.values() if len(v) > 1))
    lo = np.array([r.get("lcp_other") or 0 for r in rs], dtype=np.float64)
    ls = np.array([r.get("lcp_same") or 0 for r in rs], dtype=np.float64)
    out["lcp_same_share"] = float(np.minimum(ls, prompt).sum() / max(1, prompt.sum()))
    out["lcp_other_share"] = float(np.minimum(lo, prompt).sum() / max(1, prompt.sum()))
    return out


# ---- session entries -----------------------------------------------------------------------------------------------
def read_ids(path):
    with open(path, "rb") as f:
        head = f.read(16)
        n = int.from_bytes(head[8:16], "little")
        if n > 256 << 20:
            raise ValueError(f"{path}: header of {n} bytes")
        h = json.loads(f.read(n))
    return np.frombuffer(base64.b64decode(h["ids"]), dtype="<i4").astype(np.int64)


def roles_of(ids):
    """Per position: 1 inside an assistant span (role token excluded), 2 inside its <think>, else 0."""
    out = np.zeros(len(ids), dtype=np.int8)
    role, think = 0, False
    for i, t in enumerate(ids.tolist()):
        if t in ROLES:
            role, think = t, False
            continue
        if role == ASSISTANT:
            if t == THINK:
                think = True
            out[i] = 2 if think else 1
            if t == UNTHINK:
                think = False
    return out


def hashes(ids):
    """64-bit hash of the 8 ids ending at each position (positions < 7, and keys over a mark (-1), get 0)."""
    n = len(ids)
    h = np.zeros(n, dtype=np.uint64)
    if n < KEY:
        return h
    x = (ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    acc = np.zeros(n - KEY + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(KEY):
            acc = acc * np.uint64(0x100000001B3) ^ x[j:n - KEY + 1 + j]
    bad = np.convolve((ids < 0).astype(np.int64), np.ones(KEY, dtype=np.int64), "valid") > 0
    acc[bad] = 0
    acc[acc == 0] = 1
    acc[bad] = 0
    h[KEY - 1:] = acc
    return h


class Index:
    """Sorted (hash, position) over a stream: candidates for a key, newest first, below a limit."""

    def __init__(self, stream):
        self.s = stream
        h = hashes(stream)
        keep = np.nonzero(h)[0]
        order = np.lexsort((keep, h[keep]))
        self.h, self.p = h[keep][order], keep[order]

    def longest(self, ctx_tail, limit):
        """(len, at): ctx_tail's longest match ending before `limit` with a non-mark id after it."""
        if len(ctx_tail) < KEY:
            return 0, 0
        k = hashes(ctx_tail[-KEY:])[-1]
        a, b = np.searchsorted(self.h, k, "left"), np.searchsorted(self.h, k, "right")
        if a == b:
            return 0, 0
        ps = self.p[a:b]
        ps = ps[ps + 1 < limit][-STEPS:][::-1]
        best = (0, 0)
        s = self.s
        tail = ctx_tail[-MAX_MATCH:]
        for p in ps.tolist():
            if s[p + 1] < 0 or not np.array_equal(s[p + 1 - KEY:p + 1], tail[-KEY:]):
                continue
            m = min(MAX_MATCH, len(tail), p + 1)
            back = s[p + 1 - m:p + 1][::-1]
            want = tail[-m:][::-1]
            diff = np.nonzero((back != want) | (back < 0))[0]
            n = int(diff[0]) if len(diff) else m
            if n > best[0]:
                best = (n, p + 1)
                if n == MAX_MATCH:
                    break
        return best

    def copy(self, at, most, limit):
        c = self.s[at:min(at + most, limit)]
        stop = np.nonzero(c < 0)[0]
        return c[:stop[0]] if len(stop) else c


def entries(stores, limit):
    paths = []
    for st in stores:
        paths += glob.glob(os.path.join(st, "*", "rank0", "entries", "*.tfs"))
    paths.sort(key=os.path.getmtime)
    return paths[:limit] if limit else paths


def walk(ids, roles, start, own, rec, rec_limit, k):
    """Decode walk over the assistant positions at or past `start`: per round, own context first, then Recall."""
    st = Counter()
    j = max(start, 1)
    n = len(ids)
    while j < n:
        if roles[j] == 0:
            j += 1
            continue
        end = j
        while end < n and roles[end] != 0:
            end += 1
        while j < end:
            st["rounds"] += 1
            st["think" if roles[j] == 2 else "answer"] += 0  # keys exist
            tail = ids[max(0, j - MAX_MATCH):j]
            got, src = None, None
            ln, at = own.longest(tail, j)
            if ln >= ENTER_MATCH:
                c = own.copy(at, k, j)
                if len(c) >= 2:
                    got, src = c, "own"
            if got is None and rec is not None:
                ln, at = rec.longest(tail, rec_limit)
                if ln >= MIN_MATCH:
                    c = rec.copy(at, k, rec_limit)
                    if len(c) >= 2:
                        got, src = c, "recall"
            acc = 0
            if got is not None:
                m = min(len(got), end - j)
                eq = got[:m] == ids[j:j + m]
                acc = int(np.argmin(eq)) if not eq.all() else m
                st[src + "_copies"] += 1
                st[src + "_proposed"] += len(got)
                st[src + "_landed"] += acc
                st[src + "_missed_copies"] += int(acc == 0)
            step = min(acc + 1, end - j)
            kind = "think" if roles[j] == 2 else "answer"
            st[kind + "_tokens"] += step
            if src:
                st[kind + "_" + src + "_landed"] += min(acc, step)
            j += step
    st["assistant_tokens"] = st["think_tokens"] + st["answer_tokens"]
    return st


def entry_stats(stores, k, limit):
    paths = entries(stores, limit)
    seqs, firsts = [], defaultdict(list)
    total = Counter()
    news = []
    for path in paths:
        try:
            ids = read_ids(path)
        except (ValueError, KeyError, OSError) as e:
            total["unreadable"] += 1
            continue
        key = ids[:256].tobytes()
        lcp = 0
        for q in firsts.get(key, []):
            m = min(len(q), len(ids))
            d = np.nonzero(q[:m] != ids[:m])[0]
            lcp = max(lcp, int(d[0]) if len(d) else m)
        firsts[key].append(ids)
        total["entries"] += 1
        total["ids"] += len(ids)
        total["new_ids"] += len(ids) - lcp
        news.append((ids, lcp))
    # the record: every entry's new ids (with OVERLAP ids before them), a mark after each, in time order
    parts, limits = [], []
    at = 0
    for ids, lcp in news:
        seg = ids[max(0, lcp - OVERLAP):]
        limits.append(at)  # an entry reads the record as it was before it
        parts.append(seg)
        parts.append(np.array([-1], dtype=np.int64))
        at += len(seg) + 1
    stream = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    rec = Index(stream)
    for (ids, lcp), lim in zip(news, limits):
        if lcp >= len(ids):
            continue
        roles = roles_of(ids)
        total["new_assistant_ids"] += int((roles[lcp:] != 0).sum())
        own = Index(ids)
        total.update(walk(ids, roles, lcp, own, rec, lim, k))
    return dict(total)


def summarize(t, k):
    a = max(1, t.get("assistant_tokens", 0))
    s = {"k": k}
    for src in ("own", "recall"):
        c = t.get(src + "_copies", 0)
        s[src] = {"copies": c, "landed_per_copy": t.get(src + "_landed", 0) / max(1, c),
                  "landed_share_of_assistant_tokens": t.get(src + "_landed", 0) / a,
                  "copies_landing_nothing": t.get(src + "_missed_copies", 0) / max(1, c)}
    s["rounds_per_assistant_token_copy_only"] = t.get("rounds", 0) / a
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log")
    ap.add_argument("--store", action="append", default=[])
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--limit-entries", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = {}
    if a.log:
        res["log"] = log_stats(a.log)
        print(json.dumps(res["log"], indent=1))
    if a.store:
        t = entry_stats(a.store, a.k, a.limit_entries)
        res["entries"] = t
        res["summary"] = summarize(t, a.k)
        print(json.dumps(t, indent=1))
        print(json.dumps(res["summary"], indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
