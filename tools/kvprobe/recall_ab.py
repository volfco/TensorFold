"""The served Recall A/B (docs/recall.md "Benchmarks and gates"): real conversations replayed against the Zig server,
once with `--recall` on a fresh record and once with `--recall-read-only` on an empty one (the lead starts each).

    recall_ab.py run --base URL --label recall --store DIR [--store DIR] --out DIR [--conversations N] [--dry-run]
    recall_ab.py compare DIR_RECALL DIR_READ_ONLY [--json OUT] [--md OUT]
    recall_ab.py selftest

Conversations come from the serving stack's session store, the format recall_yield.py reads
(<store>/<compat>/rank0/entries/*.tfs: a JSON header whose `ids` are a kept conversation prefix's token ids). Entries
are taken in time order (mtime); an entry whose ids are a prefix of an earlier conversation's, or extend them, is
that conversation (the longest is kept). Sessions that share only a system prompt and tools stay apart; a
conversation the template re-rendered (earlier reasoning stripped) shows as two and replays twice. A conversation's ids split at the role tokens: the system
span becomes the system message; each run of user (and, with `--tools as-user`, observation) spans before an
assistant span becomes one user turn, its text from the server's `/detokenize`; the record's assistant spans are
dropped, as the model's own greedy reply (temperature 0, `--max-tokens` 256) feeds the next turn. The last span of
an entry is cut by the keep grid: a cut user span is not sent.

Per reply: rounds, drafted, accepted (the reply's `speculative` block, else the server log's `done` line with
`--log`), decode tok/s (`tensorfold.tokens_per_second`), the token sha256, prompt and cached tokens. After the run:
`GET /v1/recall` (ids, copies proposed and accepted, bands). Nothing of the conversations' text is written: the
output holds counts and hashes only. `--dry-run` reads the store and prints the turn structure (text pending: it
needs /detokenize) without a server.
"""
import argparse, hashlib, json, os, statistics, sys, time, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import served
from recall_yield import SYSTEM, USER, ASSISTANT, OBSERVATION, EOT, read_ids, entries

ROLES = {SYSTEM: "system", USER: "user", ASSISTANT: "assistant", OBSERVATION: "observation", EOT: "end"}


def lcp(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    ne = np.nonzero(a[:n] != b[:n])[0]
    return int(ne[0]) if len(ne) else n


def conversations(seqs):
    """`seqs`: (name, ids) in time order -> conversations, each the longest ids of its entries."""
    out = []
    for name, ids in seqs:
        for c in out:
            if lcp(c["ids"], ids) == min(len(c["ids"]), len(ids)):  # one is a prefix of the other
                c["entries"] += 1
                if len(ids) > len(c["ids"]):
                    c["ids"] = ids
                break
        else:
            out.append({"ids": ids, "entries": 1, "first": name})
    for c in out:
        c["key"] = hashlib.sha256(np.asarray(c["ids"][:256], dtype="<i4").tobytes()).hexdigest()[:12]
    return out


def spans(ids):
    """(role, start, end) for each role token's span; ids before the first role token are the template's prefix."""
    at = [i for i, t in enumerate(ids.tolist()) if t in ROLES]
    return [(ROLES[int(ids[s])], s + 1, e) for s, e in zip(at, at[1:] + [len(ids)])]


def turns(ids, tools="as-user"):
    """The replay plan: (system span or None, [user turns, each a list of (role, start, end)]). None: skipped."""
    sp = spans(ids)
    if sp and sp[-1][0] != "assistant":
        sp = sp[:-1]  # a cut user span is not sent (a cut assistant span still ends the turn before it)
    system = [s for s in sp if s[0] == "system"]
    out, pending = [], []
    for s in sp:
        if s[0] in ("user", "observation"):
            if s[0] == "observation" and tools == "skip":
                return None
            pending.append(s)
        elif s[0] == "assistant" and pending:
            out.append(pending)
            pending = []
    return system, out


def text_of(srv, ids, parts):
    if srv is None:
        return "\n\n".join(f"<{r} {e - s} ids>" for r, s, e in parts)
    return "\n\n".join(srv.text(ids[s:e].tolist()).strip() for _, s, e in parts)


def load_store(stores, limit, tools="as-user"):
    """The conversations with at least one user turn to replay (the first `limit`), and a count of the rest."""
    paths = entries(stores, 0)
    seqs = [(os.path.basename(p), read_ids(p)) for p in paths]
    convs, skipped = [], {"entries": len(paths), "no turn": 0, "tool results (--tools skip)": 0}
    for c in conversations(seqs):
        got = turns(c["ids"], tools)
        if got is None:
            skipped["tool results (--tools skip)"] += 1
        elif not got[1]:
            skipped["no turn"] += 1  # no user span before an assistant one: a single cut prompt, or no role tokens
        else:
            c["system"], c["turns"] = got
            convs.append(c)
    return (convs[:limit] if limit else convs), skipped


def run(a):
    srv = None if a.dry_run else served.Server(a.base)
    convs, skipped = load_store(a.store, a.conversations, a.tools)
    print(f"store: {skipped['entries']} entries; skipped: {skipped['no turn']} with no user turn before an assistant span, "
          f"{skipped['tool results (--tools skip)']} with tool results (--tools skip); {len(convs)} to replay")
    if not convs:
        raise SystemExit(f"recall_ab: refused: no conversation under {a.store} has a user turn followed by the model's reply "
                         "(single cut prompts and entries without role tokens are not conversations)")
    os.makedirs(a.out, exist_ok=True)
    tail = served.LogTail(a.log) if a.log and not a.dry_run else None
    rows, plan = [], []
    for ci, c in enumerate(convs):
        system, ts = c["system"], c["turns"][: a.max_turns]
        plan.append({"conv": ci, "key": c["key"], "ids": len(c["ids"]), "entries": c["entries"], "turns": len(ts),
                     "system_ids": sum(e - s for _, s, e in system), "turn_ids": [sum(e - s for _, s, e in t) for t in ts]})
        print(f"conversation {ci} ({c['key']}): {len(c['ids'])} ids from {c['entries']} entries, {len(ts)} turns", flush=True)
        if a.dry_run:
            continue
        msgs = [{"role": "system", "content": text_of(srv, c["ids"], system)}] if system else []
        for ti, parts in enumerate(ts):
            msgs.append({"role": "user", "content": text_of(srv, c["ids"], parts)})
            row = {"conv": ci, "key": c["key"], "turn": ti + 1}
            try:
                r = srv.chat(served.chat_body(list(msgs), a.max_tokens), stream=a.stream)
            except urllib.error.HTTPError as e:
                row["error"] = f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}"
                rows.append(row)
                print(f"  turn {ti + 1}: {row['error']} (conversation stops)", flush=True)
                break
            for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "rounds", "drafted", "accepted",
                      "decode_tps", "decode_s", "total_s", "ttft_s", "token_sha256"):
                row[k] = r.get(k)
            if tail is not None:
                time.sleep(0.2)
                done = served.parse_log(tail.new())["done"]
                if done:
                    row["log_done"] = {k: done[-1].get(k) for k in ("rounds", "accepted", "drafted", "sha", "tokens", "decode")}
                    for k in ("rounds", "accepted", "drafted"):
                        if row.get(k) is None:
                            row[k] = done[-1].get(k)
            rows.append(row)
            print(f"  turn {ti + 1}: prompt {row['prompt_tokens']} cached {row['cached_tokens']} tokens {row['completion_tokens']} "
                  f"rounds {row['rounds']} accepted {row['accepted']}/{row['drafted']} tok/s {row['decode_tps']} sha {(row['token_sha256'] or '')[:12]}", flush=True)
            msgs.append({"role": "assistant", "content": r["content"]})
    if a.dry_run:
        json.dump({"plan": plan, "skipped": skipped}, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
        return
    try:
        recall = served.get(srv.base, "/v1/recall")
    except (urllib.error.URLError, ValueError) as e:
        recall = {"error": str(e)}
    out = {"meta": {"label": a.label, "model": srv.name(), "max_tokens": a.max_tokens, "tools": a.tools,
                    "conversations": len(convs), "max_turns": a.max_turns, "stream": a.stream, "skipped": skipped},
           "plan": plan, "replies": rows, "recall": recall, "summary": summary(rows)}
    json.dump(out, open(os.path.join(a.out, "recall-ab.json"), "w"), indent=1)
    print(json.dumps(out["summary"]))
    print(f"recall: {json.dumps(recall.get('copies'))} ids {recall.get('ids')} state {recall.get('state')}")


def summary(rows):
    ok = [r for r in rows if "error" not in r and r.get("completion_tokens") is not None]
    tok = sum(r["completion_tokens"] for r in ok)
    rounds = sum(r.get("rounds") or 0 for r in ok)
    dec = sum(r.get("decode_s") or 0 for r in ok)
    tps = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    return {"replies": len(ok), "errors": len(rows) - len(ok), "tokens": tok, "rounds": rounds,
            "accepted": sum(r.get("accepted") or 0 for r in ok), "drafted": sum(r.get("drafted") or 0 for r in ok),
            "tokens_per_round": tok / rounds if rounds else None,
            "decode_tps_weighted": sum(max(0, r["completion_tokens"] - 1) for r in ok) / dec if dec else None,
            "decode_tps_median": statistics.median(tps) if tps else None,
            "seconds": sum(r.get("total_s") or 0 for r in ok)}


# ---- compare -------------------------------------------------------------------------------------------------------
def compare(da, db):
    A = json.load(open(os.path.join(da, "recall-ab.json")))
    B = json.load(open(os.path.join(db, "recall-ab.json")))
    kb = {(r["conv"], r["turn"]): r for r in B["replies"]}
    pairs, first = [], None
    for r in A["replies"]:
        o = kb.get((r["conv"], r["turn"]))
        if o is None or "error" in r or "error" in o:
            continue
        same = r["token_sha256"] == o["token_sha256"]
        if not same and first is None:
            first = (r["conv"], r["turn"])
        pairs.append({"conv": r["conv"], "turn": r["turn"], "same": same, "rounds": [r.get("rounds"), o.get("rounds")],
                      "tps": [r.get("decode_tps"), o.get("decode_tps")], "tokens": [r["completion_tokens"], o["completion_tokens"]]})
    return {"labels": [A["meta"]["label"], B["meta"]["label"]], "summary": [A["summary"], B["summary"]],
            "recall": [A["recall"], B["recall"]], "pairs": pairs, "same": sum(p["same"] for p in pairs),
            "first_difference": first}


def num(x, f="{:.2f}"):
    return "-" if x is None else (f.format(x) if isinstance(x, float) else str(x))


def markdown(res):
    la, lb = res["labels"]
    sa, sb = res["summary"]
    out = [f"| | {la} | {lb} | {la} / {lb} |", "|---|---|---|---|"]
    for k in ("replies", "tokens", "rounds", "accepted", "drafted", "tokens_per_round", "decode_tps_weighted", "decode_tps_median", "seconds"):
        ratio = sa[k] / sb[k] if isinstance(sa.get(k), (int, float)) and sb.get(k) else None
        out.append(f"| {k} | {num(sa.get(k))} | {num(sb.get(k))} | {num(ratio, '{:.3f}')} |")
    for lab, rc in zip(res["labels"], res["recall"]):
        cp = rc.get("copies") or {}
        bands = ", ".join(f"{b.get('matched_from')}: {num(b.get('landing_rate'), '{:.3f}')}" for b in rc.get("bands") or [])
        out.append(f"\n`GET /v1/recall` ({lab}): state {rc.get('state')}, ids {rc.get('ids')}, copies proposed "
                   f"{cp.get('proposals')} ({cp.get('proposed')} tokens), accepted {cp.get('accepted')}; landing by band {bands or '-'}")
    n = len(res["pairs"])
    out.append(f"\nreplies byte-identical: {res['same']}/{n}" + ("" if res["first_difference"] is None else
               f" (first difference: conversation {res['first_difference'][0]} turn {res['first_difference'][1]})"))
    return "\n".join(out)


def cmd_compare(a):
    res = compare(a.recall, a.read_only)
    md = markdown(res)
    print(md)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if a.md:
        open(a.md, "w").write(md + "\n")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="replay the conversations against one configuration's server")
    r.add_argument("--base", default="http://127.0.0.1:18080")
    r.add_argument("--label", required=True, help="recall or read-only (labels the run; the server's flags decide)")
    r.add_argument("--store", action="append", required=True, help="a session store root (holds <compat>/rank0/entries)")
    r.add_argument("--out", required=True)
    r.add_argument("--conversations", type=int, default=0, help="the first N conversations in time order (0: all)")
    r.add_argument("--max-turns", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--tools", choices=("as-user", "skip"), default="as-user",
                   help="observation (tool result) spans: sent as user text, or the conversation skipped")
    r.add_argument("--stream", action="store_true", help="stream each reply (adds the time to first token)")
    r.add_argument("--log", help="rank 0's server log: draft counters from its done lines when a reply lacks them")
    r.add_argument("--dry-run", action="store_true", help="the turn structure from the store, no server")
    c = sub.add_parser("compare", help="the A/B table")
    c.add_argument("recall")
    c.add_argument("read_only")
    c.add_argument("--json")
    c.add_argument("--md")
    sub.add_parser("selftest", help="synthetic store and an in-process fake server")
    a = p.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "compare":
        return cmd_compare(a)
    import fake_server
    return fake_server.selftest_recall(main)


if __name__ == "__main__":
    main()
