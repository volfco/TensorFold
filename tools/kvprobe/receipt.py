"""The lazy KV codec's receipt in the repository's style (docs/receipts/glm-kv-pages-8e705835/README.md):

    receipt.py --off DIR --eager DIR --lazy DIR [--recall DIR --read-only DIR] --commit SHA --binaries FILE
               [--repo ROOT] [--out-root docs/receipts] [--date YYYY-MM-DD] [--platform TEXT] [--checkpoint TEXT]
               [--host-tests FILE] [--extra LABEL=DIR ...] [--notes FILE.md] [--force]
    receipt.py selftest

The run directories are lazy_receipt.py's (after `collect`) and recall_ab.py's. Writes
<out-root>/glm-kv-codec-lazy-<short sha>/: README.md (Scope, What changed, Results, Reproduction, Remaining coverage
limits, Records, Publication handling), records/ (the runs' JSON and server logs, the compare outputs), commit.txt
(from `git` in --repo, read only), binaries.txt (the lead's file), redaction-report.json and SHA256SUMS (every file
but itself). Machine names, addresses, interfaces and home paths are replaced with RUNBOOK.md's placeholders.
"""
import argparse, datetime, hashlib, json, os, re, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lazy_receipt, recall_ab

REDACT = [  # (pattern, placeholder): RUNBOOK.md's names where it has one
    (re.compile(r"/home/[^/\s\"']+"), "HOME"),
    (re.compile(r"/Users/[^/\s\"']+"), "HOME"),
    (re.compile(r"/nfs\d*/[^\s\"']*"), "MODEL"),
    (re.compile(r"/tmp/[^\s\"']*"), "TMP"),
    (re.compile(r"\b(?:gx10|spark|dgx)[-\w]*\b", re.I), "HOST"),
    (re.compile(r"\b192\.168\.\d+\.\d+\b"), "LINK_ADDR"),
    (re.compile(r"\b10\.\d+\.\d+\.\d+\b"), "ADDR"),
    (re.compile(r"\b(?:enp|eth|roce|rocep|roceP)\w*\b"), "IFACE"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "EMAIL"),
]


def redact(text, extra=()):
    counts = {}
    for pat, rep in list(REDACT) + list(extra):
        text, n = pat.subn(rep, text)
        if n:
            counts[rep] = counts.get(rep, 0) + n
    return text, counts


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout.strip()


def fmt(x, f="{:.2f}"):
    return "-" if x is None else (f.format(x) if isinstance(x, float) else str(x))


def codec_row(name, totals):
    if not totals or totals[0] is None:
        return f"no KVCODEC totals ({name}: the codec is off, or the ranks were not stopped with SIGINT)"
    parts = []
    for rk, t in enumerate(totals):
        if t is None:
            parts.append(f"rank {rk}: no totals line")
            continue
        parts.append(f"rank {rk}: `encode_ms={t.get('encode_ms')}` on the prompt pass, `lazy_encode_ms={t.get('lazy_encode_ms')}` on the "
                     f"workers, `decode_ms={t.get('decode_ms')}`, `copy_ms={t.get('copy_ms', '-')}`, `lazy_copy_ms={t.get('lazy_copy_ms', '-')}`, "
                     f"`keep_io_ms={t.get('keep_io_ms', '-')}`, `wait_ms={t.get('wait_ms', '-')}`; keeps {t.get('keeps')}, compacted "
                     f"{t.get('compacted')}, ratio {t.get('ratio')}, pages {t.get('raw_pages')} -> {t.get('kept_pages')}, threads {t.get('threads')}")
    return "; ".join(parts)


def results(res, runs, ab):
    names = res["configs"]
    rows = ["| Check | Result |", "| --- | --- |"]
    n = len(res["turns"])
    same = "identical in all {} turns (token sha256)".format(n) if res["first_difference"] is None else \
        "differ from turn {} (first difference)".format(res["first_difference"])
    rows.append(f"| Replies, {' / '.join(names)} (`records/compare.json`) | {same} |")
    for name in names:
        rows.append(f"| Codec timers, {name} (`records/{name}/log-summary.json`) | {codec_row(name, res['codec'].get(name))} |")
    for name in names:
        ps = res["pages"].get(name) or []
        cells = []
        for rk, p in enumerate(ps):
            cells.append(f"rank {rk}: {p['trimmed']} trims, {p['freed']} pages freed, last `free={p['free_last']}`; the store's "
                         f"last line {p['held_mib_last']} MiB held, {p['spare_mib_last']} MiB spare")
        rows.append(f"| Pages, {name} | {'; '.join(cells) or 'no log'} |")
    lz = (res["codec"].get("lazy") or [None])[0] or {}
    forms = res["forms"].get("lazy") or {}
    rows.append(f"| Restore forms, lazy | totals: restores {lz.get('restores', '-')}, raw {lz.get('raw_restores', '-')}, part compacted "
                f"{lz.get('partial_restores', '-')}; the extra cases as the log shows them before each restore: raw case {forms.get('raw', '-')}, "
                f"partial case {forms.get('partial', '-')} |")
    refused = [line for r in runs for lg in ((r["log"] or {}).get("ranks") or []) for line in lg["refused"]]
    rows.append(f"| `KVCODEC refused` or `BadSnapshotControl` | {len(refused)} lines{'' if not refused else ': ' + refused[0][:120]} |")
    if ab:
        sa, sb = ab["summary"]
        rows.append(f"| Recall A/B, {ab['labels'][0]} vs {ab['labels'][1]} (`records/recall-ab.md`) | {sa['replies']} replies each; byte-identical "
                    f"{ab['same']}/{len(ab['pairs'])}; rounds {sa['rounds']} vs {sb['rounds']}; tokens a round {fmt(sa['tokens_per_round'])} vs "
                    f"{fmt(sb['tokens_per_round'])}; decode tok/s {fmt(sa['decode_tps_weighted'])} vs {fmt(sb['decode_tps_weighted'])} |")
    ttft = ["| Turn | Kind | Prompt | Cached " + " / ".join(names) + " | TTFT s " + " / ".join(names) + " | Total s " + " / ".join(names) + " | Same |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for t in res["turns"]:
        ttft.append(f"| {t['turn']} | {t['kind']} | {t['prompt']} | {' / '.join(str(c) for c in t['cached'])} | "
                    f"{' / '.join(fmt(x) for x in t['ttft_s'])} | {' / '.join(fmt(x) for x in t['total_s'])} | {'yes' if t['same'] else 'NO'} |")
    return "\n".join(rows), "\n".join(ttft)


def limits(res, runs, ab):
    out = []
    lz = (res["codec"].get("lazy") or [None])[0] or {}
    if not lz:
        out.append("The lazy run has no KVCODEC totals line, so its timers and restore counts are not in this receipt.")
    else:
        if not lz.get("raw_restores"):
            out.append("No raw restore was served: the raw case's resume came after its state's lazy delay.")
        if not lz.get("partial_restores"):
            out.append("No part-compacted restore was served: the partial case's restore fell before or after the "
                       "compaction (the host tests in `zig/glm_snapshot_host_lazy.zig` cover that form).")
    for r in runs:
        ranks = (r["log"] or {}).get("ranks") or []
        if len(ranks) < 2:
            out.append(f"`{r['meta']['config']}`: only rank 0's log was collected; rank 1's totals are not in this receipt.")
    out += ["One conversation of {} turns a configuration, served once each; TTFT differences of a few percent are within "
            "one serve's variation (CONTRIBUTING.md asks for alternating runs to separate them).".format(len(res["turns"])),
            "Two ranks only: the page agreement under a rank-skewed schedule and the extra gate exchange's cost are not measured.",
            "The keep's raw copy into host pages (`pool.move` and the stream's synchronize in `cuda_snapshots.keepAs`) is outside "
            "the codec and in none of its timers; it is the same in every configuration with the host cache on."]
    if not ab:
        out.append("The served Recall A/B was not run with this receipt.")
    else:
        out.append(f"Recall A/B: {ab['summary'][0]['replies']} replies from the session store's sample; observation (tool "
                   "result) spans were sent as user text, so a conversation's rendering differs from its recorded one.")
    return "\n".join("- " + x for x in out)


def reproduction(runs):
    m = {r["meta"]["config"]: r["meta"] for r in runs}
    lz = m.get("lazy") or next(iter(m.values()))
    args = (f"--lazy-ms {lz['lazy_ms']} --pause {lz['pause_s']} --max-tokens {lz['max_tokens']} --first-tokens {lz['first_tokens']} "
            f"--piece-tokens {lz['piece_tokens']} --partial-frac {lz['partial_frac']} --source {lz['source']}")
    return f"""Placeholders as in `tools/kvprobe/RUNBOOK.md`: `R0` and `R1` the two hosts (rank 0 serves HTTP), `R0_LINK` rank 0's
address on the direct link, `MODEL` the checkpoint, `TF` this commit's tree on both hosts, `OUT` a results directory
on R0. Each configuration is served alone (`tools/kvprobe/RUNBOOK-lazy.md` steps 1 and 2), both ranks under
`taskset -c 0-4,10-14`, `serve MODEL --context 32768 --parallel 1 --prompt-cache-gib 16 --no-drafts`,
`TF_GLM_HOST_CACHE=1 TF_GLM_KV_LEDGER=1 TF_GLM_KV_DUMP=OUT/dump-<cfg>` on both ranks, and:

| config | extra environment (both ranks) |
| --- | --- |
| `off` | `TF_GLM_KV_CODEC=0` |
| `eager` | `TF_GLM_KV_CODEC=1 TF_GLM_KV_CODEC_CPUS=WORKERS` |
| `lazy` | `TF_GLM_KV_CODEC=1 TF_GLM_KV_CODEC_LAZY=1 TF_GLM_KV_CODEC_LAZY_MS={lz['lazy_ms']} TF_GLM_KV_CODEC_CPUS=WORKERS` |

Per configuration, from R0 (the server on 127.0.0.1:18080):

```bash
python3 tools/kvprobe/lazy_receipt.py run --base http://127.0.0.1:18080 --config <cfg> --out OUT/<cfg> --log OUT/<cfg>-r0.log {args}
# stop rank 0 with SIGINT (both ranks tear down and print their KVCODEC totals), then
python3 tools/kvprobe/lazy_receipt.py collect OUT/<cfg> --log OUT/<cfg>-r0.log --log OUT/<cfg>-r1.log
```

Then `lazy_receipt.py compare OUT/off OUT/eager OUT/lazy --json OUT/compare.json`. The Recall A/B: two serves of the
same build, `--recall` on a fresh record and `--recall-read-only` on an empty one, each replayed with
`recall_ab.py run --base http://127.0.0.1:18080 --label <recall|read-only> --store SESSIONS --out OUT/<label>
--log OUT/<label>-r0.log`, compared with `recall_ab.py compare OUT/recall OUT/read-only`. This folder:
`receipt.py --off OUT/off --eager OUT/eager --lazy OUT/lazy [--recall OUT/recall --read-only OUT/read-only]
--commit <sha> --binaries binaries.txt`."""


def build(a):
    runs = [lazy_receipt.load(d) for d in (a.off, a.eager, a.lazy)]
    res = lazy_receipt.compare([a.off, a.eager, a.lazy])
    ab = recall_ab.compare(a.recall, a.read_only) if a.recall else None
    sha = git(a.repo, "rev-parse", a.commit)
    folder = os.path.join(a.out_root, f"glm-kv-codec-lazy-{sha[:8]}")
    if os.path.exists(folder):
        if not a.force:
            raise SystemExit(f"receipt: refused: {folder} exists (--force replaces it)")
        shutil.rmtree(folder)
    os.makedirs(os.path.join(folder, "records"))
    report = {}

    def put(rel, text):
        text, counts = redact(text)
        if counts:
            report[rel] = counts
        path = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    for r in runs:
        name = r["meta"]["config"]
        for f in sorted(os.listdir(r["dir"])):
            if f.endswith((".json", ".log")):
                put(f"records/{name}/{f}", open(os.path.join(r["dir"], f), encoding="utf-8", errors="replace").read())
    put("records/compare.json", json.dumps(res, indent=1))
    put("records/compare.md", lazy_receipt.table(res) + "\n")
    if ab:
        for label, d in zip(ab["labels"], (a.recall, a.read_only)):
            put(f"records/{label}/recall-ab.json", open(os.path.join(d, "recall-ab.json")).read())
        put("records/recall-ab.json", json.dumps(ab, indent=1))
        put("records/recall-ab.md", recall_ab.markdown(ab) + "\n")
    if a.host_tests:
        put("records/host-tests.log", open(a.host_tests, encoding="utf-8", errors="replace").read())
    extras = []  # (label, run): further lazy_receipt runs published beside the three compared ones
    for spec in a.extra or ():
        label, _, d = spec.partition("=")
        if not d or not re.fullmatch(r"[\w.-]+", label):
            raise SystemExit(f"--extra takes LABEL=DIR, got {spec!r}")
        r = lazy_receipt.load(d)
        for f in sorted(os.listdir(d)):
            if f.endswith((".json", ".log")):
                put(f"records/{label}/{f}", open(os.path.join(d, f), encoding="utf-8", errors="replace").read())
        extras.append((label, r))
    notes = open(a.notes).read().strip() + "\n\n" if a.notes else ""
    put("binaries.txt", open(a.binaries).read())
    put("commit.txt", "\n".join([f"tested implementation: {sha} ({git(a.repo, 'log', '-1', '--format=%s', sha)})",
                                 f"implementation tree: {git(a.repo, 'rev-parse', sha + '^{tree}')}",
                                 "after it: the receipt's commit adds this folder and changes no source", ""]))
    table, ttft = results(res, runs, ab)
    for label, r in extras:
        ranks = (r["log"] or {}).get("ranks") or []
        table += f"\n| Codec timers, {label} (`records/{label}/log-summary.json`) | {codec_row(label, [lg['totals'][-1] if lg['totals'] else None for lg in ranks])} |"
    if extras:
        ttft += "\n\nFurther runs (`records/<label>/turns.json`), TTFT a turn in seconds:\n\n| Turn | " + " | ".join(l for l, _ in extras) + " |\n| --- |" + " --- |" * len(extras) + "\n"
        ttft += "\n".join(f"| {i + 1} | " + " | ".join(fmt(r["turns"][i].get("ttft_s")) if i < len(r["turns"]) else "-" for _, r in extras) + " |" for i in range(max(len(r["turns"]) for _, r in extras)))
    readme = f"""# GLM-5.3-Flash on CUDA: the KV codec's lazy compaction, served, receipt

Tested implementation: the binaries built from `{sha[:8]}…` (commit.txt has the commit and its tree). The receipt's
commit adds this folder and changes no source.
Test date: {a.date}. Platform: {a.platform}.
Toolchain: Zig 0.17.0, nvcc for sm_121; binaries.txt has every binary's sha256.
Checkpoint: {a.checkpoint}.

## Scope

This qualifies the lossless KV codec on GLM's kept prompt states with the host cache on, served at TP=2: the codec
off, eager (compaction inside the keep, on the prompt pass) and lazy (`TF_GLM_KV_CODEC_LAZY=1`: the keep returns raw
and the workers compact it after the delay), one growing greedy conversation each: replies byte for byte across the
three, time to first token per turn, where the encode runs and what it costs, pages given back, and restores from
compacted, raw and part-compacted entries.{' It also carries the served Recall A/B of `docs/recall.md`.' if ab else ''}
It does not qualify the codec at other TP sizes, under concurrent streams or with drafts on.

## What changed for these runs

- The codec's timers mean the same in both modes (`tools/kvprobe/STATUS-lazy.md` "Timers"): `encode_ms` and
  `lazy_encode_ms` count the workers' encode waves only, `decode_ms` the decode waves only; the ledger's read and the
  dump's writes are `keep_io_ms`, the engine thread's page copies and staging `copy_ms`, the compactor's write-back
  `lazy_copy_ms`, and the engine thread's waits for the workers or a compaction `wait_ms`. The eager and lazy totals
  of session record 11.11 (5x apart both ways for the same bytes) had timed the same work; the eager run's workers
  were unpinned and shared the engine's five CPUs, the lazy run's had five of their own. Here eager and lazy both
  pin the workers to the same CPUs (`TF_GLM_KV_CODEC_CPUS`).
- `tools/kvprobe/lazy_receipt.py` (the conversation, its raw and part-compacted cases, `collect`, `compare`),
  `recall_ab.py` (the Recall A/B) and `receipt.py` (this folder).

## Results

{table}

Time to first token per turn (the first streamed chunk carrying text; `records/<config>/turns.json`):

{ttft}

{notes}## Reproduction

{reproduction(runs)}

## Remaining coverage limits

{limits(res, runs, ab)}

## Records

`records/<config>/turns.json` (each turn's timings, tokens, hashes and the log lines it caused),
`records/<config>/log-summary.json` and `server-r<rank>.log` (the server logs), `records/compare.json` and
`compare.md`{', `records/recall-ab.json`, `recall-ab.md` and each configuration\'s `recall-ab.json`' if ab else ''}{', `records/host-tests.log`' if a.host_tests else ''};
`commit.txt`, `binaries.txt`, `redaction-report.json`, `SHA256SUMS`.

## Publication handling

Machine names, addresses, network interfaces and home and local paths replaced with placeholders
(`redaction-report.json` counts them a file); result values and token hashes kept. The conversation's text is not
included (`lazy_receipt.py run` rebuilds it from the tree's `docs/*.md` or its fixed synthetic prose), and the
Recall records hold counts and hashes only. SHA256SUMS pins every published file except itself.
"""
    put("README.md", readme)
    with open(os.path.join(folder, "redaction-report.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    sums = []
    for root, _, files in os.walk(folder):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), folder)
            if rel != "SHA256SUMS":
                sums.append(f"{hashlib.sha256(open(os.path.join(root, f), 'rb').read()).hexdigest()}  {rel}")
    with open(os.path.join(folder, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sorted(sums, key=lambda s: s.split("  ", 1)[1])) + "\n")
    print(f"wrote {folder}")
    return folder


def selftest():
    import fake_server
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    lz = fake_server.selftest_lazy(lazy_receipt.main)["_dirs"]
    rc = fake_server.selftest_recall(recall_ab.main)["_dirs"]
    with open(os.path.join(lz["lazy"], "server-r0.log"), "a") as f:
        f.write("info: binary=/home/someone/x on gx10-abc0 via 192.168.100.1 enp1s0f0np0\n")
    tmp = tempfile.mkdtemp(prefix="receipt-")
    bins = os.path.join(tmp, "binaries.txt")
    open(bins, "w").write("0" * 64 + "  tensorfold-native\n")
    args = ["--off", lz["off"], "--eager", lz["eager"], "--lazy", lz["lazy"], "--recall", rc["recall"], "--read-only", rc["read-only"],
            "--commit", "HEAD", "--binaries", bins, "--repo", root, "--out-root", tmp]
    folder = main(args)
    text = open(os.path.join(folder, "README.md")).read()
    for sec in ("## Scope", "## What changed", "## Results", "## Reproduction", "## Remaining coverage limits", "## Records"):
        fake_server.check(sec in text, f"README has {sec}")
    fake_server.check("identical in all 9 turns" in text and "Recall A/B" in text, "README results: replies and the Recall A/B")
    every = "".join(open(os.path.join(r, f), errors="replace").read() for r, _, fs in os.walk(folder) for f in fs)
    fake_server.check(not re.search(r"/home/|gx10-|192\.168\.|enp1s0", every), "no machine names, addresses or home paths")
    rep = json.load(open(os.path.join(folder, "redaction-report.json")))
    fake_server.check(rep.get("records/lazy/server-r0.log", {}).get("HOST") == 1, "the redaction report counts them")
    for line in open(os.path.join(folder, "SHA256SUMS")):
        h, rel = line.rstrip("\n").split("  ", 1)
        fake_server.check(hashlib.sha256(open(os.path.join(folder, rel), "rb").read()).hexdigest() == h, f"SHA256SUMS: {rel}")
    try:
        main(args)
        fake_server.check(False, "an existing folder is refused")
    except SystemExit as e:
        fake_server.check("refused" in str(e), "an existing folder is refused by name")
    print(f"selftest passed ({folder})")


def main(argv=None):
    if (argv if argv is not None else sys.argv[1:])[:1] == ["selftest"]:
        return selftest()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--off", "--eager", "--lazy", "--commit", "--binaries"):
        p.add_argument(k, required=True)
    p.add_argument("--recall", help="recall_ab.py's --recall run")
    p.add_argument("--read-only", help="recall_ab.py's --recall-read-only run")
    p.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    p.add_argument("--out-root", default=None, help="default: <repo>/docs/receipts")
    p.add_argument("--date", default=datetime.date.today().isoformat())
    p.add_argument("--platform", default="two GB10 (sm_121, 128 GB shared memory) at TP=2 over a direct RoCE link, ranks pinned to CPUs 0-4,10-14")
    p.add_argument("--checkpoint", default="the GLM-5.3-Flash checkpoint of `tools/kvprobe/RUNBOOK.md` (`MODEL`)")
    p.add_argument("--host-tests", help="a log of the host tests on the tested commit")
    p.add_argument("--extra", action="append", help="LABEL=DIR: a further lazy_receipt run published under records/LABEL (repeatable)")
    p.add_argument("--notes", help="a Markdown file inserted before Reproduction (the lead's notes on the runs)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if bool(a.recall) != bool(a.read_only):
        raise SystemExit("receipt: refused: --recall and --read-only go together")
    a.out_root = a.out_root or os.path.join(a.repo, "docs", "receipts")
    return build(a)


if __name__ == "__main__":
    main()
