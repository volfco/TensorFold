"""Loader for the serving stack's draft dump with the routing records of patch 0710 (numpy only).

On-disk format (patch 0430's dump, extended by 0710; rank 0 writes):

- ``<dir>/meta.json``: dump settings (``hidden``, ``layers``, ``tap_layers``, the ``GLM53_TF_*`` environment).
- ``<dir>/index.jsonl``: one JSON object a segment. ``kind`` "p" (prefill rows) or "d" (decode rows), ``start`` (the
  first row's absolute position), ``n`` (rows), ``h0``/``h1`` (prefix hashes), ``shard`` (file name) and ``arrays``:
  name -> [byte offset in the shard, dtype, shape]. 0430's arrays: ``tokens`` int32 [n], ``hidden`` bfloat16 [n, D],
  ``taps`` bfloat16 [n, T*D] (optional), ``lp`` float16 [n, k], ``ids`` int32 [n, k] (top-k ids, largest first).
  0710 adds, on prefill segments only, when its knobs were on:
    ``route_ids`` int16 [n, Lm, 8] and ``route_w`` float32 [n, Lm, 8], with ``route_layers`` (Lm layer ids) in the
    record; ``kda_lg`` float32 [n, Lk, H] (mean log forget gate per head) with ``kda_layers`` and ``kda_heads``;
    ``dsa_sel`` int32 [N, 512] (selected pool ids, ascending), ``dsa_pos`` int32 [N] (absolute positions) and
    ``dsa_layer`` int16 [N], with ``dsa_stride``.
- ``<dir>/shard-NNNNN.bin``: the raw little-endian arrays.

``load`` returns the prefill rows of every request, in dump order:

    request [R] int64      request ordinal (a request starts at a segment with start 0 or a break in positions)
    pos [R] int64          absolute position
    token [R] int32
    top1 [R] int32         the row's greedy next token (ids[:, 0]), or absent when the dump has no ids
    experts [R, Lm, 8] int16   -1 where a segment carried no routing
    weights [R, Lm, 8] float32 0 there
    moe_layers [Lm]
    dsa_sel {layer: (sel [R', 512] int32 with -1 padding, rows [R'] int64 index into R)}
    kda_gate [R, Lk, H] float32 or None (NaN where a segment carried none); kda_layers [Lk]
    requests [(first row, rows)] per request
"""
from __future__ import annotations

import json
import os

import numpy as np

_DT = {"bfloat16": np.uint16, "int32": np.int32, "<i4": np.int32, "int16": np.int16, "int64": np.int64,
       "float16": np.float16, "float32": np.float32, "uint8": np.uint8}


class Dump:
    """Index records and memory-mapped shards of one dump directory."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.meta = {}
        if os.path.exists(os.path.join(path, "meta.json")):
            with open(os.path.join(path, "meta.json")) as f:
                self.meta = json.load(f)
        with open(os.path.join(path, "index.jsonl")) as f:
            self.recs = [json.loads(line) for line in f if line.strip()]
        self._maps: dict[str, np.memmap] = {}

    def arr(self, rec: dict, name: str) -> np.ndarray:
        off, dtype, shape = rec["arrays"][name]
        dt = np.dtype(_DT.get(dtype, dtype))
        f = self._maps.get(rec["shard"])
        if f is None:
            f = self._maps[rec["shard"]] = np.memmap(os.path.join(self.path, rec["shard"]), dtype=np.uint8, mode="r")
        n = int(np.prod(shape)) * dt.itemsize
        if off + n > f.shape[0]:
            raise ValueError(f"{rec['shard']}: array {name} at {off}+{n} past the shard's {f.shape[0]} bytes")
        return f[off:off + n].view(dt).reshape(shape)


def requests_of(recs: list[dict]) -> list[list[dict]]:
    """Prefill segments grouped into requests: a new one at start 0 or wherever positions do not continue."""

    out: list[list[dict]] = []
    end = None
    for r in recs:
        if r["kind"] != "p":
            continue
        if r["start"] == 0 or end is None or r["start"] != end:
            out.append([])
        out[-1].append(r)
        end = r["start"] + r["n"]
    return out


def load(path: str) -> dict:
    d = Dump(path)
    reqs = requests_of(d.recs)
    segs = [s for req in reqs for s in req]
    moe_layers: list[int] = []
    kda_layers: list[int] = []
    heads = 0
    for s in segs:
        if "route_ids" in s["arrays"]:
            if moe_layers and s["route_layers"] != moe_layers:
                raise ValueError(f"segment {s.get('seg')}: route layers {s['route_layers']} differ from {moe_layers}")
            moe_layers = list(s["route_layers"])
        if "kda_lg" in s["arrays"]:
            if kda_layers and s["kda_layers"] != kda_layers:
                raise ValueError(f"segment {s.get('seg')}: KDA layers differ")
            kda_layers, heads = list(s["kda_layers"]), int(s["kda_heads"])
    R = sum(s["n"] for s in segs)
    out = {
        "request": np.empty(R, np.int64), "pos": np.empty(R, np.int64), "token": np.empty(R, np.int32),
        "experts": np.full((R, len(moe_layers), 8), -1, np.int16), "weights": np.zeros((R, len(moe_layers), 8), np.float32),
        "moe_layers": moe_layers, "kda_layers": kda_layers,
        "kda_gate": np.full((R, len(kda_layers), heads), np.nan, np.float32) if kda_layers else None,
        "dsa_sel": {}, "requests": [], "meta": d.meta,
    }
    has_top1 = all("ids" in s["arrays"] for s in segs) and segs
    if has_top1:
        out["top1"] = np.empty(R, np.int32)
    sel: dict[int, list] = {}
    at = 0
    for qi, req in enumerate(reqs):
        first = at
        for s in req:
            n = s["n"]
            rows = slice(at, at + n)
            out["request"][rows] = qi
            out["pos"][rows] = s["start"] + np.arange(n)
            out["token"][rows] = d.arr(s, "tokens")
            if has_top1:
                out["top1"][rows] = d.arr(s, "ids")[:, 0]
            if "route_ids" in s["arrays"]:
                out["experts"][rows] = d.arr(s, "route_ids")
                out["weights"][rows] = d.arr(s, "route_w")
            if "kda_lg" in s["arrays"]:
                out["kda_gate"][rows] = d.arr(s, "kda_lg")
            if "dsa_sel" in s["arrays"]:
                ps, ls, ss = d.arr(s, "dsa_pos").astype(np.int64), d.arr(s, "dsa_layer"), d.arr(s, "dsa_sel")
                if ps.size and ((ps < s["start"]).any() or (ps >= s["start"] + n).any()):
                    raise ValueError(f"segment {s.get('seg')}: a selection's position lies outside its rows")
                for layer in np.unique(ls):
                    m = ls == layer
                    sel.setdefault(int(layer), []).append((np.asarray(ss[m]), at + ps[m] - s["start"]))
            at += n
        out["requests"].append((first, at - first))
    for layer, parts in sel.items():
        out["dsa_sel"][layer] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    return out


def write_synthetic(path: str, requests: list[dict], *, seg_rows: int = 512, meta: dict | None = None) -> None:
    """Write a dump in the format above (for tests). Each request: ``tokens`` [T]; optional ``top1`` [T],
    ``experts`` [T, Lm, 8], ``weights``, ``moe_layers``, ``kda_lg`` [T, Lk, H], ``kda_layers``, and
    ``dsa`` {layer: (positions [N], sel [N, 512])} with absolute positions; ``start`` (default 0)."""

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta or {"version": 1, "hidden": 0, "layers": 0}, f)
    shard = open(os.path.join(path, "shard-00001.bin"), "wb")
    index = open(os.path.join(path, "index.jsonl"), "w")
    off = 0
    seg = 0

    def put(arrays: dict, rec: dict) -> None:
        nonlocal off, seg
        where = {}
        for name, a in arrays.items():
            a = np.ascontiguousarray(a)
            where[name] = [off, str(a.dtype), list(a.shape)]
            shard.write(a.tobytes())
            off += a.nbytes
        seg += 1
        rec.update(seg=seg, shard="shard-00001.bin", arrays=where)
        index.write(json.dumps(rec) + "\n")

    for q in requests:
        toks = np.asarray(q["tokens"], np.int32)
        base = int(q.get("start", 0))
        for a in range(0, len(toks), seg_rows):
            b = min(a + seg_rows, len(toks))
            arrays = {"tokens": toks[a:b]}
            rec = {"kind": "p", "start": base + a, "n": b - a, "h0": None, "h1": None, "tap_layers": []}
            if "top1" in q:
                arrays["ids"] = np.asarray(q["top1"], np.int32)[a:b, None]
                arrays["lp"] = np.zeros((b - a, 1), np.float16)
            if "experts" in q:
                arrays["route_ids"] = np.asarray(q["experts"], np.int16)[a:b]
                arrays["route_w"] = np.asarray(q["weights"], np.float32)[a:b]
                rec["route_layers"] = list(q["moe_layers"])
            if "kda_lg" in q:
                arrays["kda_lg"] = np.asarray(q["kda_lg"], np.float32)[a:b]
                rec.update(kda_layers=list(q["kda_layers"]), kda_heads=int(np.shape(q["kda_lg"])[2]))
            if "dsa" in q:
                pos, lay, sel = [], [], []
                for layer, (ps, ss) in q["dsa"].items():
                    ps = np.asarray(ps)
                    m = (ps >= base + a) & (ps < base + b)
                    pos.append(ps[m]); sel.append(np.asarray(ss)[m]); lay.append(np.full(m.sum(), layer))
                arrays["dsa_sel"] = np.concatenate(sel).astype(np.int32).reshape(-1, 512)
                arrays["dsa_pos"] = np.concatenate(pos).astype(np.int32)
                arrays["dsa_layer"] = np.concatenate(lay).astype(np.int16)
                rec["dsa_stride"] = 1
            put(arrays, rec)
    shard.close()
    index.close()
