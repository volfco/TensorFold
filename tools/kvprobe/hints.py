"""Cache-entry hints: the external post-processor. Reads a TF_GLM_KV_DUMP keep (kvdump.py) and, when given, a routing
dump of the same prompt (routedump.py, the serving stack's patch 0710), and writes the sidecar the engine's reader
(zig/src/core/kv_hints.zig) validates. Format: tools/kvprobe/HINTS.md.

    python3 hints.py KEEP_STEM IDS_JSON OUT.tfhints --config CONFIG_JSON --checkpoint HEX64
        [--routing DUMP_DIR --request N] [--hot-rows 16] [--expert-tokens 4096] [--estimate]

IDS_JSON: the entry's prompt ids (a JSON list; the first `pos` are the kept state's). --estimate prints, from the
routing dump's DSA selections, the fraction of whole pages holding no selected pool per sampled row: an upper bound on
what an exact page prune could skip (a page with a selected pool is never prunable).
"""
import hashlib, json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvdump

MAGIC = b"TFKVHNT1"
VERSION, HEADER, ENTRY, ALL = 1, 192, 32, 0xFFFFFFFF
KIND = {"bounds": 1, "selected": 2, "hot": 3, "experts": 4}
EXPERT_ENTRY = 24
M64 = (1 << 64) - 1
SECRET = (0xa0761d6478bd642f, 0xe7037ed1a0b428db, 0x8ebc6af09c88c6e3, 0x589965cc75374cc3)


def _mix(a, b):
    x = a * b
    return (x & M64) ^ (x >> 64)


def _mum(a, b):
    x = a * b
    return x & M64, x >> 64


def _r(data, at, n):
    return int.from_bytes(data[at:at + n], "little")


def wyhash(seed, data):
    """Zig std.hash.Wyhash.hash (0.17), the sidecar's checksum and token hash."""
    data = bytes(data)
    n = len(data)
    s0 = seed ^ _mix(seed ^ SECRET[0], SECRET[1])
    st = [s0, s0, s0]
    if n <= 16:
        if n >= 4:
            end, quarter = n - 4, (n >> 3) << 2
            a = (_r(data, 0, 4) << 32) | _r(data, quarter, 4)
            b = (_r(data, end, 4) << 32) | _r(data, end - quarter, 4)
        elif n > 0:
            a, b = (data[0] << 16) | (data[n >> 1] << 8) | data[n - 1], 0
        else:
            a = b = 0
    else:
        i = 0
        if n >= 48:
            while i + 48 < n:
                for k in range(3):
                    st[k] = _mix(_r(data, i + 16 * k, 8) ^ SECRET[k + 1], _r(data, i + 16 * k + 8, 8) ^ st[k])
                i += 48
            st[0] ^= st[1] ^ st[2]
        rest = data[i:]
        j = 0
        while j + 16 < len(rest):
            st[0] = _mix(_r(rest, j, 8) ^ SECRET[1], _r(rest, j + 8, 8) ^ st[0])
            j += 16
        a, b = _r(data, n - 16, 8), _r(data, n - 8, 8)
    a ^= SECRET[1]
    b ^= st[0]
    a, b = _mum(a, b)
    return _mix(a ^ SECRET[0] ^ n, b ^ SECRET[1])


def tokens_hash(ids):
    return wyhash(0, np.asarray(ids, dtype="<u4").tobytes())


class Writer:
    def __init__(self, *, config, checkpoint, layout, page_rows, pool, i_dim, dsa_layers, experts, moe_layers,
                 token_count, tokens_hash):
        assert len(config) == 32 and len(checkpoint) == 32
        h = bytearray(HEADER)
        h[0:8] = MAGIC
        for at, v in ((8, VERSION), (12, HEADER), (16, layout), (20, page_rows), (24, pool), (28, i_dim),
                      (32, dsa_layers), (128, experts), (132, moe_layers)):
            h[at:at + 4] = struct.pack("<I", v)
        h[40:48] = struct.pack("<Q", token_count)
        h[48:56] = struct.pack("<Q", tokens_hash)
        h[56:88] = config
        h[88:120] = checkpoint
        self.head, self.table, self.body = h, [], bytearray()

    def section(self, kind, layer, count, width, data):
        data = bytes(data)
        assert len(data) == count * width
        self.table.append((KIND[kind], layer, count, width, len(self.body), len(data)))
        self.body += data
        self.body += b"\0" * ((8 - len(self.body) % 8) % 8)

    def finish(self):
        base = HEADER + ENTRY * len(self.table)
        total = base + len(self.body) + 8
        self.head[36:40] = struct.pack("<I", len(self.table))
        self.head[120:128] = struct.pack("<Q", total)
        out = bytearray(self.head)
        for kind, layer, count, width, off, n in self.table:
            out += struct.pack("<IIIIQQ", kind, layer, count, width, off + base, n)
        out += self.body
        out += struct.pack("<Q", wyhash(0, out))
        return bytes(out)


def page_bounds(pooled, page_pools, pages):
    """[pages][min | max][i_dim] bf16 codes of each whole page's pooled keys (min and max of bf16 values are exact)."""
    vals = kvdump.bf16(pooled[:pages * page_pools]).reshape(pages, page_pools, -1)
    lo, hi = vals.min(axis=1), vals.max(axis=1)
    to = lambda x: (np.ascontiguousarray(x, np.float32).view(np.uint32) >> 16).astype("<u2")
    return np.stack([to(lo), to(hi)], axis=1)


def _cache_of(layer, caches):
    """The DSA cache index of a global layer id (GLM-5.3-Flash: layers 3, 7, ..., 43 are caches 0..10)."""
    j = (layer - 3) // 4
    return j if (layer - 3) % 4 == 0 and 0 <= j < caches else None


def routing_rows(rd, request):
    rows = np.nonzero(rd["request"] == request)[0]
    if rows.size == 0:
        raise SystemExit(f"request {request} not in the routing dump")
    return rows


def selections(rd, rows):
    """Per DSA layer (global id): the sampled rows' selected pool ids [n, K] (-1 padded) of this request's rows."""
    out = {}
    keep = set(rows.tolist())
    for layer, (sel, at) in rd["dsa_sel"].items():
        m = np.array([r in keep for r in at.tolist()], dtype=bool)
        if m.any():
            out[int(layer)] = (sel[m], at[m])
    return out


def build(keep, ids, cfg_sha, ckpt, rd=None, request=None, hot_rows=16, expert_tokens=4096):
    s = keep.shape
    pos = keep.pos
    if len(ids) < pos:
        raise SystemExit(f"{len(ids)} ids for a state kept at {pos}")
    ids = list(ids[:pos])
    page_rows, pool = s["page_rows"], s["pool"]
    page_pools, pages = page_rows // pool, pos // page_rows
    caches = keep.dsa_caches
    experts, moe_layers = 288, 42
    w = Writer(config=cfg_sha, checkpoint=ckpt, layout=1, page_rows=page_rows, pool=pool, i_dim=s["i_dim"],
               dsa_layers=len(caches), experts=experts, moe_layers=moe_layers, token_count=pos, tokens_hash=tokens_hash(ids))
    for j in caches:
        if pages:
            b = page_bounds(keep.pooled(j), page_pools, pages)
            w.section("bounds", j, pages, 4 * s["i_dim"], b.tobytes())
    if rd is not None:
        rows = routing_rows(rd, request)
        if not np.array_equal(rd["token"][rows][:pos], np.asarray(ids[:len(rows)], np.int32)[:pos]):
            raise SystemExit("the routing dump's request does not hold the entry's tokens")
        sel = selections(rd, rows)
        touched = {}
        for layer, (x, at) in sel.items():
            j = _cache_of(layer, len(caches))
            if j is None:
                continue
            counts = np.zeros(pos // pool, "<u4")
            v = x[(x >= 0) & (x < counts.size)]
            np.add.at(counts, v, 1)
            w.section("selected", j, counts.size, 4, counts.tobytes())
            last = x[np.argsort(at)[-hot_rows:]]  # the last sampled rows: the round a resume continues from
            for p in (last[last >= 0] * pool // page_rows).tolist():
                touched[p] = touched.get(p, 0) + 1
        held = -(-pos // page_rows)
        order = sorted((p for p in touched if p < held), key=lambda p: (-touched[p], p))
        order += [p for p in range(held) if p not in touched]
        w.section("hot", ALL, len(order), 4, np.asarray(order, "<u4").tobytes())
        ent = expert_prior(rd["token"][rows], rd["experts"][rows], expert_tokens)
        if ent:
            w.section("experts", ALL, len(ent), EXPERT_ENTRY, b"".join(ent))
    return w.finish()


def expert_prior(tokens, experts, cap):
    """Per token id (the `cap` most frequent) and MoE layer: its 8 most often routed experts in this request."""
    tokens = np.asarray(tokens)
    uniq, counts = np.unique(tokens, return_counts=True)
    keep = uniq[np.argsort(-counts, kind="stable")[:cap]]
    out = []
    for t in sorted(keep.tolist()):
        rows = experts[tokens == t]  # [n, layers, 8]
        for layer in range(rows.shape[1]):
            v = rows[:, layer, :].ravel()
            v = v[v >= 0]
            if v.size == 0:
                continue
            ids, c = np.unique(v, return_counts=True)
            top = ids[np.argsort(-c, kind="stable")[:8]]
            pad = np.zeros(8, "<u2")
            pad[:top.size] = top
            out.append(struct.pack("<IHH", t, layer, top.size) + pad.tobytes())
    return out


def prunable_upper_bound(rd, page_rows=512, pool=4):
    """Per DSA layer: over sampled rows, the fraction of whole pages before the row holding no selected pool."""
    out = {}
    for layer, (sel, at) in rd["dsa_sel"].items():
        pos = rd["pos"][at]
        fr = []
        for x, p in zip(sel, pos):
            pages = int(p) // page_rows
            if pages == 0:
                continue
            hit = np.unique(x[x >= 0] * pool // page_rows)
            fr.append(1 - np.count_nonzero(hit < pages) / pages)
        if fr:
            out[int(layer)] = {"rows": len(fr), "p50": float(np.median(fr)), "mean": float(np.mean(fr))}
    return out


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    opts = {"--config": None, "--checkpoint": None, "--routing": None, "--request": "0", "--hot-rows": "16",
            "--expert-tokens": "4096"}
    args, it, estimate = [], iter(argv[1:]), False
    for a in it:
        if a == "--estimate":
            estimate = True
        elif a in opts:
            opts[a] = next(it)
        else:
            args.append(a)
    if not opts["--config"] or not opts["--checkpoint"]:
        raise SystemExit("--config and --checkpoint are required: the reader refuses a sidecar of another model")
    cfg = hashlib.sha256(open(opts["--config"], "rb").read()).digest()
    ckpt = bytes.fromhex(opts["--checkpoint"])
    if len(ckpt) != 32:
        raise SystemExit("--checkpoint takes the 64-hex-digit checkpoint digest")
    rd = None
    if opts["--routing"]:
        import routedump
        rd = routedump.load(opts["--routing"])
        if estimate:
            print(json.dumps(prunable_upper_bound(rd), indent=1))
    keep = kvdump.Keep(args[0])
    data = build(keep, json.load(open(args[1])), cfg, ckpt, rd, int(opts["--request"]), int(opts["--hot-rows"]),
                 int(opts["--expert-tokens"]))
    open(args[2], "wb").write(data)
    print(f"wrote {args[2]}: {len(data)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
