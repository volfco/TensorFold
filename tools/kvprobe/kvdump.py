"""Read a GLM `TF_GLM_KV_DUMP` keep back as tensors (numpy only).

A keep `<dir>/glm-rank<r>-keep<id>` is three files written by the Zig engine (zig/src/families/glm/
cuda_snapshot_codec.zig, cuda_snapshot_layout.zig `manifest`):

- `.bin`   the kept state's raw bytes in plan order (the snapshot pages' order, rows ascending).
- `.parts` one `class length row` line per codec cut (1 MiB blocks inside each plan part); `tf-kv-codec-bench --file`
           reads it. Classes: prefix, latent, scale, ring, pooled, tail (zig/src/core/kv_ledger.zig).
- `.plan`  the manifest: `tfkvplan 1`, then
           `rank R world W pos P mtp_len M capacity C base B mtp_base MB paged 0|1 len N`
           `shape kv_lora .. i_dim .. hidden .. kda_layers .. heads .. dk 128 dv 128 conv_c .. rec_layer .. proj_row ..
                  ring_rows .. ring_row .. page_rows .. pool .. caches ..`
           `cur <one 0/1 per KDA layer>`        which half of the double-buffered recurrent state is live
           `half <h> <offset> <bytes>`          TF_GLM_KV_LIVE_HALF keeps: the dead half of `rec` left out of the dump
                                                (arena-head offsets; the head is then two prefix parts around it)
           `field <name> <offset in the arena head> <bytes>` for pos_dev conv rec proj save_k save_v save_g save_b
           `cache <j> kv8 0|1 pk_rows <n> [mtp]` one a DSA cache, the MTP layer's last
           `part <i> <class> <len> <row bytes> <cache j or ->` the plan's parts before cutting
           `end`
Layouts (zig/src/families/glm/cuda_state.zig State): conv bf16 [kda layers][3][conv_c]; rec fp32
[2][kda layers][heads][dk][dv], live half `cur[layer]`; latent rows bf16 [pos][kv_lora] (kv8: e4m3 [pos][kv_lora] then
a scale part, fp32 [pos]); rings bf16 [rows][i_dim], the index keys then the index gates, ring row r holding position
`base + r` (`mtp_base` for the MTP cache's); pooled keys bf16 [pools][i_dim], pool p covering positions 4p..4p+3.
Every reader check refuses by raising DumpError with the reason; nothing is guessed.
"""
import os
import numpy as np

CLASSES = ("prefix", "latent", "scale", "ring", "pooled", "tail")
FIELDS = ("pos_dev", "conv", "rec", "proj", "save_k", "save_v", "save_g", "save_b")


class DumpError(Exception):
    pass


def bf16(u16):
    """bf16 bit patterns (uint16) to float32."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 0xF and m == 7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_table()


def stems(directory):
    """Every keep in a dump directory, as (rank, id, stem), in rank then id order."""
    out = []
    for name in os.listdir(directory):
        if name.startswith("glm-rank") and name.endswith(".plan"):
            r, k = name[len("glm-rank"):-len(".plan")].split("-keep")
            out.append((int(r), int(k), os.path.join(directory, name[:-len(".plan")])))
    return sorted(out)


class Keep:
    def __init__(self, stem, mmap=True):
        self.stem = stem
        with open(stem + ".plan") as f:
            self.plan_text = f.read()
        self._parse(self.plan_text)
        size = os.path.getsize(stem + ".bin")
        if size != self.len:
            raise DumpError(f"{stem}.bin holds {size} bytes, the manifest's plan {self.len}")
        self.raw = np.memmap(stem + ".bin", dtype=np.uint8, mode="r") if mmap else np.fromfile(stem + ".bin", dtype=np.uint8)
        self._check_cuts()

    # ---- parsing --------------------------------------------------------------------------------------------------
    def _parse(self, text):
        lines = [l.split() for l in text.splitlines() if l.strip()]
        if not lines or lines[0] != ["tfkvplan", "1"]:
            raise DumpError("not a tfkvplan 1 manifest")
        if lines[-1] != ["end"]:
            raise DumpError("manifest truncated (no end line)")
        self.fields, self.caches, self.parts = {}, [], []
        self.meta, self.shape, self.cur, self.half = {}, {}, None, None
        for w in lines[1:-1]:
            k = w[0]
            if k == "rank":
                self.meta = {w[i]: int(w[i + 1]) for i in range(0, len(w), 2)}
            elif k == "shape":
                self.shape = {w[i]: int(w[i + 1]) for i in range(1, len(w), 2)}
            elif k == "cur":
                self.cur = [int(c) for c in (w[1] if len(w) > 1 else "")]
            elif k == "half":
                self.half = (int(w[1]), int(w[2]), int(w[3]))
            elif k == "field":
                self.fields[w[1]] = (int(w[2]), int(w[3]))
            elif k == "cache":
                self.caches.append({"j": int(w[1]), "kv8": int(w[3]), "pk_rows": int(w[5]), "mtp": len(w) > 6})
            elif k == "part":
                if w[2] not in CLASSES:
                    raise DumpError(f"unknown class {w[2]}")
                self.parts.append({"i": int(w[1]), "class": w[2], "len": int(w[3]), "row": int(w[4]),
                                   "cache": None if w[5] == "-" else int(w[5])})
            else:
                raise DumpError(f"unknown manifest line {k}")
        for need in ("rank", "world", "pos", "len", "base", "mtp_base"):
            if need not in self.meta:
                raise DumpError(f"manifest lacks {need}")
        for f in FIELDS:
            if f not in self.fields:
                raise DumpError(f"manifest lacks field {f}")
        self.rank, self.world, self.pos, self.len = (self.meta[k] for k in ("rank", "world", "pos", "len"))
        if self.cur is None or len(self.cur) != self.shape["kda_layers"]:
            raise DumpError("cur bits do not match kda_layers")
        off = 0
        for p in self.parts:
            p["off"] = off
            off += p["len"]
        if off != self.len:
            raise DumpError(f"manifest parts sum to {off}, its plan to {self.len}")
        if not self.parts or self.parts[0]["class"] != "prefix":
            raise DumpError("the first part is not the arena head")
        head = self.parts[0]["len"]
        if self.half is not None:  # the head in two prefix parts around the dead KDA half the keep left out
            h, hole, hl = self.half
            if len(self.parts) < 2 or self.parts[1]["class"] != "prefix" or hole != head:
                raise DumpError("a half keep's arena head is not two prefix parts around its hole")
            ro, rn = self.fields["rec"]
            if h not in (0, 1) or rn % 2 or hole != ro + (1 - h) * (rn // 2) or hl != rn // 2:
                raise DumpError(f"half {h} hole {hole}+{hl} is not the dead half of rec {ro}+{rn}")
            if any(b != h for b in self.cur):
                raise DumpError("a half keep's cur bits do not all select the kept half")
            head += hl + self.parts[1]["len"]
        for f, (o, n) in self.fields.items():
            if o + n > head:
                raise DumpError(f"field {f} ends at {o + n}, past the arena head's {head} bytes")

    def _check_cuts(self):
        with open(self.stem + ".parts") as f:
            cuts = [(w[0], int(w[1]), int(w[2]) if len(w) > 2 else None) for w in (l.split() for l in f) if w]
        if sum(c[1] for c in cuts) != self.len:
            raise DumpError(f"{self.stem}.parts sums to {sum(c[1] for c in cuts)}, the plan to {self.len}")
        # the cuts refine the parts: each part is one or more consecutive cuts of its class
        i = 0
        for p in self.parts:
            left = p["len"]
            while left > 0:
                if i >= len(cuts) or cuts[i][0] != p["class"] or cuts[i][1] > left:
                    raise DumpError(f"cut {i} does not lie inside part {p['i']} ({p['class']})")
                if cuts[i][2] is not None and cuts[i][2] != p["row"]:
                    raise DumpError(f"cut {i} row {cuts[i][2]} differs from part {p['i']}'s {p['row']}")
                left -= cuts[i][1]
                i += 1
        if i != len(cuts):
            raise DumpError("cuts past the last part")

    # ---- accessors ------------------------------------------------------------------------------------------------
    def _bytes(self, off, n):
        if off < 0 or off + n > self.len:
            raise DumpError(f"read {off}+{n} outside the dump's {self.len} bytes")
        return self.raw[off:off + n]

    def _head(self, off, n):
        """Bytes of the arena head at `off` (the manifest's field offsets), mapped past the dead half a half keep left out."""
        if self.half is not None:
            _, hole, hl = self.half
            if off < hole + hl and off + n > hole:
                raise DumpError(f"head bytes {off}+{n} lie in the dead half the keep left out ({hole}+{hl})")
            if off >= hole + hl:
                off -= hl
        return self._bytes(off, n)

    def field(self, name):
        o, n = self.fields[name]
        return self._head(o, n)

    def cache_parts(self, j):
        ps = [p for p in self.parts if p["cache"] == j]
        if not ps:
            raise DumpError(f"no parts for cache {j}")
        out = {"latent": None, "scale": None, "rings": [], "pooled": None}
        for p in ps:
            if p["class"] == "ring":
                out["rings"].append(p)
            else:
                out[p["class"]] = p
        if len(out["rings"]) not in (0, 2):
            raise DumpError(f"cache {j} has {len(out['rings'])} ring parts, not the index keys and gates")
        return out

    @property
    def dsa_caches(self):
        """Indices of the main (non-MTP) DSA caches."""
        return [c["j"] for c in self.caches if not c["mtp"]]

    def latent(self, j):
        """Cache j's latent rows as float32 [rows, kv_lora], and the raw codes (uint16 bf16, or uint8 e4m3)."""
        cp = self.cache_parts(j)
        p = cp["latent"]
        kv = self.shape["kv_lora"]
        kv8 = self.caches[j]["kv8"]
        if p is None:
            return np.zeros((0, kv), np.float32), np.zeros((0, kv), np.uint16)
        width = 1 if kv8 else 2
        if p["row"] != kv * width or p["len"] % p["row"]:
            raise DumpError(f"cache {j} latent rows of {p['row']} bytes, kv_lora {kv} x {width}")
        b = self._bytes(p["off"], p["len"])
        if kv8:
            codes = np.asarray(b).reshape(-1, kv)
            s = cp["scale"]
            if s is None or s["len"] != 4 * codes.shape[0]:
                raise DumpError(f"cache {j}: an e4m3 latent without one fp32 scale a row")
            scale = np.asarray(self._bytes(s["off"], s["len"])).view(np.float32)
            return E4M3[codes] * scale[:, None], codes
        codes = np.asarray(b).view(np.uint16).reshape(-1, kv)
        return bf16(codes), codes

    def rings(self, j):
        """Cache j's index-key and index-gate rows (bf16 codes [rows, i_dim] each) and the position of row 0."""
        cp = self.cache_parts(j)
        if not cp["rings"]:
            return None, None, None
        i_dim = self.shape["i_dim"]
        out = []
        for p in cp["rings"]:
            if p["row"] != 2 * i_dim:
                raise DumpError(f"cache {j} ring rows of {p['row']} bytes, i_dim {i_dim}")
            out.append(np.asarray(self._bytes(p["off"], p["len"])).view(np.uint16).reshape(-1, i_dim))
        first = self.meta["mtp_base"] if self.caches[j]["mtp"] else self.meta["base"]
        return out[0], out[1], first

    def pooled(self, j):
        """Cache j's pooled index keys, bf16 codes [pools, i_dim] (pool p: positions 4p..4p+3)."""
        p = self.cache_parts(j)["pooled"]
        if p is None:
            return np.zeros((0, self.shape["i_dim"]), np.uint16)
        if p["row"] != 2 * self.shape["i_dim"]:
            raise DumpError(f"cache {j} pooled rows of {p['row']} bytes")
        return np.asarray(self._bytes(p["off"], p["len"])).view(np.uint16).reshape(-1, self.shape["i_dim"])

    def kda_state(self, layer):
        """KDA layer `layer`'s live recurrent state, float32 [heads, dk, dv] (this rank's heads)."""
        s = self.shape
        nk, heads, dk, dv = s["kda_layers"], s["heads"], s["dk"], s["dv"]
        per = s["rec_layer"]
        if per != heads * dk * dv * 4:
            raise DumpError(f"rec_layer {per} is not heads {heads} x {dk} x {dv} fp32")
        o, n = self.fields["rec"]
        if n != 2 * nk * per:
            raise DumpError(f"rec field {n} bytes, not 2 x {nk} x {per}")
        if not 0 <= layer < nk:
            raise DumpError(f"KDA layer {layer} of {nk}")
        if self.half is not None and self.cur[layer] != self.half[0]:
            raise DumpError(f"KDA layer {layer}'s live half {self.cur[layer]} is not the half the keep carries")
        at = o + (self.cur[layer] * nk + layer) * per
        return np.asarray(self._head(at, per)).view(np.float32).reshape(heads, dk, dv)

    def conv(self, layer):
        """KDA layer `layer`'s short-conv history, float32 [3, conv_c]."""
        s = self.shape
        o, n = self.fields["conv"]
        per = 3 * s["conv_c"] * 2
        if n != s["kda_layers"] * per:
            raise DumpError(f"conv field {n} bytes, not {s['kda_layers']} x {per}")
        return bf16(np.asarray(self._head(o + layer * per, per)).view(np.uint16).reshape(3, s["conv_c"]))


# ---- a writer for synthetic captures (host tests); the Zig host test checks the reader against the engine's own ---
def write(stem, *, rank=0, world=1, pos, kv_lora, i_dim, heads, kda_layers, latents, rings=None, pooled=None,
          rec=None, cur=None, conv=None, base=0, hidden=64, mtp_base=0, block=1 << 20, half=None):
    """Write a dump in the engine's format. latents: list of bf16 code arrays [pos, kv_lora] (one a DSA cache, no MTP
    cache); rec: float32 [kda_layers, heads, 128, 128] live states (written to half `cur`); conv: float32 [layers, 3, c];
    half: a TF_GLM_KV_LIVE_HALF keep carrying that half of `rec` alone (every cur bit must select it)."""
    dk = dv = 128
    conv_c = 3 * heads * dk
    rec_layer = heads * dk * dv * 4
    cur = cur if cur is not None else [0] * kda_layers
    sizes = [("pos_dev", 4), ("conv", kda_layers * 3 * conv_c * 2), ("rec", 2 * kda_layers * rec_layer),
             ("proj", 0), ("save_k", 0), ("save_v", 0), ("save_g", 0), ("save_b", 0)]
    head = bytearray()
    offs = {}
    for name, n in sizes:
        at = (len(head) + 255) // 256 * 256
        head.extend(b"\0" * (at - len(head)))
        offs[name] = (at, n)
        head.extend(b"\0" * n)
    head[0:4] = np.int32(pos).tobytes()
    if conv is not None:
        o, n = offs["conv"]
        head[o:o + n] = np.asarray(conv, np.float32).view(np.uint32).__rshift__(16).astype(np.uint16).tobytes()
    if rec is not None:
        o, _ = offs["rec"]
        for li in range(kda_layers):
            at = o + (cur[li] * kda_layers + li) * rec_layer
            head[at:at + rec_layer] = np.asarray(rec[li], np.float32).tobytes()
    if half is None:
        parts = [("prefix", bytes(head), 0, None)]
    else:
        if any(c != half for c in cur):
            raise ValueError("half: every layer's cur bit must select the kept half")
        dead, hole = offs["rec"][0] + (1 - half) * kda_layers * rec_layer, kda_layers * rec_layer
        parts = [("prefix", bytes(head[:dead]), 0, None), ("prefix", bytes(head[dead + hole:]), 0, None)]
    ring_rows = 0
    for j, lat in enumerate(latents):
        parts.append(("latent", np.asarray(lat, np.uint16).tobytes(), 2 * kv_lora, j))
        if rings is not None:
            for r in rings[j]:
                parts.append(("ring", np.asarray(r, np.uint16).tobytes(), 2 * i_dim, j))
                ring_rows = len(r)
        pk = pooled[j] if pooled is not None else np.zeros((pos // 4 + 2, i_dim), np.uint16)
        parts.append(("pooled", np.asarray(pk, np.uint16).tobytes(), 2 * i_dim, j))
    parts.append(("tail", b"\0" * 8, 0, None))
    total = sum(len(p[1]) for p in parts)
    with open(stem + ".bin", "wb") as f:
        for p in parts:
            f.write(p[1])
    with open(stem + ".parts", "w") as f:
        for cls, data, row, _ in parts:
            for a in range(0, len(data), block):
                f.write(f"{cls} {min(block, len(data) - a)} {row}\n")
    with open(stem + ".plan", "w") as f:
        f.write("tfkvplan 1\n")
        f.write(f"rank {rank} world {world} pos {pos} mtp_len 0 capacity {pos} base {base} mtp_base {mtp_base} paged 0 len {total}\n")
        f.write(f"shape kv_lora {kv_lora} i_dim {i_dim} hidden {hidden} kda_layers {kda_layers} heads {heads} dk 128 dv 128 "
                f"conv_c {conv_c} rec_layer {rec_layer} proj_row 0 ring_rows {ring_rows} ring_row {2 * i_dim} page_rows 512 "
                f"pool 4 caches {len(latents)}\n")
        f.write("cur " + "".join(str(c) for c in cur) + "\n")
        if half is not None:
            f.write(f"half {half} {dead} {hole}\n")
        for name, (o, n) in offs.items():
            f.write(f"field {name} {o} {n}\n")
        for j in range(len(latents)):
            f.write(f"cache {j} kv8 0 pk_rows {pos // 4 + 2}\n")
        for i, (cls, data, row, j) in enumerate(parts):
            f.write(f"part {i} {cls} {len(data)} {row} {'-' if j is None else j}\n")
        f.write("end\n")
