"""hints.py against the Zig reader's golden sidecar (zig/src/core/kv_hints_test.zig pins the same sha256), Zig's
Wyhash test vectors, and the post-processor on a synthetic keep and routing dump. Run: python3 tools/kvprobe/test_hints.py"""
import hashlib, os, shutil, struct, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hints, kvdump

GOLDEN = "0b40a058ad3fbd9b9736ab2cf398efc6e4de94228b3d3dda23db28214ddb1df1"


def bf(x):
    return struct.pack("<H", np.float32(x).view(np.uint32) >> 16)


class Format(unittest.TestCase):
    def test_wyhash_vectors(self):
        # std/hash/wyhash.zig's test vectors (seeds 0..6)
        for seed, text, want in ((0, b"", 0x409638ee2bde459), (1, b"a", 0xa8412d091b5fe0a9), (2, b"abc", 0x32dd92e4b2915153),
                                 (3, b"message digest", 0x8619124089a3a16b), (4, b"abcdefghijklmnopqrstuvwxyz", 0x7a43afb61d7f5f40),
                                 (5, b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0xff42329b90e50d58),
                                 (6, b"12345678901234567890123456789012345678901234567890123456789012345678901234567890", 0xc39cab13b115aad3)):
            self.assertEqual(hints.wyhash(seed, text), want, (seed, text))

    def test_golden_matches_the_zig_writer(self):
        ids = [7 + i % 3 for i in range(1026)]
        w = hints.Writer(config=b"\xc0" * 32, checkpoint=b"\xc7" * 32, layout=1, page_rows=512, pool=4, i_dim=2, dsa_layers=11,
                         experts=288, moe_layers=42, token_count=len(ids), tokens_hash=hints.tokens_hash(ids))
        w.section("bounds", 3, 2, 8, b"".join(bf(x) for x in (-1, -2, 1, 2, 0, -0.5, 0.25, 0.5)))
        sel = bytearray(256 * 4)
        sel[0] = 5
        w.section("selected", 3, 256, 4, sel)
        w.section("hot", hints.ALL, 2, 4, struct.pack("<II", 1, 0))
        w.section("experts", hints.ALL, 1, 24, struct.pack("<IHH", 9, 41, 2) + struct.pack("<8H", 287, 3, 0, 0, 0, 0, 0, 0))
        self.assertEqual(hashlib.sha256(w.finish()).hexdigest(), GOLDEN)


class PostProcessor(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_bounds_are_sound_for_every_pool(self):
        rng = np.random.default_rng(3)
        pos, i_dim = 1500, 8
        pk = (np.asarray(rng.standard_normal((pos // 4 + 2, i_dim)), np.float32).view(np.uint32) >> 16).astype(np.uint16)
        stem = os.path.join(self.dir, "glm-rank0-keep1")
        kvdump.write(stem, pos=pos, kv_lora=16, i_dim=i_dim, heads=1, kda_layers=1, latents=[np.zeros((pos, 16), np.uint16)], pooled=[pk])
        k = kvdump.Keep(stem)
        data = hints.build(k, list(range(pos)), b"\0" * 32, b"\1" * 32)
        n = struct.unpack_from("<I", data, 36)[0]
        kind, layer, count, width, off, size = struct.unpack_from("<IIIIQQ", data, hints.HEADER)
        self.assertEqual((n, kind, layer, count, width), (1, 1, 0, pos // 512, 4 * i_dim))
        b = kvdump.bf16(np.frombuffer(data, "<u2", count * 2 * i_dim, off)).reshape(count, 2, i_dim)
        vals = kvdump.bf16(pk)
        for p in range(count):
            page = vals[p * 128:(p + 1) * 128]
            self.assertTrue(np.all(page >= b[p, 0]) and np.all(page <= b[p, 1]))
            self.assertTrue(np.all(page.min(0) == b[p, 0]) and np.all(page.max(0) == b[p, 1]))  # tight, not just sound
        self.assertEqual(struct.unpack_from("<Q", data, len(data) - 8)[0], hints.wyhash(0, data[:-8]))

    def test_routing_sections(self):
        pos = 1024
        stem = os.path.join(self.dir, "glm-rank0-keep1")
        kvdump.write(stem, pos=pos, kv_lora=16, i_dim=8, heads=1, kda_layers=1, latents=[np.zeros((pos, 16), np.uint16)] * 11)
        ids = [i % 50 for i in range(pos)]
        experts = np.tile(np.arange(8, dtype=np.int16), (pos, 42, 1))
        sel = np.full((4, 512), -1, np.int32)
        sel[:, 0] = 200  # pool 200: page 1 (pools 128..255)
        rd = {"request": np.zeros(pos, int), "pos": np.arange(pos), "token": np.asarray(ids, np.int32), "experts": experts,
              "dsa_sel": {7: (sel, np.array([600, 700, 800, 900]))}}
        data = hints.build(kvdump.Keep(stem), ids, b"\0" * 32, b"\1" * 32, rd, 0)
        n = struct.unpack_from("<I", data, 36)[0]
        table = [struct.unpack_from("<IIIIQQ", data, hints.HEADER + 32 * i) for i in range(n)]
        kinds = [(t[0], t[1]) for t in table]
        self.assertIn((2, 1), kinds)  # selected counts for layer 7 = cache 1
        sel_t = next(t for t in table if t[0] == 2)
        counts = np.frombuffer(data, "<u4", sel_t[2], sel_t[4])
        self.assertEqual(int(counts[200]), 4)
        hot = next(t for t in table if t[0] == 3)
        self.assertEqual(list(np.frombuffer(data, "<u4", hot[2], hot[4])), [1, 0])
        ex = next(t for t in table if t[0] == 4)
        self.assertEqual(ex[2], 50 * 42)
        # rows 600..900 see one whole page (page 0), which holds no selected pool: all of it could be skipped
        self.assertEqual(hints.prunable_upper_bound(rd)[7]["p50"], 1.0)
        sel[:, 1] = 10  # now page 0 holds a selected pool on every row: nothing is prunable
        self.assertEqual(hints.prunable_upper_bound(rd)[7]["p50"], 0.0)

    def test_refuses_mismatched_routing(self):
        pos = 512
        stem = os.path.join(self.dir, "glm-rank0-keep1")
        kvdump.write(stem, pos=pos, kv_lora=16, i_dim=8, heads=1, kda_layers=1, latents=[np.zeros((pos, 16), np.uint16)])
        rd = {"request": np.zeros(pos, int), "pos": np.arange(pos), "token": np.ones(pos, np.int32),
              "experts": np.zeros((pos, 42, 8), np.int16), "dsa_sel": {}}
        with self.assertRaises(SystemExit):
            hints.build(kvdump.Keep(stem), [2] * pos, b"\0" * 32, b"\1" * 32, rd, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
