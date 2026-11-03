"""kvdump.py on a TF_GLM_KV_LIVE_HALF=1 keep: the arena head in two parts around the dead KDA half, the fields past
the hole mapped, the live states and the conv history exact, the dead half refused, and a manifest whose parity does
not select the kept half refused. Run: python3 tools/kvprobe/test_kvdump_half.py"""
import os, shutil, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvdump


class Half(unittest.TestCase):
    kv, i_dim, heads, layers, pos = 16, 8, 2, 3, 64

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        self.rec = rng.standard_normal((self.layers, self.heads, 128, 128)).astype(np.float32)
        self.conv = rng.standard_normal((self.layers, 3, 3 * self.heads * 128)).astype(np.float32)
        self.lat = (rng.standard_normal((self.pos, self.kv)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def dump(self, name, cur, half):
        stem = os.path.join(self.dir, name)
        kvdump.write(stem, pos=self.pos, kv_lora=self.kv, i_dim=self.i_dim, heads=self.heads, kda_layers=self.layers,
                     latents=[self.lat], rec=self.rec, cur=cur, conv=self.conv, half=half, block=4096)
        return stem

    def test_half_keep_reads_back_exactly_and_is_smaller_by_the_dead_half(self):
        for h in (0, 1):
            full = kvdump.Keep(self.dump(f"glm-rank0-keep{1 + 2 * h}", [h] * self.layers, None))
            part = kvdump.Keep(self.dump(f"glm-rank0-keep{2 + 2 * h}", [h] * self.layers, h))
            rec_layer = self.heads * 128 * 128 * 4
            self.assertEqual(part.half, (h, full.fields["rec"][0] + (1 - h) * self.layers * rec_layer, self.layers * rec_layer))
            self.assertEqual(full.len - part.len, self.layers * rec_layer)
            self.assertEqual([p["class"] for p in part.parts[:3]], ["prefix", "prefix", "latent"])
            for li in range(self.layers):
                np.testing.assert_array_equal(part.kda_state(li), self.rec[li])
                np.testing.assert_array_equal(part.kda_state(li), full.kda_state(li))
                # conv is bf16 in the dump: both keeps hold the same codes
                np.testing.assert_array_equal(part.conv(li), full.conv(li))
            np.testing.assert_array_equal(part.latent(0)[1], full.latent(0)[1])
            np.testing.assert_array_equal(np.asarray(part.field("pos_dev")), np.asarray(full.field("pos_dev")))
            # the whole rec field spans the hole: refused by name
            with self.assertRaises(kvdump.DumpError):
                part.field("rec")
            ro, rn = part.fields["rec"]
            with self.assertRaises(kvdump.DumpError):
                part._head(ro + (1 - h) * (rn // 2), rec_layer)

    def test_writer_refuses_a_layer_on_the_other_half(self):
        with self.assertRaises(ValueError):
            self.dump("glm-rank0-keep9", [1, 0, 1], 1)

    def test_reader_refuses_a_parity_that_does_not_select_the_kept_half(self):
        stem = self.dump("glm-rank0-keep7", [1, 1, 1], 1)
        text = open(stem + ".plan").read().replace("cur 111\n", "cur 101\n")
        open(stem + ".plan", "w").write(text)
        with self.assertRaises(kvdump.DumpError):
            kvdump.Keep(stem)


if __name__ == "__main__":
    unittest.main()
