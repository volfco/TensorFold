"""Synthetic tests for replay_check.py: head-only dump pairs with planted per-head differences, and (with
KVPROBE_REPLAY_FIXTURE set to the directory the Zig host test `TF_GLM_KDA_REPLAY_CHECK reports ...` filled through
TF_KVPROBE_REPLAY_DUMP) the engine's own format.

    python3 tools/kvprobe/test_replay_check.py
"""
import os, shutil, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvdump, replay_check

HEADS, LAYERS = 3, 2
ROUTING = """# report

## E. KDA decay horizon in tokens

| layer | p10 | p50 | p90 | A+S vs B+S rel dev |
|---|---|---|---|---|
| 0 | 1.2 | 2.0 | 40.0 | 0.000 |
| 1 | 1.1 | 8.0 | 10.0 | 0.003 |

## F. other

| 7 | 1 | 2 | 3 |
"""


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        self.rec = rng.standard_normal((LAYERS, HEADS, 128, 128)).astype(np.float32)
        self.conv = rng.standard_normal((LAYERS, 3, 3 * HEADS * 128)).astype(np.float32)
        conv_b = (np.asarray(self.conv, np.float32).view(np.uint32) >> 16 << 16).view(np.float32)  # bf16-exact
        rec2 = self.rec.copy()
        rec2[0, 1] *= 1.25  # layer 0 head 1: rel 0.25
        rec2[1, 2, 5, 7] += 3.0  # layer 1 head 2: one element, max abs 3
        conv2 = conv_b.copy()
        conv2[1, 2, 1 * HEADS * 128 + 0 * 128 + 4] = 9.0  # layer 1, row 2, the k block of head 0
        self.stored = os.path.join(self.dir, "kda-replay-rank1-keep7-w64-stored")
        self.replayed = os.path.join(self.dir, "kda-replay-rank1-keep7-w64-replayed")
        # the replayed state lives in the other half (the replay flips every layer's parity)
        kvdump.write(self.stored, rank=1, world=2, pos=4096, kv_lora=16, i_dim=8, heads=HEADS, kda_layers=LAYERS,
                     latents=[], rec=self.rec, cur=[0, 1], conv=conv_b)
        kvdump.write(self.replayed, rank=1, world=2, pos=4096, kv_lora=16, i_dim=8, heads=HEADS, kda_layers=LAYERS,
                     latents=[], rec=rec2, cur=[1, 0], conv=conv2)
        self.routing = os.path.join(self.dir, "routing.md")
        with open(self.routing, "w") as f:
            f.write(ROUTING)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_planted_heads(self):
        r = replay_check.compare(self.stored, self.replayed, horizons=replay_check.read_horizons(self.routing))
        self.assertEqual(r["window"], 64)  # from the name
        l0, l1 = r["layers"]
        self.assertEqual(l0["rec_same"], 2)
        self.assertEqual(l1["rec_same"], 2)
        self.assertEqual(l0["heads"][0]["rec_rel"], 0.0)
        self.assertAlmostEqual(l0["heads"][1]["rec_rel"], 0.25, places=6)
        self.assertAlmostEqual(l1["heads"][2]["rec_maxabs"], 3.0, places=5)
        self.assertEqual(l0["conv_same"], HEADS)
        self.assertEqual([h["conv_same"] for h in l1["heads"]], [False, True, True])
        self.assertEqual(r["total"]["rec_same"], 4)
        # the bound (1 - 1/h)^W at the table's horizons
        self.assertAlmostEqual(l0["bound"]["p90"], (1 - 1 / 40.0) ** 64)
        self.assertAlmostEqual(l1["bound"]["p50"], (1 - 1 / 8.0) ** 64)
        self.assertIn("| 0 | 2/3 |", replay_check.markdown(r))

    def test_dir_pairs_and_refusals(self):
        self.assertEqual(replay_check.pairs(self.dir), [(self.stored, self.replayed)])
        os.rename(self.replayed + ".plan", self.replayed + ".plan.x")
        self.assertEqual(replay_check.pairs(self.dir), [])
        os.rename(self.replayed + ".plan.x", self.replayed + ".plan")
        other = os.path.join(self.dir, "kda-replay-rank1-keep8-w64-replayed")
        kvdump.write(other, rank=1, world=2, pos=2048, kv_lora=16, i_dim=8, heads=HEADS, kda_layers=LAYERS, latents=[])
        with self.assertRaisesRegex(kvdump.DumpError, "pos differs"):
            replay_check.compare(self.stored, other)
        bare = os.path.join(self.dir, "x-stored")
        for ext in (".bin", ".parts", ".plan"):
            shutil.copy(self.stored + ext, bare + ext)
        with self.assertRaisesRegex(kvdump.DumpError, "pass --window"):
            replay_check.compare(bare, self.replayed)
        self.assertEqual(replay_check.compare(bare, self.replayed, window=32)["window"], 32)
        empty = os.path.join(self.dir, "empty.md")
        open(empty, "w").close()
        with self.assertRaisesRegex(kvdump.DumpError, "no section E"):
            replay_check.read_horizons(empty)

    def test_engine_format(self):
        d = os.environ.get("KVPROBE_REPLAY_FIXTURE")
        if not d:
            self.skipTest("KVPROBE_REPLAY_FIXTURE unset")
        found = replay_check.pairs(d)
        self.assertTrue(found)
        r = replay_check.compare(*found[0])
        t = r["total"]
        # the host test's pass reproduces the stored state but one element of one head, conv history exactly
        self.assertEqual(t["rec_same"], t["heads"] - 1)
        self.assertEqual(t["conv_same"], t["heads"])


if __name__ == "__main__":
    unittest.main()
