"""Host tests for kvdump.py and kvstate.py: synthetic captures with planted structure, and (with KVPROBE_FIXTURE set to
a directory the Zig host test `TF_GLM_KV_DUMP writes ...` filled through TF_KVPROBE_FIXTURE_DUMP) the engine's own
format. Run: python3 tools/kvprobe/test_kvstate.py"""
import json, os, re, shutil, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvdump, kvstate


def to_bf16(x):
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = np.random.default_rng(1)
        self.n, kv, i_dim = 256, 16, 8
        S = rng.standard_normal((self.n, kv)).astype(np.float32)
        noise = rng.standard_normal((self.n, kv)).astype(np.float32)
        u = rng.standard_normal((2, 2, 128, 3))
        v = rng.standard_normal((2, 2, 3, 128))
        state = (u @ v).astype(np.float32)  # rank 3 a head
        bump = (rng.standard_normal((2, 2, 128, 1)) @ rng.standard_normal((2, 2, 1, 128))).astype(np.float32)
        probe = {"captures": []}
        # S alone at 0, A+S and B+S at 64; A and B prefixes differ, S rows are the same in S and A+S, perturbed in B+S
        for ctx, start, rows, st in (("S", 0, S, state), ("A+S", 64, S, state), ("B+S", 64, S + 0.1 * noise, state + 0.5 * bump)):
            pre = rng.standard_normal((start, kv)).astype(np.float32)
            lat = to_bf16(np.concatenate([pre, rows]))
            ik = to_bf16(np.concatenate([rng.standard_normal((start, i_dim)), rows[:, :i_dim]]).astype(np.float32))
            pos = start + self.n
            pk = to_bf16(np.concatenate([rng.standard_normal((start // 4, i_dim)), rows[::4, :i_dim], np.zeros((2, i_dim))]).astype(np.float32))
            stem = os.path.join(self.dir, f"glm-rank0-keep{len(probe['captures']) + 1}")
            kvdump.write(stem, pos=pos, kv_lora=kv, i_dim=i_dim, heads=2, kda_layers=2, latents=[lat], rings=[[ik, ik]],
                         pooled=[pk], rec=st, cur=[1, 0], conv=np.ones((2, 3, 3 * 2 * 128), np.float32), block=4096)
            probe["captures"].append({"type": "prose", "context": ctx, "s_start": start, "s_len": self.n, "keeps": {"0": [stem]}})
        self.probe = probe
        self.noise_rel = float(np.median(np.linalg.norm(0.1 * noise, axis=1) / np.linalg.norm(S, axis=1)))

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_identical_rows_align_exactly(self):
        # S and A+S hold the same rows at different starts: any off-by-one in the alignment shows as a deviation
        r = kvstate.analyze(self.probe, pairs=[("S", "A+S")])["types"]["prose"]["pairs"]["S vs A+S"]
        lat = r["latent"][0]
        self.assertEqual(r["rows"], self.n)
        self.assertEqual(lat["rel_l2"]["p50"], 0.0)
        self.assertEqual(lat["identical_rows"], self.n)
        self.assertEqual(lat["entropy"]["xor"], 0.0)
        self.assertGreater(lat["other_token"]["p50"], 0.5)
        self.assertEqual(r["ring"][0]["rel_l2"]["p50"], 0.0)
        self.assertEqual(r["pooled"][0]["rel_l2"]["p50"], 0.0)
        self.assertEqual(r["kda"][0]["rel"]["p50"], 0.0)

    def test_planted_deviation_and_rank(self):
        r = kvstate.analyze(self.probe, pairs=[("A+S", "B+S")])["types"]["prose"]["pairs"]["A+S vs B+S"]
        lat = r["latent"][0]
        self.assertAlmostEqual(lat["rel_l2"]["p50"], self.noise_rel, delta=0.03)  # bf16 rounding only
        self.assertLess(lat["entropy"]["raw"], 1.0)
        k = r["kda"][0]
        self.assertGreater(k["rel"]["p50"], 0.0)
        self.assertLessEqual(k["rank90"]["p50"], 3)  # the planted rank-3 state
        self.assertGreater(k["diff_energy8"]["p50"], 0.999)  # a rank-1 difference
        self.assertLess(k["diff_eff_rank"]["p50"], 1.01)

    def test_live_half_is_read(self):
        k = kvdump.Keep(self.probe["captures"][0]["keeps"]["0"][0])
        self.assertEqual(k.cur, [1, 0])
        self.assertGreater(np.abs(k.kda_state(0)).sum(), 0)
        o, _ = k.fields["rec"]
        dead = np.asarray(k.raw[o:o + k.shape["rec_layer"]]).view(np.float32)  # half 0 of layer 0: never written
        self.assertEqual(np.abs(dead).sum(), 0)

    def test_refusals(self):
        stem = self.probe["captures"][0]["keeps"]["0"][0]
        plan = open(stem + ".plan").read()
        for bad, why in ((plan.replace("tfkvplan 1", "tfkvplan 2"), "not a tfkvplan"), (plan.replace("end\n", ""), "truncated"),
                         (re.sub(r"part 1 latent (\d+)", lambda m: f"part 1 latent {int(m[1]) + 2}", plan), "sum")):
            open(stem + ".plan", "w").write(bad)
            with self.assertRaises(kvdump.DumpError, msg=why):
                kvdump.Keep(stem)
        open(stem + ".plan", "w").write(plan)
        parts = open(stem + ".parts").read()
        open(stem + ".parts", "w").write(parts.replace("latent", "ring", 1))
        with self.assertRaises(kvdump.DumpError):
            kvdump.Keep(stem)
        open(stem + ".parts", "w").write(parts)
        kvdump.Keep(stem)

    def test_markdown(self):
        md = kvstate.markdown(kvstate.analyze(self.probe))
        self.assertIn("A+S vs B+S", md)
        self.assertIn("KDA layer", md)


class Formats(unittest.TestCase):
    def test_e4m3_and_bf16(self):
        self.assertEqual(kvdump.E4M3[0x38], 1.0)
        self.assertEqual(kvdump.E4M3[0x7E], 448.0)
        self.assertEqual(kvdump.E4M3[0xB8], -1.0)
        self.assertTrue(np.isnan(kvdump.E4M3[0x7F]))
        self.assertEqual(kvdump.E4M3[0x01], 2.0 ** -9)
        self.assertEqual(float(kvdump.bf16(np.uint16(0x3F80))), 1.0)

    @unittest.skipUnless(os.environ.get("KVPROBE_FIXTURE"), "KVPROBE_FIXTURE unset: no engine-written dump to read")
    def test_engine_fixture(self):
        keeps = kvdump.stems(os.environ["KVPROBE_FIXTURE"])
        self.assertTrue(keeps)
        for _, _, stem in keeps:
            k = kvdump.Keep(stem)
            for j in range(len(k.caches)):
                lat, _ = k.latent(j)
                self.assertEqual(lat.shape[1], k.shape["kv_lora"])
                ik, ig, first = k.rings(j)
                self.assertEqual(ik.shape, ig.shape)
            self.assertEqual(k.kda_state(0).shape, (k.shape["heads"], 128, 128))
            self.assertEqual(k.conv(0).shape, (3, k.shape["conv_c"]))


class FakeServer:
    """Byte tokens and a per-message template, as GLM's renders messages one by one."""

    def __init__(self, base=None):
        pass

    def ids(self, text):
        return list(text.encode("utf-8", "replace"))

    def text(self, ids):
        return bytes(ids).decode("utf-8", "ignore")

    def render(self, messages, tools=None):
        out = [300]
        for m in messages:
            out += [301 + ["system", "user", "assistant", "tool"].index(m["role"])] + self.ids(m.get("content") or "")
            for c in m.get("tool_calls", []):
                out += [310] + self.ids(c["function"]["arguments"])
        return out + [399]

    def clip(self, text, n, skip=0):
        ids = self.ids(text)[skip:skip + n]
        if len(ids) < n:
            raise SystemExit("short")
        return self.text(ids)


class Driver(unittest.TestCase):
    def test_build_aligns_s_on_the_grid(self):
        import kv_probe_send as d
        d.Server = FakeServer
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        out = os.path.join(tempfile.mkdtemp(), "prompts.json")
        d.build(root, out, None, ["prose", "boilerplate"])
        P = json.load(open(out))
        srv = FakeServer()
        for t in ("prose", "boilerplate"):
            ps = [p for p in P["prompts"] if p["type"] == t]
            self.assertEqual([p["context"] for p in ps], ["S", "A+S", "B+S", "L+S"])
            for p in ps:
                ids = srv.render(p["messages"], p["tools"])
                self.assertEqual(ids[p["s_start"]:p["s_start"] + p["s_len"]], p["s_ids"])
                self.assertEqual(p["s_start"] % d.GRID, ps[0]["s_start"] % d.GRID)
                self.assertEqual(p["suffix_len"], ps[0]["suffix_len"])  # the same filler after S
                self.assertEqual(p["s_ids"], ps[0]["s_ids"])
                self.assertGreater(p["suffix_len"], d.GRID)  # the history mark lands past S


if __name__ == "__main__":
    unittest.main(verbosity=2)
