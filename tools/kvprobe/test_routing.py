"""Synthetic tests for routedump.py and routing.py: planted structure with known answers.

    python3 tools/kvprobe/test_routing.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routedump  # noqa: E402
import routing  # noqa: E402

E = routing.EXPERTS
L = 6


def random_sets(rng, n, layers=L):
    return np.stack([np.stack([rng.choice(E, 8, replace=False) for _ in range(layers)]) for _ in range(n)]).astype(np.int16)


def probe_prompts(rng, nS=256, nA=128):
    S = rng.integers(0, 50000, nS).tolist()
    A = rng.integers(50000, 60000, nA).tolist()
    B = rng.integers(60000, 70000, nA).tolist()
    prompts = {"S": S, "A+S": A + S, "B+S": B + S}
    meta = {k: {"len": len(v), "s_start": len(v) - nS} for k, v in prompts.items()}
    return {"S": S, "prompts": prompts, "meta": meta, "pairs": [["A+S", "B+S"], ["S", "A+S"]]}


class Loader(unittest.TestCase):
    def test_round_trip_and_missing_routing(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as t:
            ex = random_sets(rng, 300)
            w = rng.random((300, L, 8)).astype(np.float32)
            kda = -rng.random((300, 3, 4)).astype(np.float32)
            sel = np.sort(rng.choice(5000, (10, 512)), 1)
            q1 = {"tokens": np.arange(300), "top1": np.arange(1, 301), "experts": ex, "weights": w,
                  "moe_layers": list(range(3, 3 + L)), "kda_lg": kda, "kda_layers": [0, 1, 2],
                  "dsa": {3: (np.arange(10) * 16 + 100, sel)}}
            q2 = {"tokens": np.arange(50), "top1": np.arange(50)}               # a request without 0710 records
            routedump.write_synthetic(t, [q1, q2], seg_rows=128)
            d = routedump.load(t)
        self.assertEqual(d["requests"], [(0, 300), (300, 50)])
        np.testing.assert_array_equal(d["experts"][:300], ex)
        np.testing.assert_array_equal(d["weights"][:300], w)
        self.assertTrue((d["experts"][300:] == -1).all())
        np.testing.assert_array_equal(d["top1"][:300], np.arange(1, 301))
        np.testing.assert_array_equal(d["kda_gate"][:300], kda)
        self.assertTrue(np.isnan(d["kda_gate"][300:]).all())
        s, rows = d["dsa_sel"][3]
        np.testing.assert_array_equal(rows, np.arange(10) * 16 + 100)
        np.testing.assert_array_equal(s, sel)
        self.assertEqual(d["moe_layers"], [3, 4, 5, 6, 7, 8])

    def test_selection_outside_its_segment_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            routedump.write_synthetic(t, [{"tokens": np.arange(20), "dsa": {3: (np.array([5]), np.zeros((1, 512)))}}])
            import json
            with open(os.path.join(t, "index.jsonl")) as f:
                recs = [json.loads(x) for x in f]
            recs[0]["start"] = 100                                              # forge: rows 100..119, selection at 5
            with open(os.path.join(t, "index.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(r) for r in recs) + "\n")
            with self.assertRaisesRegex(ValueError, "outside its rows"):
                routedump.load(t)


def build(rng, P, *, same=True, sel_same=True):
    """Three requests of probe_prompts; S rows route identically across contexts when ``same``."""

    nS = len(P["S"])
    s_ex = random_sets(rng, nS)
    s_w = rng.random((nS, L, 8)).astype(np.float32)
    s_sel = {}
    reqs = []
    for name, ids in P["prompts"].items():
        s0 = P["meta"][name]["s_start"]
        ex = random_sets(rng, len(ids))
        w = rng.random((len(ids), L, 8)).astype(np.float32)
        if same:
            ex[s0:], w[s0:] = s_ex, s_w
        pos = np.arange(s0, len(ids), 16)
        sel = []
        for p in pos:
            k = p - s0
            if k not in s_sel:
                s_sel[k] = np.sort(rng.choice(max(1, (nS + 64) // 4), 512 if nS >= 2048 else 32, replace=False))
            rel = s_sel[k] if sel_same else np.sort(rng.choice(max(1, (nS + 64) // 4), s_sel[k].size, replace=False))
            row = np.full(512, -1)
            row[:rel.size] = rel + s0 // 4
            sel.append(row)
        reqs.append({"tokens": ids, "experts": ex, "weights": w, "moe_layers": list(range(3, 3 + L)),
                     "kda_lg": np.full((len(ids), 2, 4), -0.01, np.float32), "kda_layers": [0, 1],
                     "dsa": {3: (pos, np.array(sel))}})
    return reqs


class Analyses(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.P = probe_prompts(self.rng)

    def load(self, reqs):
        with tempfile.TemporaryDirectory() as t:
            routedump.write_synthetic(t, reqs, seg_rows=100)
            return routedump.load(t)

    def test_identical_contexts_agree_fully(self):
        d = self.load(build(self.rng, self.P, same=True))
        names = routing.name_requests(d, self.P)
        self.assertEqual(sorted(names), ["A+S", "B+S", "S"])
        A = routing.routing_agreement(d, names, self.P)
        for v in A.values():
            np.testing.assert_allclose(v["jaccard"], 1.0)
            np.testing.assert_allclose(v["weighted"], 1.0, atol=1e-6)
            np.testing.assert_allclose(v["top1"], 1.0)
        B = routing.selection_stability(d, names, self.P)
        for by in B["cross"].values():
            for v in by.values():
                self.assertAlmostEqual(v["jaccard_in_S"], 1.0)
                self.assertAlmostEqual(v["share_in_S"], 1.0)
        K = routing.kda_horizons(d, names, self.P)
        np.testing.assert_allclose(K["p50"], 1 / (1 - np.exp(-0.01)), rtol=1e-4)
        for v in K["cross"].values():
            np.testing.assert_allclose(v, 0.0)

    def test_independent_contexts_agree_at_chance(self):
        d = self.load(build(self.rng, self.P, same=False, sel_same=False))
        names = routing.name_requests(d, self.P)
        A = routing.routing_agreement(d, names, self.P)
        j = A["A+S vs B+S"]["jaccard"].mean()
        # two random 8-sets of 288: expected intersection 8*8/288 = 0.22, Jaccard about 0.014
        self.assertLess(j, 0.03)
        self.assertLess(A["A+S vs B+S"]["top1"].mean(), 0.02)

    def test_planted_layer_map_is_recalled_and_random_is_chance(self):
        rng = self.rng
        n = 4000
        perm = [rng.permutation(E) for _ in range(L)]
        ex = np.zeros((n, L, 8), np.int16)
        ex[:, 0] = random_sets(rng, n, 1)[:, 0]
        for l in range(1, L):
            ex[:, l] = perm[l][ex[:, l - 1]]
        q = [{"tokens": rng.integers(0, 10**6, n // 2), "experts": ex[:n // 2], "weights": np.ones((n // 2, L, 8)),
              "moe_layers": list(range(L))},
             {"tokens": rng.integers(0, 10**6, n // 2), "experts": ex[n // 2:], "weights": np.ones((n // 2, L, 8)),
              "moe_layers": list(range(L))}]
        D = routing.predictors(self.load(q))
        r = np.asarray(D["recall"]["layer L-1"][8])[1:]
        np.testing.assert_allclose(r, 1.0)
        # the same rows with independent layers: recall@8 near 8/288
        ex2 = random_sets(rng, n)
        q[0]["experts"], q[1]["experts"] = ex2[:n // 2], ex2[n // 2:]
        D = routing.predictors(self.load(q))
        r = np.asarray(D["recall"]["layer L-1"][8])[1:]
        self.assertTrue((np.abs(r - 8 / E) < 0.03).all(), r)
        self.assertTrue((np.abs(np.asarray(D["recall"]["random"][8]) - 8 / E) < 1e-12).all())

    def test_token_keyed_routing_is_recalled_by_the_token_prior(self):
        rng = self.rng
        vocab = 50
        table = random_sets(rng, vocab)
        toks = rng.integers(0, vocab, 3000)
        q = [{"tokens": toks[:1500], "experts": table[toks[:1500]], "weights": np.ones((1500, L, 8)),
              "moe_layers": list(range(L))},
             {"tokens": toks[1500:], "experts": table[toks[1500:]], "weights": np.ones((1500, L, 8)),
              "moe_layers": list(range(L))}]
        D = routing.predictors(self.load(q))
        self.assertAlmostEqual(D["token_seen_share"], 1.0)
        np.testing.assert_allclose(np.asarray(D["recall"]["token"][8]), 1.0)

    def test_repeated_sets_have_full_locality_and_headroom(self):
        rng = self.rng
        one = random_sets(rng, 1)
        ex = np.repeat(one, 400, 0)
        q = [{"tokens": np.arange(200), "experts": ex[:200], "weights": np.ones((200, L, 8)), "moe_layers": list(range(L))},
             {"tokens": np.arange(200), "experts": ex[200:], "weights": np.ones((200, L, 8)), "moe_layers": list(range(L))}]
        d = self.load(q)
        C = routing.locality(d)
        for w, cov in C["coverage"].items():
            np.testing.assert_allclose(cov, 1.0)
        np.testing.assert_allclose(C["distinct"][16], 8.0)
        D = routing.predictors(d)
        np.testing.assert_allclose(np.asarray(D["recall"]["previous row"][8]), 1.0)
        H = routing.headroom(D, 1000)
        self.assertAlmostEqual(H[("previous row", 8)]["hit_bytes"], 8000.0)
        self.assertAlmostEqual(H[("previous row", 16)]["wasted_bytes"], 8000.0)
        self.assertEqual(H["all_miss_bytes"], 8000)

    def test_consecutive_selection_stability(self):
        rng = self.rng
        base = np.sort(rng.choice(4000, 512, replace=False))
        pos = np.arange(2052, 2052 + 160, 16)
        same = np.tile(base, (len(pos), 1))
        q = [{"tokens": np.arange(2300), "dsa": {7: (pos, same)}}]
        B = routing.selection_stability(self.load(q), {}, None)
        self.assertAlmostEqual(B["consecutive"][7]["jaccard"], 1.0)
        self.assertAlmostEqual(B["consecutive"][7]["in_previous"], 1.0)
        self.assertEqual(B["consecutive"][7]["gap"], 16)
        disjoint = np.stack([np.arange(512) + 512 * (i % 2) for i in range(len(pos))])
        q[0]["dsa"] = {7: (pos, disjoint)}
        B = routing.selection_stability(self.load(q), {}, None)
        self.assertAlmostEqual(B["consecutive"][7]["jaccard"], 0.0)

    def test_probe_split_never_trains_on_the_shared_chunk(self):
        d = self.load(build(self.rng, self.P, same=True))
        names = routing.name_requests(d, self.P)
        train, test = routing.split(d, names)
        first, n = names["A+S"]
        self.assertEqual(test.sum(), n)
        self.assertTrue(test[first:first + n].all())
        for f, m in names.values():
            self.assertFalse(train[f:f + m].any())
        self.assertEqual(routing.repeated_context(d, train, test), 0.0)
        # the even / odd request split trains on S's text: every held-out S row repeats a training context
        tr2, te2 = routing.split(d)
        self.assertGreater(routing.repeated_context(d, tr2, te2), 0.4)

    def test_report_renders_every_section(self):
        d = self.load(build(self.rng, self.P, same=True))
        text = routing.report(d, self.P, routing.PRESETS["exl3-4bpw-rank"])
        for head in ("## A.", "## B.", "## C.", "## D.", "## E."):
            self.assertIn(head, text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
