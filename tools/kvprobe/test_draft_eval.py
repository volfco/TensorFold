"""Host tests for draft_eval.py on synthetic dumps in the patches/0430 format. Run: python3 test_draft_eval.py"""
import json, os, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draft_eval as de


def write_dump(d, reqs, seg=5):
    """reqs: list of (tokens, greedy, completion or None). Prefill rows in segments of `seg`, a shard per dump."""
    index, blob = [], bytearray()

    def put(a):
        off = len(blob)
        blob.extend(np.ascontiguousarray(a).tobytes())
        while len(blob) % 64:
            blob.append(0)
        return off

    for toks, greedy, comp in reqs:
        toks, greedy = np.asarray(toks, np.int32), np.asarray(greedy, np.int32)
        for a in range(0, len(toks), seg):
            t, g = toks[a:a + seg], greedy[a:a + seg].reshape(-1, 1)
            index.append({"kind": "p", "start": a, "n": len(t), "shard": "shard-00001.bin", "tap_layers": [],
                          "arrays": {"tokens": [put(t), "int32", [len(t)]], "ids": [put(g), "int32", [len(t), 1]],
                                     "lp": [put(np.zeros((len(t), 1), np.float16)), "float16", [len(t), 1]]}})
        if comp:
            c = np.asarray(comp, np.int32)
            index.append({"kind": "d", "start": len(toks), "n": len(c), "shard": "shard-00001.bin",
                          "arrays": {"tokens": [put(c), "int32", [len(c)]]}})
    open(f"{d}/shard-00001.bin", "wb").write(bytes(blob))
    with open(f"{d}/index.jsonl", "w") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")
    json.dump({"version": 1, "topk": 1}, open(f"{d}/meta.json", "w"))


def oracle_greedy(toks):
    """A model that always predicts the true next token (and 0 after the last)."""
    return list(toks[1:]) + [0]


class Fixed:
    """A source that always proposes the given continuation of the true text from position i."""
    name = "fixed"

    def __init__(self, text, shift=0):
        self.text, self.shift = text, shift

    def start(self, req):
        pass

    def finish(self, req):
        pass

    def propose(self, ctx, most):
        i = len(ctx) - 1 + self.shift
        return list(self.text[i + 1:i + 1 + most])


class T(unittest.TestCase):
    def test_acceptance_is_not_off_by_one(self):
        toks = list(range(100, 140))
        req = {"index": 0, "tokens": toks, "greedy": oracle_greedy(toks)}
        res = de.evaluate([req], Fixed(toks), ks=(4,))[4]
        # positions 0..35 accept all 4; 36 accepts 3 (39 is the last real token), 37: 2, 38: 1, 39: none proposed
        self.assertEqual(res["proposals"], 39)
        self.assertEqual(res["strict"], 36 * 4 + 3 + 2 + 1)
        self.assertEqual(res["loose"], res["strict"])
        # a source one token late (it proposes the token after the next) accepts nothing: the comparison is
        # draft[0] against row i's own pick, so shifting by one row must zero it
        late = de.evaluate([req], Fixed(toks, shift=1), ks=(4,))[4]
        self.assertEqual(late["loose"], 0)
        early = de.evaluate([req], Fixed(toks, shift=-1), ks=(4,))[4]
        self.assertEqual(early["loose"], 0)

    def test_strict_needs_the_prompt_to_follow_the_draft(self):
        toks = [1, 2, 3, 4, 5, 6]
        greedy = [2, 9, 4, 5, 6, 0]  # row 1 mispredicts: the prompt says 3, the model 9
        # draft from row 0: [2, 9, 4]; loose accepts 2 then 9 (row 1's pick) then 4 (row 2's): 3;
        # strict stops after 2: row 1's pick 9 was computed on the prompt's 2, which the draft also has, so 9 counts;
        # row 2's pick was computed on prompt token 3, but the draft put 9 there: stop at 2
        self.assertEqual(de.accepted([2, 9, 4], toks, greedy, 0, False), 3)
        self.assertEqual(de.accepted([2, 9, 4], toks, greedy, 0, True), 2)
        self.assertEqual(de.accepted([7], toks, greedy, 0, True), 0)

    def test_recall_mirrors_min_match_and_marks(self):
        r = de.Record()
        seg = list(range(1000, 1100))
        r.add(seg)
        src = de.RecallSource()
        src.record = r
        self.assertEqual(src.propose(seg[:70], 6), seg[70:76])
        self.assertEqual(src.propose(seg[4:18], 6), [])  # 14 matched ids: under 16
        self.assertEqual(src.propose(seg[:16], 6), seg[16:22])  # exactly 16
        self.assertEqual(src.propose(seg[90:99], 6), [])  # 9 matched ids
        self.assertEqual(src.propose(seg[:98], 6), seg[98:100])  # two ids left before the mark
        self.assertEqual(src.propose(seg[:99], 6), [])  # one id left before the mark: under 2
        n, _ = r.longest(seg[:100])
        self.assertEqual(n, 0)  # the last id has a mark after it: no match

    def test_recall_prefers_newest_of_equal_length(self):
        a = list(range(1, 30)) + [500, 501, 502]
        b = list(range(1, 30)) + [600, 601, 602]
        r = de.Record()
        r.add(a)
        r.add(b)
        src = de.RecallSource()
        src.record = r
        self.assertEqual(src.propose(list(range(1, 30)), 3), [600, 601, 602])

    def test_dump_roundtrip_and_cross_request_recall(self):
        rng = np.random.default_rng(1)
        S = [int(x) for x in rng.integers(1000, 2000, 300)]
        A = [int(x) for x in rng.integers(3000, 4000, 50)]
        B = [int(x) for x in rng.integers(5000, 6000, 70)]
        reqs = [(A + S, oracle_greedy(A + S), None), (B + S, oracle_greedy(B + S), [7, 8, 9])]
        with tempfile.TemporaryDirectory() as d:
            write_dump(d, reqs, seg=37)
            got = de.load_dump(d)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[1]["tokens"], B + S)
        self.assertEqual(got[1]["completion"], [7, 8, 9])
        res = de.evaluate(got, de.RecallSource(), ks=(8,))[8]
        # request 0: nothing in the record. Request 1: from 16 tokens into S every row matches A+S's copy of S,
        # except the last 9 rows whose copies run into the record's end of segment
        self.assertEqual(res["positions"], len(A + S) + len(B + S))
        self.assertGreater(res["proposals"], 250)
        self.assertGreater(res["mean_strict"], 7.0)
        ctx = de.evaluate(got, de.ContextCopy(), ks=(8,))[8]
        self.assertEqual(ctx["proposals"], 0)  # random text never repeats inside one request

    def test_token_prior_leaves_its_own_request_out(self):
        reqs = [{"index": 0, "tokens": [1, 2, 1, 2, 1, 2], "greedy": [2, 1, 2, 1, 2, 0]},
                {"index": 1, "tokens": [1, 3, 1, 3], "greedy": [3, 1, 3, 0]}]
        p = de.TokenPrior(reqs, order=1)
        p.start(reqs[1])
        self.assertEqual(p.propose([1], 1), [])  # most 1: under 2, nothing
        self.assertEqual(p.propose([1], 2)[0], 2)  # from request 0 only
        p.start(reqs[0])
        self.assertEqual(p.propose([1], 2), [3, 1])

    def test_gate_stops_proposing_after_misses(self):
        r = de.Record()
        seg = list(range(1, 200))
        r.add(seg)
        src = de.RecallSource(gate=True, head=2.0)
        src.record = r
        fired = 0
        for _ in range(200):
            d = src.propose(seg[:40], 7)
            if d:
                fired += 1
                src.observe(0)  # every copy misses
        self.assertLess(fired, 20)


class Yield(unittest.TestCase):
    def test_numpy_index_matches_the_record(self):
        import recall_yield as ry
        rng = np.random.default_rng(3)
        base = [int(x) for x in rng.integers(1, 40, 400)]
        segs = [base[:200], base[50:300], [int(x) for x in rng.integers(1, 40, 150)], base[120:400]]
        r = de.Record()
        parts = []
        for sgm in segs:
            r.add(sgm)
            parts += sgm + [-1]
        idx = ry.Index(np.asarray(parts, dtype=np.int64))
        checked = 0
        for t in range(300):
            a = int(rng.integers(0, 380))
            ctx = base[a:a + int(rng.integers(8, 20))]
            n1, at1 = r.longest(ctx)
            n2, at2 = idx.longest(np.asarray(ctx, dtype=np.int64), len(parts))
            self.assertEqual(n1, n2)
            if n1:
                self.assertEqual(r.copy(at1, 7), [int(x) for x in idx.copy(at2, 7, len(parts))])
                checked += 1
        self.assertGreater(checked, 100)

    def test_walk_counts_a_repeated_answer_as_copies(self):
        import recall_yield as ry
        ans = list(range(1000, 1100))
        ids = np.asarray([ry.USER, 5, 6, ry.ASSISTANT] + ans + [ry.USER, 7, ry.ASSISTANT] + ans, dtype=np.int64)
        roles = ry.roles_of(ids)
        self.assertEqual(int((roles != 0).sum()), 200)
        st = ry.walk(ids, roles, 0, ry.Index(ids), None, 0, 7)
        self.assertEqual(st["assistant_tokens"], 200)
        # the second answer: copies start once 8 of its ids are in context; each lands 7 and moves 8
        self.assertGreater(st["own_landed"], 80)
        # a round is its landed copy plus the target's own token, except a copy that ends the span exactly
        self.assertIn(st["own_landed"] + st["rounds"], (200, 201))


if __name__ == "__main__":
    unittest.main(verbosity=1)
