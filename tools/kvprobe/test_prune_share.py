"""prune_share.py on a synthetic keep: every bound sound, the shares ordered as the bounds are (oracle >= head, box/12 >=
box/8, exact threshold >= the device's cut), and the device cut's threshold at most the row's 512th best score.
Run: python3 tools/kvprobe/test_prune_share.py"""
import os, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prune_share as ps


class Shares(unittest.TestCase):
    def test_synthetic_keep(self):
        with tempfile.TemporaryDirectory() as tmp:
            t, unsound = ps.synthetic(tmp)
        self.assertEqual(unsound, 0)
        self.assertTrue(t)
        for e in t.values():
            self.assertGreaterEqual(e["oracle"] + 1e-9, e["head"][1])
            self.assertGreaterEqual(e["box/12"][1] + 1e-9, e["box/8"][1])
            self.assertGreater(e["head"][1], 0)  # clustered pages: something is excludable
            for n in ps.BOUNDS:
                self.assertGreaterEqual(e[n][1] + 1e-9, e[n][0])

    def test_bounds_hold_and_the_cut_is_below_the_kth(self):
        rng = np.random.default_rng(3)
        keys = (rng.normal(size=(20 * ps.PP + 37, ps.D)) + 3 * rng.normal(size=(21, 1, ps.D)).repeat(ps.PP, 1).reshape(-1, ps.D)[:20 * ps.PP + 37])
        pg = ps.Pages(keys.astype(np.float32), 4)
        q, w = rng.normal(size=(8, ps.D)), rng.normal(size=8)
        npool = keys.shape[0]
        b, scores, best = ps.row_bounds(pg, keys, q, w, ps.D ** -0.5, npool)
        for n in ps.BOUNDS:
            self.assertTrue(np.all(b[n] >= best), n)
        cut, seeds = ps.device_cut(b["box/8"], scores, npool // ps.PP, 8)
        self.assertEqual(len(seeds), 8)
        self.assertLessEqual(cut, ps.kth(scores))
        # the planted fault of the kernels' tests: negative head weights bounded from above, visible as unsound
        whole = npool // ps.PP
        d = (keys[:whole * ps.PP] @ q.T).reshape(whole, ps.PP, -1).max(axis=1)  # the heads' exact maxima
        self.assertTrue(np.any(ps.weigh(d, d, w, ps.D ** -0.5) < best))


if __name__ == "__main__":
    unittest.main()
