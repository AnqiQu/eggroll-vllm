"""Tests for population_eval.py's metrics (pure python) and eggroll_noise.py (torch; skipped otherwise).
Run:  python -m unittest tests.test_population_metrics"""
import itertools
import math
import unittest

import population_eval as P

try:
    import torch
    import eggroll_noise as N
except Exception:  # pragma: no cover
    torch = None


def brute(n, c, k, pred):
    items = [1] * c + [0] * (n - c)
    groups = list(itertools.combinations(range(n), k))
    return sum(pred([items[i] for i in g]) for g in groups) / len(groups)


class TestMetrics(unittest.TestCase):
    def test_pass_at_k_matches_enumeration(self):
        for n, c, k in [(6, 0, 2), (6, 1, 1), (6, 2, 3), (6, 6, 4), (8, 3, 5)]:
            self.assertAlmostEqual(P.pass_at_k(n, c, k), brute(n, c, k, lambda g: any(g)))

    def test_group_signal_matches_enumeration(self):
        for n, c, k in [(6, 0, 2), (6, 1, 2), (6, 3, 3), (8, 7, 4), (8, 8, 8)]:
            self.assertAlmostEqual(P.group_signal(n, c, k), brute(n, c, k, lambda g: 0 < sum(g) < len(g)))

    def test_summarize(self):
        # 4 members (2 antithetic pairs) x 3 problems
        correct = [[1, 0, 0],
                   [1, 1, 0],
                   [0, 0, 0],
                   [1, 0, 0]]
        centre = [1, 0, 1]
        s = P.summarize(correct, centre, antithetic=True, group_k=2)
        self.assertEqual(s["counts"], [3, 1, 0])
        self.assertAlmostEqual(s["centre_acc"], 2 / 3)
        self.assertAlmostEqual(s["member_acc_mean"], (1 / 3 + 2 / 3 + 0 + 1 / 3) / 4)
        self.assertAlmostEqual(s["coverage"]["4"], 2 / 3)       # problems 0 and 1 solved by someone
        self.assertAlmostEqual(s["coverage"]["1"], 4 / 12)      # = mean member accuracy
        self.assertAlmostEqual(s["rescued"], 1 / 3)             # problem 1
        self.assertAlmostEqual(s["broken"], 1 / 3)             # problem 2: centre right, 0/4 members
        self.assertAlmostEqual(s["frac_problems_mixed"], 2 / 3)
        # pairs (0,1) and (2,3): disagreements on (p1) and (p0) -> 2 of 6 pair-problems
        self.assertAlmostEqual(s["pair_disagree"], 2 / 6)
        self.assertAlmostEqual(s["frac_problems_with_pair_signal"], 2 / 3)
        self.assertAlmostEqual(s["group_signal@2"], (P.group_signal(4, 3, 2) + P.group_signal(4, 1, 2)) / 3)


@unittest.skipIf(torch is None, "torch not installed")
class TestNoise(unittest.TestCase):
    def test_scale_rank_and_determinism(self):
        d = N.lowrank_delta((256, 192), sigma=1e-3, rank=1, seed=3, pair=5, layer_idx=2)
        self.assertEqual(tuple(d.shape), (256, 192))
        self.assertAlmostEqual(float(d.std()), 1e-3, delta=2e-4)
        sv = torch.linalg.svdvals(d.double())
        self.assertLess(float(sv[1] / sv[0]), 1e-5)  # rank 1 up to fp32 rounding
        self.assertTrue(torch.equal(d, N.lowrank_delta((256, 192), 1e-3, 1, 3, 5, 2)))
        self.assertFalse(torch.equal(d, N.lowrank_delta((256, 192), 1e-3, 1, 3, 6, 2)))
        d4 = N.lowrank_delta((256, 192), sigma=1e-3, rank=4, seed=3, pair=5, layer_idx=2)
        self.assertAlmostEqual(float(d4.std()), 1e-3, delta=2e-4)

    def test_is_target_weight(self):
        self.assertTrue(N.is_target_weight("model.layers.0.self_attn.q_proj.weight", 2))
        self.assertTrue(N.is_target_weight("model.layers.3.mlp.down_proj.weight", 2))
        self.assertFalse(N.is_target_weight("model.layers.0.self_attn.q_proj.bias", 1))
        self.assertFalse(N.is_target_weight("model.embed_tokens.weight", 2))
        self.assertFalse(N.is_target_weight("lm_head.weight", 2))


if __name__ == "__main__":
    unittest.main()
