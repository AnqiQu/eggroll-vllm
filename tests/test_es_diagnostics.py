"""Unit tests for es_diagnostics (numpy only).  Run:  python -m unittest tests.test_es_diagnostics"""
import math
import unittest

import numpy as np

import es_diagnostics as D


class FakeLogprob:
    def __init__(self, lp):
        self.logprob = lp


class TestPairs(unittest.TestCase):
    def test_pair_split_and_disagreement(self):
        arr = np.array([[1, 0], [0, 0], [1, 1], [1, 1]], dtype=float)  # pairs (0,1), (2,3)
        plus, minus = D.pair_split(arr)
        np.testing.assert_array_equal(plus, [[1, 0], [1, 1]])
        np.testing.assert_array_equal(minus, [[0, 0], [1, 1]])
        s = D.pair_disagreement(arr)
        self.assertAlmostEqual(s["disagree_rate"], 0.25)   # only (0,1) on prompt 0
        self.assertAlmostEqual(s["absdiff_mean"], 0.25)

    def test_center_per_prompt_matches_trainer(self):
        rng = np.random.default_rng(0)
        f = rng.random((8, 3))
        expected = np.mean(f - np.mean(f, axis=0, keepdims=True), axis=1)
        np.testing.assert_allclose(D.center_per_prompt(f), expected)


class TestSignalDecomposition(unittest.TestCase):
    def test_format_only_signal(self):
        # Fitness = correctness + format, but correctness is identical within
        # every pair -> the update carries zero correctness information.
        rng = np.random.default_rng(1)
        P, N = 16, 4
        correct = rng.integers(0, 2, size=(P // 2, N)).repeat(2, axis=0).astype(float)  # same in pair
        fmt = rng.integers(0, 2, size=(P, N)).astype(float)
        fit = 2.0 * correct + fmt
        out = D.signal_decomposition(fit, {"reward/frac_correct": correct, "reward/frac_format_ok": fmt})
        self.assertAlmostEqual(out["frac_correct/pair_disagree_rate"], 0.0)
        self.assertAlmostEqual(out["frac_correct/cos_with_fitness"], 0.0)
        self.assertAlmostEqual(out["frac_correct/share_of_update"], 0.0)
        self.assertAlmostEqual(out["frac_format_ok/cos_with_fitness"], 1.0, places=6)
        self.assertAlmostEqual(out["frac_format_ok/share_of_update"], 1.0, places=6)

    def test_additive_shares_sum_to_one(self):
        rng = np.random.default_rng(2)
        P, N = 32, 5
        c = rng.integers(0, 2, size=(P, N)) * 2.0
        f = rng.random((P, N))
        fit = c + f
        out = D.signal_decomposition(fit, {"reward/correctness": c, "reward/format": f})
        self.assertAlmostEqual(out["correctness/share_of_update"] + out["format/share_of_update"], 1.0, places=6)
        self.assertIn("fitness/pairs_with_signal", out)

    def test_gated_fitness_alignment(self):
        # Gated = correct AND format. When format is (nearly) always ok the
        # gated pair-differences align with the correctness pair-differences.
        rng = np.random.default_rng(3)
        P, N = 64, 8
        correct = rng.integers(0, 2, size=(P, N)).astype(float)
        fmt = np.ones((P, N))
        gated = correct * fmt
        out = D.signal_decomposition(gated, {"reward/frac_correct": correct, "reward/frac_format_ok": fmt})
        self.assertAlmostEqual(out["frac_correct/cos_with_fitness"], 1.0, places=6)
        self.assertAlmostEqual(out["frac_format_ok/pairs_with_signal"], 0.0)


class TestCollapse(unittest.TestCase):
    def test_first_divergence(self):
        self.assertIsNone(D.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(D.first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(D.first_divergence([1, 2], [1, 2, 3]), 2)

    def test_greedy_collapse_stats(self):
        texts = [["a b c", "x"], ["a b c", "y"], ["q", "z"], ["r", "z"]]  # pairs (0,1),(2,3)
        ids = [[[1, 2, 3], [7]], [[1, 2, 3], [8]], [[5], [9]], [[6], [9]]]
        s = D.greedy_collapse_stats(texts, ids)
        # identical: (0,1) prompt0 yes, prompt1 no; (2,3) prompt0 no, prompt1 yes -> 0.5
        self.assertAlmostEqual(s["pair_identical_rate"], 0.5)
        # divergence for the 2 non-identical: ([7],[8]) -> 0/1, ([5],[6]) -> 0/1
        self.assertAlmostEqual(s["pair_first_div_frac"], 0.0)
        # distinct per prompt: prompt0 {a b c, q, r}=3/4, prompt1 {x,y,z}=3/4
        self.assertAlmostEqual(s["distinct_outputs_frac"], 0.75)

    def test_fully_collapsed(self):
        texts = [["same"] * 2] * 4
        s = D.greedy_collapse_stats(texts, [[[1, 2]] * 2] * 4)
        self.assertAlmostEqual(s["pair_identical_rate"], 1.0)
        self.assertAlmostEqual(s["distinct_outputs_frac"], 0.25)
        self.assertAlmostEqual(s["pair_first_div_frac"], 1.0)


class TestEntropy(unittest.TestCase):
    def test_topk_rows_from_vllm_like_objects(self):
        per_token = [
            {5: FakeLogprob(math.log(0.7)), 9: FakeLogprob(math.log(0.2)), 1: FakeLogprob(math.log(0.1))},
            None,
            {2: FakeLogprob(math.log(0.5)), 3: FakeLogprob(math.log(0.5))},
        ]
        rows = D.topk_logprob_rows(per_token, k=2)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0][0], math.log(0.7))
        self.assertAlmostEqual(rows[0][1], math.log(0.2))
        stats = D.entropy_stats_from_rows(rows)
        self.assertEqual(stats["n_tokens"], 2.0)
        # token 1 margin = log(0.7/0.2); token 2 margin = 0 -> frac low = 0.5
        self.assertAlmostEqual(stats["frac_margin_lt_0p5"], 0.5)
        self.assertAlmostEqual(stats["margin"], (math.log(0.7 / 0.2) + 0.0) / 2)
        self.assertAlmostEqual(stats["top1_prob"], (0.7 + 0.5) / 2)

    def test_peaked_vs_flat(self):
        flat = D.entropy_stats_from_rows([[math.log(0.25)] * 4])
        peaked = D.entropy_stats_from_rows([[math.log(0.97), math.log(0.01), math.log(0.01), math.log(0.01)]])
        self.assertGreater(flat["entropy_topk"], peaked["entropy_topk"])
        self.assertGreater(peaked["margin"], flat["margin"])

    def test_aggregate_token_weighted(self):
        a = {"n_tokens": 1.0, "entropy_topk": 1.0, "top1_prob": 0.5, "margin": 1.0, "frac_margin_lt_0p5": 0.0}
        b = {"n_tokens": 3.0, "entropy_topk": 0.0, "top1_prob": 1.0, "margin": 3.0, "frac_margin_lt_0p5": 1.0}
        agg = D.aggregate_entropy([a, b])
        self.assertAlmostEqual(agg["entropy/entropy_topk"], 0.25)
        self.assertAlmostEqual(agg["entropy/margin"], 2.5)
        self.assertAlmostEqual(agg["entropy/n_tokens"], 4.0)


class TestStep(unittest.TestCase):
    def test_step_diagnostics_prefix_and_format(self):
        rng = np.random.default_rng(4)
        fit = rng.integers(0, 2, size=(8, 2)).astype(float)
        comps = {"reward/frac_correct": fit.copy(), "reward/frac_format_ok": np.ones((8, 2))}
        texts = [["t%d" % (p // 2)] * 2 for p in range(8)]
        d = D.step_diagnostics(fit, comps, texts, None, [{"n_tokens": 2.0, "entropy_topk": 0.3, "top1_prob": 0.9, "margin": 2.0, "frac_margin_lt_0p5": 0.1}])
        self.assertTrue(all(k.startswith("diag/") for k in d))
        self.assertIn("diag/entropy/margin", d)
        self.assertIn("diag/pair_identical_rate", d)
        line = D.format_diagnostics(d)
        self.assertTrue(line.startswith("DIAG:"))


if __name__ == "__main__":
    unittest.main()
