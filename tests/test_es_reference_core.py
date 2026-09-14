"""Tests for the ES core of es_reference.py (requires torch; skipped otherwise).
Run:  python -m unittest tests.test_es_reference_core"""
import math
import unittest

import numpy as np

try:
    import torch
    import es_reference as R
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestLowRankES(unittest.TestCase):
    def _params(self, shapes, dtype=torch.float32, seed=0):
        g = torch.Generator().manual_seed(seed)
        return [(f"w{i}", torch.nn.Parameter(torch.randn(*s, generator=g) * 0.02, requires_grad=False)
                 .to(dtype)) for i, s in enumerate(shapes)]

    def test_perturbation_scale_and_antithetic(self):
        params = self._params([(64, 48)])
        es = R.LowRankES(params, sigma=1e-3, rank=1, seed=1)
        d = es.delta(0, 3, 0, "cpu")
        self.assertEqual(tuple(d.shape), (64, 48))
        # per-entry std of B@A should be ~sigma (rank-1 product of N(0,sigma) x N(0,sigma))
        self.assertAlmostEqual(float(d.std()), 1e-3, delta=3e-4)
        base = params[0][1].detach().clone()
        es.set_member(0, 3, +1); plus = params[0][1].detach().clone()
        es.set_member(0, 3, -1); minus = params[0][1].detach().clone()
        torch.testing.assert_close(plus - base, -(minus - base), atol=1e-6, rtol=0)
        es.restore()
        torch.testing.assert_close(params[0][1].detach(), base, atol=0, rtol=0)

    def test_restore_is_exact_in_bf16(self):
        # bf16 W + d - d != W in general; master weights make restore exact.
        params = self._params([(32, 32)], dtype=torch.bfloat16)
        es = R.LowRankES(params, sigma=1e-3, rank=2, seed=2)
        base = params[0][1].detach().clone()
        for k in range(4):
            es.set_member(0, k, +1)
        es.restore()
        self.assertTrue(torch.equal(params[0][1].detach(), base))

    def test_shape_fitness_matches_trainer_formula(self):
        rng = np.random.default_rng(0)
        F = rng.random((8, 3))
        expect = np.mean(F - F.mean(axis=0, keepdims=True), axis=1)
        np.testing.assert_allclose(R.shape_fitness(F, False), expect)
        z = R.shape_fitness(F, True)
        self.assertAlmostEqual(float(z.std()), 1.0, places=5)
        np.testing.assert_allclose(R.pair_diffs(z), z[0::2] - z[1::2])

    def _quadratic(self, seed=3):
        params = self._params([(16, 8)], seed=seed)
        W = params[0][1]
        g = torch.Generator().manual_seed(seed + 100)
        x = torch.randn(8, generator=g)
        target = torch.randn(16, generator=g)

        def loss():
            return float(((W.detach() @ x - target) ** 2).sum())

        def true_grad():  # d loss / d W = 2 (W x - t) x^T
            return 2.0 * torch.outer(W.detach() @ x - target, x)

        return params, W, loss, true_grad

    def _es_step(self, es, W, loss, step, pop):
        F = np.zeros((pop, 1))
        for m in range(pop):
            es.set_member(step, m // 2, +1 if m % 2 == 0 else -1)
            F[m, 0] = -loss()
        es.restore()
        return R.pair_diffs(R.shape_fitness(F, True))

    def test_es_update_points_downhill(self):
        # The ES step must be positively aligned with -grad (sign + construction check).
        params, W, loss, true_grad = self._quadratic()
        es = R.LowRankES(params, sigma=1e-2, rank=2, seed=4)
        pop = 64
        cosines = []
        for step in range(10):
            before = W.detach().clone()
            g = true_grad()
            d = self._es_step(es, W, loss, step, pop)
            es.update(step, d, lr=1e-3, pop_size=pop)
            upd = (W.detach() - before).flatten()
            cosines.append(float(torch.dot(upd, -g.flatten()) / (upd.norm() * g.norm() + 1e-12)))
        self.assertGreater(float(np.mean(cosines)), 0.2, f"cosines with -grad: {cosines}")

    def test_es_descends_on_quadratic(self):
        params, W, loss, _ = self._quadratic(seed=7)
        es = R.LowRankES(params, sigma=1e-2, rank=2, seed=8)
        pop = 64
        l0 = loss()
        for step in range(60):
            d = self._es_step(es, W, loss, step, pop)
            stats = es.update(step, d, lr=2e-2, pop_size=pop)
            self.assertIn("update_rms", stats)
        l1 = loss()
        self.assertLess(l1, 0.5 * l0, f"ES did not descend: {l0:.4f} -> {l1:.4f}")

    def test_fp32_master_accumulates_sub_ulp_steps(self):
        # In bf16 a step of 1e-6 on weights ~0.02 rounds away; the fp32 master keeps it.
        params = self._params([(64, 64)], dtype=torch.bfloat16, seed=5)
        es = R.LowRankES(params, sigma=1e-3, rank=1, seed=6)
        m0 = es.master[0].clone()
        d = np.ones(1) * 1e-3   # one pair, tiny coefficient
        for step in range(20):
            es.update(0, d, lr=1e-3, pop_size=2)
        self.assertGreater(float((es.master[0] - m0).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
