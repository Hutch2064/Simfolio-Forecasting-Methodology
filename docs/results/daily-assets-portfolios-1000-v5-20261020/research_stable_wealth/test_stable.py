import math
import unittest
from types import SimpleNamespace
import numpy as np
import streaming
from . import kernels, adapter
from simfolio_forecasting_methodology.runner import TrainingData, PortfolioPolicy
from simfolio_forecasting_methodology.french_panel.runner import _readonly_copy


def states(n):
    return (np.ones(n), np.zeros(n), np.zeros(n, dtype=bool), np.zeros(n), np.zeros(n))


class StableTests(unittest.TestCase):
    def test_raw_finite_paths_exact_and_chunk_invariant(self):
        rng = np.random.default_rng(41)
        history = rng.normal(.001, .04, 600)
        source = np.expm1(history)
        ix = rng.integers(0, 600, (1000, 257))
        delta = rng.normal(0, .001, 1000)
        original = streaming.native_block(history, source, ix, delta, np.ones(1000),
                                          np.zeros(1000), np.zeros(1000))[0]
        state = states(1000)
        chunks = [kernels.raw_block(history, source, ix[:, i:i+17], delta, *state)[0]
                  for i in range(0, 257, 17)]
        np.testing.assert_array_equal(np.concatenate(chunks, axis=1), original)

    def test_FHS_finite_paths_exact(self):
        rng = np.random.default_rng(42)
        pool = rng.normal(size=300)
        ix = rng.integers(0, 300, (1000, 251))
        means = rng.normal(.001, .0001, 1000)
        old = streaming.fhs_block(pool, ix, means, .0004, .05, .90, np.ones(1000),
                                  np.ones(1000), np.zeros(1000), np.zeros(1000))[0]
        state = states(1000)
        variance = np.ones(1000)
        new = [kernels.fhs_block(pool, ix[:, i:i+19], means, .0004, .05, .90, variance, *state)[0]
               for i in range(0, 251, 19)]
        np.testing.assert_array_equal(np.concatenate(new, axis=1), old)

    def test_positive_overflow_continues_without_new_cap(self):
        history = np.array([700.])
        source = np.expm1(history)
        state = states(1)
        actual, transitions = kernels.raw_block(history, source, np.zeros((1, 6), dtype=int), np.zeros(1), *state)
        np.testing.assert_allclose(actual[0], 700.*np.arange(1, 7), rtol=0, atol=2e-12)
        self.assertEqual(transitions, 1)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(actual[0, -1], math.log(np.finfo(float).max))

    def test_native_underflow_remains_absorbing(self):
        history = np.array([math.log(1e-6), 700.])
        source = np.array([-.999999, np.expm1(700.)])
        indices = np.array([[0]*100+[1]])
        old = streaming.native_block(history, source, indices, np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1))[0]
        actual = kernels.raw_block(history, source, indices, np.zeros(1), *states(1))[0]
        np.testing.assert_array_equal(actual, old)

    def test_full_adapter_regular_1000_path_exact(self):
        x = np.random.default_rng(42).normal(.001, .012, 900)
        train = TrainingData(portfolio_log_returns=_readonly_copy(x),
            asset_log_returns=_readonly_copy(x.reshape(-1, 1)), policy=PortfolioPolicy(('X',), (1.,), 'none'))
        context = SimpleNamespace(simulations=1000, horizon_days=171, seed=528)
        original = list(streaming.terminal_chunks(streaming.BEST, x, 171, 1000, 528))
        model = adapter.build()
        actual = list(model.iter_terminal_log_returns(train, context))
        for (i, a), (j, b) in zip(original, actual, strict=True):
            self.assertEqual(i, j)
            np.testing.assert_array_equal(a, b)
        self.assertEqual(model.last_metadata['positive_wealth_overflow_transitions'], 0)


if __name__ == '__main__':
    unittest.main()
