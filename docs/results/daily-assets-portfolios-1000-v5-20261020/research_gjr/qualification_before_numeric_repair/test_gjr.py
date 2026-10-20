import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from . import gjr_fit as g
import streaming
from unittest.mock import patch
from .adapter import build
from types import SimpleNamespace
from simfolio_forecasting_methodology.french_panel.runner import _readonly_copy
from simfolio_forecasting_methodology.runner import TrainingData, PortfolioPolicy


class GjrTests(unittest.TestCase):
    def test_analytic_gradient(self):
        y = np.random.default_rng(777).standard_t(7, 400)
        y = (y-y.mean())/y.std()
        kn = float(np.mean(np.where(y < 0, y*y, 0.)))
        for theta in np.random.default_rng(12).normal(0, 2, (15, 3)):
            _, gradient = g.objective(theta, y, 1.-kn, kn)
            numeric = approx_derivative(lambda t: g.objective(t, y, 1.-kn, kn)[0], theta, method='3-point').reshape(-1)
            np.testing.assert_allclose(gradient, numeric, rtol=2e-5, atol=1e-8)

    def test_global_empirical_pool_stationarity_and_target(self):
        y = np.random.default_rng(32).standard_t(4, 700)
        y = (y-y.mean())/y.std()
        for theta in np.random.default_rng(33).normal(0, 5, (30, 3)):
            ap, am, beta, _ = g.parameters(theta)
            self.assertLess(beta+max(ap, am), 1.)
            _, _, variance, _ = g.filter_loss(y, ap, am, beta, .4, .6)
            pool, omega, persistence, kn = g.target_pool(y, variance, ap, am, beta)
            self.assertGreater(omega, 0)
            self.assertLess(persistence, 1)
            np.testing.assert_allclose(pool.mean(), 0, atol=1e-15)
            np.testing.assert_allclose(np.mean(pool*pool), 1, atol=1e-15)
            self.assertAlmostEqual(omega+beta+ap*(1-kn)+am*kn, 1., places=15)

    def test_symmetric_filter_and_recursion_match_parent(self):
        y = np.random.default_rng(778).normal(size=500)
        _, _, actual, next_v = g.filter_loss(y, .05, .05, .90, .5, .5)
        _, _, expected, expected_next = streaming.incumbent.retained.FHS.base.filter_loss(y, .05, .90)
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-15)
        self.assertAlmostEqual(next_v, expected_next, places=14)
        pool = np.array([-.8, -.1, .2, 1.3])
        indices = np.random.default_rng(779).integers(0, 4, (24, 79))
        mean = np.linspace(-.001, .001, 24)
        states_a = (np.full(24, .9), np.ones(24), np.zeros(24), np.zeros(24))
        states_b = tuple(x.copy() for x in states_a)
        first = streaming.fhs_block(pool, indices, mean, .0004, .05, .90, *states_a)[0]
        second = g.terminal_block(pool, indices, mean, .0004, .05, .05, .90, 1.-.05-.90, *states_b)
        np.testing.assert_array_equal(first, second)

    def test_fitted_asymmetric_data_and_scale_equivariance(self):
        rng = np.random.default_rng(15)
        n = 7000
        x = np.empty(n)
        variance = 1.
        for i in range(n):
            x[i] = np.sqrt(variance)*rng.normal()
            variance = .11+(.16 if x[i] < 0 else .02)*x[i]**2+.80*variance
        fitted = g.fit(x*.01)
        self.assertGreater(fitted['alpha_negative'], fitted['alpha_positive'])
        self.assertLessEqual(fitted['objective'], fitted['nested_parent_objective'])
        scaled = g.fit(x*.02)
        for key in ['alpha_positive', 'alpha_negative', 'beta', 'next_variance_normalized']:
            self.assertAlmostEqual(fitted[key], scaled[key], places=9)
        self.assertAlmostEqual(scaled['target_variance']/fitted['target_variance'], 4., places=12)
        self.assertTrue(any(x['eligible'] for x in fitted['fixed_start_results']))

    def test_invalid_input(self):
        for x in [[0., 0.], [float('nan'), 1.], [0.]]:
            with self.assertRaises(ValueError):
                g.fit(x)

    def test_nested_adapter_exact_1000_path_parent_and_unchanged_raw_half(self):
        x = np.random.default_rng(72).normal(.0004, .013, 700)
        training = TrainingData(portfolio_log_returns=_readonly_copy(x),
            asset_log_returns=_readonly_copy(x.reshape(-1, 1)),
            policy=PortfolioPolicy(('X',), (1.,), 'none'))
        context = SimpleNamespace(simulations=1000, horizon_days=139, seed=99)
        history = np.log1p(np.clip(np.expm1(x), -.999999, np.finfo(float).max/10))
        fitted = streaming.incumbent.retained.FHS.base.fit_variance_target(history)
        forced = {**fitted, 'alpha_positive': fitted['alpha'], 'alpha_negative': fitted['alpha'],
                  'omega_forecast': fitted['omega']}
        parent = list(streaming.terminal_chunks(streaming.BEST, x, 139, 1000, 99))
        with patch.object(g, 'fit', return_value=forced):
            actual = list(build().iter_terminal_log_returns(training, context))
        for (i, a), (j, b) in zip(actual, parent, strict=True):
            self.assertEqual(i, j)
            np.testing.assert_array_equal(a, b)
        candidate = list(build().iter_terminal_log_returns(training, context))
        for (_, a), (_, b) in zip(candidate, parent, strict=True):
            self.assertEqual(a.shape[0], 1000)
            np.testing.assert_array_equal(a[:500], b[:500])


if __name__ == '__main__':
    unittest.main()
