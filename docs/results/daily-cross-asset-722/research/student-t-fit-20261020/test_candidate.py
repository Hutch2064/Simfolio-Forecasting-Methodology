from types import SimpleNamespace
import numpy as np
from scipy.optimize import approx_fprime
from arch.univariate.distribution import StudentsT

import candidate
import t_fit


def test_likelihood_matches_independent_library_and_gradient():
    y = np.random.default_rng(1234).standard_t(7, size=500)*np.sqrt(5/7)
    theta = np.array([-.4, 1.4, 6.5])
    value, gradient = t_fit.objective(theta, y)
    a, b, _ = t_fit.gaussian.parameters(theta[:2])
    _, _, variances, _ = t_fit.conditional_loss(y, a, b, theta[2])
    expected = -StudentsT().loglikelihood([theta[2]], y, variances)/len(y)
    np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)
    numeric = approx_fprime(theta, lambda q: t_fit.objective(q, y)[0], 1e-6)
    np.testing.assert_allclose(gradient, numeric, rtol=3e-4, atol=2e-7)


def test_same_variance_recursion_as_frozen_gaussian_filter():
    y = np.random.default_rng(76).normal(size=300)
    for a, b in [(.05, .9), (0., 0.)]:
        _, _, vt, nt = t_fit.conditional_loss(y, a, b, 8.)
        _, _, vg, ng = t_fit.gaussian.filter_loss(y, a, b)
        np.testing.assert_array_equal(vt, vg)
        assert nt == ng


def test_fit_is_translation_and_scale_equivariant_and_empirical_pool_normalized():
    y = np.random.default_rng(88).standard_t(5, size=800)*.01
    a = t_fit.fit_student(y)
    b = t_fit.fit_student(2*y+.02)
    np.testing.assert_allclose([a['alpha'], a['beta']], [b['alpha'], b['beta']], rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(a['pool'], b['pool'], rtol=3e-5, atol=3e-6)
    assert abs(a['pool'].mean()) < 1e-14
    assert abs(np.mean(a['pool']**2)-1) < 1e-14
    assert a['alpha'] >= 0 and a['beta'] >= 0 and a['alpha']+a['beta'] < 1
    assert any(x['eligible'] for x in a['student_attempts'])


def test_gaussian_boundary_replays_entire_retained_forecast_exactly(monkeypatch):
    import sweep as api
    panel = api.load_panel(candidate.ROOT/'inputs')
    row = panel['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    origin = SimpleNamespace(**origins[0])
    monkeypatch.setattr(candidate, 'fit_student', candidate.retained.FHS.base.fit_variance_target)
    model = candidate.build_candidate()
    assert type(model.components[0]) is candidate.retained.GaussianComponent
    assert candidate.retained.GaussianPool.simulate_daily_log_returns is model.simulate_daily_log_returns.__func__
    expected, _ = api.original._forecast(candidate.build_gaussian_equal_pool(), row['series_id'], values, dates, origin, 531)
    actual, _ = api.original._forecast(model, row['series_id'], values, dates, origin, 531)
    np.testing.assert_array_equal(actual, expected)
