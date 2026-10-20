from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import norm
from scipy.integrate import quad
import candidate
import crps_fit


def test_training_CRPS_matches_independent_CDF_integral():
    y = np.array([0., -.5, 1., -3., .2])
    value, _, v, _ = crps_fit.conditional_loss(y, .1, .8)
    independent = []
    for obs, variance in zip(y, v):
        sd = np.sqrt(variance)
        independent.append(quad(lambda x: norm.cdf(x/ sd)**2, -np.inf, obs)[0]
            +quad(lambda x: (1-norm.cdf(x/sd))**2, obs, np.inf)[0])
    assert value == pytest.approx(np.mean(independent), rel=2e-12, abs=2e-12)


def test_recursive_gradient_matches_numerical_differences():
    y = np.random.default_rng(1234).standard_t(7, size=500)*np.sqrt(5/7)
    theta = np.array([-.4, 1.4])
    _, gradient = crps_fit.objective(theta, y)
    numeric = approx_fprime(theta, lambda q: crps_fit.objective(q, y)[0], 1e-6)
    np.testing.assert_allclose(gradient, numeric, rtol=3e-4, atol=2e-7)


def test_same_variance_recursion_and_constant_boundary_as_frozen_filter():
    y = np.random.default_rng(76).normal(size=300)
    for a, b in [(.05, .9), (0., 0.)]:
        _, _, vt, nt = crps_fit.conditional_loss(y, a, b)
        _, _, vg, ng = crps_fit.gaussian.filter_loss(y, a, b)
        np.testing.assert_array_equal(vt, vg)
        assert nt == ng


def test_fit_equivariance_pool_normalization_and_eligible_convergence():
    y = np.random.default_rng(88).standard_t(5, size=800)*.01
    a = crps_fit.fit_crps(y)
    b = crps_fit.fit_crps(2*y+.02)
    np.testing.assert_allclose([a['alpha'], a['beta']], [b['alpha'], b['beta']], rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(a['pool'], b['pool'], rtol=3e-5, atol=3e-6)
    assert abs(a['pool'].mean()) < 1e-14
    assert abs(np.mean(a['pool']**2)-1) < 1e-14
    assert a['alpha'] >= 0 and a['beta'] >= 0 and a['alpha']+a['beta'] < 1
    assert any(x['eligible'] for x in a['fixed_start_results'])
    assert a['objective'] <= a['constant_variance_objective']+1e-12


def test_likelihood_fit_boundary_replays_entire_current_winner(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    monkeypatch.setattr(candidate, 'fit_crps', candidate.retained.FHS.base.fit_variance_target)
    model = candidate.build_candidate()
    assert type(model.components[0]) is candidate.parent.ShrunkComponent
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 531)
    actual, _ = api.original._forecast(model, row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 531)
    np.testing.assert_array_equal(actual, expected)
