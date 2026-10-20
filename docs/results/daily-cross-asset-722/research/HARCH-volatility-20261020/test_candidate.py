from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import approx_fprime, minimize
from arch.univariate import HARCH
import harch
import candidate


def test_lag_features_and_variance_match_independent_arch_package():
    y = np.random.default_rng(22).normal(size=200)
    features = harch.lag_features(y)
    a = np.array([.05, .15, .7])
    expected = HARCH(lags=[1, 5, 22]).compute_variance(np.r_[1-a.sum(), a], y,
        np.empty(len(y)), 1., np.column_stack((np.zeros(len(y)), np.full(len(y), np.inf))))
    np.testing.assert_allclose(1+(features-1)@a, expected, atol=1e-14)
    assert np.all(features[0] == 1)
    assert features[10, 2] == pytest.approx((sum(y[:10]**2)+12)/22)


def test_analytic_gradient_and_independent_constrained_fit():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, .01, 400), rng.normal(0, .04, 400)]
    y = (x-x.mean())/np.std(x)
    features = harch.lag_features(y)
    theta = np.array([-2., -1., 1.])
    _, gradient = harch.objective(theta, y, features)
    numerical = approx_fprime(theta, lambda t: harch.objective(t, y, features)[0], 1e-6)
    np.testing.assert_allclose(gradient, numerical, atol=1e-7)
    fitted = harch.fit(x)
    def objective_a(a):
        v = 1+(features-1)@a
        if np.min(v) <= 0:
            return 1e6
        return .5*np.mean(np.log(v)+y*y/v)
    independent = minimize(objective_a, [.1, .1, .3], method='SLSQP', bounds=[(0, 1)]*3,
        constraints=[{'type': 'ineq', 'fun': lambda a: 1-harch.MARGIN-a.sum()}],
        options={'ftol': 1e-12, 'maxiter': 1000})
    assert independent.success
    assert fitted['objective'] == pytest.approx(independent.fun, abs=2e-7)


def test_training_equivariance_pool_moments_and_no_failed_fit_fallback(monkeypatch):
    x = np.random.default_rng(65).standard_t(5, size=700)*.01
    fit = harch.fit(x)
    moved = harch.fit(2*x+.02)
    np.testing.assert_allclose(moved['alpha'], fit['alpha'], atol=1e-8)
    np.testing.assert_allclose(moved['pool'], fit['pool'], atol=1e-8)
    assert fit['pool'].mean() == pytest.approx(0, abs=1e-14)
    assert np.mean(fit['pool']**2) == pytest.approx(1, abs=1e-14)
    assert 0 <= fit['alpha'].sum() < 1
    original = harch.minimize
    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        result.success = False
        return result
    monkeypatch.setattr(harch, 'minimize', fail)
    with pytest.raises(ArithmeticError, match='Neither fixed'):
        harch.fit(x)


def test_multiscale_forecast_matches_independent_explicit_history_recursion():
    rng = np.random.default_rng(6)
    pool = np.array([-1.5, -.3, .2, .8, 1.4])
    indices = rng.integers(0, len(pool), size=(3, 80))
    last = rng.random(22)
    a = np.array([.05, .15, .7])
    delta = np.array([-.0001, 0, .0001])
    out, clips, floors = harch.native_paths(pool, indices, .0003, .0004, a, last, delta)
    golden = np.empty_like(out)
    for p in range(3):
        history = list(last)
        wealth = 1.
        previous = 0.
        for t in range(80):
            variance = 1-a.sum()+sum(a[j]*np.mean(history[-lag:]) for j, lag in enumerate(harch.LAGS))
            innovation = np.sqrt(variance)*pool[indices[p, t]]
            raw = .0003+delta[p]+.02*innovation
            wealth *= 1+np.expm1(raw)
            current = np.log(max(wealth, 1e-300))
            golden[p, t] = current-previous
            previous = current
            history.append(innovation**2)
    np.testing.assert_allclose(out, golden, atol=1e-14)
    assert (clips, floors) == (0, 0)


def test_ARCH1_and_constant_variance_forecast_boundaries_match_original_kernel():
    rng = np.random.default_rng(7)
    pool = rng.normal(size=600)
    pool -= pool.mean()
    pool /= np.std(pool)
    last = rng.random(22)
    indices = rng.integers(0, len(pool), size=(4, 60))
    delta = np.zeros(4)
    for a in [0., .4]:
        actual = harch.native_paths(pool, indices, .0003, .0004, np.array([a, 0, 0]), last, delta)
        expected = candidate.retained.FHS.base.recursive_paths(pool, indices, .0003, .0004, a, 0., 1-a+a*last[-1])
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-14)
        assert actual[1:] == expected[1:]


def test_complete_pool_wrapper_identity_when_original_FHS_is_substituted(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    monkeypatch.setattr(candidate, 'MultiscaleComponent', lambda: candidate.parent.ShrunkComponent('FHS'))
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 193)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 193)
    np.testing.assert_array_equal(actual, expected)
