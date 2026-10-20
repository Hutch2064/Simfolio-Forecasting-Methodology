from types import SimpleNamespace
from itertools import product
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
import candidate
import joint_fit


def test_innovation_derivative_and_joint_gradient():
    y = np.random.default_rng(71).normal(size=600)
    errors, derivative = joint_fit.innovations(y, .3)
    numeric = (joint_fit.innovations(y, .3+1e-6)[0]-joint_fit.innovations(y, .3-1e-6)[0])/2e-6
    np.testing.assert_allclose(derivative, numeric, rtol=1e-7, atol=2e-9)
    for parameters in (np.array([-.4, 1.4, .3]), np.array([1.2, -.7, -.5])):
        _, gradient = joint_fit.objective(parameters, y)
        numeric = approx_derivative(lambda x: joint_fit.objective(x, y)[0], parameters, method='3-point').ravel()
        np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=2e-8)


def test_joint_likelihood_against_independent_conditional_recursion():
    y = np.random.default_rng(82).standard_t(7, size=400)
    a, b, theta = .08, .84, -.2
    loss, _, errors, variance, next_v = joint_fit.filter_loss(y, a, b, theta)
    independent = []
    previous = 0.
    for observation in y:
        epsilon = observation-theta*previous
        independent.append(epsilon)
        previous = epsilon
    independent = np.array(independent)
    target = np.mean(independent**2)
    conditional = [target]
    for epsilon in independent:
        conditional.append(target*(1-a-b)+a*epsilon**2+b*conditional[-1])
    np.testing.assert_array_equal(errors, independent)
    np.testing.assert_allclose(variance, conditional[:-1], rtol=1e-14, atol=1e-14)
    assert next_v == pytest.approx(conditional[-1], rel=1e-14)
    expected = .5*np.mean(np.log(conditional[:-1])+independent**2/conditional[:-1])
    assert loss == pytest.approx(expected, rel=1e-14)


def test_zero_MA_training_boundary_matches_frozen_GARCH():
    y = np.random.default_rng(22).normal(size=500)
    y = y/np.sqrt(np.mean(y*y))
    loss, _, _, v, next_v = joint_fit.filter_loss(y, .05, .9, 0.)
    expected, _, vg, ng = joint_fit.gaussian.filter_loss(y, .05, .9)
    np.testing.assert_allclose(v, vg, rtol=2e-14, atol=2e-14)
    assert loss == pytest.approx(expected, abs=1e-14)
    assert next_v == pytest.approx(ng, rel=2e-14)


def test_fit_equivariance_and_recovery_of_synthetic_dependence():
    rng = np.random.default_rng(71)
    eps = rng.normal(size=6001)
    history = (eps[1:]+.3*eps[:-1])*.01
    fit = joint_fit.fit_joint(history)
    transformed = joint_fit.fit_joint(2*history+.02)
    assert fit['theta'] == pytest.approx(.3, abs=.035)
    np.testing.assert_allclose([fit['theta'], fit['alpha'], fit['beta']],
        [transformed['theta'], transformed['alpha'], transformed['beta']], atol=2e-5, rtol=1e-4)
    assert transformed['target_variance'] == pytest.approx(4*fit['target_variance'], rel=2e-5)
    assert abs(fit['pool'].mean()) < 1e-14
    assert np.mean(fit['pool']**2) == pytest.approx(1., abs=1e-14)
    assert any(x['eligible'] for x in fit['joint_attempts'])


def test_new_kernel_zero_MA_matches_native_accounting_exactly():
    fit = joint_fit.gaussian.fit_variance_target(np.random.default_rng(31).normal(size=500)*.01)
    indices = np.random.default_rng(45).integers(0, len(fit['pool']), size=(2, 250))
    expected, c, f = joint_fit.gaussian.recursive_paths(fit['pool'], indices, fit['mean'],
        fit['target_variance'], fit['alpha'], fit['beta'], fit['next_variance_normalized'])
    actual, cc, ff = joint_fit.recursive_paths(fit['pool'], indices, fit['mean'],
        fit['target_variance'], fit['alpha'], fit['beta'], fit['next_variance_normalized'], 0., .02)
    np.testing.assert_array_equal(actual, expected)
    assert (cc, ff) == (c, f)


def test_finite_memory_cumulative_moments_and_known_boundary_shock():
    pool = np.array([-1., 1.])
    indices = np.array(list(product([0, 1], repeat=3)))
    theta, last, sd, mu = .3, -.01, .02, .001
    daily, clips, floors = joint_fit.recursive_paths(pool, indices, mu, sd*sd, 0., 0., 1., theta, last)
    cumulative = daily.cumsum(axis=1)
    expected = np.array([mu*h+theta*last for h in (1, 2, 3)])
    np.testing.assert_allclose(cumulative.mean(axis=0), expected, atol=1e-14, rtol=0)
    expected_variance = np.array([sd*sd*(1+(h-1)*(1+theta)**2) for h in (1, 2, 3)])
    np.testing.assert_allclose(cumulative.var(axis=0), expected_variance, atol=1e-14, rtol=1e-12)
    assert clips == floors == 0


def test_full_current_forecast_boundary_replays_exactly(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    def no_MA(history):
        return {**joint_fit.gaussian.fit_variance_target(history), 'theta': 0.,
                'last_innovation': float(history[-1]-history.mean())}
    monkeypatch.setattr(joint_fit, 'fit_joint', no_MA)
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 151)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 151)
    np.testing.assert_array_equal(actual, expected)
