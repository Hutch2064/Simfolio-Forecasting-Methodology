import numpy as np
import pytest
from statsmodels.stats.meta_analysis import _fit_tau_iterative
import random_effects as re


def test_equal_variance_closed_form_and_zero_boundary():
    means = np.array([-.2, .1, .4, 1.])
    variance = .03
    assert re.tau_squared(means, np.full(4, variance)) == pytest.approx(np.var(means, ddof=1)-variance, rel=1e-13)
    assert re.tau_squared(means*.01, np.full(4, variance)) == 0
    assert re.tau_squared(np.ones(4), np.zeros(4)) == 0
    assert re.tau_squared([0., 1.], [0., 0.]) == pytest.approx(.5)


def test_matches_independent_statsmodels_iterative_algorithm():
    rng = np.random.default_rng(553)
    for _ in range(30):
        variance = rng.uniform(.02, .2, 25)
        means = rng.normal(size=25)
        # Use the library's documented tolerance: tighter tolerances can
        # trigger its ee<0 reset after floating-point Newton overshoot.
        independent, converged = _fit_tau_iterative(means, variance, atol=1e-5, maxiter=100)
        assert converged
        assert re.tau_squared(means, variance) == pytest.approx(independent, abs=2e-6)


def test_translation_scale_and_order_equivariance():
    means = np.array([-.03, .01, .002, .05])
    variance = np.array([.0001, .0002, .0003, .0001])
    fitted = re.tau_squared(means, variance)
    assert re.tau_squared(-3*means+.02, 9*variance) == pytest.approx(9*fitted, rel=1e-13)
    assert re.tau_squared(means[::-1], variance[::-1]) == pytest.approx(fitted, rel=1e-13)


def test_batches_use_every_training_observation_and_no_future():
    history = np.random.default_rng(61).normal(size=537)
    fit = re.fit_training_batches(history)
    groups = np.array_split(history, int(np.sqrt(len(history))))
    assert sum(fit['batch_sizes']) == len(history)
    assert max(fit['batch_sizes'])-min(fit['batch_sizes']) <= 1
    np.testing.assert_array_equal(fit['batch_means'], [g.mean() for g in groups])
    np.testing.assert_array_equal(fit['within_batch_sampling_variances'], [np.var(g, ddof=1)/len(g) for g in groups])
    for bad in [np.full(100, np.nan), np.arange(10.)]:
        with pytest.raises(ValueError):
            re.fit_training_batches(bad)


def test_zero_variance_component_recovers_exact_parent_native_forecast(monkeypatch):
    from types import SimpleNamespace
    import candidate
    import sweep
    x = np.random.default_rng(953).normal(.0002, .01, 650)
    dates = np.arange(np.datetime64('2000-01-01'), np.datetime64('2000-01-01')+np.timedelta64(650, 'D'))
    origin = SimpleNamespace(position=599, max_horizon=30, origin_date=str(dates[599]), origin_label='unit')
    monkeypatch.setattr(candidate.random_effects, 'fit_training_batches', lambda history: {'tau_squared': 0.})
    expected, _ = sweep.original._forecast(candidate.build_smooth_mean_pool(), 'toy', x, dates, origin, 55)
    actual, _ = sweep.original._forecast(candidate.build_candidate(), 'toy', x, dates, origin, 55)
    np.testing.assert_array_equal(actual, expected)
    for component in candidate.build_candidate().components:
        assert component.kind in ('WR', 'FHS')


def test_mean_uncertainty_is_exact_gaussian_convolution_with_unchanged_draws():
    import candidate
    parent = candidate.parent.retained
    se = parent.gaussian_sd(15, .001)
    original = parent.parameter_draws(15, .001, 120, 47)
    actual = candidate.mean_draws(15, .001, 120, 47, 3*se**2)
    np.testing.assert_allclose(actual, 2*original, atol=1e-17)
    np.testing.assert_array_equal(candidate.mean_draws(15, .001, 120, 47, 0), original)
