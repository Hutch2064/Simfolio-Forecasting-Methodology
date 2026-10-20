from types import SimpleNamespace
import numpy as np
import pytest
import candidate
import posterior


def test_single_coefficient_reconstructs_exact_native_pool_and_variance():
    history = np.random.default_rng(72).standard_t(7, size=800)*.01
    original = candidate.retained.FHS.base.fit_variance_target(history)
    rebuilt = candidate.conditional_fit(candidate.basis(history), original['alpha'], original['beta'])
    np.testing.assert_array_equal(rebuilt['pool'], original['pool'])
    assert rebuilt['next_variance_normalized'] == original['next_variance_normalized']
    assert rebuilt['target_variance'] == original['target_variance']


def test_next_variance_against_native_filter_for_importance_coordinates():
    y = np.random.default_rng(73).normal(size=200)
    points = np.array([[-.5, 2.], [-3., -.1], [1., 1.]])
    _, values = posterior.targets_and_next_variance(points, y)
    for coordinates, value in zip(points, values):
        a, b, _ = posterior.gaussian.parameters(coordinates)
        assert value == pytest.approx(posterior.gaussian.filter_loss(y, a, b)[3], rel=2e-14)


def test_precision_gate_rejects_concentrated_or_disagreeing_importance_results():
    left = {'mean': np.array([.1, .8]), 'covariance': np.diag([.01, .02]),
            'next_variance_mean': 1., 'ESS': 1000., 'largest_normalized_weight': .01}
    assert posterior.precision_check([left, left])['qualified']
    assert not posterior.precision_check([left, {**left, 'ESS': 100.}])['qualified']
    assert not posterior.precision_check([left, {**left, 'largest_normalized_weight': .05}])['qualified']
    assert not posterior.precision_check([left, {**left, 'next_variance_mean': 1.05}])['qualified']
    assert not posterior.precision_check([left, {**left, 'mean': np.array([.2, .8])}])['qualified']


def test_posterior_draws_deterministic_stationary_and_affine_invariant():
    history = np.random.default_rng(74).standard_t(7, size=700)*.01
    one, diagnostic = posterior.parameter_draws(history, 120, 77)
    two, _ = posterior.parameter_draws(history, 120, 77)
    transformed, _ = posterior.parameter_draws(2*history+.02, 120, 77)
    np.testing.assert_array_equal(one, two)
    np.testing.assert_allclose(one, transformed, atol=2e-6, rtol=2e-5)
    assert diagnostic['qualified']
    assert np.all(one >= 0) and np.all(one.sum(axis=1) < 1.)


def test_collapsed_coefficient_posterior_replays_entire_current_forecast(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    def point_posterior(history, paths, seed):
        fit = candidate.retained.FHS.base.fit_variance_target(history)
        return np.tile([fit['alpha'], fit['beta']], (paths, 1)), {'qualified': True, 'collapsed_boundary': True}
    monkeypatch.setattr(posterior, 'parameter_draws', point_posterior)
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 173)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 173)
    np.testing.assert_array_equal(actual, expected)


def test_failed_integration_does_not_fall_back_to_a_point_fit(monkeypatch):
    calls = []
    monkeypatch.setattr(posterior, 'fit_proposal', lambda y: (np.zeros(2), np.eye(2)))
    def unqualified(y, mode, covariance, seed, power):
        calls.append(power)
        return {'mean': np.array([.1, .8]), 'covariance': np.eye(2),
                'next_variance_mean': 1., 'ESS': 1., 'largest_normalized_weight': 1.}
    monkeypatch.setattr(posterior, 'importance_nodes', unqualified)
    with pytest.raises(ArithmeticError, match='failed frozen importance precision'):
        posterior.parameter_draws(np.array([-.02, .01, .03, -.01, .02]), 120, 31)
    assert calls == [power for power in posterior.NODE_POWERS for _ in range(2)]
