from types import SimpleNamespace
import numpy as np
import pytest
import candidate


def test_precision_addition_gives_existing_center_and_new_variance():
    for mu, se in [(2., 1.), (-.01, .02), (.004, .0001)]:
        variance = 1/(1/se**2+1/mu**2)
        mean = variance*mu/se**2
        assert candidate.parent.regularized_mean(mu, se) == pytest.approx(mean)
        assert (se*candidate.contraction_factor(mu, se))**2 == pytest.approx(variance)


def test_finite_boundaries_and_invalid_inputs():
    assert candidate.contraction_factor(0., 1.) == 0.
    assert candidate.contraction_factor(1., 0.) == 1.
    assert candidate.contraction_factor(1e300, 1e-300) == 1.
    assert candidate.contraction_factor(1e-300, 1e300) == 0.
    for mu, se in [(np.nan, 1), (1, -1), (np.inf, 1), (1, np.inf)]:
        with pytest.raises(ValueError):
            candidate.contraction_factor(mu, se)


def test_scale_sign_and_noise_monotonicity():
    a = candidate.contraction_factor(-.02, .01)
    assert a == candidate.contraction_factor(.02, .01)
    assert a == pytest.approx(candidate.contraction_factor(2., 1.))
    assert 0 < candidate.contraction_factor(.02, .03) < a < 1


def test_same_random_nodes_and_center_with_derived_variance_change():
    mu, se = .02, .01
    factor = candidate.contraction_factor(mu, se)
    draws = candidate.retained.parameter_draws(20, .01, 120, 121)
    shift = candidate.parent.regularized_mean(mu, se)-mu
    new = draws*factor+shift
    assert np.mean(new) == pytest.approx(shift, abs=1e-17)
    assert np.var(new) == pytest.approx(np.var(draws)*factor**2, rel=1e-14)


def test_no_contraction_boundary_replays_complete_current_winner(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    monkeypatch.setattr(candidate, 'contraction_factor', lambda mu, se: 1.)
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 1391)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 1391)
    np.testing.assert_array_equal(actual, expected)
