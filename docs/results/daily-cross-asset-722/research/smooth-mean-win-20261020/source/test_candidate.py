from types import SimpleNamespace
import numpy as np
import pytest
import candidate


def test_formula_known_values_zero_information_and_zero_noise():
    assert candidate.regularized_mean(2, 1) == pytest.approx(1.6)
    assert candidate.regularized_mean(1, 1) == .5
    assert candidate.regularized_mean(-1, 1) == -.5
    assert candidate.regularized_mean(0, 1) == 0
    assert candidate.regularized_mean(2, 0) == 2
    assert np.isfinite(candidate.regularized_mean(1e300, 1e-300))
    assert candidate.regularized_mean(1e-300, 1e300) == 0


def test_formula_is_odd_scale_equivariant_and_monotone_in_noise():
    values = [candidate.regularized_mean(.02, se) for se in (.001, .01, .1, 1)]
    assert all(0 <= values[i+1] < values[i] <= .02 for i in range(3))
    for mu, se in [(.02, .01), (-.01, .03)]:
        assert candidate.regularized_mean(-mu, se) == -candidate.regularized_mean(mu, se)
        assert candidate.regularized_mean(100*mu, 100*se) == pytest.approx(100*candidate.regularized_mean(mu, se))


def test_invalid_mean_or_uncertainty_is_rejected():
    for mu, se in [(np.nan, 1), (1, -1), (np.inf, 1), (1, np.inf)]:
        with pytest.raises(ValueError):
            candidate.regularized_mean(mu, se)


def test_fixed_shift_does_not_change_Gaussian_parameter_dispersion():
    draws = candidate.retained.parameter_draws(20, .01, 120, 551)
    shift = candidate.regularized_mean(.02, .01)-.02
    np.testing.assert_allclose(np.var(draws+shift), np.var(draws), rtol=1e-14)
    assert np.mean(draws+shift) == pytest.approx(shift, abs=1e-17)


def test_no_shrinkage_boundary_replays_full_native_winner_exactly(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    monkeypatch.setattr(candidate, 'regularized_mean', lambda mu, se: mu)
    expected, _ = api.original._forecast(candidate.build_gaussian_equal_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 1191)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 1191)
    np.testing.assert_array_equal(actual, expected)
