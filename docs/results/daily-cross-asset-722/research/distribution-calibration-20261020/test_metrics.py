import numpy as np
import pytest
import metrics


def test_midpoint_ranks_endpoints_ties_and_empirical_quantiles():
    ensemble = np.zeros((240, 3))
    out = metrics.diagnostics(ensemble, [-1, 0, 1])
    np.testing.assert_array_equal(out['CDF_midpoint'], [0, .5, 1])
    np.testing.assert_array_equal(out['forecast_tie_fraction'], [0, 1, 0])
    np.testing.assert_array_equal(out['central_interval_coverage_90'], [0, 1, 0])
    np.testing.assert_array_equal(out['below_quantile_0.05'], [1, 0, 0])
    ensemble = np.repeat(np.arange(1., 241.)[:, None], 3, axis=1)
    out = metrics.diagnostics(ensemble, [12., 120., 228.])
    np.testing.assert_array_equal(out['equal_quantile_0.05'], [1, 0, 0])
    np.testing.assert_array_equal(out['equal_quantile_0.50'], [0, 1, 0])
    np.testing.assert_array_equal(out['equal_quantile_0.95'], [0, 0, 1])


def test_diagnostics_affine_equivariance_and_model_path_permutation():
    rng = np.random.default_rng(21)
    ensemble = rng.normal(size=(240, 50))
    y = rng.normal(size=50)
    original = metrics.diagnostics(ensemble, y)
    shifted = metrics.diagnostics(3*ensemble+2, 3*y+2)
    permuted = metrics.diagnostics(ensemble[rng.permutation(240)], y)
    for metric in original:
        np.testing.assert_array_equal(original[metric], shifted[metric])
        np.testing.assert_array_equal(original[metric], permuted[metric])


def test_calendar_band_normalization_and_no_score_fields():
    ensemble = np.repeat(np.arange(1., 241.)[:, None], 3, axis=1)
    out = metrics.diagnostics(ensemble, [12., 120., 228.])
    summary = metrics.weighted_summary(out, [1, 31, 366], [.2, .3, .5])
    assert summary['all']['CDF_midpoint'] == pytest.approx(np.dot(out['CDF_midpoint'], [.2, .3, .5]))
    assert len(summary) == 4
    assert all('score' not in k for k in summary['all'])
    assert summary['31_365_days']['CDF_midpoint'] == out['CDF_midpoint'][1]
    with pytest.raises(ValueError, match='positive normalized'):
        metrics.weighted_summary(out, [1, 31, 366], [0, .5, .5])
