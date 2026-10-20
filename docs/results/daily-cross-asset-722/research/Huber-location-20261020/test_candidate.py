from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from statsmodels.robust.norms import HuberT
import location
import candidate


def test_location_matches_independent_Huber_objective():
    x = np.r_[np.random.default_rng(33).normal(size=1500)*.01, -.4, .3]
    estimate = location.estimate(x)
    scale = np.median(np.abs(x-np.median(x)))/location.NORMAL_MAD
    huber = HuberT(t=location.C)
    result = minimize_scalar(lambda m: np.sum(huber.rho((x-m)/scale)),
        bounds=(float(x.min()), float(x.max())), method='bounded', options={'xatol': 1e-14})
    assert result.success
    assert estimate == pytest.approx(result.x, abs=2e-10)
    assert abs(np.sum(huber.psi((x-estimate)/scale))) < 1e-9


def test_equivariance_constant_and_zero_MAD_boundaries():
    x = np.r_[np.random.default_rng(34).normal(size=600)*.01, .8]
    estimate = location.estimate(x)
    assert location.estimate(2*x+.02) == pytest.approx(2*estimate+.02, abs=1e-14)
    assert location.estimate(-x) == pytest.approx(-estimate, abs=1e-14)
    assert location.estimate(np.full(10, .003)) == .003
    assert np.isfinite(location.estimate(np.r_[np.zeros(10), .1]))
    assert location.estimate(x, c=np.inf) == float(x.mean())


def test_fixed_tuning_has_canonical_normal_efficiency_and_bounded_influence():
    c = location.C
    center_probability = 2*norm.cdf(c)-1
    score_variance = center_probability-2*c*norm.pdf(c)+2*c*c*(1-norm.cdf(c))
    assert center_probability**2/score_variance == pytest.approx(.95, abs=.0001)
    x = np.linspace(-1, 1, 1001)
    moved = np.r_[x, 1000.]
    assert abs(location.estimate(moved)-location.estimate(x)) < .005
    assert abs(moved.mean()-x.mean()) > .9


def test_native_full_forecast_without_robustification_is_current_winner(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    monkeypatch.setattr(candidate, 'location_estimate', lambda x: float(x.mean()))
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 193)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 193)
    np.testing.assert_array_equal(actual, expected)
