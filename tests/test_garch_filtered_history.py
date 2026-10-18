"""Bounded numerical/scope checks; these do not establish OOS performance."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('arch')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/mcmc_runtime')]
spec = importlib.util.spec_from_file_location('tested_gjr_filtered_history',
    ROOT/'tools/garch_filtered_history/models.py')
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


def test_recursion_matches_scalar_oracle_and_keeps_negative_asymmetry():
    params = np.array([[.00001, .05, .12, .8], [.00002, .1, .04, .75]])
    initial_v = np.array([.0001, .0002])
    initial_e = np.array([-.01, .02])
    pool = np.array([[-1., .5], [1., -1.5], [.3, .7]])
    indices = np.array([[0, 1, 2], [2, 0, 1]])
    means = np.array([[.001, .002], [.002, .003], [.003, .004]])
    actual = models.forecast_paths(params, initial_v, initial_e, pool, indices, means)
    expected = np.empty_like(actual)
    for s in range(2):
        v, e = initial_v.copy(), initial_e.copy()
        for d in range(3):
            v = params[:, 0]+params[:, 1]*e**2+params[:, 2]*(e < 0)*e**2+params[:, 3]*v
            e = np.sqrt(v)*pool[indices[s, d]]
            expected[s, d] = np.clip(means[d]+e, -1., 1.)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-17)
    with pytest.raises(ArithmeticError, match='future GJR'):
        models.forecast_paths(-np.ones((2, 4)), initial_v, initial_e, pool, indices, means)


def test_common_stationary_rows_repeat_and_wrap():
    indices = models.stationary_indices(4, 240, 15, 1e100, 129)
    np.testing.assert_array_equal(indices[:, 1:], (indices[:, :-1]+1) % 4)
    np.testing.assert_array_equal(indices, models.stationary_indices(4, 240, 15, 1e100, 129))


def test_full_training_fit_constraints_and_normalized_pool():
    x = np.random.default_rng(513).standard_t(7, size=504)*.008 + .0002
    params, v, e, pool, receipt = models.fit_asset(x.tobytes())
    assert receipt['training_rows'] == len(x) and receipt['converged']
    assert receipt['nu'] > 2 and receipt['persistence'] < 1.
    assert params[0] > 0 and params[1] >= 0 and params[3] >= 0
    assert v > 0 and np.isfinite(e)
    assert abs(pool.mean()) < 1e-15
    assert abs(np.mean(pool*pool)-1.) < 1e-15


def test_iid_boundary_solver_roundoff_is_explicit_and_refiltered():
    x = np.random.default_rng(520).normal(.0003, .01, (504, 3))
    receipts = [models.fit_asset(np.ascontiguousarray(a).tobytes())[4] for a in x.T]
    assert any(r['maximum_parameter_feasibility_correction'] > 0. for r in receipts)
    for r in receipts:
        assert r['alpha'] >= 0 and r['alpha']+r['gamma'] >= 0 and r['beta'] >= 0
        assert r['maximum_parameter_feasibility_correction'] <= r['solver_feasibility_tolerance']
        assert r['maximum_variance_refilter_change_percent_units'] < 1e-8


@pytest.mark.parametrize('gamma', [.1, -.1])
def test_closed_stationarity_boundary_projects_only_within_solver_tolerance(gamma):
    alpha = .2
    raw = np.array([.01, alpha, gamma, 1.-alpha-gamma/2., 7.])
    used, correction, original_persistence = models.feasible_parameters(raw)
    assert original_persistence >= 1.
    assert used[1]+used[2]/2.+used[3] < 1.
    assert np.sign(used[2]) == np.sign(gamma)
    assert 0. < correction <= 1e-9
    assert used[0] == raw[0] and used[4] == raw[4]
    np.testing.assert_allclose(used[1:4]/raw[1:4], np.full(3, used[1]/raw[1]), rtol=0., atol=3e-16)
    invalid = raw.copy()
    invalid[3] += 1e-7
    with pytest.raises(ArithmeticError, match='substantive stationarity'):
        models.feasible_parameters(invalid)
    invalid = raw.copy()
    invalid[2] = -invalid[1]-1e-7
    with pytest.raises(ArithmeticError, match='substantive positivity'):
        models.feasible_parameters(invalid)


def test_path_clear_retains_bounded_training_fit_cache():
    x = np.random.default_rng(291).normal(0., .01, 504)
    fitted = models.fit_asset(x.tobytes())
    before = models.fit_asset.cache_info()
    models.clear_path_cache()
    assert models.fit_asset(x.tobytes()) is fitted
    after = models.fit_asset.cache_info()
    assert after.hits == before.hits+1 and after.maxsize == 64


def test_constrained_retry_uses_same_likelihood_and_records_failed_attempt(monkeypatch):
    original = models.arch_model
    calls = []
    def builder(*args, **kwargs):
        model = original(*args, **kwargs)
        fit = model.fit
        def retry_fit(**options):
            calls.append(options)
            result = fit(**options)
            if len(calls) == 1:
                result.params.iloc[3] += 2.
            return result
        model.fit = retry_fit
        return model
    monkeypatch.setattr(models, 'arch_model', builder)
    x = np.random.default_rng(391).standard_t(7, 504)*.01
    receipt = models.fit_asset(x.tobytes())[4]
    assert receipt['retry_count'] >= 1
    assert calls[0]['options']['ftol'] == 1e-9 and calls[1]['options']['ftol'] == 1e-12
    assert calls[1]['starting_values'] is not None
    assert receipt['fit_attempts'][0]['accepted'] is False
    assert receipt['fit_attempts'][-1]['accepted'] is True
    assert receipt['fit_attempts'][0]['used_params_percent_units'] is None


def test_ppw_is_training_only_and_joint_rows(monkeypatch):
    assets = np.random.default_rng(42).normal(size=(40, 3))
    seen = []
    def fit(data):
        x = np.frombuffer(data, np.float64)
        seen.append(x.copy())
        return np.ones(4), 1., 0., x, {'training_rows': len(x)}
    def selector(x):
        np.testing.assert_array_equal(x[:, :3], assets)
        np.testing.assert_array_equal(x[:, 3:], assets**2)
        return pd.DataFrame({'stationary': [1., 2., 3., 4., 5., 6.]})
    monkeypatch.setattr(models, 'fit_asset', fit)
    monkeypatch.setattr(models, 'optimal_block_length', selector)
    models.configuration.cache_clear()
    _, pool, length, receipt = models.configuration(assets.shape, assets.tobytes())
    np.testing.assert_array_equal(np.column_stack(seen), assets)
    np.testing.assert_array_equal(pool, assets)
    assert length == 6. and receipt['joint_row_resampling']
    models.configuration.cache_clear()


def test_exact_original_mean_function(monkeypatch):
    marker = object()
    seen = []
    def original(data):
        seen.append(data)
        return marker
    def curves(fit, horizon):
        assert fit is marker
        return np.arange(horizon, dtype=float), np.zeros(horizon)
    monkeypatch.setattr(models.parent.parent.body.learned.student, 'original_fit', original)
    monkeypatch.setattr(models, 'moment_return_curves', curves)
    np.testing.assert_array_equal(models.canonical_mean(b'training_prefix_only', 7), np.arange(7))
    assert seen == [b'training_prefix_only']
    assert models.parent.VARIANT == 'existing_sv_rate'


@pytest.mark.parametrize('assets_count', [1, 3])
def test_candidate_small_training_and_policy_rejoin(assets_count, monkeypatch):
    from simfolio_forecasting_methodology.runner import TrainingData, PortfolioPolicy, ForecastContext
    rng = np.random.default_rng(520)
    x = np.empty((504, assets_count))
    v = np.full(assets_count, .0001)
    e = np.zeros(assets_count)
    for day in range(len(x)):
        v = .000005 + .12*e*e + .2*(e < 0)*e*e + .7*v
        e = np.sqrt(v)*rng.standard_t(7, assets_count)*np.sqrt(5./7.)
        x[day] = .0003+e
    dates = pd.bdate_range('2020-01-01', periods=511)
    weights = tuple(np.full(assets_count, 1./assets_count))
    training = TrainingData(x@np.array(weights), x,
        PortfolioPolicy(tuple(f'A{i}' for i in range(assets_count)), weights, 'monthly'),
        dates[:504].to_numpy())
    context = ForecastContext(models.MODEL_ID, 'toy', 'toy', 7, 240, 1,
        dates[504:].to_numpy(), str(dates[503].date()))
    monkeypatch.setattr(models, 'canonical_mean', lambda data, horizon: np.full(horizon, .0002))
    models.clear_path_cache()
    a = models.CANDIDATES[0].simulate_daily_log_returns(training, context)
    models.clear_path_cache()
    b = models.CANDIDATES[0].simulate_daily_log_returns(training, context)
    assert a.shape == (240, 7) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert all(record['gjr_student_fit']['training_rows'] == 504
               for record in models.FIT_DIAGNOSTICS.values())
