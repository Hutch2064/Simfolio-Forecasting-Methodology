"""Location-only contracts for the exact M261 mean ablation."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pyvinecopulib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/mcmc_runtime"))
spec = importlib.util.spec_from_file_location(
    "tested_static_mean_models", ROOT / "tools/expanded_static_mean/models.py"
)
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


def test_translation_preserves_centered_asset_paths():
    paths = np.random.default_rng(29).normal(0.002, 0.02, (12, 19, 3))
    before = paths.copy()
    old_mean = np.linspace(-0.001, 0.003, 57).reshape(19, 3)
    new_mean = np.array([0.0007, -0.0002, 0.0003])
    translation = new_mean - old_mean
    identity = id(paths)
    models.translate_asset_paths_inplace(paths, translation)
    assert id(paths) == identity
    np.testing.assert_array_equal(paths, before + translation)
    np.testing.assert_allclose(paths - new_mean, before - old_mean, rtol=0, atol=2e-17)
    np.testing.assert_allclose(paths.var(axis=0), before.var(axis=0), rtol=0, atol=1e-18)


def test_mean_translation_reads_original_fit_without_mutating_sd(monkeypatch):
    assets = np.arange(30.0).reshape(10, 3) / 10000
    fits = [object(), object(), object()]
    seen = []
    sd = np.array([0.01, 0.02])

    def original_fit(data):
        seen.append(data)
        return fits[len(seen) - 1]

    def moments(fit, horizon):
        assert fit in fits and horizon == 2
        return np.array([0.001, 0.002]), sd

    monkeypatch.setattr(models.parent.parent.body.learned.student, "original_fit", original_fit)
    monkeypatch.setattr(models, "moment_return_curves", moments)
    actual = models.mean_translation(assets, 2)
    np.testing.assert_array_equal(actual, assets.mean(axis=0) - np.array([[0.001], [0.002]]))
    np.testing.assert_array_equal(sd, [0.01, 0.02])
    assert seen == [np.ascontiguousarray(x).tobytes() for x in assets.T]


def test_asset_translation_precedes_original_rejoin_and_restores_callback(monkeypatch):
    assets = np.array([[0.01, -0.02], [0.03, 0.01]])
    training = SimpleNamespace(asset_log_returns=assets)
    context = SimpleNamespace(horizon_days=2)
    original_paths = np.array([[[0.1, -0.2], [0.3, 0.4]]])
    delta = np.array([[0.001, -0.002], [0.003, 0.004]])
    weights, mask = object(), object()
    seen = []

    def rejoin(paths, observed_weights, observed_mask):
        seen.append(paths.copy())
        assert observed_weights is weights and observed_mask is mask
        return paths[:, :, 0]

    def simulate(self, observed_training, observed_context):
        assert observed_training is training and observed_context is context
        for history in assets.T:
            identity = models.hashlib.sha256(np.ascontiguousarray(history).tobytes()).hexdigest()
            models.FIT_DIAGNOSTICS[self.model_id, identity] = {}
        return models.shell.rejoin(original_paths.copy(), weights, mask)

    monkeypatch.setattr(models.shell, "rejoin", rejoin)
    monkeypatch.setattr(models.parent.Candidate, "simulate_daily_log_returns", simulate)
    monkeypatch.setattr(models, "mean_translation", lambda a, h: delta)
    models.CANDIDATES[0].simulate_daily_log_returns(training, context)
    np.testing.assert_array_equal(seen[0], original_paths + delta)
    assert models.shell.rejoin is rejoin

    def fail(*args):
        raise RuntimeError("synthetic simulation failure")

    monkeypatch.setattr(models.parent.Candidate, "simulate_daily_log_returns", fail)
    with pytest.raises(RuntimeError, match="synthetic simulation failure"):
        models.CANDIDATES[0].simulate_daily_log_returns(training, context)
    assert models.shell.rejoin is rejoin


def test_actual_m261_multiasset_paths_keep_fits_uniforms_sd_and_centered_residuals():
    # Retained adapters mutate shared numerical kernels during initialization;
    # isolate the actual mechanism smoke from other adapter tests.
    code = r'''
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'tools/mcmc_runtime'))
spec = importlib.util.spec_from_file_location('actual_static_mean', root/'tools/expanded_static_mean/models.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m.initialize(None, 1, 1)
rng = np.random.default_rng(516)
assets = rng.normal(size=(504, 2)) * np.array([.009, .014]) + [.0006, -.0001]
dates = pd.bdate_range('2019-01-01', periods=len(assets) + 17).to_numpy()
training = TrainingData(assets @ [.6, .4], assets,
    PortfolioPolicy(('A', 'B'), (.6, .4), 'monthly'), dates[:len(assets)])
context = ForecastContext('mechanism', 'synthetic', 'synthetic', 17, 8, 927,
    dates[len(assets):], str(pd.Timestamp(dates[len(assets)-1]).date()))
captured = []
original_rejoin = m.shell.rejoin
def capture(paths, weights, mask):
    captured.append(paths.copy())
    return original_rejoin(paths, weights, mask)
m.shell.rejoin = capture
before_result = m.parent.CANDIDATES[0].simulate_daily_log_returns(training, context)
data = [np.ascontiguousarray(x).tobytes() for x in assets.T]
fit_function = m.parent.parent.body.learned.student.original_fit
fits = [fit_function(d) for d in data]
configuration = m.parent.parent.body.configuration
configurations = [configuration(d, context.horizon_days) for d in data]
old_means = np.column_stack([m.moment_return_curves(f, context.horizon_days)[0] for f in fits])
sd = [m.moment_return_curves(f, context.horizon_days)[1].copy() for f in fits]
seed = m.shell.bd.deterministic_seed('copula_alternatives', m.shell.bd.FRONTIER_DEPENDENCE_ID,
    str(context.origin_date), context.horizon_days, context.simulations)
uniforms = m.shell.controls.gaussian_uniforms(assets.shape, assets.tobytes(), 8, 17, seed)
uniform_bytes = uniforms.tobytes()
after_result = m.CANDIDATES[0].simulate_daily_log_returns(training, context)
assert m.shell.rejoin is capture
assert len(captured) == 2
static_means = np.array([history.mean() for history in assets.T])
expected = captured[0] + (static_means - old_means)
np.testing.assert_array_equal(captured[1], expected)
np.testing.assert_allclose(captured[1] - static_means, captured[0] - old_means,
    rtol=0, atol=3e-17)
np.testing.assert_allclose(captured[1].var(axis=0), captured[0].var(axis=0), rtol=0, atol=1e-18)
for a, d in enumerate(data):
    assert fit_function(d) is fits[a]
    assert configuration(d, 17) is configurations[a]
    np.testing.assert_array_equal(m.moment_return_curves(fits[a], 17)[1], sd[a])
    assert m.FIT_DIAGNOSTICS[m.MODEL_ID, m.hashlib.sha256(d).hexdigest()][
        'return_mean_curve'] == 'constant_raw_historical_asset_log_mean'
assert m.shell.controls.gaussian_uniforms(assets.shape, assets.tobytes(), 8, 17, seed) is uniforms
assert uniforms.tobytes() == uniform_bytes
np.testing.assert_array_equal(after_result, original_rejoin(expected, training.policy.weights,
    np.array([date in m.shell._historical_rebalance_dates(
        pd.DatetimeIndex(dates), training.policy.rebalance) for date in pd.DatetimeIndex(dates[-17:])])) )
assert np.isfinite(before_result).all() and np.isfinite(after_result).all()
print('actual M261 multiasset mean-only mechanism passed')
'''
    subprocess.run([sys.executable, "-c", code, str(ROOT)], check=True)
