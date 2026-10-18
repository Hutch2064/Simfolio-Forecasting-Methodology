"""Bounded checks that expanded controls preserve exact catalogue execution."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.numerical.factors import load_packaged_factor_frame
from simfolio_forecasting_methodology.models.registry import build_model
from simfolio_forecasting_methodology.runner import (
    ForecastContext,
    OriginTask,
    PortfolioPolicy,
    TrainingData,
    evaluate_origin_task,
)

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tools/expanded_controls/models.py"
IDS = (
    "naive_iid_historical_portfolio_bootstrap",
    "constant_mean_gaussian",
    "bayesian_sbb_overlay_ewma_absolute",
    "factor_ff6_ridge_residual_sbb_none",
)


def load_adapter(monkeypatch, model_id):
    monkeypatch.setenv("SIMFOLIO_EXPANDED_CONTROL", model_id)
    spec = importlib.util.spec_from_file_location("expanded_control_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.initialize(None, 1, 1)
    return module


@pytest.fixture(scope="module")
def dated_fixture():
    # Use actual packaged market dates to exercise the unchanged factor join.
    dates = load_packaged_factor_frame("ff6").loc["2018-01-01":"2020-01-10"].index
    dates = dates.to_numpy(dtype="datetime64[ns]")
    train_dates, future_dates = dates[-228:-8], dates[-8:]
    rng = np.random.default_rng(42017)
    assets = rng.normal(0.0002, 0.009, (len(train_dates), 3))
    assets[:, 1:] += assets[:, :1] * 0.35
    return train_dates, future_dates, assets


def training_fixture(dated_fixture, asset_count):
    dates, future, assets = dated_fixture
    assets = assets[:, :asset_count].copy()
    weights = np.full(asset_count, 1.0 / asset_count)
    portfolio = np.log(np.exp(assets) @ weights)
    return TrainingData(
        portfolio_log_returns=portfolio,
        asset_log_returns=assets,
        policy=PortfolioPolicy(tuple(f"TOY{i}" for i in range(asset_count)), tuple(weights), "daily"),
        training_dates=dates,
    ), future


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("asset_count", [1, 3])
def test_exact_registered_forecast_bytes(monkeypatch, dated_fixture, model_id, asset_count):
    adapter = load_adapter(monkeypatch, model_id)
    selected, = adapter.CANDIDATES
    direct = build_model(model_id)
    assert type(selected) is type(direct)
    training, future = training_fixture(dated_fixture, asset_count)
    context = ForecastContext(model_id, f"toy{asset_count}", "calendar-origin", len(future),
                              7, 193746, future_dates=future,
                              origin_date=str(training.training_dates[-1])[:10])
    method = ("simulate_terminal_log_returns" if hasattr(direct, "simulate_terminal_log_returns")
              else "simulate_daily_log_returns")
    expected = getattr(direct, method)(training, context)
    actual = getattr(selected, method)(training, context)
    assert actual.shape == (7, len(future))
    assert actual.dtype == expected.dtype == np.float64
    assert actual.tobytes() == expected.tobytes()
    task = OriginTask(context.portfolio_id, context.origin_label, training,
                      np.zeros(len(future)), context.seed, future, context.origin_date)
    assert evaluate_origin_task(selected, task, simulations=7).tobytes() == \
        evaluate_origin_task(direct, task, simulations=7).tobytes()
    adapter.clear_path_cache()
    assert adapter.TIMINGS == adapter.FIT_DIAGNOSTICS == {}


@pytest.mark.parametrize("model_id", IDS)
def test_canonical_runner_forwards_exact_input_seed_and_calendar(
        monkeypatch, dated_fixture, model_id):
    selected, = load_adapter(monkeypatch, model_id).CANDIDATES
    training, future = training_fixture(dated_fixture, 3)
    seen = []

    def capture(self, received_training, context):
        seen.append((received_training, context))
        return np.zeros((context.simulations, context.horizon_days))

    method = ("simulate_terminal_log_returns" if hasattr(selected, "simulate_terminal_log_returns")
              else "simulate_daily_log_returns")
    monkeypatch.setattr(type(selected), method, capture)
    task = OriginTask("exact-portfolio", "exact-origin", training, np.zeros(len(future)),
                      384729, future, "2019-12-30")
    evaluate_origin_task(selected, task, simulations=7)
    received_training, context = seen[0]
    assert received_training is training
    assert context.seed == task.seed
    assert context.model_id == model_id
    assert context.origin_date == task.origin_date
    assert context.origin_label == task.origin_label
    assert context.portfolio_id == task.portfolio_id
    assert context.horizon_days == len(future)
    assert context.simulations == 7
    np.testing.assert_array_equal(context.future_dates, future)


@pytest.mark.parametrize("invalid", ["", "website_naive_stationary_bootstrap", "unknown"])
def test_unregistered_or_unselected_control_fails_closed(monkeypatch, invalid):
    with pytest.raises(ValueError, match="SIMFOLIO_EXPANDED_CONTROL"):
        load_adapter(monkeypatch, invalid)


def test_import_has_no_production_or_source_checkout_dependency():
    script = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('control_dependency_check', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forbidden = [str(getattr(value, '__file__', '') or '') for value in sys.modules.values()
             if '/simfolio-engine/' in str(getattr(value, '__file__', '') or '')
             or '/source-research/' in str(getattr(value, '__file__', '') or '')]
forbidden += [name for name in sys.modules if name == 'app' or name.startswith('app.')]
print(json.dumps(forbidden))
"""
    env = dict(os.environ, SIMFOLIO_EXPANDED_CONTROL=IDS[0], PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", script, str(SOURCE)], env=env,
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []
