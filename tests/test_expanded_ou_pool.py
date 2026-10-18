import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology.runner import ForecastContext

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "tools/mcmc_runtime"), str(ROOT / "tools/expanded_panel")]
spec = importlib.util.spec_from_file_location(
    "ou_pool_test", ROOT / "tools/expanded_ou_pool/models.py"
)
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


def test_original_components_and_fit_budget_are_selected():
    assert models.FIRST.model_id == "asset_map_stationary_ar1_ou_residual_volatility"
    assert models.FIRST.covariance_family == "ou"
    assert models.FIRST.fit_seconds == 10.0
    assert models.SECOND.model_id == "bayesian_sbb_overlay_ewma_absolute"


def test_equal_complete_paths_preserve_component_context_and_training(monkeypatch):
    seen = []

    class Component:
        def __init__(self, model_id, offset):
            self.model_id = model_id
            self.offset = offset

        def simulate_daily_log_returns(self, training, context):
            seen.append((training, context))
            return (np.arange(context.simulations * context.horizon_days).reshape(
                context.simulations, context.horizon_days
            ) + self.offset)

    first = Component("original_ou_id", 10.0)
    second = Component("original_ewma_id", -1000.0)
    monkeypatch.setattr(models, "FIRST", first)
    monkeypatch.setattr(models, "SECOND", second)
    dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[D]")
    context = ForecastContext(models.MODEL_ID, "portfolio", "origin", 3, 240, 123,
                              dates, "2020-01-01")
    training = object()
    paths = models.Candidate().simulate_daily_log_returns(training, context)
    expected = np.arange(120 * 3).reshape(120, 3)
    np.testing.assert_array_equal(paths[:120], expected + 10.0)
    np.testing.assert_array_equal(paths[120:], expected - 1000.0)
    assert paths.shape == (240, 3)
    for (observed_training, observed_context), component in zip(seen, (first, second)):
        assert observed_training is training
        assert observed_context == replace(context, model_id=component.model_id,
                                           simulations=120)
        assert observed_context.future_dates is dates


@pytest.mark.parametrize("simulations", [0, 1, 3, 239])
def test_rejects_counts_that_cannot_preserve_equal_weights(simulations):
    context = ForecastContext(models.MODEL_ID, "p", "o", 3, simulations, 123)
    with pytest.raises(ValueError, match="even simulation count"):
        models.Candidate().simulate_daily_log_returns(None, context)
