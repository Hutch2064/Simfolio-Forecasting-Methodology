import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from simfolio_forecasting_methodology.runner import ForecastContext

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "tools/mcmc_runtime"), str(ROOT / "tools/expanded_panel")]
spec = importlib.util.spec_from_file_location("linear_pool_test", ROOT / "tools/expanded_linear_pool/models.py")
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


def test_entire_paths_are_combined_without_averaging_or_changing_context(monkeypatch):
    seen = []
    class Component:
        def __init__(self, value):
            self.value = value
            self.model_id = str(value)
        def simulate_daily_log_returns(self, training, context):
            seen.append((training, context))
            return np.full((context.simulations, context.horizon_days), self.value)
    monkeypatch.setattr(models, "FIRST", Component(1.0))
    monkeypatch.setattr(models, "SECOND", Component(-2.0))
    context = ForecastContext("pool", "portfolio", "origin", 3, 8, 123)
    training = object()
    paths = models.Candidate().simulate_daily_log_returns(training, context)
    np.testing.assert_array_equal(paths[:4], np.ones((4, 3)))
    np.testing.assert_array_equal(paths[4:], np.full((4, 3), -2.0))
    assert all(t is training and c.seed == context.seed and c.origin_label == context.origin_label
               and c.simulations == 4 for t, c in seen)


def test_does_not_silently_approximate_unequal_counts():
    with pytest.raises(ValueError):
        models.Candidate().simulate_daily_log_returns(None, ForecastContext("p", "p", "o", 3, 7, 123))
