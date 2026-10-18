import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools/mcmc_runtime"))
spec = importlib.util.spec_from_file_location("raw_sbb_test", ROOT / "tools/raw_stationary_bootstrap/models.py")
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


def test_one_asset_and_portfolio_versions_have_identical_paths():
    dates = pd.bdate_range("1979-12-31", periods=600).to_numpy()
    x = np.random.default_rng(12).normal(0.0003, 0.01, 600)
    training = TrainingData(x, x[:, None], PortfolioPolicy(("A",), (1.0,), "none"), dates)
    context = ForecastContext("test", "single_A", "origin", 63, 32, 123,
        pd.bdate_range(pd.Timestamp(dates[-1]) + pd.offsets.BDay(1), periods=63).to_numpy(),
        str(pd.Timestamp(dates[-1]).date()))
    a = models.Candidate(models.MODEL_IDS[0]).simulate_daily_log_returns(training, context)
    b = models.Candidate(models.MODEL_IDS[1]).simulate_daily_log_returns(training, context)
    np.testing.assert_array_equal(a, b)


def test_index_assembly_matches_scalar_geometric_restart():
    seed, n, length, sims, horizon = 42, 19, 4.25, 8, 100
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, n, sims, dtype=np.int32)
    flags = rng.random((sims, horizon - 1)) < 1.0 / length
    points = rng.integers(0, n, flags.shape, dtype=np.int32)
    expected = np.empty((sims, horizon), dtype=np.int32)
    expected[:, 0] = initial
    for t in range(1, horizon):
        expected[:, t] = np.where(flags[:, t-1], points[:, t-1], (expected[:, t-1] + 1) % n)
    np.testing.assert_array_equal(models.indices(n, length, sims, horizon, seed), expected)


def test_block_choice_uses_only_supplied_history_and_is_finite():
    history = np.random.default_rng(17).normal(size=(600, 3))
    length = models.block_length(history)
    assert 1 <= length < len(history)
    assert models.block_length(history[:, :1]) == models.block_length(history[:, 0:1].copy())
