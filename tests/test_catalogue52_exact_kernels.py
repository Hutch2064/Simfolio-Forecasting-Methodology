"""Complete fit/path parity and restoration for the scoped runtime adapter."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/catalogue52"))
import legacy_exact_kernels as kernels

from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg
from simfolio_forecasting_methodology.models.numerical import inla_bdes as ib
from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData


@pytest.mark.parametrize("n", [96, 1536])
@pytest.mark.parametrize("model_class,fit,normalize", [
    (kernels.frontier.HistoricalFrontierModel, bd.fit_bdes_fastmap, lambda x: x),
    (kernels.exact_kalman.ExactKalmanDynamicGaussianFactorModel, ib.fit_bdes_full_inla,
     kernels.exact_kalman._source_engine_log_returns),
])
def test_complete_fit_daily_and_terminal_bytes(n, model_class, fit, normalize):
    x = np.random.default_rng(83).normal(0.0003, 0.009, n)
    dates = pd.bdate_range("1990-01-01", periods=n)
    training = TrainingData(x, x[:, None], PortfolioPolicy(("TEST",), (1.0,), "monthly"),
                            dates.to_numpy())
    reference_fit = fit(normalize(x))
    with kernels.accelerated():
        actual_fit = fit(normalize(x))
    assert pickle.dumps(actual_fit, protocol=5) == pickle.dumps(reference_fit, protocol=5)
    model = model_class()
    for horizon in (126, 2922):
        future = pd.bdate_range(dates[-1] + pd.offsets.BDay(1), periods=horizon).to_numpy()
        context = ForecastContext(model.model_id, "test", "test", horizon, 240, 83,
                                  future, str(dates[-1].date()))
        reference = model.simulate_daily_log_returns(training, context)
        actual = kernels.ExactKernelModel(model).simulate_daily_log_returns(training, context)
        assert actual.tobytes() == reference.tobytes()
        assert np.cumsum(actual, axis=1).tobytes() == np.cumsum(reference, axis=1).tobytes()


@pytest.mark.parametrize("assets", [1, 8, 25, 129])
def test_full_rejoin_preserves_numpy_reduction_order(assets):
    paths = np.random.default_rng(12).normal(0, 0.03, (4, 32, assets))
    weights = np.random.default_rng(19).random(assets)
    mask = np.arange(32) % 5 == 0
    assert kernels._rejoin(paths, weights, mask).tobytes() == (
        dg.rebalanced_portfolio_log_paths(paths, weights, mask).tobytes())


def test_all_globals_restored_after_exception_and_nested_context():
    kernels.prepare()
    before = [(module, name, getattr(module, name))
              for module, name, _ in kernels._REPLACEMENTS]
    with (pytest.raises(RuntimeError, match="intentional"),
          kernels.accelerated(), kernels.accelerated()):
        raise RuntimeError("intentional")
    for module, name, original in before:
        assert getattr(module, name) is original
    assert all(len(value) == 64 for value in kernels.source_version().values())


@pytest.mark.parametrize("paths,weights,mask", [
    (np.zeros((2, 2)), (1,), np.zeros(2, bool)),
    (np.full((2, 2, 1), np.nan), (1,), np.zeros(2, bool)),
    (np.zeros((2, 2, 1)), (0,), np.zeros(2, bool)),
    (np.zeros((2, 2, 1)), (1, 1), np.zeros(2, bool)),
    (np.zeros((2, 2, 1)), (1,), np.zeros(3, bool)),
])
def test_rejoin_keeps_original_validation(paths, weights, mask):
    with pytest.raises(ValueError) as original:
        dg.rebalanced_portfolio_log_paths(paths, weights, mask)
    with pytest.raises(ValueError, match=str(original.value)):
        kernels._rejoin(paths, weights, mask)
