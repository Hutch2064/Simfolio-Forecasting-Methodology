"""Source-bound product controls and a separately identified pure EWMA control.

French series are already portfolio returns. Each is supplied to the native
product kernels as one asset with unit weight and no turnover costs. This is
an exact one-column specialization, retaining native means, row draws, block
selection, clipping and wealth accounting. No historical OOS scores are used.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

from ..runner import ForecastContext, TrainingData

WEBSITE_DEFAULT_ID = "asset_level_naive_iid_mean_shrunk_rebalanced"
TESTING_NAIVE_ID = "model_testing_joint_stationary_bootstrap_v1"
EWMA_QMLE_ID = "FRENCH1926-EWMA-QMLE"
CONTROL_IDS = (WEBSITE_DEFAULT_ID, TESTING_NAIVE_ID, EWMA_QMLE_ID)
_SOURCE_PACKAGE = "simfolio_forecasting_methodology.resources.french1926.control_sources"


@lru_cache(maxsize=1)
def frozen_control_manifest() -> dict:
    """Verify the shipped executable closure before admitting native controls."""
    root = files(_SOURCE_PACKAGE)
    manifest = json.loads(root.joinpath("manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen control source checksum mismatch: {name}")
    return manifest


@lru_cache(maxsize=1)
def _native_kernel():
    frozen_control_manifest()
    module = importlib.import_module(f"{_SOURCE_PACKAGE}.smart_iid_forecast")
    helpers = importlib.import_module(f"{_SOURCE_PACKAGE}.engine_helpers")
    if (module._fused_no_rebalance_terminal_wealth is None
            or helpers._stationary_bootstrap_indices_from_draws_numba is None):
        raise ImportError("Exact native product controls require the all-models numba dependency")
    return module


def _validate(training: TrainingData, context: ForecastContext) -> np.ndarray:
    training.validate()
    if context.horizon_days < 1 or context.simulations < 1:
        raise ValueError("control forecasts require positive horizon and path counts")
    if context.future_dates is not None:
        future = np.asarray(context.future_dates, dtype="datetime64[ns]")
        if future.shape != (context.horizon_days,) or np.any(np.isnat(future)):
            raise ValueError("future dates must match the forecast horizon")
        if np.any(future[1:] <= future[:-1]):
            raise ValueError("future dates must increase")
        if training.training_dates is not None and future[0] <= training.training_dates[-1]:
            raise ValueError("forecast dates must follow the training boundary")
    return np.asarray(training.portfolio_log_returns, dtype=np.float64)


def _no_rebalance_dates(dates, frequency, offset):
    if frequency != "none" or offset != 0:
        raise ValueError("French unit-weight control only supports no rebalancing")
    return set()


def native_control_arguments(training: TrainingData, context: ForecastContext) -> dict:
    """Construct only the prepared one-column training panel; never read targets."""
    history = _validate(training, context)
    if training.asset_log_returns is None or training.policy is None:
        raise ValueError("native French controls require a one-column asset panel")
    if (training.asset_log_returns.shape != (history.size, 1)
            or tuple(training.policy.weights) != (1.0,)
            or training.policy.rebalance != "none"
            or not np.array_equal(training.asset_log_returns[:, 0], history)):
        raise ValueError("native French controls require the unit-weight portfolio series")
    if training.training_dates is None or context.future_dates is None:
        raise ValueError("native French controls require training and forecast dates")
    dates = pd.DatetimeIndex(training.training_dates)
    column = training.policy.tickers[0]
    # Both production kernels consume simple asset returns and recover their
    # historical log rows internally. Preserve that native round trip exactly.
    simple = np.expm1(history)
    return {
        "strategy_returns": pd.Series(simple, index=dates),
        "asset_returns": pd.DataFrame({column: simple}, index=dates),
        "weights": {column: 1.0},
        "historical_dates": dates,
        "future_dates": pd.DatetimeIndex(context.future_dates),
        "n_paths": int(context.simulations),
        "rng": np.random.default_rng(context.seed),
        "rebalance_frequency": "none",
        "rebalance_offset_days": 0,
        "cost_per_turnover_bps": 0.0,
        "date_selector": _no_rebalance_dates,
    }


class _NativeControl:
    model_id: str

    @property
    def dependency_identity(self):
        return {"frozen_product_control": frozen_control_manifest(),
                "adaptation": "one_French_series_as_unit_weight_asset_no_costs"}

    def simulate_with_metadata(self, training: TrainingData, context: ForecastContext):
        arguments = native_control_arguments(training, context)
        arguments["_naive_stationary_bootstrap"] = self.model_id == TESTING_NAIVE_ID
        return _native_kernel().simulate_smart_iid_log_returns(**arguments)

    def simulate_daily_log_returns(self, training: TrainingData,
                                   context: ForecastContext) -> np.ndarray:
        return self.simulate_with_metadata(training, context)[0]


@dataclass(frozen=True)
class WebsiteDefaultControl(_NativeControl):
    """Exact current website Default joint-IID/mean-shrinkage kernel."""
    model_id: ClassVar[str] = WEBSITE_DEFAULT_ID


@dataclass(frozen=True)
class TestingNaiveControl(_NativeControl):
    """Exact current Testing/whitepaper unshifted joint stationary bootstrap."""
    model_id: ClassVar[str] = TESTING_NAIVE_ID


@dataclass(frozen=True)
class EwmaQmleFit:
    mean: float
    decay: float | None
    initial_variance: float
    forecast_variance: float
    objective: float | None
    fit_status: str


def _ewma_variances(residuals: np.ndarray, initial_variance: float,
                    decay: float) -> tuple[np.ndarray, float]:
    """Predict before observing a residual; then update for the following day."""
    updated, _ = lfilter([1.0 - decay], [1.0, -decay], residuals * residuals,
                         zi=[decay * initial_variance])
    predictive = np.concatenate(([initial_variance], updated[:-1]))
    return predictive, float(updated[-1])


def _ewma_objective(decay: float, residuals: np.ndarray, initial_variance: float) -> float:
    variances, _ = _ewma_variances(residuals, initial_variance, decay)
    variances = np.maximum(variances, np.finfo(np.float64).tiny)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(np.sum(np.log(variances) + residuals * residuals / variances))


def fit_ewma_qmle(history: np.ndarray) -> EwmaQmleFit:
    """Fit decay by Gaussian predictive likelihood using this training set only.

    The deterministic constant case has no identifiable decay and is reported
    explicitly. Nonconstant fits compare the bounded optimum with lambda=0 and
    lambda=1. Failed optimization raises; there is no fixed-decay fallback.
    """
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("EWMA requires a finite nonempty training vector")
    if np.all(values == values[0]):
        return EwmaQmleFit(float(values[0]), None, 0.0, 0.0, None,
                           "deterministic_constant_history")
    mean = float(np.mean(values))
    residuals = values - mean
    initial = float(np.var(values, ddof=1))
    opt = minimize_scalar(_ewma_objective, args=(residuals, initial), bounds=(0.0, 1.0),
                          method="bounded", options={"xatol": 1e-8})
    if not opt.success or not np.isfinite(opt.fun):
        raise ValueError("EWMA Gaussian QMLE decay fit failed")
    decay = min((0.0, float(opt.x), 1.0),
                key=lambda value: _ewma_objective(value, residuals, initial))
    _, forecast_variance = _ewma_variances(residuals, initial, decay)
    return EwmaQmleFit(mean, decay, initial, forecast_variance,
                       _ewma_objective(decay, residuals, initial), "complete")


@dataclass(frozen=True)
class EwmaQmleControl:
    """Standalone Gaussian EWMA, with a fitted mean and decay, and recursive paths.

    This is a new method. It does not inherit scores or identity from the historic
    Bayesian SBB + deterministic EWMA volatility-overlay candidate.
    """
    model_id: ClassVar[str] = EWMA_QMLE_ID
    dependency_identity: ClassVar[dict] = {
        "method": EWMA_QMLE_ID,
        "mean": "training_sample_log_return_mean",
        "decay_fit": "predictive_Gaussian_QMLE_lambda_0_to_1_xatol_1e-8_endpoints",
        "variance_initialization": "training_sample_variance_ddof_1",
        "variance_evolution": "h_next=lambda*h+(1-lambda)*simulated_residual_squared",
        "innovations": "independent_standard_Gaussian",
        "constant_history": "deterministic_training_return_no_identifiable_decay",
        "historical_scores_inherited": False,
    }

    def simulate_daily_log_returns(self, training: TrainingData,
                                   context: ForecastContext) -> np.ndarray:
        history = _validate(training, context)
        fit = fit_ewma_qmle(history)
        shape = (int(context.simulations), int(context.horizon_days))
        if fit.decay is None:
            return np.full(shape, fit.mean, dtype=np.float64)
        rng = np.random.default_rng(context.seed)
        paths = np.empty(shape, dtype=np.float64)
        variance = np.full(shape[0], fit.forecast_variance, dtype=np.float64)
        for day in range(shape[1]):
            residual = np.sqrt(variance) * rng.standard_normal(shape[0])
            paths[:, day] = fit.mean + residual
            variance = fit.decay * variance + (1.0 - fit.decay) * residual * residual
        return paths


def build_control_model(model_id: str):
    factories = {WEBSITE_DEFAULT_ID: WebsiteDefaultControl,
                 TESTING_NAIVE_ID: TestingNaiveControl,
                 EWMA_QMLE_ID: EwmaQmleControl}
    try:
        factory = factories[model_id]
    except KeyError:
        raise ValueError(f"unknown French control: {model_id}") from None
    return factory()
