"""Date-aware Frontier adapter for the standalone OOS harness.

The numerical Frontier module intentionally does not know about pandas or the
experiment calendar.  This adapter supplies the actual future business dates so
monthly/quarterly/annual portfolio rebalancing follows calendar boundaries
rather than fixed 21/63/252-day approximations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..runner import ForecastContext, TrainingData
from ..seeds import deterministic_seed
from .frontier import (
    FRONTIER_MODEL_ID,
    _rank_rejoin,
    _simulate_dependent_uniforms,
    _simulate_fastmap_marginal,
    fit_dynamic_gaussian_factor_model,
)


def _calendar_rebalance_mask(dates: np.ndarray | None, horizon: int, frequency: str) -> np.ndarray:
    mask = np.zeros(int(horizon), dtype=bool)
    if dates is None:
        return mask
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    if len(index) != int(horizon):
        raise ValueError("future date calendar does not match forecast horizon")
    normalized = str(frequency or "none").lower()
    if normalized == "monthly":
        labels = index.to_period("M")
    elif normalized == "quarterly":
        labels = index.to_period("Q")
    elif normalized in {"annually", "annual", "yearly"}:
        labels = index.to_period("Y")
    else:
        return mask
    # Rebalance after the last observed trading day of each completed period.
    if len(index) > 1:
        mask[:-1] = np.asarray(labels[:-1] != labels[1:], dtype=bool)
    return mask


def _portfolio_rejoin_calendar(
    asset_paths: np.ndarray,
    weights: np.ndarray,
    frequency: str,
    future_dates: np.ndarray | None,
) -> np.ndarray:
    simulations, horizon, assets = asset_paths.shape
    target = np.asarray(weights, dtype=np.float64)
    holdings = np.broadcast_to(target, (simulations, assets)).copy()
    output = np.empty((simulations, horizon), dtype=np.float64)
    mask = _calendar_rebalance_mask(future_dates, horizon, frequency)
    for day in range(horizon):
        previous = np.sum(holdings, axis=1)
        holdings *= np.exp(np.clip(asset_paths[:, day, :], -745.0, 50.0))
        ending = np.sum(holdings, axis=1)
        output[:, day] = np.log(np.maximum(ending, 1e-300) / np.maximum(previous, 1e-300))
        if mask[day]:
            holdings = ending[:, None] * target[None, :]
    return output


@dataclass(frozen=True)
class CalendarFrontierModel:
    """Standalone Frontier with exact calendar-aware portfolio rebalancing."""

    model_id: str = FRONTIER_MODEL_ID

    def simulate_daily_log_returns(
        self, training: TrainingData, context: ForecastContext
    ) -> np.ndarray:
        training.validate()
        if training.asset_log_returns is None or training.policy is None:
            raise ValueError("Frontier requires asset-level training data and a portfolio policy")
        assets = np.asarray(training.asset_log_returns, dtype=np.float64)
        simulations = int(context.simulations)
        horizon = int(context.horizon_days)
        marginal = np.empty((simulations, horizon, assets.shape[1]), dtype=np.float64)
        for asset, ticker in enumerate(training.policy.tickers):
            asset_seed = deterministic_seed(
                "frontier_fastmap_asset", ticker, horizon, simulations, int(context.seed)
            )
            marginal[:, :, asset] = _simulate_fastmap_marginal(
                assets[:, asset], simulations, horizon, asset_seed
            )
        dependence = fit_dynamic_gaussian_factor_model(assets)
        dependence_seed = deterministic_seed(
            "frontier_dynamic_gaussian_factor",
            context.portfolio_id,
            horizon,
            simulations,
            context.seed,
        )
        uniforms = _simulate_dependent_uniforms(
            dependence, simulations, horizon, dependence_seed
        )
        dependent_assets = _rank_rejoin(marginal, uniforms)
        return _portfolio_rejoin_calendar(
            dependent_assets,
            np.asarray(training.policy.weights, dtype=np.float64),
            training.policy.rebalance,
            context.future_dates,
        )
