"""Historical Frontier asset-level forecast model.

This adapter composes the extracted source-research FastMAP marginal fitter
with the original temporary-panel Gaussian factor/Kalman dependence layer. It
requires the explicit historical and future calendars so seed identity and
rebalancing dates cannot silently change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...runner import ForecastContext, TrainingData
from ..numerical.bdes_fastmap import (
    FRONTIER_CANDIDATE,
    FRONTIER_DEPENDENCE_ID,
    FRONTIER_MARGINAL_ID,
    deterministic_seed,
    fit_bdes_fastmap,
    simulate_fastmap_marginal,
)
from ..numerical.dynamic_gaussian import (
    fit_dynamic_gaussian_factor_model,
    map_uniforms_to_marginal_paths,
    rebalanced_portfolio_log_paths,
    simulate_future_gaussian_uniforms,
)

FRONTIER_MODEL_ID = "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced"
TURNOVER_COST_BPS = 15.0


def _historical_rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> set[pd.Timestamp]:
    normalized = str(frequency or "none").lower()
    if normalized in {"none", ""}:
        return set()
    if normalized == "daily":
        return set(dates)
    freq_map = {
        "weekly": "W",
        "biweekly": "2W",
        "monthly": "ME",
        "quarterly": "QE",
        "annually": "YE",
        "annual": "YE",
        "yearly": "YE",
    }
    if normalized not in freq_map:
        raise ValueError(f"frontier_unknown_rebalance_frequency:{frequency}")
    target_dates = pd.date_range(start=dates.min(), end=dates.max(), freq=freq_map[normalized])
    if target_dates.empty:
        return set()
    positions = np.unique(dates.searchsorted(target_dates, side="right") - 1)
    positions = positions[positions >= 0]
    return {pd.Timestamp(value) for value in dates.take(positions)}


def _validate_calendar(training: TrainingData, context: ForecastContext) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if training.training_dates is None:
        raise ValueError("frontier_requires_explicit_training_dates")
    if context.origin_date is None or not str(context.origin_date).strip():
        raise ValueError("frontier_requires_explicit_origin_date")
    if context.future_dates is None:
        raise ValueError("frontier_requires_explicit_future_dates")
    train_dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(training.training_dates), errors="raise"))
    future_dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(context.future_dates), errors="raise"))
    if len(train_dates) != len(training.portfolio_log_returns):
        raise ValueError("frontier_training_dates_do_not_align")
    if len(future_dates) != int(context.horizon_days):
        raise ValueError("frontier_future_dates_do_not_align")
    if len(train_dates) == 0 or len(future_dates) == 0 or train_dates.hasnans or future_dates.hasnans:
        raise ValueError("frontier_calendar_contains_missing_dates")
    if not train_dates.is_monotonic_increasing or not train_dates.is_unique:
        raise ValueError("frontier_training_dates_must_be_strictly_increasing")
    if not future_dates.is_monotonic_increasing or not future_dates.is_unique:
        raise ValueError("frontier_future_dates_must_be_strictly_increasing")
    if future_dates[0] <= train_dates[-1]:
        raise ValueError("frontier_future_dates_must_follow_training_dates")
    expected_future = pd.bdate_range(train_dates[-1] + pd.offsets.BDay(1), periods=int(context.horizon_days))
    if not future_dates.equals(expected_future):
        raise ValueError("frontier_future_dates_must_follow_source_business_day_policy")
    try:
        origin = pd.Timestamp(str(context.origin_date)).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("frontier_invalid_origin_date") from exc
    if origin != train_dates[-1].normalize():
        raise ValueError("frontier_origin_date_must_equal_training_end")
    return train_dates, future_dates


@dataclass(frozen=True)
class HistoricalFrontierModel:
    """The source-derived FastMAP + exact Gaussian/Kalman Frontier model."""

    model_id: str = FRONTIER_MODEL_ID

    def simulate_daily_log_returns(self, training: TrainingData, context: ForecastContext) -> np.ndarray:
        training.validate()
        if self.model_id != FRONTIER_MODEL_ID:
            raise ValueError(f"frontier_unknown_model_id:{self.model_id}")
        if training.asset_log_returns is None or training.policy is None:
            raise ValueError("frontier_requires_asset_level_training_data")
        if int(context.simulations) < 2:
            raise ValueError("frontier_requires_at_least_two_simulations")
        if int(context.horizon_days) < 1:
            raise ValueError("frontier_requires_positive_horizon")
        train_dates, future_dates = _validate_calendar(training, context)
        assets = np.asarray(training.asset_log_returns, dtype=np.float64)
        tickers = tuple(str(value) for value in training.policy.tickers)
        if len(tickers) != assets.shape[1]:
            raise ValueError("frontier_asset_order_does_not_match_policy")
        if len(train_dates) < 80:
            raise ValueError("frontier_requires_80_historical_observations")
        horizon, simulations = int(context.horizon_days), int(context.simulations)

        marginal = np.empty((simulations, horizon, assets.shape[1]), dtype=np.float64)
        fits = []
        for asset_index, ticker in enumerate(tickers):
            fit = fit_bdes_fastmap(assets[:, asset_index], FRONTIER_CANDIDATE)
            fits.append(fit)
            asset_seed = deterministic_seed(
                "asset_level_current_engine",
                ticker,
                str(context.origin_date),
                FRONTIER_MARGINAL_ID,
                horizon,
                simulations,
            )
            marginal[:, :, asset_index] = simulate_fastmap_marginal(fit, simulations, horizon, asset_seed)
        if assets.shape[1] == 1:
            dependent_assets = marginal
        else:
            dependence = fit_dynamic_gaussian_factor_model(assets)
            dependence_seed = deterministic_seed(
                "copula_alternatives",
                FRONTIER_DEPENDENCE_ID,
                str(context.origin_date),
                horizon,
                simulations,
            )
            uniforms = simulate_future_gaussian_uniforms(dependence, simulations, horizon, np.random.default_rng(dependence_seed))
            dependent_assets = map_uniforms_to_marginal_paths(marginal, uniforms)
        full_dates = train_dates.append(future_dates)
        rebalance_dates = _historical_rebalance_dates(full_dates, training.policy.rebalance)
        rebalance_mask = np.asarray([date in rebalance_dates for date in future_dates], dtype=bool)
        output = rebalanced_portfolio_log_paths(
            dependent_assets,
            training.policy.weights,
            rebalance_mask,
            cost_per_turnover_bps=TURNOVER_COST_BPS,
        )
        if output.shape != (simulations, horizon) or not np.all(np.isfinite(output)):
            raise ValueError("frontier_nonfinite_portfolio_paths")
        return np.asarray(output, dtype=np.float64)


# Short alias for callers that use the model name from the public registry.
FrontierAssetLevelModel = HistoricalFrontierModel

__all__ = ["FRONTIER_MODEL_ID", "FrontierAssetLevelModel", "HistoricalFrontierModel"]
