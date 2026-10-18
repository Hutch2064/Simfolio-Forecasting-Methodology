"""Whitepaper stationary-bootstrap benchmark, retaining its seed and block rule."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from simfolio_forecasting_methodology.models.asset_level.frontier import (
    _historical_rebalance_dates,
    _validate_calendar,
)
from simfolio_forecasting_methodology.models.numerical.base_models import (
    _politis_white_block_length,
    _stationary_bootstrap_indices,
)
from simfolio_forecasting_methodology.models.numerical.bdes_fastmap import deterministic_seed
from simfolio_forecasting_methodology.runner import TERMINAL_FORECAST_SEMANTICS

SimfolioEngine = Any
_REBALANCE_MASK_CACHE = {}
TIMINGS = {}
FIT_DIAGNOSTICS = {}
def _normalized_weights(weights: Sequence[float], count: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size != int(count):
        raise ValueError("portfolio weights do not match asset count")
    values = np.maximum(values, 0.0)
    total = float(np.sum(values, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(int(count), 1.0 / max(int(count), 1), dtype=np.float64)
    return values / total

def _no_rebalance_terminal_logs(
    asset_log_paths: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    """Compute no-rebalance portfolio terminal logs without a daily Python loop."""
    paths = np.asarray(asset_log_paths, dtype=np.float64)
    weights_array = _normalized_weights(weights, int(paths.shape[2]))
    # Match the existing rejoin helper's per-day exponentiation guard while
    # staying in log space so long horizons do not overflow.
    cumulative_asset_logs = np.cumsum(np.clip(paths, -745.0, 50.0), axis=1, dtype=np.float64)
    row_max = np.max(cumulative_asset_logs, axis=2, keepdims=True)
    weighted_sum = np.sum(
        np.exp(cumulative_asset_logs - row_max) * weights_array[None, None, :],
        axis=2,
    )
    return row_max[:, :, 0] + np.log(np.maximum(weighted_sum, 1e-300))

def _future_rebalance_mask(
    historical_dates: pd.DatetimeIndex,
    horizon: int,
    rebalance: str,
    engine: SimfolioEngine,
) -> np.ndarray:
    """Reuse the engine's date policy, caching only the small boolean mask."""
    last_date = pd.Timestamp(historical_dates[-1]).normalize()
    key = (last_date, int(horizon), str(rebalance))
    cached = _REBALANCE_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=int(horizon),
    )
    policy_dates = pd.DatetimeIndex(historical_dates).append(future_dates)
    rebalance_dates = set(pd.Timestamp(value) for value in engine._get_dates_for_freq(policy_dates, str(rebalance)))  # noqa: C401 - retain whitepaper helper verbatim
    mask = np.asarray([date in rebalance_dates for date in future_dates], dtype=bool)
    # The canonical panel has a bounded horizon/origin grid.  Keep the cache
    # bounded as a guard for interactive research runs with arbitrary dates.
    if len(_REBALANCE_MASK_CACHE) >= 512:
        _REBALANCE_MASK_CACHE.pop(next(iter(_REBALANCE_MASK_CACHE)))
    _REBALANCE_MASK_CACHE[key] = mask
    return mask

def _segmented_rebalanced_terminal_logs(
    asset_log_paths: np.ndarray,
    weights: Sequence[float],
    historical_dates: pd.DatetimeIndex,
    rebalance: str,
    engine: SimfolioEngine,
    cost_per_turnover_bps: float = 15.0,
) -> np.ndarray:
    """Rejoin paths by rebalance segment, preserving the existing semantics."""
    paths = np.asarray(asset_log_paths, dtype=np.float64)
    n_sims, horizon, n_assets = paths.shape
    weights_array = _normalized_weights(weights, int(n_assets))
    mask = _future_rebalance_mask(historical_dates, int(horizon), str(rebalance), engine)
    rebalance_indices = np.flatnonzero(mask)
    ends = [int(value) for value in rebalance_indices.tolist()]
    if not ends or ends[-1] != int(horizon) - 1:
        ends.append(int(horizon) - 1)

    clipped = np.clip(paths, -745.0, 50.0)
    holdings = np.broadcast_to(weights_array, (int(n_sims), int(n_assets))).copy()
    start_value = np.sum(holdings, axis=1)
    cumulative_log_value = np.zeros(int(n_sims), dtype=np.float64)
    terminal = np.empty((int(n_sims), int(horizon)), dtype=np.float64)
    cost_rate = max(float(cost_per_turnover_bps), 0.0) / 10000.0
    segment_start = 0

    for segment_end in ends:
        if segment_end < segment_start:
            continue
        segment_logs = clipped[:, segment_start : segment_end + 1, :]
        segment_cumulative = np.cumsum(segment_logs, axis=1, dtype=np.float64)
        segment_holdings = holdings[:, None, :] * np.exp(segment_cumulative)
        ending_values = np.sum(segment_holdings, axis=2)
        terminal[:, segment_start : segment_end + 1] = (
            cumulative_log_value[:, None]
            + np.log(
                np.maximum(ending_values, 1e-300)
                / np.maximum(start_value[:, None], 1e-300)
            )
        )

        if bool(mask[segment_end]):
            ending_value = ending_values[:, -1]
            drifted_weights = segment_holdings[:, -1, :] / np.maximum(ending_value[:, None], 1e-300)
            one_way_turnover = 0.5 * np.sum(
                np.abs(weights_array[None, :] - drifted_weights),
                axis=1,
            )
            after_cost = np.maximum(
                ending_value * (1.0 - one_way_turnover * cost_rate),
                0.0,
            )
            cumulative_log_value = cumulative_log_value + np.log(
                np.maximum(after_cost, 1e-300)
                / np.maximum(start_value, 1e-300)
            )
            # The existing helper records the post-cost value on the
            # rebalance day; replace the pre-cost segment endpoint.
            terminal[:, segment_end] = cumulative_log_value
            start_value = after_cost
            holdings = after_cost[:, None] * weights_array[None, :]
        segment_start = segment_end + 1

    return terminal

def _asset_level_politis_white_block_length(
    history: np.ndarray,
) -> tuple[int, Sequence[int], Sequence[int]]:
    """Select one shared block length and retain per-asset diagnostics.

    The current engine applies the Politis–White rule to both a centered
    return series and its squared centered return series.  Apply that exact
    rule independently to each asset, then use the maximum selected length so
    every asset can share the same sampled historical row sequence.  This
    keeps the contemporaneous cross-asset dependence from the IID candidate.
    """
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        asset_count = int(values.shape[1]) if values.ndim == 2 else 0
        return 1, [1] * asset_count, [1] * asset_count

    return_lengths = []
    squared_return_lengths = []
    for asset_index in range(int(values.shape[1])):
        centered = values[:, asset_index] - float(np.mean(values[:, asset_index]))
        return_lengths.append(int(max(1, _politis_white_block_length(centered))))
        squared_return_lengths.append(
            int(max(1, _politis_white_block_length(centered * centered)))
        )
    block_length = int(
        max(
            1,
            min(
                int(values.shape[0]),
                max(return_lengths + squared_return_lengths),
            ),
        )
    )
    return block_length, return_lengths, squared_return_lengths

class CalendarPolicy:
    @staticmethod
    def _get_dates_for_freq(dates, frequency):
        return _historical_rebalance_dates(pd.DatetimeIndex(dates), frequency)


@dataclass(frozen=True)
class Candidate:
    model_id: str = "asset_level_naive_politis_white_stationary_bootstrap_rebalanced"
    forecast_output_semantics: str = TERMINAL_FORECAST_SEMANTICS

    def simulate_terminal_log_returns(self, training, context):
        training.validate()
        past, _ = _validate_calendar(training, context)
        history = np.asarray(training.asset_log_returns, dtype=np.float64)
        length, return_lengths, squared_lengths = _asset_level_politis_white_block_length(history)
        seed = deterministic_seed(self.model_id, "forecast_oos_all_daily_shared_stationary_bootstrap",
                                  context.origin_date, int(context.horizon_days), int(context.simulations), int(length))
        sampled = history[_stationary_bootstrap_indices(n=len(history), block_length=length,
            total_days=context.horizon_days, n_paths=context.simulations, rng=np.random.default_rng(seed))]
        FIT_DIAGNOSTICS[self.model_id] = {"block_length":length, "return_block_lengths":return_lengths,
            "squared_return_block_lengths":squared_lengths, "seed":seed, "training_rows":len(history),
            "selector":"whitepaper engine Politis-White rule; not arch.bootstrap.optimal_block_length"}
        if training.policy.rebalance == "none":
            return _no_rebalance_terminal_logs(sampled, training.policy.weights)
        return _segmented_rebalanced_terminal_logs(sampled, training.policy.weights, past,
            training.policy.rebalance, CalendarPolicy(), cost_per_turnover_bps=15.0)


CANDIDATES = (Candidate(),)


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    pass


def clear_path_cache():
    pass
