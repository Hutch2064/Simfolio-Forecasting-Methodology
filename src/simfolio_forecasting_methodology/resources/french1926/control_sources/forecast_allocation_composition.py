from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range


MAX_COMPOSITION_POINTS = 480
MAX_COMPOSITION_CELLS = 20_000


def _sample_positions(horizon: int, asset_count: int) -> np.ndarray:
    budget = min(
        MAX_COMPOSITION_POINTS,
        max(2, MAX_COMPOSITION_CELLS // max(int(asset_count), 1)),
    )
    if horizon <= budget:
        return np.arange(horizon, dtype=np.int64)
    return np.unique(np.linspace(0, horizon - 1, budget, dtype=np.int64))


if njit is not None:

    @njit(parallel=True, cache=True, fastmath=False)
    def _smart_weight_samples(
        source_simple: np.ndarray,
        sampled_indices: np.ndarray,
        target: np.ndarray,
        rebalance_mask: np.ndarray,
        sample_positions: np.ndarray,
        band_enabled: bool,
        absolute_lower: float,
        absolute_upper: float,
        relative_lower: float,
        relative_upper: float,
    ) -> np.ndarray:
        path_count, horizon = sampled_indices.shape
        asset_count = target.size
        output = np.empty(
            (path_count, sample_positions.size, asset_count), dtype=np.float32
        )
        for path in prange(path_count):
            holdings = target.copy()
            sample_index = 0
            for day in range(horizon):
                total = 0.0
                for asset in range(asset_count):
                    holdings[asset] *= 1.0 + source_simple[
                        sampled_indices[path, day], asset
                    ]
                    total += holdings[asset]
                denominator = max(total, 1e-300)
                should_rebalance = rebalance_mask[day]
                if band_enabled and day > 0:
                    for asset in range(asset_count):
                        drift = holdings[asset] / denominator - target[asset]
                        if (
                            (absolute_lower > 0.0 and drift < -absolute_lower)
                            or (absolute_upper > 0.0 and drift > absolute_upper)
                            or (
                                relative_lower > 0.0
                                and drift < -relative_lower * max(abs(target[asset]), 1e-300)
                            )
                            or (
                                relative_upper > 0.0
                                and drift > relative_upper * max(abs(target[asset]), 1e-300)
                            )
                        ):
                            should_rebalance = True
                            break
                if should_rebalance:
                    for asset in range(asset_count):
                        holdings[asset] = denominator * target[asset]
                if sample_index < sample_positions.size and day == sample_positions[sample_index]:
                    for asset in range(asset_count):
                        output[path, sample_index, asset] = holdings[asset] / denominator
                    sample_index += 1
        return output

    @njit(parallel=True, cache=True, fastmath=False)
    def _frontier_weight_samples(
        asset_log_paths: np.ndarray,
        target: np.ndarray,
        rebalance_mask: np.ndarray,
        sample_positions: np.ndarray,
        band_enabled: bool,
        absolute_lower: float,
        absolute_upper: float,
        relative_lower: float,
        relative_upper: float,
    ) -> np.ndarray:
        asset_count, path_count, horizon = asset_log_paths.shape
        output = np.empty(
            (path_count, sample_positions.size, asset_count), dtype=np.float32
        )
        for path in prange(path_count):
            holdings = target.copy()
            sample_index = 0
            for day in range(horizon):
                total = 0.0
                for asset in range(asset_count):
                    log_return = asset_log_paths[asset, path, day]
                    log_return = min(max(log_return, -745.0), 50.0)
                    holdings[asset] *= np.exp(log_return)
                    total += holdings[asset]
                denominator = max(total, 1e-300)
                should_rebalance = rebalance_mask[day]
                if band_enabled and day > 0:
                    for asset in range(asset_count):
                        drift = holdings[asset] / denominator - target[asset]
                        if (
                            (absolute_lower > 0.0 and drift < -absolute_lower)
                            or (absolute_upper > 0.0 and drift > absolute_upper)
                            or (
                                relative_lower > 0.0
                                and drift < -relative_lower * max(abs(target[asset]), 1e-300)
                            )
                            or (
                                relative_upper > 0.0
                                and drift > relative_upper * max(abs(target[asset]), 1e-300)
                            )
                        ):
                            should_rebalance = True
                            break
                if should_rebalance:
                    for asset in range(asset_count):
                        holdings[asset] = denominator * target[asset]
                if sample_index < sample_positions.size and day == sample_positions[sample_index]:
                    for asset in range(asset_count):
                        output[path, sample_index, asset] = holdings[asset] / denominator
                    sample_index += 1
        return output

else:  # pragma: no cover
    _smart_weight_samples = None
    _frontier_weight_samples = None


def _band_parameters(settings: Mapping[str, Any] | None) -> tuple[bool, float, float, float, float]:
    if not isinstance(settings, Mapping):
        return False, 0.0, 0.0, 0.0, 0.0
    nested = settings.get("rebalance_band_settings") or settings.get(
        "forecast_rebalance_band_settings"
    )
    if isinstance(nested, Mapping):
        settings = nested
    enabled = settings.get("enabled") or settings.get("rebalance_bands_enabled")
    enabled = enabled is True or str(enabled or "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return False, 0.0, 0.0, 0.0, 0.0

    def pct(key: str) -> float:
        try:
            return max(float(settings.get(key, 0.0) or 0.0), 0.0) / 100.0
        except (TypeError, ValueError):
            return 0.0

    mode = str(settings.get("mode") or "symmetric").lower()
    basis = str(settings.get("basis") or "absolute").lower()
    symmetric = pct("value")
    lower = pct("lower") if mode == "asymmetric" else symmetric
    upper = pct("upper") if mode == "asymmetric" else symmetric
    absolute_value = pct("absolute_value") or (symmetric if basis == "absolute" else 0.0)
    relative_value = pct("relative_value") or (symmetric if basis == "relative" else 0.0)
    values = (
        pct("absolute_lower") or (lower if basis == "absolute" else 0.0) if mode == "asymmetric" else absolute_value,
        pct("absolute_upper") or (upper if basis == "absolute" else 0.0) if mode == "asymmetric" else absolute_value,
        pct("relative_lower") or (lower if basis == "relative" else 0.0) if mode == "asymmetric" else relative_value,
        pct("relative_upper") or (upper if basis == "relative" else 0.0) if mode == "asymmetric" else relative_value,
    )
    return any(value > 0.0 for value in values), *values


def build_forecast_allocation_composition(
    *,
    smart_context: Mapping[str, Any] | None,
    frontier_context: Mapping[str, Any] | None,
    forecast_weights: Mapping[str, float] | None,
    future_dates: Sequence[Any],
    rebalance_mask: np.ndarray | None,
    band_settings: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Reduce exact asset scenarios to bounded cross-path mean realized weights."""

    context = smart_context if isinstance(smart_context, Mapping) else frontier_context
    if not isinstance(context, Mapping) or not isinstance(forecast_weights, Mapping):
        return None
    tickers = [str(value) for value in (context.get("tickers") or [])]
    horizon = len(future_dates)
    if not tickers or horizon < 1 or rebalance_mask is None:
        return None
    target = np.asarray([float(forecast_weights.get(ticker, 0.0)) for ticker in tickers], dtype=np.float64)
    total = float(np.sum(target, dtype=np.float64))
    mask = np.asarray(rebalance_mask, dtype=np.bool_).reshape(-1)
    if not np.all(np.isfinite(target)) or abs(total) <= 1e-12 or mask.shape != (horizon,):
        return None
    target /= total
    positions = _sample_positions(horizon, len(tickers))
    bands = _band_parameters(band_settings)
    if smart_context is not None and _smart_weight_samples is not None:
        samples = _smart_weight_samples(
            np.asarray(smart_context.get("source_simple"), dtype=np.float64),
            np.asarray(smart_context.get("sampled_indices"), dtype=np.int32),
            target,
            mask,
            positions,
            *bands,
        )
        mean_weights = np.mean(samples, axis=0, dtype=np.float64)
    elif frontier_context is not None and _frontier_weight_samples is not None:
        weighted_sum = np.zeros((positions.size, len(tickers)), dtype=np.float64)
        path_total = 0
        for chunk in frontier_context.get("chunks") or []:
            paths = np.asarray(chunk.get("asset_log_paths"), dtype=np.float32)
            if paths.ndim != 3 or paths.shape[0] != len(tickers) or paths.shape[2] != horizon:
                return None
            samples = _frontier_weight_samples(paths, target, mask, positions, *bands)
            weighted_sum += np.sum(samples, axis=0, dtype=np.float64)
            path_total += int(paths.shape[1])
        if path_total < 1:
            return None
        mean_weights = weighted_sum / float(path_total)
    else:
        return None
    return {
        "schema_version": "forecast-allocation-composition-v1",
        "dates": [str(future_dates[int(position)])[:10] for position in positions],
        "assets": tickers,
        "weights": mean_weights.tolist(),
        "path_count": int(
            np.asarray(smart_context.get("sampled_indices")).shape[0]
            if smart_context is not None
            else sum(int(chunk.get("path_count", 0)) for chunk in frontier_context.get("chunks") or [])
        ),
    }

def smart_composition_from_samples(
    *, source_simple: np.ndarray, sampled_indices: np.ndarray, target: np.ndarray,
    rebalance_mask: np.ndarray, future_dates: Sequence[Any], tickers: Sequence[str],
    band_settings: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if _smart_weight_samples is None:
        return None
    positions = _sample_positions(len(future_dates), len(tickers))
    samples = _smart_weight_samples(
        source_simple, sampled_indices, target, rebalance_mask, positions,
        *_band_parameters(band_settings),
    )
    return {
        "schema_version": "forecast-allocation-composition-v1",
        "dates": [str(future_dates[int(position)])[:10] for position in positions],
        "assets": [str(value) for value in tickers],
        "weights": np.mean(samples, axis=0, dtype=np.float64).tolist(),
        "path_count": int(sampled_indices.shape[0]),
    }


def frontier_composition_from_paths(
    *, asset_log_paths: np.ndarray, target: np.ndarray, rebalance_mask: np.ndarray,
    future_dates: Sequence[Any], tickers: Sequence[str],
    band_settings: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if _frontier_weight_samples is None:
        return None
    positions = _sample_positions(len(future_dates), len(tickers))
    samples = _frontier_weight_samples(
        asset_log_paths, target, rebalance_mask, positions,
        *_band_parameters(band_settings),
    )
    return {
        "schema_version": "forecast-allocation-composition-v1",
        "dates": [str(future_dates[int(position)])[:10] for position in positions],
        "assets": [str(value) for value in tickers],
        "weights": np.mean(samples, axis=0, dtype=np.float64).tolist(),
        "path_count": int(asset_log_paths.shape[1]),
    }
