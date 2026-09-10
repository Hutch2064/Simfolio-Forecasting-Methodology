"""Exact Gaussian factor/Kalman dependence and portfolio rejoin closure.

The implementation follows the Gaussian special case in the historical
``simfolio_adaptive_pgas.py`` and ``simfolio_oos_copula_alternatives.py``
wrappers.  It is intentionally serial and cache-free; serial arithmetic is
the reference path used for source parity probes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata


def pseudo_observations(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("frontier_invalid_asset_history")
    if not np.all(np.isfinite(values)):
        raise ValueError("frontier_nonfinite_asset_history")
    ranks = np.column_stack([rankdata(values[:, col], method="average") for col in range(values.shape[1])])
    return np.clip(ranks / float(values.shape[0] + 1), 1e-8, 1.0 - 1e-8).astype(np.float64)


def fit_dynamic_gaussian_factor_model(history: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 80 or values.shape[1] < 2:
        raise ValueError("frontier_requires_80_observations_and_two_assets")
    if not np.all(np.isfinite(values)):
        raise ValueError("frontier_nonfinite_asset_history")
    observations = ndtri(pseudo_observations(values))
    mean = np.mean(observations, axis=0)
    centered = observations - mean[None, :]
    covariance = np.asarray(np.cov(centered, rowvar=False, ddof=1), dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= -1e-7:
        raise ValueError("frontier_non_psd_score_covariance")
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    cumulative = np.cumsum(eigenvalues) / max(float(np.sum(eigenvalues)), 1e-12)
    factor_count = max(1, min(int(np.searchsorted(cumulative, 0.80) + 1), 3, int(values.shape[1])))
    loading = eigenvectors[:, :factor_count] * np.sqrt(eigenvalues[:factor_count])[None, :]
    residual_variance = np.diag(covariance) - np.sum(loading * loading, axis=1)
    if not np.all(np.isfinite(residual_variance)) or np.any(residual_variance <= 1e-7):
        raise ValueError("frontier_nonpositive_factor_residual_variance")
    with np.errstate(all="ignore"):
        factor_scores = centered @ loading @ np.linalg.inv(loading.T @ loading)
    phi = np.empty(factor_count, dtype=np.float64)
    innovation_variance = np.empty(factor_count, dtype=np.float64)
    for factor in range(factor_count):
        left, right = factor_scores[:-1, factor], factor_scores[1:, factor]
        denominator = float(np.dot(left, left))
        fitted = float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
        phi[factor] = float(np.clip(fitted, -0.995, 0.995))
        innovation_variance[factor] = max(float(np.var(right - phi[factor] * left, ddof=1)), 1e-7)
    initial_variance = innovation_variance / np.maximum(1.0 - phi * phi, 1e-7)
    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "observations": np.asarray(observations, dtype=np.float64),
        "loading": np.asarray(loading, dtype=np.float64),
        "residual_variance": np.asarray(residual_variance, dtype=np.float64),
        "phi": phi,
        "innovation_variance": innovation_variance,
        "initial_variance": initial_variance,
        "factor_count": int(factor_count),
        "observation_count": int(values.shape[0]),
        "asset_count": int(values.shape[1]),
    }


def kalman_terminal_posterior(model: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(model["observations"], dtype=np.float64)
    loading = np.asarray(model["loading"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    phi = np.asarray(model["phi"], dtype=np.float64)
    q = np.asarray(model["innovation_variance"], dtype=np.float64)
    residual = np.asarray(model["residual_variance"], dtype=np.float64)
    factor_count = int(loading.shape[1])
    if factor_count < 1 or factor_count > 3:
        raise ValueError("frontier_unsupported_factor_dimension")
    state = np.zeros(factor_count, dtype=np.float64)
    covariance = np.diag(np.asarray(model["initial_variance"], dtype=np.float64))
    observation_information = loading.T @ (loading / residual[:, None])
    for index, row in enumerate(observations):
        if index:
            state = phi * state
            covariance = phi[:, None] * covariance * phi[None, :] + np.diag(q)
        prior_precision = np.linalg.inv(covariance)
        posterior_precision = prior_precision + observation_information
        covariance = np.linalg.inv(posterior_precision)
        information = prior_precision @ state + loading.T @ ((row - mean) / residual)
        state = covariance @ information
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
        raise ValueError("frontier_nonfinite_kalman_terminal_posterior")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("frontier_nonpositive_kalman_terminal_posterior") from exc
    return state, covariance


def simulate_future_gaussian_uniforms(
    model: Dict[str, Any],
    simulations: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Use the original day-major draw order and Gaussian state recurrence."""
    count, days = int(simulations), int(horizon)
    if count <= 0 or days <= 0:
        return np.empty((max(count, 0), max(days, 0), int(model["asset_count"])), dtype=np.float64)
    loading = np.asarray(model["loading"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    phi = np.asarray(model["phi"], dtype=np.float64)
    q_scale = np.sqrt(np.asarray(model["innovation_variance"], dtype=np.float64))
    residual_scale = np.sqrt(np.asarray(model["residual_variance"], dtype=np.float64))
    posterior_mean, posterior_covariance = kalman_terminal_posterior(model)
    factor_state = np.asarray(rng.multivariate_normal(posterior_mean, posterior_covariance, size=count, check_valid="raise"), dtype=np.float64)
    factor_count, asset_count = int(loading.shape[1]), int(loading.shape[0])
    random_draws = np.asarray(rng.normal(size=(days, count * (factor_count + asset_count))), dtype=np.float64)
    output = np.empty((count, days, asset_count), dtype=np.float64)
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    # This mirrors _simulate_future_gaussian_uniforms_flat_numba from the
    # historical wrapper.  It deliberately consumes the factor and asset
    # draw blocks in the same row-major positions.
    for simulation in range(count):
        state = factor_state[simulation].copy()
        state = state * phi + random_draws[0, simulation * factor_count : (simulation + 1) * factor_count] * q_scale
        for day in range(days):
            if day > 0:
                factor_start = simulation * factor_count
                state = state * phi + random_draws[day, factor_start : factor_start + factor_count] * q_scale
            asset_start = count * factor_count + simulation * asset_count
            values = mean + random_draws[day, asset_start : asset_start + asset_count] * residual_scale + state @ loading.T
            output[simulation, day, :] = 0.5 * (1.0 + np.vectorize(math.erf)(values * inv_sqrt_two))
    if not np.all(np.isfinite(output)):
        raise ValueError("frontier_nonfinite_future_uniforms")
    return np.clip(output, 1e-8, 1.0 - 1e-8)


def map_uniforms_to_marginal_paths(source: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if source.ndim != 3 or source.shape != uniforms.shape or source.shape[0] < 2:
        raise ValueError("frontier_source_uniform_shape_mismatch")
    n_sims, _, n_assets = source.shape
    positions = np.clip(uniforms, 0.0, 1.0) * float(n_sims - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, n_sims - 1)
    fraction = positions - lower
    output = np.empty_like(source)
    for asset in range(n_assets):
        ordered = np.sort(source[:, :, asset], axis=0)
        low = np.take_along_axis(ordered, lower[:, :, asset], axis=0)
        high = np.take_along_axis(ordered, upper[:, :, asset], axis=0)
        output[:, :, asset] = low * (1.0 - fraction[:, :, asset]) + high * fraction[:, :, asset]
    return output


def rebalanced_portfolio_log_paths(
    log_paths: np.ndarray,
    weights: Sequence[float],
    rebalance_mask: np.ndarray,
    *,
    cost_per_turnover_bps: float = 15.0,
) -> np.ndarray:
    paths = np.asarray(log_paths, dtype=np.float64)
    if paths.ndim != 3 or not np.all(np.isfinite(paths)):
        raise ValueError("frontier_invalid_rejoin_paths")
    target = np.asarray(weights, dtype=np.float64)
    if target.ndim != 1 or target.size != paths.shape[2] or not np.all(np.isfinite(target)):
        raise ValueError("frontier_invalid_rejoin_weights")
    target = np.maximum(target, 0.0)
    total_weight = float(target.sum())
    if total_weight <= 0.0:
        raise ValueError("frontier_nonpositive_rejoin_weights")
    target = target / total_weight
    mask = np.asarray(rebalance_mask, dtype=bool)
    if mask.shape != (paths.shape[1],):
        raise ValueError("frontier_invalid_rebalance_mask")
    n_sims, horizon, n_assets = paths.shape
    holdings_state = np.broadcast_to(target, (n_sims, n_assets)).copy()
    out = np.empty((n_sims, horizon), dtype=np.float64)
    cost = max(float(cost_per_turnover_bps), 0.0) / 10000.0
    for simulation in range(n_sims):
        holdings = holdings_state[simulation].copy()
        for day in range(horizon):
            previous = float(np.sum(holdings))
            log_returns = np.clip(paths[simulation, day], -745.0, 50.0)
            holdings *= np.exp(log_returns)
            ending = float(np.sum(holdings))
            if bool(mask[day]):
                denominator = max(ending, 1e-300)
                turnover = float(np.sum(np.abs(target - holdings / denominator)))
                ending = max(ending - ending * 0.5 * turnover * cost, 0.0)
                holdings = ending * target
            out[simulation, day] = math.log(max(ending, 1e-300) / max(previous, 1e-300))
    return out


def rejoin_sorted_uniform_paths(
    source: np.ndarray,
    uniforms: np.ndarray,
    weights: Sequence[float],
    rebalance_mask: np.ndarray,
    *,
    cost_per_turnover_bps: float = 15.0,
) -> np.ndarray:
    mapped = map_uniforms_to_marginal_paths(source, uniforms)
    return rebalanced_portfolio_log_paths(mapped, weights, rebalance_mask, cost_per_turnover_bps=cost_per_turnover_bps)


__all__ = [
    "fit_dynamic_gaussian_factor_model",
    "kalman_terminal_posterior",
    "map_uniforms_to_marginal_paths",
    "pseudo_observations",
    "rebalanced_portfolio_log_paths",
    "simulate_future_gaussian_uniforms",
]
