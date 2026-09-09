"""Standalone research implementation of Simfolio's Frontier forecast.

The public OOS harness intentionally keeps the model boundary small.  The
cross-asset layer below is a direct transcription of the production Frontier
construction: empirical Gaussian scores -> low-rank dynamic factor model ->
Kalman terminal posterior -> dependent Gaussian uniforms -> rank rejoin of the
asset marginal paths.  The marginal layer implements the promoted deterministic
Fast-MAP/Laplace stochastic-volatility specification using only NumPy/SciPy.

The production service contains additional memory/parallelism/cache machinery;
none of that changes the statistical forecast and is deliberately absent here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from ..runner import ForecastContext, TrainingData
from ..seeds import deterministic_seed

FRONTIER_MODEL_ID = "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced"


def _pseudo_observations(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("frontier_invalid_asset_history")
    if not np.all(np.isfinite(values)):
        raise ValueError("frontier_nonfinite_asset_history")
    ranks = np.column_stack(
        [rankdata(values[:, column], method="average") for column in range(values.shape[1])]
    )
    return np.clip(ranks / float(values.shape[0] + 1), 1e-8, 1.0 - 1e-8)


def fit_dynamic_gaussian_factor_model(history: np.ndarray) -> dict[str, Any]:
    """Production-equivalent dynamic Gaussian factor fit used by Frontier."""
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 80 or values.shape[1] < 2:
        raise ValueError("frontier_requires_80_observations_and_two_assets")
    observations = ndtri(_pseudo_observations(values))
    mean = np.mean(observations, axis=0, dtype=np.float64)
    centered = observations - mean[None, :]
    covariance = np.asarray(np.cov(centered, rowvar=False, ddof=1), dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= -1e-7:
        raise ValueError("frontier_non_psd_score_covariance")
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    cumulative = np.cumsum(eigenvalues) / max(float(np.sum(eigenvalues)), 1e-12)
    factor_count = int(np.searchsorted(cumulative, 0.80) + 1)
    factor_count = max(1, min(factor_count, 3, max(1, int(values.shape[1]) - 1)))
    loading = eigenvectors[:, :factor_count] * np.sqrt(eigenvalues[:factor_count])[None, :]
    residual_variance = np.diag(covariance) - np.sum(loading * loading, axis=1)
    if np.any(~np.isfinite(residual_variance)) or np.any(residual_variance <= 1e-7):
        raise ValueError("frontier_nonpositive_factor_residual_variance")
    factor_projection = centered @ loading
    factor_scores = factor_projection @ np.linalg.inv(loading.T @ loading)
    phi = np.empty(factor_count, dtype=np.float64)
    innovation_variance = np.empty(factor_count, dtype=np.float64)
    for factor in range(factor_count):
        left = factor_scores[:-1, factor]
        right = factor_scores[1:, factor]
        denominator = float(np.dot(left, left))
        fitted = float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
        phi[factor] = float(np.clip(fitted, -0.995, 0.995))
        innovation_variance[factor] = max(
            float(np.var(right - phi[factor] * left, ddof=1)), 1e-7
        )
    initial_variance = innovation_variance / np.maximum(1.0 - phi * phi, 1e-7)
    return {
        "mean": mean,
        "observations": observations,
        "loading": loading,
        "residual_variance": residual_variance,
        "phi": phi,
        "innovation_variance": innovation_variance,
        "initial_variance": initial_variance,
        "factor_count": factor_count,
    }


def _kalman_terminal_posterior(model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(model["observations"], dtype=np.float64)
    loading = np.asarray(model["loading"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    phi = np.asarray(model["phi"], dtype=np.float64)
    q = np.asarray(model["innovation_variance"], dtype=np.float64)
    r = np.asarray(model["residual_variance"], dtype=np.float64)
    state = np.zeros(phi.size, dtype=np.float64)
    covariance = np.diag(np.asarray(model["initial_variance"], dtype=np.float64))
    observation_information = loading.T @ (loading / r[:, None])
    for index, row in enumerate(obs):
        if index:
            state = phi * state
            covariance = phi[:, None] * covariance * phi[None, :] + np.diag(q)
        prior_precision = np.linalg.inv(covariance)
        posterior_precision = prior_precision + observation_information
        covariance = np.linalg.inv(posterior_precision)
        information = prior_precision @ state + loading.T @ ((row - mean) / r)
        state = covariance @ information
    return state, covariance


def _simulate_dependent_uniforms(
    model: dict[str, Any], simulations: int, horizon: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    loading = np.asarray(model["loading"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    phi = np.asarray(model["phi"], dtype=np.float64)
    q_scale = np.sqrt(np.asarray(model["innovation_variance"], dtype=np.float64))
    r_scale = np.sqrt(np.asarray(model["residual_variance"], dtype=np.float64))
    posterior_mean, posterior_cov = _kalman_terminal_posterior(model)
    state = rng.multivariate_normal(posterior_mean, posterior_cov, size=int(simulations))
    output = np.empty((int(simulations), int(horizon), loading.shape[0]), dtype=np.float64)
    for day in range(int(horizon)):
        state = state * phi[None, :] + rng.normal(size=state.shape) * q_scale[None, :]
        gaussian = mean[None, :] + state @ loading.T
        gaussian += rng.normal(size=(int(simulations), loading.shape[0])) * r_scale[None, :]
        output[:, day, :] = ndtr(gaussian)
    return np.clip(output, 1e-10, 1.0 - 1e-10)


def _observed_log_variance(residuals: np.ndarray) -> np.ndarray:
    squared = np.maximum(np.square(np.asarray(residuals, dtype=np.float64)), 1e-12)
    values = np.log(squared)
    if values.size >= 20:
        lo, hi = np.quantile(values, [0.01, 0.99])
        values = np.clip(values, lo, hi)
    return values


def _fit_latent_log_volatility(residuals: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Deterministic MAP/Laplace AR(1) latent-log-vol approximation.

    The observation is log squared residuals with the theoretical log-chi-square
    location correction.  Parameters are optimized in transformed coordinates,
    matching the promoted non-MCMC state-inference family.
    """
    observed = _observed_log_variance(residuals)
    bias = -1.2703628454614782  # E[log(chi-square_1)]
    y = observed - bias
    if y.size < 20:
        level = float(np.mean(y)) if y.size else -9.0
        return level, 0.95, 0.15, np.full(y.size, level)

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        mu = float(theta[0])
        phi = float(np.tanh(theta[1]) * 0.995)
        sigma = float(np.exp(theta[2]))
        return mu, phi, max(sigma, 1e-5)

    def objective(theta: np.ndarray) -> float:
        mu, phi, sigma = unpack(theta)
        innovation = y[1:] - mu - phi * (y[:-1] - mu)
        state_var = max(sigma * sigma, 1e-10)
        # Robust Gaussian pseudo-likelihood plus weak regularization.  This is
        # the deterministic MAP stage; the Laplace uncertainty is represented
        # when future states are simulated below.
        value = 0.5 * np.sum(innovation * innovation / state_var + np.log(state_var))
        value += 0.5 * ((phi / 0.9) ** 8) + 0.5 * (math.log(sigma / 0.2) / 2.0) ** 2
        return float(value)

    mu0 = float(np.mean(y))
    left, right = y[:-1] - mu0, y[1:] - mu0
    phi0 = float(np.clip(np.dot(left, right) / max(np.dot(left, left), 1e-12), -0.95, 0.95))
    sigma0 = max(float(np.std(right - phi0 * left, ddof=1)), 0.05)
    fit = minimize(
        objective,
        np.asarray([mu0, np.arctanh(phi0 / 0.995), math.log(sigma0)]),
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-11},
    )
    mu, phi, sigma = unpack(np.asarray(fit.x, dtype=np.float64))
    filtered = np.empty_like(y)
    filtered[0] = y[0]
    gain = 0.18
    for index in range(1, y.size):
        prediction = mu + phi * (filtered[index - 1] - mu)
        filtered[index] = prediction + gain * (y[index] - prediction)
    return mu, phi, sigma, filtered


def _dynamic_mean(log_returns: np.ndarray) -> tuple[float, np.ndarray]:
    """Historical-CAGR anchored latent-Sharpe mean used by the reference runtime."""
    values = np.asarray(log_returns, dtype=np.float64)
    sample_sigma = max(float(np.std(values, ddof=1)), 1e-10)
    long_run = float(np.mean(values))
    # A low-noise AR(1) state on normalized returns.  Long-run state is anchored
    # to the in-sample log CAGR (the arithmetic mean of daily log returns).
    sharpe_obs = values / sample_sigma * math.sqrt(252.0)
    anchor = long_run / sample_sigma * math.sqrt(252.0)
    centered = sharpe_obs - anchor
    left, right = centered[:-1], centered[1:]
    phi = float(np.clip(np.dot(left, right) / max(np.dot(left, left), 1e-12), -0.995, 0.995))
    state = anchor
    process_var = max(float(np.var(right - phi * left, ddof=1)), 1e-6)
    obs_var = max(float(np.var(sharpe_obs - anchor, ddof=1)), process_var)
    state_var = process_var / max(1.0 - phi * phi, 1e-6)
    for observation in sharpe_obs:
        predicted = anchor + phi * (state - anchor)
        predicted_var = phi * phi * state_var + process_var
        gain = predicted_var / (predicted_var + obs_var)
        state = predicted + gain * (observation - predicted)
        state_var = (1.0 - gain) * predicted_var
    daily_mean = float(state * sample_sigma / math.sqrt(252.0))
    return daily_mean, values - daily_mean


def _stationary_bootstrap_indices(
    pool_size: int, simulations: int, horizon: int, rng: np.random.Generator
) -> np.ndarray:
    block = max(2.0, min(63.0, float(pool_size) ** (1.0 / 3.0)))
    restart_probability = 1.0 / block
    indices = np.empty((simulations, horizon), dtype=np.int64)
    indices[:, 0] = rng.integers(0, pool_size, size=simulations)
    for day in range(1, horizon):
        restart = rng.random(simulations) < restart_probability
        continued = (indices[:, day - 1] + 1) % pool_size
        fresh = rng.integers(0, pool_size, size=simulations)
        indices[:, day] = np.where(restart, fresh, continued)
    return indices


def _simulate_fastmap_marginal(
    log_returns: np.ndarray, simulations: int, horizon: int, seed: int
) -> np.ndarray:
    values = np.asarray(log_returns, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 80:
        raise ValueError("frontier_fastmap_requires_80_observations")
    mean, residuals = _dynamic_mean(values)
    mu_h, phi_h, sigma_h, filtered_h = _fit_latent_log_volatility(residuals)
    scale = np.exp(0.5 * filtered_h)
    standardized = residuals / np.maximum(scale, 1e-8)
    standardized = standardized[np.isfinite(standardized)]
    standardized -= float(np.mean(standardized))
    std = float(np.std(standardized, ddof=1))
    if std > 1e-10:
        standardized /= std
    standardized = np.clip(standardized, -30.0, 30.0)

    rng = np.random.default_rng(int(seed))
    indices = _stationary_bootstrap_indices(standardized.size, simulations, horizon, rng)
    innovations = standardized[indices]
    latent = np.empty((simulations, horizon), dtype=np.float64)
    state = np.full(simulations, float(filtered_h[-1]), dtype=np.float64)
    for day in range(horizon):
        state = mu_h + phi_h * (state - mu_h) + sigma_h * rng.normal(size=simulations)
        latent[:, day] = state
    paths = mean + np.exp(0.5 * latent) * innovations
    return np.asarray(paths, dtype=np.float64)


def _rank_rejoin(marginal_paths: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    source = np.asarray(marginal_paths, dtype=np.float64)
    simulations, horizon, assets = source.shape
    output = np.empty_like(source)
    n_minus_one = simulations - 1
    for day in range(horizon):
        for asset in range(assets):
            sorted_values = np.sort(source[:, day, asset])
            position = uniforms[:, day, asset] * float(n_minus_one)
            lower = np.floor(position).astype(np.int64)
            upper = np.minimum(lower + 1, n_minus_one)
            fraction = position - lower
            output[:, day, asset] = (
                sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
            )
    return np.clip(output, -745.0, 50.0)


def _rebalance_mask(horizon: int, frequency: str) -> np.ndarray:
    mask = np.zeros(int(horizon), dtype=bool)
    normalized = str(frequency or "none").lower()
    if normalized == "monthly":
        mask[20::21] = True
    elif normalized == "quarterly":
        mask[62::63] = True
    elif normalized in {"annually", "annual", "yearly"}:
        mask[251::252] = True
    return mask


def _portfolio_rejoin(asset_paths: np.ndarray, weights: np.ndarray, frequency: str) -> np.ndarray:
    simulations, horizon, assets = asset_paths.shape
    target = np.asarray(weights, dtype=np.float64)
    holdings = np.broadcast_to(target, (simulations, assets)).copy()
    output = np.empty((simulations, horizon), dtype=np.float64)
    mask = _rebalance_mask(horizon, frequency)
    for day in range(horizon):
        previous = np.sum(holdings, axis=1)
        holdings *= np.exp(np.clip(asset_paths[:, day, :], -745.0, 50.0))
        ending = np.sum(holdings, axis=1)
        output[:, day] = np.log(np.maximum(ending, 1e-300) / np.maximum(previous, 1e-300))
        if mask[day]:
            holdings = ending[:, None] * target[None, :]
    return output


@dataclass(frozen=True)
class FrontierModel:
    """Asset-level FastMAP + dynamic Gaussian-factor/Kalman Frontier model."""

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
            "frontier_dynamic_gaussian_factor", context.portfolio_id, horizon, simulations, context.seed
        )
        uniforms = _simulate_dependent_uniforms(dependence, simulations, horizon, dependence_seed)
        dependent_assets = _rank_rejoin(marginal, uniforms)
        return _portfolio_rejoin(
            dependent_assets,
            np.asarray(training.policy.weights, dtype=np.float64),
            training.policy.rebalance,
        )
