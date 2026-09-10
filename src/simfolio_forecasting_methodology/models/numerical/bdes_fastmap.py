"""Historical BDES FastMAP/Laplace marginal forecast closure.

This module is a deliberately small extraction of the numerical functions used
by the 2026-08-23 academic Frontier wrapper.  It keeps the historical
candidate's float64 arithmetic, deterministic seed alias, DLM mean state,
Laplace stochastic-log-volatility fit, multiscale BDES state, stationary
bootstrap, and serial path recurrence.  Service, catalogue, cache, and
MCMC-control code is intentionally outside this closure.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


SV_LOG_CHI_SQUARE_MEAN = -1.2703628454614782
SV_LOG_CHI_SQUARE_VAR = (math.pi * math.pi) / 2.0
RANK1_SV_FORECAST_MIN_OBS = 5
FULL_MCMC_SV_MIN_OBS = RANK1_SV_FORECAST_MIN_OBS

BDES_MULTISCALE_GRID_FIXED = "fixed_four_scale"
BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC = "history_gated_calendar_geometric_without_daily"

FRONTIER_MARGINAL_ID = (
    "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_fast_map_laplace_sigma_points"
)
FRONTIER_DEPENDENCE_ID = "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced"

# These are the candidate dictionary values from the source-research registry.
# The public model accepts no alternate values because silently changing one of
# them would create a different historical candidate.
FRONTIER_CANDIDATE: Dict[str, Any] = {
    "ablation_family": "current_production_bdes_cagr_fast_map_laplace",
    "id": FRONTIER_MARGINAL_ID,
    "type": "bdes_non_mcmc_sv_overlay",
    "return_target": "portfolio_daily_log_return",
    "forecast_level": "portfolio_return",
    "factor_model": "not_applicable",
    "regime_model": "not_applicable",
    "sv_research_baseline": False,
    "mean_component": "portfolio_likelihood_estimated_latent_sharpe_dlm_to_historical_cagr_anchor",
    "mean_model_ablation_of": "bayesian_sbb_overlay_gjr_garch_1_1_empirical_bayes_sharpe",
    "innovation_component": "bdes_multiscale_latent_log_volatility_with_conditional_sharpe_mean_scaling",
    "measurement_equation": "log_chi_square_mean_corrected_winsorized",
    "residual_tail_method": "raw_empirical_untruncated",
    "tail_method": "filtered_empirical_tail",
    "vol_anchor_model": "none",
    "vol_anchor_target": "none",
    "state_innovation_distribution": "gaussian",
    "state_innovation_source": "map_laplace_smoothed_state_path",
    "state_innovation_standardization": "mean_std",
    "state_innovation_resampling": "iid",
    "state_innovation_coupling": "correlation_mixture",
    "mean_model": "evidence_estimated_sharpe_dlm_historical_cagr_anchor",
    "vol_model": "map_laplace_stochastic_log_volatility_overlay",
    "vol_path_model": "bdes_multiscale_log_vol",
    "vol_path_blend": "none",
    "bdes_multiscale_k_star": 4,
    "bdes_multiscale_shock_coupling": "convex_state_independent_mix",
    "innovation_standardization": "mean_std",
    "innovation_resampling": "stationary_bootstrap",
    "innovation_pool_source": "sample_standardized",
    "path_generator": "stationary_bootstrap_standardized_residuals_scaled_by_bdes_multiscale_latent_sv_paths",
    "leverage": True,
    "leverage_alignment": "lagged_return",
    "leverage_correlation_method": "pearson",
    "leverage_correlation_scope": "deterministic_selected_state_path",
    "state_inference_model": "map_laplace_sigma_points",
    "unclipped_sv_measurement": False,
    "unclipped_empirical_innovations": True,
    "mean_state_scaling": "conditional_sv_sharpe",
    "clip_simulated_returns": True,
    "innovation_method": "empirical_standardized_residuals",
    "innovation_conditioning": "none",
    "innovation_tail_splice": "none",
    "latent_vol_persistence": "ar1",
    "overlay_model": FRONTIER_MARGINAL_ID,
    "vol_overlay_model": "map_laplace_stochastic_log_volatility_overlay",
    "sv_measurement_bias": "log_chi_square_theoretical",
    "sv_sigma_scale_method": "none",
    "sv_research_baseline": False,
    "simulation_method": FRONTIER_MARGINAL_ID,
    "selection_method": "pre_specified_bdes_cagr_fast_map_laplace_sv_overlay_validation",
    "ablation_family": "current_production_bdes_cagr_fast_map_laplace",
    "feature_family": "current_production_bdes_cagr_non_mcmc_state_inference",
    "comparison_target": "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_optimal_block_adaptive_metropolis_proposal_mcmc",
    "previous_incumbent_id": "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_optimal_block_adaptive_metropolis_proposal_mcmc",
    "rollback_env_var": "SIMFOLIO_FORECAST_MODEL=bdes_mcmc",
    "validation_status": "paired_80_portfolio_fast_map_laplace_sigma_point_sv_overlay_candidate",
    "production_status": "promoted_to_live_engine_20260703",
    "parameter_sampler": "deterministic_map_laplace_sigma_points",
    "state_inference": "map_laplace_sigma_points",
    "transformed_parameter_mcmc": False,
    "mcmc_iterations": 0,
    "mcmc_initial_iterations": 0,
    "mcmc_burn": 0,
    "mcmc_thin": 0,
    "mcmc_stopping": "none",
    "mcmc_min_iterations": 0,
    "mcmc_check_interval": 0,
    "mcmc_min_ess": None,
    "mcmc_rhat_threshold": None,
    "mcmc_forecast_stability_tolerance": None,
    "mcmc_extend_iterations": 0,
    "mcmc_max_iterations": 0,
    "mcmc_chains": 0,
    "mcmc_proposal_adaptation": "none",
    "mcmc_adaptation_iterations": 0,
    "mcmc_adaptation_target_acceptance": None,
}


def deterministic_seed(*parts: Any) -> int:
    """Historical source-research seed derivation, including its alias."""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


def politis_white_block_length(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 4:
        return 1
    arr = arr - float(np.mean(arr))
    max_lag = int(min(max(5, math.ceil((n ** (1.0 / 3.0)) * 2.0)), max(2, n // 4)))
    if max_lag < 2:
        return 1
    gamma0 = float(np.dot(arr, arr) / n)
    if not np.isfinite(gamma0) or gamma0 <= 0.0:
        return 1
    gamma = np.empty(max_lag + 1, dtype=np.float64)
    gamma[0] = gamma0
    for lag in range(1, max_lag + 1):
        gamma[lag] = float(np.dot(arr[lag:], arr[:-lag]) / n)
    weights = 1.0 - np.arange(1, max_lag + 1, dtype=np.float64) / (max_lag + 1.0)
    g_hat = float(np.sum(2.0 * weights * np.arange(1, max_lag + 1) * gamma[1:]))
    d_hat = float(gamma0 + 2.0 * np.sum(weights * gamma[1:]))
    if not np.isfinite(g_hat) or not np.isfinite(d_hat) or abs(d_hat) < 1e-12:
        return 1
    b_opt = ((2.0 * g_hat * g_hat) / (d_hat * d_hat) * n) ** (1.0 / 3.0)
    if not np.isfinite(b_opt) or b_opt < 1.0:
        return 1
    return int(max(1, round(float(b_opt))))


def stationary_bootstrap_indices(
    n: int,
    block_length: int,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(n_paths) <= 0 or int(total_days) <= 0:
        return np.empty((max(int(n_paths), 0), max(int(total_days), 0)), dtype=np.int32)
    if int(n) <= 0:
        raise ValueError("frontier_stationary_bootstrap_requires_observations")
    block = max(int(block_length), 1)
    probability = 1.0 / float(block)
    initial = rng.integers(0, int(n), size=int(n_paths), dtype=np.int32)
    if int(total_days) == 1:
        return initial.reshape(int(n_paths), 1)
    flags = rng.random((int(n_paths), int(total_days) - 1)) < probability
    points = rng.integers(0, int(n), size=(int(n_paths), int(total_days) - 1), dtype=np.int32)
    out = np.empty((int(n_paths), int(total_days)), dtype=np.int32)
    out[:, 0] = initial
    for day in range(1, int(total_days)):
        out[:, day] = np.where(flags[:, day - 1], points[:, day - 1], (out[:, day - 1] + 1) % int(n))
    return out


def _hac_mean_standard_error(values: np.ndarray) -> Tuple[float, int, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0, 0, 0.0
    centered = x - float(np.mean(x))
    sample_var = float(np.var(centered, ddof=1))
    if not np.isfinite(sample_var) or sample_var <= 0.0:
        return 0.0, 0, 0.0
    bandwidth = int(max(politis_white_block_length(centered), politis_white_block_length(centered * centered)))
    bandwidth = int(max(0, min(bandwidth, x.size - 1)))
    long_run_var = float(np.dot(centered, centered) / x.size)
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / float(bandwidth + 1)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / x.size)
        if np.isfinite(gamma):
            long_run_var += 2.0 * weight * gamma
    if not np.isfinite(long_run_var) or long_run_var <= 0.0:
        long_run_var = sample_var
    return float(math.sqrt(max(long_run_var, 0.0) / float(x.size))), bandwidth, float(long_run_var)


DLM_DRIFT_PHI_MIN = 0.0
DLM_DRIFT_PHI_MAX = 0.9995
DLM_DRIFT_STATE_VAR_RATIO_MIN = 1.0e-10
DLM_DRIFT_MAX_ANNUAL_DRIFT_SHARPE_SD = 2.5
DLM_DRIFT_STATE_VAR_RATIO_MAX = float((DLM_DRIFT_MAX_ANNUAL_DRIFT_SHARPE_SD**2) / 252.0)


def _overlay_half_life_days(persistence: float) -> Optional[float]:
    p = abs(float(persistence))
    if not np.isfinite(p) or p <= 0.0 or p >= 0.999999:
        return None
    return float(math.log(0.5) / math.log(p))


def _dlm_ar1_loglik(y: np.ndarray, obs_var: float, phi: float, state_var_ratio: float) -> float:
    values = np.asarray(y, dtype=np.float64)
    obs = float(obs_var)
    phi_value = float(phi)
    ratio = float(state_var_ratio)
    if values.size < 2 or not np.isfinite(obs) or obs <= 0.0:
        return -math.inf
    if not np.isfinite(phi_value) or phi_value < 0.0 or phi_value >= 1.0:
        return -math.inf
    if not np.isfinite(ratio) or ratio <= 0.0:
        return -math.inf
    stationary_var = max(obs * ratio, obs * 1.0e-14)
    state_noise_var = max((1.0 - phi_value * phi_value) * stationary_var, obs * 1.0e-14)
    mean = 0.0
    variance = stationary_var
    loglik = 0.0
    for value in values:
        predicted = phi_value * mean
        predicted_var = phi_value * phi_value * variance + state_noise_var
        forecast_var = predicted_var + obs
        if not np.isfinite(forecast_var) or forecast_var <= 0.0:
            return -math.inf
        innovation = float(value) - predicted
        loglik += -0.5 * (math.log(2.0 * math.pi) + math.log(forecast_var) + innovation * innovation / forecast_var)
        gain = predicted_var / forecast_var
        mean = predicted + gain * innovation
        variance = max((1.0 - gain) * predicted_var, obs * 1.0e-14)
    return float(loglik) if np.isfinite(loglik) else -math.inf


def _kalman_ar1_drift_filter(
    y: np.ndarray,
    *,
    obs_var: float,
    phi: float,
    state_var_ratio: float,
) -> Dict[str, Any]:
    values = np.asarray(y, dtype=np.float64)
    obs = float(obs_var)
    phi_value = float(np.clip(phi, DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX))
    ratio = float(np.clip(state_var_ratio, DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX))
    stationary_var = max(obs * ratio, obs * 1.0e-14)
    state_noise_var = max((1.0 - phi_value * phi_value) * stationary_var, obs * 1.0e-14)
    predicted_mean = np.empty(values.size, dtype=np.float64)
    predicted_var = np.empty(values.size, dtype=np.float64)
    filtered_mean = np.empty(values.size, dtype=np.float64)
    filtered_var = np.empty(values.size, dtype=np.float64)
    mean = 0.0
    variance = stationary_var
    loglik = 0.0
    for idx, value in enumerate(values):
        predicted = phi_value * mean
        predicted_var_value = phi_value * phi_value * variance + state_noise_var
        forecast_var = predicted_var_value + obs
        if not np.isfinite(forecast_var) or forecast_var <= 0.0:
            return {"loglik": -math.inf}
        innovation = float(value) - predicted
        loglik += -0.5 * (math.log(2.0 * math.pi) + math.log(forecast_var) + innovation * innovation / forecast_var)
        gain = predicted_var_value / forecast_var
        mean = predicted + gain * innovation
        variance = max((1.0 - gain) * predicted_var_value, obs * 1.0e-14)
        predicted_mean[idx] = predicted
        predicted_var[idx] = predicted_var_value
        filtered_mean[idx] = mean
        filtered_var[idx] = variance
    return {
        "loglik": float(loglik),
        "predicted_mean": predicted_mean,
        "predicted_var": predicted_var,
        "filtered_mean": filtered_mean,
        "filtered_var": filtered_var,
        "state_noise_var": float(state_noise_var),
        "stationary_var": float(stationary_var),
        "final_mean": float(mean),
        "final_var": float(variance),
    }


def _estimate_evidence_dlm_drift_params(
    centered_returns: np.ndarray,
    *,
    obs_var: float,
    signal_ratio_start: float,
) -> Dict[str, float]:
    values = np.asarray(centered_returns, dtype=np.float64)
    values = values[np.isfinite(values)]
    ratio_start = float(np.clip(signal_ratio_start, DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX))
    best_phi = 0.95
    best_ratio = ratio_start
    best_loglik = -math.inf

    def score(phi_value: float, ratio_value: float) -> float:
        return _dlm_ar1_loglik(values, float(obs_var), float(phi_value), float(ratio_value))

    for phi0 in (0.25, 0.75, 0.95, 0.99):
        for ratio0 in (ratio_start, min(ratio_start * 10.0, DLM_DRIFT_STATE_VAR_RATIO_MAX), max(ratio_start * 0.10, DLM_DRIFT_STATE_VAR_RATIO_MIN)):
            start = np.asarray([float(np.clip(phi0, DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX)), math.log(float(np.clip(ratio0, DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX)))], dtype=np.float64)

            def objective(theta: np.ndarray) -> float:
                value = score(float(theta[0]), math.exp(float(theta[1])))
                return float(-value) if np.isfinite(value) else 1.0e100

            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=((DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX), (math.log(DLM_DRIFT_STATE_VAR_RATIO_MIN), math.log(DLM_DRIFT_STATE_VAR_RATIO_MAX))),
                options={"maxiter": 45, "ftol": 1.0e-7, "gtol": 1.0e-5},
            )
            if not np.isfinite(result.fun):
                continue
            phi_hat = float(np.clip(result.x[0], DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX))
            ratio_hat = float(np.clip(math.exp(float(result.x[1])), DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX))
            loglik = score(phi_hat, ratio_hat)
            if np.isfinite(loglik) and loglik > best_loglik:
                best_phi, best_ratio, best_loglik = phi_hat, ratio_hat, loglik
    if not np.isfinite(best_loglik):
        best_loglik = score(best_phi, best_ratio)
    return {"phi": float(best_phi), "state_var_ratio": float(best_ratio), "loglik": float(best_loglik)}


def dlm_mu_draw_paths(fit: Mapping[str, Any], total_days: int, n_paths: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    if not bool(fit.get("dlm_drift_paths", False)) or int(total_days) <= 0 or int(n_paths) <= 0:
        return None
    anchor_mu = float(fit.get("dlm_long_run_anchor_mean", 0.0))
    phi = float(fit.get("dlm_state_transition_phi", 0.0))
    state_noise_var = float(max(fit.get("dlm_state_noise_var", 0.0), 0.0))
    state_mean = float(fit.get("dlm_state_posterior_deviation_mean", 0.0))
    state_var = float(max(fit.get("dlm_state_posterior_deviation_var", 0.0), 0.0))
    if not all(np.isfinite(value) for value in (anchor_mu, phi, state_noise_var, state_mean, state_var)):
        return None
    phi = float(np.clip(phi, DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX))
    states = rng.normal(state_mean, math.sqrt(state_var), size=int(n_paths))
    noise_sd = math.sqrt(state_noise_var)
    out = np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    for day in range(int(total_days)):
        states = phi * states + (rng.normal(0.0, noise_sd, size=int(n_paths)) if noise_sd > 0.0 else 0.0)
        out[:, day] = anchor_mu + states
    return out if np.all(np.isfinite(out)) else None


def _sv_observed_log_variance(train_values: np.ndarray, mu: float, *, winsorize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    eps_x = (x - float(mu)) * 100.0
    squared = eps_x * eps_x
    if winsorize:
        positive = squared[np.isfinite(squared) & (squared > 0.0)]
        floor = float(max(np.quantile(positive, 0.001) * 0.1, 1e-10)) if positive.size else 1e-10
        observed = np.log(np.maximum(squared, floor)) - SV_LOG_CHI_SQUARE_MEAN
        observed = observed[np.isfinite(observed)]
    else:
        mask = np.isfinite(squared) & (squared > 0.0)
        observed = np.log(squared[mask]) - SV_LOG_CHI_SQUARE_MEAN
        eps_x = eps_x[mask]
        observed = observed[np.isfinite(observed)]
        eps_x = eps_x[: observed.size]
    if winsorize and observed.size >= 80:
        lo, hi = np.quantile(observed, [0.005, 0.995])
        observed = np.clip(observed, lo, hi)
    return observed.astype(np.float64), eps_x[: observed.size].astype(np.float64)


def _sv_kalman_filter(observed_log_var: np.ndarray, level: float, phi: float, eta: float, *, return_path: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    y = np.asarray(observed_log_var, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0 or not all(np.isfinite(v) for v in (level, phi, eta)):
        return -math.inf, np.empty(0), np.empty(0)
    if phi < 0.0 or phi >= 0.999 or eta <= 1e-6:
        return -math.inf, np.empty(0), np.empty(0)
    q = float(eta * eta)
    mean = float(level)
    variance = float(max(q / max(1.0 - phi * phi, 1e-4), 1e-6))
    loglik = 0.0
    filtered = np.empty(y.size if return_path else 1, dtype=np.float64)
    filtered_var = np.empty(y.size if return_path else 1, dtype=np.float64)
    for idx, obs in enumerate(y):
        forecast_var = float(max(variance + SV_LOG_CHI_SQUARE_VAR, 1e-8))
        innovation = float(obs - mean)
        loglik += -0.5 * (math.log(2.0 * math.pi * forecast_var) + innovation * innovation / forecast_var)
        gain = variance / forecast_var
        updated_mean = mean + gain * innovation
        updated_var = float(max((1.0 - gain) * variance, 1e-8))
        if return_path:
            filtered[idx] = updated_mean
            filtered_var[idx] = updated_var
        mean = float(level + phi * (updated_mean - level))
        variance = float(phi * phi * updated_var + q)
    if not return_path:
        filtered[0], filtered_var[0] = updated_mean, updated_var
    return float(loglik), filtered, filtered_var


def _initial_sv_state_space_params(observed_log_var: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(observed_log_var, dtype=np.float64)
    y = y[np.isfinite(y)]
    level = float(np.mean(y)) if y.size else 0.0
    if y.size < 30:
        return level, 0.94, 0.25
    variance = float(np.var(y, ddof=1))
    covariance = float(np.mean((y[:-1] - np.mean(y[:-1])) * (y[1:] - np.mean(y[1:]))))
    signal_variance = max(variance - SV_LOG_CHI_SQUARE_VAR, 0.05 * max(variance, 1e-8))
    phi = float(np.clip(covariance / max(signal_variance, 1e-8), 0.05, 0.985))
    eta = float(np.sqrt(max(signal_variance * (1.0 - phi * phi), 0.02 * signal_variance, 1e-4)))
    return level, phi, float(np.clip(eta, 0.03, 2.0))


def _logit(value: float) -> float:
    eps = np.finfo(np.float64).eps
    p = float(min(max(value, eps), 1.0 - eps))
    return float(math.log(p / (1.0 - p)))


def _inv_logit(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-float(value))
        return float(1.0 / (1.0 + z))
    z = math.exp(float(value))
    return float(z / (1.0 + z))


def _sv_transformed_log_posterior(y: np.ndarray, level: float, phi_logit: float, log_eta: float) -> float:
    if not all(np.isfinite(v) for v in (level, phi_logit, log_eta)):
        return -math.inf
    phi = _inv_logit(phi_logit)
    eta = math.exp(log_eta)
    likelihood, _, _ = _sv_kalman_filter(y, level, phi, eta)
    if not np.isfinite(likelihood):
        return -math.inf
    prior_level = -0.5 * ((level - float(np.mean(y))) / 4.0) ** 2
    prior_phi = -0.5 * ((phi - 0.94) / 0.20) ** 2
    prior_eta = -0.5 * ((math.log(eta) - math.log(0.35)) / 1.0) ** 2
    return float(likelihood + prior_level + prior_phi + prior_eta + math.log(max(phi * (1.0 - phi), 1e-300)) + log_eta)


def _fit_sv_map_state_space_params(observed_log_var: np.ndarray, *, start_count: int = 1, maxiter: int = 60) -> Tuple[float, float, float]:
    y = np.asarray(observed_log_var, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 30:
        return _initial_sv_state_space_params(y)
    init_level, init_phi, init_eta = _initial_sv_state_space_params(y)
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])

    def objective(params: np.ndarray) -> float:
        value = _sv_transformed_log_posterior(y, float(params[0]), float(params[1]), float(params[2]))
        return float(-value) if np.isfinite(value) else 1e100

    starts = [(init_level, init_phi, init_eta), (init_level, 0.90, max(init_eta, 0.10)), (init_level, 0.97, max(init_eta * 0.5, 0.08)), (float(np.median(y)), 0.985, 0.12)]
    best: Optional[Tuple[float, np.ndarray]] = None
    for level, phi, eta in starts[: int(np.clip(int(start_count), 1, len(starts)))]:
        x0 = np.asarray([level, _logit(float(np.clip(phi, 0.001, 0.995))), math.log(float(np.clip(eta, 0.02, 2.5)))], dtype=np.float64)
        result = minimize(objective, x0, method="L-BFGS-B", bounds=((float(y_lo - 4.0), float(y_hi + 4.0)), (-7.0, 7.0), (math.log(0.02), math.log(2.5))), options={"maxiter": int(max(1, maxiter))})
        value = float(result.fun) if np.isfinite(result.fun) else math.inf
        if result.success and (best is None or value < best[0]):
            best = (value, np.asarray(result.x, dtype=np.float64))
    if best is None:
        return init_level, init_phi, init_eta
    params = best[1]
    return float(params[0]), float(_inv_logit(float(params[1]))), float(math.exp(float(params[2])))


def _sv_kalman_rts_smoother_mean(observed_log_var: np.ndarray, level: float, phi: float, eta: float) -> Tuple[float, np.ndarray, np.ndarray]:
    y = np.asarray(observed_log_var, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n <= 0 or not all(np.isfinite(v) for v in (level, phi, eta)) or phi < 0.0 or phi >= 0.999 or eta <= 1e-6:
        return -math.inf, np.empty(0), np.empty(0)
    state_var = float(max(eta * eta, 1e-10))
    init_var = float(state_var / max(1.0 - phi * phi, 1e-4))
    pred_mean = np.empty(n); pred_var = np.empty(n); filt_mean = np.empty(n); filt_var = np.empty(n)
    mean_prev, var_prev, loglik = float(level), init_var, 0.0
    for idx in range(n):
        mean_pred = float(level) if idx == 0 else float(level + phi * (mean_prev - level))
        var_pred = float(init_var) if idx == 0 else float(phi * phi * var_prev + state_var)
        var_pred = max(var_pred, 1e-10)
        forecast_var = max(var_pred + SV_LOG_CHI_SQUARE_VAR, 1e-10)
        innovation = float(y[idx] - mean_pred)
        loglik += -0.5 * (math.log(2.0 * math.pi * forecast_var) + innovation * innovation / forecast_var)
        gain = var_pred / forecast_var
        mean_filt = mean_pred + gain * innovation
        var_filt = max((1.0 - gain) * var_pred, 1e-10)
        pred_mean[idx], pred_var[idx], filt_mean[idx], filt_var[idx] = mean_pred, var_pred, mean_filt, var_filt
        mean_prev, var_prev = mean_filt, var_filt
    smooth_mean = np.empty(n); smooth_var = np.empty(n)
    smooth_mean[-1], smooth_var[-1] = filt_mean[-1], filt_var[-1]
    for idx in range(n - 2, -1, -1):
        gain = filt_var[idx] * phi / max(pred_var[idx + 1], 1e-10)
        smooth_mean[idx] = filt_mean[idx] + gain * (smooth_mean[idx + 1] - pred_mean[idx + 1])
        smooth_var[idx] = max(filt_var[idx] + gain * gain * (smooth_var[idx + 1] - pred_var[idx + 1]), 1e-10)
    return float(loglik), np.clip(smooth_mean, -18.0, 18.0), smooth_var


def _fast_bdes_delta_method_sigma_samples(*, state_level: float, state_phi: float, state_eta: float, last_log_var: float, rho: float, h_path: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
    h = np.asarray(h_path, dtype=np.float64)
    h = h[np.isfinite(h)]
    baseline = (float(state_level), float(np.clip(state_phi, 0.0, 0.995)), float(np.clip(state_eta, 0.02, 2.50)), float(np.clip(last_log_var, -18.0, 18.0)), float(np.clip(rho, -0.95, 0.95)))
    if h.size < 60:
        return [baseline]
    phi, eta, n = baseline[1], baseline[2], float(h.size)
    n_eff = max(8.0, min(n, n * (1.0 - phi) / max(1.0 + phi, 1e-6)))
    level_se = eta / (max(1.0 - phi, 0.01) * math.sqrt(n_eff))
    phi_se = math.sqrt(max(1.0 - phi * phi, 1e-6) / n_eff)
    eta_log_se = math.sqrt(1.0 / (2.0 * n_eff))
    last_state_se = math.sqrt(max(1.0 / ((1.0 / max(eta * eta, 1e-10)) + (1.0 / max(SV_LOG_CHI_SQUARE_VAR, 1e-10))), 1e-10))
    rho_se = math.sqrt(max((1.0 - baseline[4] * baseline[4]) ** 2 / max(n_eff - 1.0, 1.0), 1e-10))
    raw = [baseline, (baseline[0] - level_se, baseline[1], baseline[2], baseline[3], baseline[4]), (baseline[0] + level_se, baseline[1], baseline[2], baseline[3], baseline[4]), (baseline[0], baseline[1] - phi_se, baseline[2], baseline[3], baseline[4]), (baseline[0], baseline[1] + phi_se, baseline[2], baseline[3], baseline[4]), (baseline[0], baseline[1], baseline[2] * math.exp(-eta_log_se), baseline[3], baseline[4]), (baseline[0], baseline[1], baseline[2] * math.exp(eta_log_se), baseline[3], baseline[4]), (baseline[0], baseline[1], baseline[2], baseline[3] - last_state_se, baseline[4]), (baseline[0], baseline[1], baseline[2], baseline[3] + last_state_se, baseline[4]), (baseline[0], baseline[1], baseline[2], baseline[3], baseline[4] - rho_se), (baseline[0], baseline[1], baseline[2], baseline[3], baseline[4] + rho_se)]
    samples: List[Tuple[float, float, float, float, float]] = []
    seen: set[Tuple[float, float, float, float, float]] = set()
    for values in raw:
        sample = (float(np.clip(values[0], -18.0, 18.0)), float(np.clip(values[1], 0.0, 0.995)), float(np.clip(values[2], 0.02, 2.50)), float(np.clip(values[3], -18.0, 18.0)), float(np.clip(values[4], -0.95, 0.95)))
        key = tuple(round(value, 12) for value in sample)
        if all(np.isfinite(value) for value in sample) and key not in seen:
            seen.add(key); samples.append(sample)
    return samples or [baseline]


def _standardized_empirical_innovation_pool(values: np.ndarray, *, clip: Optional[float] = None, method: str = "mean_std") -> Optional[np.ndarray]:
    z = np.asarray(values, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    if str(method) == "median_mad":
        z = z - float(np.median(z)); sd = 1.4826 * float(np.median(np.abs(z)))
    else:
        z = z - float(np.mean(z)); sd = float(np.std(z, ddof=1)) if z.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    z = z / sd
    return np.clip(z, -float(clip), float(clip)).astype(np.float64) if clip is not None else z.astype(np.float64)


def _finite_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=np.float64); y = np.asarray(y_values, dtype=np.float64); n = min(x.size, y.size)
    if n < 3: return math.nan
    mask = np.isfinite(x[:n]) & np.isfinite(y[:n]); x = x[:n][mask]; y = y[:n][mask]
    if x.size < 3 or np.std(x, ddof=1) <= 1e-12 or np.std(y, ddof=1) <= 1e-12: return math.nan
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else math.nan


def _bdes_multiscale_half_lives(history_length: int, k_star: int, scale_grid: str) -> np.ndarray:
    n = int(history_length)
    if scale_grid == BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC:
        requested = [5.0, 21.0, 63.0, 126.0, 252.0]
        next_scale = 504.0
        while next_scale <= float(n): requested.append(next_scale); next_scale *= 2.0
        half_lives = np.asarray([scale for scale in requested if scale <= float(n)], dtype=float)
        if half_lives.size == 0: raise ValueError("frontier_calendar_multiscale_requires_observations")
        return half_lives
    half_lives = np.array([5.0, 21.0, 63.0, 252.0, 756.0, 1512.0], dtype=float)[: int(k_star)]
    return np.minimum(half_lives, max(float(n) / 2.0, 5.0))


def _bdes_ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(x, dtype=float); out = np.empty_like(values); finite = values[np.isfinite(values)]; level = float(np.nanmedian(finite)) if finite.size else 0.0
    for idx, value in enumerate(values):
        if np.isfinite(value): level = alpha * float(value) + (1.0 - alpha) * level
        out[idx] = level
    return out


def _bdes_multiscale_components(h_path: np.ndarray, k_star: int, scale_grid: str = BDES_MULTISCALE_GRID_FIXED) -> Dict[str, Any]:
    h = np.asarray(h_path, dtype=float); finite = h[np.isfinite(h)]; fill = float(np.nanmedian(finite)) if finite.size else 0.0
    h = np.clip(np.nan_to_num(h, nan=fill, posinf=fill, neginf=fill), -18.0, 18.0); n = h.size; ell = float(np.nanmean(h)); centered = h - ell
    half_lives = _bdes_multiscale_half_lives(n, k_star, scale_grid); phis = np.exp(-np.log(2.0) / np.maximum(half_lives, 2.0)); q = np.empty((n, phis.size), dtype=float)
    for k, phi in enumerate(phis):
        q[:, k] = _bdes_ewma(centered, 1.0 - phi); q[:, k] -= float(np.nanmean(q[:, k]))
    ridge = 0.05 * max(float(np.nanvar(centered)), 1e-8) * np.eye(phis.size)
    try: b = np.linalg.solve(q.T @ q + ridge, q.T @ centered); b = b if np.all(np.isfinite(b)) and float(np.max(np.abs(b))) <= 1e6 else np.zeros(phis.size)
    except np.linalg.LinAlgError: b = np.zeros(phis.size)
    with np.errstate(all="ignore"):
        component = q @ b
    component = component if np.all(np.isfinite(component)) else np.zeros_like(centered); resid = centered - component
    signal = np.abs(b) * np.nanstd(q, axis=0)
    if np.sum(signal) > 0.0:
        shrink = signal / (signal + np.nanmedian(signal[signal > 0.0]) + 1e-8); b = b * shrink
        with np.errstate(all="ignore"):
            component = q @ b
        component = component if np.all(np.isfinite(component)) else np.zeros_like(centered); resid = centered - component
    component_var = max(float(np.nanvar(component)), 0.0); resid_var = max(float(np.nanvar(resid)), 1e-8); reliability_weight = float(component_var / max(component_var + resid_var, 1e-8)); post_signal = np.abs(b) * np.nanstd(q, axis=0)
    dominant_half_life = float(np.sum(post_signal * half_lives) / np.sum(post_signal)) if np.sum(post_signal) > 0.0 else float(np.median(half_lives))
    inverse_mse_weight = reliability_weight
    if n > 8:
        lagged, actual = centered[:-1], centered[1:]; denom = float(np.dot(lagged, lagged)); ar_phi = float(np.clip(np.dot(lagged, actual) / denom, -0.999, 0.999)) if denom > 1e-12 else 0.0; ar_pred = ar_phi * lagged
        with np.errstate(all="ignore"):
            multiscale_pred = (q[:-1, :] * phis.reshape(1, -1)) @ b
        finite_mask = np.isfinite(actual) & np.isfinite(ar_pred) & np.isfinite(multiscale_pred)
        if np.count_nonzero(finite_mask) >= 8:
            ar_mse = max(float(np.mean((actual[finite_mask] - ar_pred[finite_mask]) ** 2)), 1e-8); multi_mse = max(float(np.mean((actual[finite_mask] - multiscale_pred[finite_mask]) ** 2)), 1e-8); inverse_mse_weight = float(ar_mse / (ar_mse + multi_mse))
    q_var = np.empty(phis.size); q_innov = np.empty((max(n - 1, 0), phis.size))
    for k, phi in enumerate(phis): q_innov[:, k] = q[1:, k] - phi * q[:-1, k]; q_var[k] = max(float(np.nanvar(q_innov[:, k])), 1e-8)
    residual = resid - float(np.nanmean(resid)); residual_lag, residual_next = residual[:-1], residual[1:]; denom = float(np.dot(residual_lag, residual_lag)); residual_phi = float(np.clip(np.dot(residual_lag, residual_next) / denom, -0.999999, 0.999999)) if denom > np.finfo(np.float64).tiny else 0.0; residual_innovations = residual_next - residual_phi * residual_lag; residual_innovation_sd = max(float(np.nanstd(residual_innovations, ddof=1)), np.finfo(np.float64).tiny); q_innovation_sd = np.maximum(np.nanstd(q_innov, axis=0, ddof=1), np.finfo(np.float64).tiny); standardized_q = q_innov / q_innovation_sd.reshape(1, -1); common_innovation = np.nanmean(standardized_q, axis=1); common_innovation -= float(np.nanmean(common_innovation)); common_sd = float(np.nanstd(common_innovation, ddof=1)); common_innovation = common_innovation / common_sd if common_sd > np.finfo(np.float64).tiny else np.zeros_like(common_innovation); standardized_residual = residual_innovations / residual_innovation_sd; common_denom = float(np.dot(common_innovation, common_innovation)); residual_common_loading = float(np.clip(np.dot(common_innovation, standardized_residual) / common_denom, -1.0, 1.0)) if common_denom > np.finfo(np.float64).tiny else 0.0
    h_q005, h_q25, h_q75, h_q995 = np.quantile(h, [0.005, 0.25, 0.75, 0.995]); h_iqr = max(float(h_q75 - h_q25), 1e-6)
    return {"ell": ell, "phis": phis, "b": b, "q_last": q[-1, :].copy(), "q_var": q_var, "half_lives": half_lives.copy(), "scale_count": int(half_lives.size), "hbar": float(np.nanmean(h)), "h_low": float(h_q005 - h_iqr), "h_high": float(h_q995 + h_iqr), "component_low": float(np.quantile(component, 0.01)), "component_high": float(np.quantile(component, 0.99)), "resid_var": resid_var, "residual_last": float(residual[-1]), "residual_phi": residual_phi, "residual_innovation_sd": residual_innovation_sd, "residual_common_loading": residual_common_loading, "component_var": component_var, "reliability_weight": reliability_weight, "inverse_mse_weight": inverse_mse_weight, "dominant_half_life_days": dominant_half_life, "max_half_life_days": float(np.max(half_lives))}


def fit_bdes_fastmap(log_returns: np.ndarray, candidate: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Fit the exact promoted historical FastMAP candidate for one asset."""
    settings = dict(FRONTIER_CANDIDATE if candidate is None else candidate)
    if settings != FRONTIER_CANDIDATE:
        unknown = sorted(set(settings) - set(FRONTIER_CANDIDATE))
        mismatched = sorted(key for key in FRONTIER_CANDIDATE if settings.get(key) != FRONTIER_CANDIDATE[key])
        raise ValueError(f"frontier_unknown_or_mismatched_candidate:{unknown + mismatched}")
    x = np.asarray(log_returns, dtype=np.float64); x = x[np.isfinite(x)]
    if x.size < FULL_MCMC_SV_MIN_OBS:
        raise ValueError("frontier_fastmap_requires_five_observations")
    sample_mu = float(np.mean(x)); sample_sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sample_sigma) or sample_sigma <= 1e-10:
        raise ValueError("frontier_fastmap_degenerate_history")
    daily_to_annual_sharpe = math.sqrt(252.0) / sample_sigma; annual_sharpe_to_daily_mu = sample_sigma / math.sqrt(252.0); anchor_sharpe = sample_mu * daily_to_annual_sharpe; observed_sharpe = x * daily_to_annual_sharpe; centered_sharpe = observed_sharpe - anchor_sharpe; obs_var = float(np.var(centered_sharpe, ddof=1)); obs_var = obs_var if np.isfinite(obs_var) and obs_var > 1e-16 else 252.0
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x); se_mu = se_mu if np.isfinite(se_mu) and se_mu > 0.0 else sample_sigma / math.sqrt(float(x.size)); se_sharpe = se_mu * daily_to_annual_sharpe; params = _estimate_evidence_dlm_drift_params(centered_sharpe, obs_var=obs_var, signal_ratio_start=(se_sharpe / max(math.sqrt(obs_var), np.finfo(np.float64).tiny)) ** 2); phi_dlm = float(params["phi"]); ratio = float(params["state_var_ratio"]); filtered = _kalman_ar1_drift_filter(centered_sharpe, obs_var=obs_var, phi=phi_dlm, state_var_ratio=ratio)
    if not np.isfinite(float(filtered.get("loglik", -math.inf))):
        raise ValueError("frontier_fastmap_dlm_fit_failed")
    predicted_daily_mean = (anchor_sharpe + np.asarray(filtered["predicted_mean"], dtype=np.float64)) * annual_sharpe_to_daily_mu; residuals = x - predicted_daily_mean; z = residuals / sample_sigma; z = z[np.isfinite(z)]; z -= float(np.mean(z)); z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0; z = z / z_sd if np.isfinite(z_sd) and z_sd > 1e-12 else z
    posterior_mean = float(sample_mu + float(filtered["final_mean"]) * annual_sharpe_to_daily_mu); posterior_sd = float(math.sqrt(max(float(filtered["final_var"]) * annual_sharpe_to_daily_mu**2, 0.0)))
    observed_log_var, eps_x = _sv_observed_log_variance(x, sample_mu, winsorize=True)
    if observed_log_var.size < FULL_MCMC_SV_MIN_OBS or eps_x.size < FULL_MCMC_SV_MIN_OBS: raise ValueError("frontier_fastmap_invalid_sv_observations")
    level, phi, eta = _fit_sv_map_state_space_params(observed_log_var, start_count=1, maxiter=60); loglik, filtered_log_var, filtered_var = _sv_kalman_filter(observed_log_var, level, phi, eta, return_path=True)
    if not np.isfinite(loglik) or filtered_log_var.size < FULL_MCMC_SV_MIN_OBS: raise ValueError("frontier_fastmap_sv_fit_failed")
    smoother_loglik, smoother_path, smoother_var = _sv_kalman_rts_smoother_mean(observed_log_var, level, phi, eta); state_path, state_var_path, state_loglik = (smoother_path, smoother_var, smoother_loglik) if np.isfinite(smoother_loglik) and smoother_path.size >= FULL_MCMC_SV_MIN_OBS else (filtered_log_var, filtered_var, loglik)
    z_pool = _standardized_empirical_innovation_pool(eps_x, clip=None, method="mean_std")
    if z_pool is None or z_pool.size < FULL_MCMC_SV_MIN_OBS: raise ValueError("frontier_fastmap_invalid_innovation_pool")
    state_innov = (state_path[1:] - level - phi * (state_path[:-1] - level)) / max(eta, 1e-8); rho = _finite_correlation(z_pool[: state_innov.size], state_innov) if state_innov.size >= 4 else 0.0; rho = float(rho if np.isfinite(rho) and abs(rho) < 1.0 else np.clip(rho if np.isfinite(rho) else 0.0, -0.95, 0.95))
    posterior_samples = _fast_bdes_delta_method_sigma_samples(state_level=level, state_phi=phi, state_eta=eta, last_log_var=float(state_path[-1]), rho=rho, h_path=state_path); bdes = _bdes_multiscale_components(state_path, 4, BDES_MULTISCALE_GRID_FIXED)
    return {**settings, "mu": posterior_mean, "residuals": (x - sample_mu).astype(np.float64), "base_fit": {"sample_mu": sample_mu, "sigma": sample_sigma, "posterior_mean": posterior_mean, "posterior_sd": posterior_sd, "posterior_mu_draws": False, "dlm_drift_paths": True, "dlm_long_run_anchor_mean": sample_mu, "dlm_state_transition_phi": phi_dlm, "dlm_state_noise_var": float(filtered["state_noise_var"]) * annual_sharpe_to_daily_mu**2, "dlm_state_posterior_deviation_mean": float(filtered["final_mean"]) * annual_sharpe_to_daily_mu, "dlm_state_posterior_deviation_var": float(filtered["final_var"]) * annual_sharpe_to_daily_mu**2, "standardized_residuals": np.clip(z, -20.0, 20.0), "sample_mean": sample_mu}, "innovation_pool": z_pool, "posterior_samples": posterior_samples, "posterior_center": (float(level), float(phi), float(eta), float(state_path[-1]), float(rho)), "bdes_multiscale_vol": bdes, "unclipped_empirical_innovations": True, "leverage": True, "leverage_alignment": "lagged_return", "sv_sigma_scale": 1.0, "state_loglikelihood": float(state_loglik), "state_path_variance_last": float(state_var_path[-1]), "n_obs": int(x.size), "mean_meta": {"method": "evidence_estimated_ar1_latent_sharpe_dlm_with_historical_cagr_anchor", "sample_mean": sample_mu, "sample_sigma": sample_sigma, "hac_bandwidth": int(bandwidth), "hac_long_run_variance": float(long_run_var), "historical_cagr_anchor": True}}


def simulate_full_mcmc_sv_bdes_log_paths_serial_numba(
    z_draws: np.ndarray,
    path_mu: np.ndarray,
    levels: np.ndarray,
    phis: np.ndarray,
    etas: np.ndarray,
    initial_log_var: np.ndarray,
    rhos: np.ndarray,
    prev_z_values: np.ndarray,
    base_state_shocks: np.ndarray,
    residual_shocks: np.ndarray,
    q_last: np.ndarray,
    q_phis: np.ndarray,
    q_b: np.ndarray,
    q_sd: np.ndarray,
    q_ell: float,
    residual_last: float,
    residual_phi: float,
    residual_innovation_sd: float,
    residual_common_loading: float,
    q_low: float,
    q_high: float,
    center_level: float,
    center_phi: float,
    center_eta: float,
    center_initial_log_var: float,
    center_rho: float,
    sigma_scale: float,
    mean_sigma_denom: float,
    use_leverage: bool,
    same_period_leverage: bool,
    posterior_centered_multiscale: bool,
    mean_state_scaling_code: int,
    unclipped_sv_measurement: bool,
    clip_simulated_returns: bool,
) -> np.ndarray:
    """Serial alias of the historical BDES path kernel.

    The later production wrapper exposes this name when nested parallel
    workers are active.  Keeping the same positional contract here avoids an
    accidental switch to a different recurrence if a caller requests it.
    """
    z_values = np.asarray(z_draws, dtype=np.float64)
    path_values = np.asarray(path_mu, dtype=np.float64)
    n_paths, total_days = z_values.shape
    if path_values.shape != (n_paths, total_days):
        raise ValueError("frontier_serial_bdes_mean_path_shape_mismatch")
    q_last = np.asarray(q_last, dtype=np.float64)
    q_phis = np.asarray(q_phis, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    q_sd = np.asarray(q_sd, dtype=np.float64)
    out = np.empty((n_paths, total_days), dtype=np.float64)
    for row_idx in range(n_paths):
        level = float(levels[row_idx]); phi = float(phis[row_idx]); eta = float(etas[row_idx]); log_var = float(initial_log_var[row_idx]); rho = float(rhos[row_idx]); prev_z = float(prev_z_values[row_idx]); q_state = q_last.copy(); residual_state = float(residual_last)
        leverage_scale = math.sqrt(max(1.0 - rho * rho, 1e-8)); center_leverage_scale = math.sqrt(max(1.0 - center_rho * center_rho, 1e-8)); sample_ar1_log_var = log_var; center_ar1_log_var = float(center_initial_log_var)
        for day_idx in range(total_days):
            z = float(z_values[row_idx, day_idx]); base_state_shock = float(base_state_shocks[row_idx, day_idx])
            if use_leverage:
                leverage_driver = z if same_period_leverage else prev_z; state_shock = rho * leverage_driver + leverage_scale * base_state_shock
            else:
                state_shock = base_state_shock
            if posterior_centered_multiscale:
                if use_leverage:
                    leverage_driver = z if same_period_leverage else prev_z; center_state_shock = center_rho * leverage_driver + center_leverage_scale * base_state_shock
                else:
                    center_state_shock = base_state_shock
            else:
                center_state_shock = state_shock
            if q_last.size > 0:
                residual_share = max(1.0 - residual_common_loading * residual_common_loading, 0.0)
                residual_shock = residual_common_loading * center_state_shock + math.sqrt(residual_share) * float(residual_shocks[row_idx, day_idx])
                residual_state = residual_phi * residual_state + residual_innovation_sd * residual_shock
                log_var_day = float(q_ell + residual_state)
                for q_idx in range(q_last.size):
                    q_state[q_idx] = q_phis[q_idx] * q_state[q_idx] + q_sd[q_idx] * center_state_shock
                    log_var_day += q_state[q_idx] * q_b[q_idx]
                log_var_day = float(np.clip(log_var_day, q_low, q_high))
                if posterior_centered_multiscale:
                    sample_ar1_log_var = level + phi * (sample_ar1_log_var - level) + eta * state_shock
                    center_ar1_log_var = center_level + center_phi * (center_ar1_log_var - center_level) + center_eta * center_state_shock
                    log_var = log_var_day + sample_ar1_log_var - center_ar1_log_var
                else:
                    log_var = log_var_day
            else:
                log_var = level + phi * (log_var - level) + eta * state_shock
            if not unclipped_sv_measurement:
                log_var = float(np.clip(log_var, -18.0, 18.0))
            sigma = sigma_scale * math.exp(0.5 * log_var) / 100.0
            day_mu = float(path_values[row_idx, day_idx])
            if mean_state_scaling_code == 1:
                day_mu = day_mu * sigma / mean_sigma_denom
            elif mean_state_scaling_code == 2:
                ratio = sigma / mean_sigma_denom; day_mu = day_mu * ratio * ratio
            value = day_mu + sigma * z
            out[row_idx, day_idx] = float(np.clip(value, -1.0, 1.0)) if clip_simulated_returns else value
            prev_z = z
    return out


def simulate_fastmap_marginal(fit: Mapping[str, Any], simulations: int, horizon: int, seed: int) -> np.ndarray:
    n_paths, total_days = int(simulations), int(horizon)
    if n_paths <= 0 or total_days <= 0: return np.empty((max(n_paths, 0), max(total_days, 0)), dtype=np.float64)
    samples = list(fit.get("posterior_samples") or []); z_pool = np.asarray(fit.get("innovation_pool"), dtype=np.float64); z_pool = z_pool[np.isfinite(z_pool)]
    if not samples or z_pool.size < FULL_MCMC_SV_MIN_OBS: raise ValueError("frontier_fastmap_missing_fit_state")
    rng = np.random.default_rng(int(seed)); block_length = max(politis_white_block_length(z_pool), politis_white_block_length(z_pool * z_pool)); z_indices = stationary_bootstrap_indices(z_pool.size, block_length, total_days, n_paths, rng); z_draws = z_pool[z_indices]
    sample_idx = rng.integers(0, len(samples), size=n_paths); levels = np.asarray([samples[int(idx)][0] for idx in sample_idx]); phis = np.asarray([samples[int(idx)][1] for idx in sample_idx]); etas = np.asarray([samples[int(idx)][2] for idx in sample_idx]); log_var = np.asarray([samples[int(idx)][3] for idx in sample_idx]); rhos = np.asarray([samples[int(idx)][4] for idx in sample_idx]); rhos = np.where(np.isfinite(rhos), rhos, 0.0)
    # The historical simulator draws the DLM mean paths before the leverage
    # and residual shock matrices.  Keep that ordering so the seed alias is
    # byte-for-byte compatible with the academic wrapper.
    path_mu = dlm_mu_draw_paths(fit["base_fit"], total_days, n_paths, rng)
    if path_mu is None:
        raise ValueError("frontier_fastmap_missing_dlm_mean_paths")
    prev_z = rng.choice(z_pool, size=n_paths, replace=True); base_state_shocks = rng.normal(0.0, 1.0, size=(n_paths, total_days)); residual_shocks = rng.normal(0.0, 1.0, size=(n_paths, total_days)); q = fit["bdes_multiscale_vol"]; q_state = np.tile(np.asarray(q["q_last"], dtype=np.float64).reshape(1, -1), (n_paths, 1)); q_phis = np.asarray(q["phis"], dtype=np.float64).reshape(1, -1); q_b = np.asarray(q["b"], dtype=np.float64); q_sd = np.sqrt(np.maximum(np.asarray(q["q_var"], dtype=np.float64), 1e-10)).reshape(1, -1); residual_state = np.full(n_paths, float(q["residual_last"]))
    out = np.empty((n_paths, total_days), dtype=np.float64); mean_sigma = max(float(fit["base_fit"]["sigma"]), np.finfo(np.float64).tiny); residual_phi = float(q["residual_phi"]); residual_sd = float(q["residual_innovation_sd"]); common_loading = float(np.clip(q["residual_common_loading"], -1.0, 1.0)); level_q, high_q = float(q["h_low"]), float(q["h_high"]); q_ell = float(q["ell"])
    for day in range(total_days):
        z = z_draws[:, day]; state_shock = rhos * prev_z + np.sqrt(np.maximum(1.0 - rhos * rhos, 1e-8)) * base_state_shocks[:, day]; q_state = q_phis * q_state + q_sd * state_shock[:, None]; residual_shock = common_loading * state_shock + math.sqrt(max(1.0 - common_loading**2, 0.0)) * residual_shocks[:, day]; residual_state = residual_phi * residual_state + residual_sd * residual_shock; log_var_day = np.clip(q_ell + q_state @ q_b + residual_state, level_q, high_q); sigma = np.exp(0.5 * log_var_day) / 100.0; day_mu = path_mu[:, day] * sigma / mean_sigma; out[:, day] = np.clip(day_mu + sigma * z, -1.0, 1.0); prev_z = z
    if not np.all(np.isfinite(out)): raise ValueError("frontier_fastmap_nonfinite_paths")
    return out


# Compatibility aliases used by the historical/source-main numerical-kernel
# selector.  They point to the same serial recurrence and do not introduce a
# second implementation or a cache-writing JIT dependency.
_simulate_full_mcmc_sv_bdes_log_paths_numba = simulate_full_mcmc_sv_bdes_log_paths_serial_numba
_simulate_full_mcmc_sv_bdes_log_paths_serial_numba = simulate_full_mcmc_sv_bdes_log_paths_serial_numba


__all__ = ["FRONTIER_CANDIDATE", "FRONTIER_DEPENDENCE_ID", "FRONTIER_MARGINAL_ID", "deterministic_seed", "fit_bdes_fastmap", "simulate_fastmap_marginal", "simulate_full_mcmc_sv_bdes_log_paths_serial_numba"]
