"""Exact source port of the 40 canonical full latent MCMC SV overlays.

The numerical kernels below are extracted from the pinned research harness
(source-research/scripts/forecast_oos_research_gate.py) and retain its
priors, filtering equations, Metropolis proposals, stopping gates, and RNG
seed contexts.  They are deliberately standalone: no source checkout or live
provider is imported at runtime.
"""
# ruff: noqa

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...runner import ForecastContext, ForecastModel, TrainingData
from ...seeds import deterministic_seed

# Pinned source constants.
_FULL_MCMC_SV_PARAMETRIC_INNOVATION_META: Dict[str, Tuple[str, str]] = {
    "jones_faddy_skew_t_iid": (
        "jones_faddy_skew_t",
        "jones_faddy_skew_t_innovations_scaled_by_latent_sv_paths",
    ),
    "genhyperbolic_iid": (
        "generalized_hyperbolic_skew_t",
        "generalized_hyperbolic_innovations_scaled_by_latent_sv_paths",
    ),
    "normal_inverse_gaussian_iid": (
        "normal_inverse_gaussian",
        "normal_inverse_gaussian_innovations_scaled_by_latent_sv_paths",
    ),
}

SV_LOG_CHI_SQUARE_MEAN = -1.2703628454614782
SV_LOG_CHI_SQUARE_VAR = (math.pi * math.pi) / 2.0
FULL_MCMC_SV_MIN_OBS = 60
DLM_DRIFT_PHI_MIN = 0.0
DLM_DRIFT_PHI_MAX = 0.9995
DLM_DRIFT_STATE_VAR_RATIO_MIN = 1.0e-10
DLM_DRIFT_MAX_ANNUAL_DRIFT_SHARPE_SD = 2.5
DLM_DRIFT_STATE_VAR_RATIO_MAX = float((DLM_DRIFT_MAX_ANNUAL_DRIFT_SHARPE_SD**2) / 252.0)
CONTINUOUS_META_DECAY_SOURCE_RESULT = (
    ".forecast-oos-runs/consolidated-no-crypto/"
    "targeted_horizon_drift_instability_controls_33orig_300sims_equal_class_1979_80p_20260528.json"
)
CONTINUOUS_META_DECAY_FEATURE_NAMES: Tuple[str, ...] = (
    "log_years", "log1p_abs_t", "sign_t", "abs_t_cap5_scaled", "log_ann_vol",
    "ac1", "skew_scaled", "exk_scaled", "sign_stability_minus_half",
    "log1p_block_mean_sd_over_se", "log1p_maxdd", "log1p_hac_bandwidth",
)
CONTINUOUS_META_DECAY_COEFFICIENTS: Tuple[float, ...] = (
    0.248559776, -0.741480825, 0.036090925, 0.102097678, 0.911871872,
    0.170420659, 0.034359381, 0.042140635, -0.028533942, 0.054180728,
    -0.087225952, 0.153349228, 0.130478606,
)
CONTINUOUS_META_DECAY_FEATURE_MEANS: Tuple[float, ...] = (
    2.956206828, 1.470444913, 0.963636364, 0.714927087, -2.169969251,
    0.064039619, -0.056602222, 0.421587682, 0.198331028, 2.126861217,
    0.329667776, 4.450456529,
)
CONTINUOUS_META_DECAY_FEATURE_SDS: Tuple[float, ...] = (
    0.78008805, 0.413778809, 0.267217063, 0.278421883, 0.233213381,
    0.057595401, 0.069899272, 0.255845881, 0.089217851, 0.190956608,
    0.11049454, 0.478150744,
)
CONTINUOUS_META_DECAY_MIN_SCALE = 0.25
CONTINUOUS_META_DECAY_MAX_SCALE = 6.0

def _full_mcmc_sv_parametric_innovation_meta(innovation_resampling: str) -> Optional[Tuple[str, str]]:
    return _FULL_MCMC_SV_PARAMETRIC_INNOVATION_META.get(str(innovation_resampling))

def _fit_full_mcmc_sv_parametric_innovation_distribution(
    z_pool: np.ndarray,
    innovation_resampling: str,
) -> Optional[Tuple[float, ...]]:
    if _full_mcmc_sv_parametric_innovation_meta(innovation_resampling) is None:
        return None
    try:
        from scipy import stats

        fit_pool = np.asarray(z_pool, dtype=np.float64)
        fit_pool = np.clip(fit_pool[np.isfinite(fit_pool)], -30.0, 30.0)
        if fit_pool.size < 60:
            return None
        if innovation_resampling == "jones_faddy_skew_t_iid":
            params = stats.jf_skew_t.fit(fit_pool)
        elif innovation_resampling == "genhyperbolic_iid":
            params = stats.genhyperbolic.fit(fit_pool)
        elif innovation_resampling == "normal_inverse_gaussian_iid":
            params = stats.norminvgauss.fit(fit_pool)
        else:
            return None
        innovation_distribution_params = tuple(float(value) for value in params)
        if not all(np.isfinite(value) for value in innovation_distribution_params):
            return None
        return innovation_distribution_params
    except Exception:
        return None

def _draw_full_mcmc_sv_parametric_innovations(
    raw_params: Sequence[float],
    innovation_resampling: str,
    shape: Tuple[int, int],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    try:
        from scipy import stats

        params = tuple(float(value) for value in raw_params)
        if not params or not all(np.isfinite(value) for value in params):
            return None
        if innovation_resampling == "jones_faddy_skew_t_iid":
            draws = stats.jf_skew_t.rvs(*params, size=shape, random_state=rng)
        elif innovation_resampling == "genhyperbolic_iid":
            draws = stats.genhyperbolic.rvs(*params, size=shape, random_state=rng)
        elif innovation_resampling == "normal_inverse_gaussian_iid":
            draws = stats.norminvgauss.rvs(*params, size=shape, random_state=rng)
        else:
            return None
        z_draws = np.asarray(draws, dtype=np.float64)
        return np.clip(np.where(np.isfinite(z_draws), z_draws, 0.0), -30.0, 30.0)
    except Exception:
        return None

def _mean_schedule_from_fit(fit: Dict[str, Any], total_days: int) -> Optional[np.ndarray]:
    total_days = int(total_days)
    if total_days <= 0:
        return None
    decay_meta = fit.get("posterior_mean_decay_meta")
    if isinstance(decay_meta, dict):
        short_mu = float(decay_meta.get("short_posterior_mean", fit.get("posterior_mean", 0.0)))
        long_mu = float(decay_meta.get("long_posterior_mean", short_mu))
        uncertainty_ratio = float(decay_meta.get("mean_uncertainty_to_process_variance_ratio", 0.0))
        if not np.isfinite(short_mu) or not np.isfinite(long_mu) or not np.isfinite(uncertainty_ratio):
            return None
        day_index = np.arange(1, total_days + 1, dtype=np.float64)
        weight = 1.0 / (1.0 + day_index * max(uncertainty_ratio, 0.0))
        schedule = long_mu + weight * (short_mu - long_mu)
        return schedule.astype(np.float64) if np.all(np.isfinite(schedule)) else None

    segments = fit.get("posterior_mean_schedule_segments")
    if not isinstance(segments, list) or not segments:
        return None
    schedule = np.empty(total_days, dtype=np.float64)
    last_mu = float(fit.get("posterior_mean", 0.0))
    start = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        mu = float(segment.get("posterior_mean", last_mu))
        through_day = segment.get("through_day")
        if through_day is None:
            end = total_days
        else:
            end = int(max(start, min(total_days, int(through_day))))
        if end > start:
            schedule[start:end] = mu
            start = end
        last_mu = mu
        if start >= total_days:
            break
    if start < total_days:
        schedule[start:] = last_mu
    if not np.all(np.isfinite(schedule)):
        return None
    return schedule

def _posterior_decay_mu_draw_paths(
    fit: Dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if not bool(fit.get("posterior_mu_draws_with_horizon_decay", False)):
        return None
    total_days = int(total_days)
    n_paths = int(n_paths)
    if total_days <= 0 or n_paths <= 0:
        return None
    decay_meta = fit.get("posterior_mean_decay_meta")
    if not isinstance(decay_meta, dict):
        return None
    short_mu = float(fit.get("posterior_mean", decay_meta.get("short_posterior_mean", 0.0)))
    long_mu = float(decay_meta.get("long_posterior_mean", 0.0))
    uncertainty_ratio = float(decay_meta.get("mean_uncertainty_to_process_variance_ratio", 0.0))
    posterior_sd = float(max(fit.get("posterior_sd", 0.0), 0.0))
    if not np.isfinite(short_mu) or not np.isfinite(long_mu) or not np.isfinite(uncertainty_ratio):
        return None
    mu_draws = rng.normal(short_mu, posterior_sd, size=n_paths)
    cap = float(max(fit.get("mu_cap", 0.0), 0.0))
    if cap > 0.0:
        mu_draws = np.clip(mu_draws, -cap, cap)
    if bool(fit.get("nonnegative_drift", False)):
        mu_draws = np.maximum(mu_draws, 0.0)
    day_index = np.arange(1, total_days + 1, dtype=np.float64)
    weight = 1.0 / (1.0 + day_index * max(uncertainty_ratio, 0.0))
    paths = long_mu + weight[None, :] * (mu_draws[:, None] - long_mu)
    return paths.astype(np.float64) if np.all(np.isfinite(paths)) else None

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
    m = 0.0
    p = stationary_var
    loglik = 0.0
    log_two_pi = math.log(2.0 * math.pi)
    for idx, value in enumerate(values):
        a = phi_value * m
        r = phi_value * phi_value * p + state_noise_var
        f = r + obs
        if not np.isfinite(f) or f <= 0.0:
            return {"loglik": -math.inf}
        innovation = float(value) - a
        loglik += -0.5 * (log_two_pi + math.log(f) + (innovation * innovation) / f)
        k = r / f
        m = a + k * innovation
        p = max((1.0 - k) * r, obs * 1.0e-14)
        predicted_mean[idx] = a
        predicted_var[idx] = r
        filtered_mean[idx] = m
        filtered_var[idx] = p
    return {
        "loglik": float(loglik),
        "predicted_mean": predicted_mean,
        "predicted_var": predicted_var,
        "filtered_mean": filtered_mean,
        "filtered_var": filtered_var,
        "state_noise_var": float(state_noise_var),
        "stationary_var": float(stationary_var),
        "final_mean": float(m),
        "final_var": float(p),
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
    phi_starts = (0.25, 0.75, 0.95, 0.99)
    ratio_starts = (
        ratio_start,
        min(ratio_start * 10.0, DLM_DRIFT_STATE_VAR_RATIO_MAX),
        max(ratio_start * 0.10, DLM_DRIFT_STATE_VAR_RATIO_MIN),
    )
    best_phi = 0.95
    best_ratio = ratio_start
    best_loglik = -math.inf

    def score(phi_value: float, ratio_value: float) -> float:
        return float(_dlm_ar1_loglik(values, float(obs_var), float(phi_value), float(ratio_value)))

    try:
        from scipy import optimize

        bounds = (
            (DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX),
            (math.log(DLM_DRIFT_STATE_VAR_RATIO_MIN), math.log(DLM_DRIFT_STATE_VAR_RATIO_MAX)),
        )
        for phi0 in phi_starts:
            for ratio0 in ratio_starts:
                start = np.asarray(
                    [
                        float(np.clip(phi0, DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX)),
                        math.log(float(np.clip(ratio0, DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX))),
                    ],
                    dtype=np.float64,
                )

                def objective(theta: np.ndarray) -> float:
                    phi_value = float(theta[0])
                    ratio_value = float(math.exp(float(theta[1])))
                    loglik = score(phi_value, ratio_value)
                    return float(-loglik) if np.isfinite(loglik) else 1.0e100

                result = optimize.minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 45, "ftol": 1.0e-7, "gtol": 1.0e-5},
                )
                if not result.success and not np.isfinite(result.fun):
                    continue
                phi_hat = float(np.clip(result.x[0], DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX))
                ratio_hat = float(np.clip(math.exp(float(result.x[1])), DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX))
                loglik = score(phi_hat, ratio_hat)
                if np.isfinite(loglik) and loglik > best_loglik:
                    best_phi = phi_hat
                    best_ratio = ratio_hat
                    best_loglik = loglik
    except Exception:
        for phi_hat in (0.0, 0.50, 0.90, 0.97, 0.995):
            for ratio_hat in np.geomspace(DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX, 10):
                loglik = score(float(phi_hat), float(ratio_hat))
                if np.isfinite(loglik) and loglik > best_loglik:
                    best_phi = float(phi_hat)
                    best_ratio = float(ratio_hat)
                    best_loglik = float(loglik)

    if not np.isfinite(best_loglik):
        best_loglik = score(best_phi, best_ratio)
    return {
        "phi": float(best_phi),
        "state_var_ratio": float(best_ratio),
        "loglik": float(best_loglik),
    }

def _dlm_mu_draw_paths(
    fit: Dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if not bool(fit.get("dlm_drift_paths", False)):
        return None
    total_days = int(total_days)
    n_paths = int(n_paths)
    if total_days <= 0 or n_paths <= 0:
        return None
    anchor_mu = float(fit.get("dlm_long_run_anchor_mean", 0.0))
    phi = float(fit.get("dlm_state_transition_phi", 0.0))
    state_noise_var = float(max(fit.get("dlm_state_noise_var", 0.0), 0.0))
    state_mean = float(fit.get("dlm_state_posterior_deviation_mean", 0.0))
    state_var = float(max(fit.get("dlm_state_posterior_deviation_var", 0.0), 0.0))
    if not all(np.isfinite(value) for value in (anchor_mu, phi, state_noise_var, state_mean, state_var)):
        return None
    phi = float(np.clip(phi, DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX))
    states = rng.normal(state_mean, math.sqrt(state_var), size=n_paths)
    noise_sd = math.sqrt(state_noise_var)
    out = np.empty((n_paths, total_days), dtype=np.float64)
    for day in range(total_days):
        if noise_sd > 0.0:
            states = phi * states + rng.normal(0.0, noise_sd, size=n_paths)
        else:
            states = phi * states
        out[:, day] = anchor_mu + states
    return out.astype(np.float64) if np.all(np.isfinite(out)) else None

def _hac_mean_standard_error(values: np.ndarray) -> Tuple[float, int, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0, 0, 0.0
    centered = x - float(np.mean(x))
    sample_var = float(np.var(centered, ddof=1))
    if not np.isfinite(sample_var) or sample_var <= 0.0:
        return 0.0, 0, 0.0
    try:
        bandwidth = int(max(_politis_white_block_length(centered), _politis_white_block_length(centered * centered)))
    except Exception:
        bandwidth = int(round(float(x.size) ** (1.0 / 3.0)))
    bandwidth = int(max(0, min(bandwidth, x.size - 1)))
    long_run_var = float(np.dot(centered, centered) / x.size)
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / float(bandwidth + 1)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / x.size)
        if np.isfinite(gamma):
            long_run_var += 2.0 * weight * gamma
    if not np.isfinite(long_run_var) or long_run_var <= 0.0:
        long_run_var = sample_var
    se = math.sqrt(max(long_run_var, 0.0) / float(x.size))
    return float(se), int(bandwidth), float(long_run_var)

def _fit_empirical_bayes_sharpe_sbb(
    train_values: np.ndarray,
    *,
    nonnegative_sharpe: bool = False,
    posterior_mu_draws: bool = False,
) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None

    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = sigma / math.sqrt(float(x.size))
        bandwidth = 0
        long_run_var = sigma * sigma
    sample_sr = float(sample_mu / sigma * math.sqrt(252.0))
    se_sr = float(se_mu / sigma * math.sqrt(252.0))
    if not np.isfinite(sample_sr) or not np.isfinite(se_sr) or se_sr <= 0.0:
        return None

    # Marginal-MLE empirical-Bayes normal means shrinkage in Sharpe units:
    # observed_SR | true_SR ~ N(true_SR, se_SR^2), true_SR ~ N(0, tau^2).
    # tau^2 is estimated from this portfolio's own signal-to-noise.
    se_sr2 = float(se_sr * se_sr)
    tau2 = float(max(sample_sr * sample_sr - se_sr2, 0.0))
    shrink_weight = float(tau2 / (tau2 + se_sr2)) if tau2 > 0.0 else 0.0
    posterior_sr = float(shrink_weight * sample_sr)
    if nonnegative_sharpe:
        posterior_sr = float(max(0.0, posterior_sr))
    posterior_var_sr = float((tau2 * se_sr2) / (tau2 + se_sr2)) if tau2 > 0.0 else 0.0
    posterior_mean = float(posterior_sr * sigma / math.sqrt(252.0))
    posterior_sd = float(math.sqrt(max(posterior_var_sr, 0.0)) * sigma / math.sqrt(252.0))
    cap_sr = float(max(abs(posterior_sr) + 1.6448536269514722 * math.sqrt(max(posterior_var_sr, 0.0)), 1e-12))
    mu_cap = cap_sr * sigma / math.sqrt(252.0)

    residuals = x - sample_mu
    z = residuals / sigma
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    z = z - float(np.mean(z))
    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1e-12:
        z = z / z_sd
    return {
        "sample_mu": sample_mu,
        "sigma": sigma,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "mu_cap": mu_cap,
        "sample_mu_days": 0,
        "nonnegative_drift": bool(nonnegative_sharpe),
        "posterior_mu_draws": bool(posterior_mu_draws),
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "meta": {
            "method": "empirical_bayes_hac_sharpe_normal_means_shrinkage",
            "prior_source": "portfolio_empirical_bayes_zero_centered_sharpe",
            "cap_method": "empirical_bayes_90pct_posterior_sharpe_bound",
            "fixed_sharpe_cap": False,
            "nonnegative_sharpe_restriction": bool(nonnegative_sharpe),
            "sample_mean": sample_mu,
            "sample_sharpe_annualized": sample_sr,
            "hac_mean_standard_error": se_mu,
            "hac_bandwidth": int(bandwidth),
            "hac_long_run_variance": long_run_var,
            "sharpe_standard_error": se_sr,
            "empirical_bayes_tau2_sharpe": tau2,
            "shrink_weight": shrink_weight,
            "posterior_sharpe_annualized": posterior_sr,
            "posterior_sharpe_cap_annualized": cap_sr,
            "posterior_mean": posterior_mean,
            "posterior_sd": posterior_sd,
            "posterior_mu_draws": bool(posterior_mu_draws),
        },
    }

def _standardized_residual_fit_payload(
    x: np.ndarray,
    *,
    posterior_mean: float,
    posterior_sd: float,
    mu_cap: float,
    meta: Dict[str, Any],
    nonnegative_drift: bool = False,
    posterior_mu_draws: bool = False,
) -> Optional[Dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    residuals = x - sample_mu
    z = residuals / sigma
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    z = z - float(np.mean(z))
    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1e-12:
        z = z / z_sd
    return {
        "sample_mu": sample_mu,
        "sigma": sigma,
        "posterior_mean": float(posterior_mean),
        "posterior_sd": float(max(posterior_sd, 0.0)),
        "mu_cap": float(max(mu_cap, 0.0)),
        "sample_mu_days": 0,
        "nonnegative_drift": bool(nonnegative_drift),
        "posterior_mu_draws": bool(posterior_mu_draws),
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "meta": {
            **meta,
            "sample_mean": sample_mu,
            "posterior_mean": float(posterior_mean),
            "posterior_sd": float(max(posterior_sd, 0.0)),
            "fixed_sharpe_cap": False,
        },
    }

def _fit_hierarchical_empirical_bayes_sharpe_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 120:
        return _fit_empirical_bayes_sharpe_sbb(x)
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = sigma / math.sqrt(float(x.size))
        bandwidth = 0
        long_run_var = sigma * sigma
    sample_sr = float(sample_mu / sigma * math.sqrt(252.0))
    se_sr = float(se_mu / sigma * math.sqrt(252.0))
    if not np.isfinite(sample_sr) or not np.isfinite(se_sr) or se_sr <= 0.0:
        return None

    centered = x - sample_mu
    try:
        block_length = int(max(_politis_white_block_length(centered), _politis_white_block_length(centered * centered)))
    except Exception:
        block_length = int(round(math.sqrt(float(x.size))))
    block_length = int(max(20, min(block_length, max(20, x.size // 4))))
    block_sharpes: List[float] = []
    block_se2: List[float] = []
    for start in range(0, x.size - block_length + 1, block_length):
        block = x[start : start + block_length]
        block_sigma = float(np.std(block, ddof=1)) if block.size > 1 else 0.0
        if np.isfinite(block_sigma) and block_sigma > 1e-10:
            block_sharpes.append(float(np.mean(block) / block_sigma * math.sqrt(252.0)))
            block_se2.append(float(252.0 / max(block.size, 1)))
    if len(block_sharpes) >= 3:
        observed_var = float(np.var(np.asarray(block_sharpes, dtype=np.float64), ddof=1))
        noise_var = float(np.median(np.asarray(block_se2, dtype=np.float64)))
        tau2 = float(max(observed_var - noise_var, 0.0))
    else:
        tau2 = float(max(sample_sr * sample_sr - se_sr * se_sr, 0.0))
    shrink_weight = float(tau2 / (tau2 + se_sr * se_sr)) if tau2 > 0.0 else 0.0
    posterior_sr = float(shrink_weight * sample_sr)
    posterior_sd_sr = math.sqrt(float((tau2 * se_sr * se_sr) / (tau2 + se_sr * se_sr))) if tau2 > 0.0 else 0.0
    return _standardized_residual_fit_payload(
        x,
        posterior_mean=float(posterior_sr * sigma / math.sqrt(252.0)),
        posterior_sd=float(posterior_sd_sr * sigma / math.sqrt(252.0)),
        mu_cap=0.0,
        posterior_mu_draws=False,
        meta={
            "method": "hierarchical_empirical_bayes_sharpe_shrinkage",
            "prior_source": "portfolio_time_block_hierarchical_empirical_bayes",
            "cap_method": "none_empirical_bayes_shrinkage_only",
            "sample_sharpe_annualized": sample_sr,
            "sharpe_standard_error": se_sr,
            "hac_mean_standard_error": se_mu,
            "hac_bandwidth": int(bandwidth),
            "hac_long_run_variance": long_run_var,
            "hierarchical_block_length": int(block_length),
            "hierarchical_block_count": int(len(block_sharpes)),
            "hierarchical_tau2_sharpe": tau2,
            "shrink_weight": shrink_weight,
            "posterior_sharpe_annualized": posterior_sr,
        },
    )

def _fit_hierarchical_empirical_bayes_sharpe_mu_uncertainty_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    fit = _fit_hierarchical_empirical_bayes_sharpe_sbb(train_values)
    if fit is None:
        return None
    out = dict(fit)
    out["posterior_mu_draws"] = True
    out["meta"] = {
        **dict(fit.get("meta", {}) or {}),
        "posterior_mu_draws": True,
        "posterior_uncertainty_source": "hierarchical_empirical_bayes_posterior_sharpe_standard_deviation",
    }
    return out

def _fit_prequential_crps_shrinkage_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 120:
        return _fit_empirical_bayes_sharpe_sbb(x)
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    start = 60
    if x.size <= start + 5:
        return _fit_empirical_bayes_sharpe_sbb(x)
    prefix = np.cumsum(x, dtype=np.float64)
    history_counts = np.arange(start, x.size, dtype=np.float64)
    mu_hat = prefix[start - 1 : x.size - 1] / history_counts
    realized = x[start:]
    mask = np.isfinite(mu_hat) & np.isfinite(realized)
    mu_hat = mu_hat[mask]
    realized = realized[mask]
    if mu_hat.size < 10 or float(np.dot(mu_hat, mu_hat)) <= 1e-20:
        shrink_weight = 0.0
    else:
        # Deterministic CRPS for the drift component is absolute error; the
        # least-squares solution is used as a stable prequential score proxy.
        shrink_weight = float(np.clip(float(np.dot(mu_hat, realized) / np.dot(mu_hat, mu_hat)), 0.0, 1.0))
    posterior_mean = float(shrink_weight * sample_mu)
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    posterior_sd = float(abs(sample_mu) * math.sqrt(max(shrink_weight * (1.0 - shrink_weight), 0.0)))
    return _standardized_residual_fit_payload(
        x,
        posterior_mean=posterior_mean,
        posterior_sd=posterior_sd,
        mu_cap=0.0,
        posterior_mu_draws=False,
        meta={
            "method": "prequential_score_selected_sample_mean_shrinkage",
            "prior_source": "expanding_origin_prequential_return_score",
            "cap_method": "none_empirical_bayes_shrinkage_only",
            "prequential_score": "deterministic_crps_absolute_error_proxy_with_least_squares_stabilization",
            "prequential_min_history_days": int(start),
            "prequential_validation_count": int(mu_hat.size),
            "prequential_shrink_weight": shrink_weight,
            "sample_sharpe_annualized": float(sample_mu / sigma * math.sqrt(252.0)),
            "posterior_sharpe_annualized": float(posterior_mean / sigma * math.sqrt(252.0)),
            "hac_mean_standard_error": se_mu,
            "hac_bandwidth": int(bandwidth),
            "hac_long_run_variance": long_run_var,
        },
    )

def _fit_positive_sample_mean_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    base = _fit_historical_realized_sharpe_sbb(train_values)
    if base is None:
        return None
    out = dict(base)
    out["posterior_mean"] = float(max(0.0, float(base["posterior_mean"])))
    out["nonnegative_drift"] = True
    out["meta"] = {
        **dict(base.get("meta", {}) or {}),
        "method": "positive_part_historical_sample_mean_stationary_bootstrap_residuals",
        "restriction_source": "merton_campbell_thompson_nonnegative_expected_excess_return",
        "raw_sample_mean": float(base["sample_mu"]),
        "posterior_mean": float(out["posterior_mean"]),
    }
    return out

def _fit_merton_positive_hac_drift_uncertainty_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    base = _fit_positive_sample_mean_sbb(train_values)
    if base is None:
        return None
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    sigma = float(base.get("sigma", 0.0))
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = float(sigma / math.sqrt(float(max(x.size, 1))))
        bandwidth = 0
        long_run_var = float(sigma * sigma)
    out = dict(base)
    out["posterior_sd"] = float(max(se_mu, 0.0))
    out["posterior_mu_draws"] = True
    out["nonnegative_drift"] = True
    out["mu_cap"] = 0.0
    out["meta"] = {
        **dict(base.get("meta", {}) or {}),
        "method": "merton_positive_sample_mean_with_hac_drift_uncertainty",
        "restriction_source": "merton_campbell_thompson_nonnegative_expected_excess_return",
        "posterior_uncertainty_source": "newey_west_hac_long_run_mean_standard_error",
        "posterior_mean": float(out["posterior_mean"]),
        "posterior_sd": float(out["posterior_sd"]),
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": int(bandwidth),
        "hac_long_run_variance": float(long_run_var),
        "cap_method": "none",
    }
    return out

def _fit_unconstrained_hac_drift_uncertainty_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    base = _fit_historical_realized_sharpe_sbb(train_values)
    if base is None:
        return None
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    sigma = float(base.get("sigma", 0.0))
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = float(sigma / math.sqrt(float(max(x.size, 1))))
        bandwidth = 0
        long_run_var = float(sigma * sigma)
    out = dict(base)
    out["posterior_sd"] = float(max(se_mu, 0.0))
    out["posterior_mu_draws"] = True
    out["nonnegative_drift"] = False
    out["mu_cap"] = 0.0
    out["meta"] = {
        **dict(base.get("meta", {}) or {}),
        "method": "unconstrained_sample_mean_with_hac_drift_uncertainty",
        "restriction_source": "none_sign_allowed_realized_drift",
        "posterior_uncertainty_source": "newey_west_hac_long_run_mean_standard_error",
        "posterior_mean": float(out["posterior_mean"]),
        "posterior_sd": float(out["posterior_sd"]),
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": int(bandwidth),
        "hac_long_run_variance": float(long_run_var),
        "cap_method": "none",
    }
    return out

def _trimmed_mean(values: np.ndarray, proportion_to_cut: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    cut = int(math.floor(float(x.size) * float(np.clip(proportion_to_cut, 0.0, 0.49))))
    if cut <= 0 or 2 * cut >= x.size:
        return float(np.mean(x))
    ordered = np.sort(x)
    return float(np.mean(ordered[cut : x.size - cut]))

def _fit_robust_realized_mean_sbb(train_values: np.ndarray, trim_fraction: float = 0.10) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    robust_mu = _trimmed_mean(x, trim_fraction)
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    residuals = x - robust_mu
    z = residuals / sigma
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    z = z - float(np.mean(z))
    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1e-12:
        z = z / z_sd
    return {
        "sample_mu": robust_mu,
        "sigma": sigma,
        "posterior_mean": robust_mu,
        "posterior_sd": 0.0,
        "mu_cap": 0.0,
        "sample_mu_days": 0,
        "nonnegative_drift": False,
        "posterior_mu_draws": False,
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "meta": {
            "method": "trimmed_realized_mean_stationary_bootstrap_residuals",
            "prior_source": "raw_portfolio_history_robust_location",
            "cap_method": "none",
            "trim_fraction": float(trim_fraction),
            "sample_mean": float(np.mean(x)),
            "robust_mean": robust_mu,
            "sample_sharpe_annualized": float(robust_mu / sigma * math.sqrt(252.0)),
            "posterior_mean": robust_mu,
            "posterior_sd": 0.0,
        },
    }

def _estimate_random_effects_drift_instability(
    values: np.ndarray,
    *,
    fallback_sigma: float,
) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    sigma = float(fallback_sigma)
    empty = {
        "drift_instability_variance": 0.0,
        "block_mean_count": 0,
        "block_mean_length": 0,
        "block_mean_q_statistic": 0.0,
        "block_mean_q_degrees_freedom": 0,
        "block_mean_method": "der_simonian_laird_random_effects_on_portfolio_time_blocks",
    }
    if x.size < 120 or not np.isfinite(sigma) or sigma <= 1e-12:
        return empty

    centered = x - float(np.mean(x))
    try:
        block_length = int(
            max(
                _politis_white_block_length(centered),
                _politis_white_block_length(centered * centered),
            )
        )
    except Exception:
        block_length = int(round(math.sqrt(float(x.size))))
    block_length = int(max(20, min(block_length, max(20, x.size // 4))))
    block_count = int(x.size // block_length)
    if block_count < 3:
        return {**empty, "block_mean_length": int(block_length)}

    block_means: List[float] = []
    block_variances: List[float] = []
    block_bandwidths: List[int] = []
    eps = float(np.finfo(np.float64).tiny)
    for start in range(0, block_count * block_length, block_length):
        block = x[start : start + block_length]
        if block.size < 2:
            continue
        block_mean = float(np.mean(block))
        block_sigma = float(np.std(block, ddof=1)) if block.size > 1 else sigma
        se_block, bandwidth, _ = _hac_mean_standard_error(block)
        if not np.isfinite(se_block) or se_block <= 0.0:
            se_block = float(block_sigma / math.sqrt(float(max(block.size, 1))))
            bandwidth = 0
        if np.isfinite(block_mean) and np.isfinite(se_block) and se_block > 0.0:
            block_means.append(block_mean)
            block_variances.append(float(max(se_block * se_block, eps)))
            block_bandwidths.append(int(bandwidth))

    k = int(len(block_means))
    if k < 3:
        return {**empty, "block_mean_length": int(block_length), "block_mean_count": k}
    means = np.asarray(block_means, dtype=np.float64)
    variances = np.asarray(block_variances, dtype=np.float64)
    weights = 1.0 / np.maximum(variances, eps)
    sum_w = float(np.sum(weights))
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        return {**empty, "block_mean_length": int(block_length), "block_mean_count": k}
    fixed_mean = float(np.sum(weights * means) / sum_w)
    q_stat = float(np.sum(weights * (means - fixed_mean) ** 2))
    c_value = float(sum_w - np.sum(weights * weights) / sum_w)
    tau2 = float(max((q_stat - float(k - 1)) / c_value, 0.0)) if c_value > 0.0 else 0.0
    if not np.isfinite(tau2) or tau2 < 0.0:
        tau2 = 0.0
    return {
        "drift_instability_variance": tau2,
        "block_mean_count": k,
        "block_mean_length": int(block_length),
        "block_mean_q_statistic": q_stat,
        "block_mean_q_degrees_freedom": int(k - 1),
        "block_mean_fixed_effect_mean": fixed_mean,
        "block_mean_observed_variance": float(np.var(means, ddof=1)) if k > 1 else 0.0,
        "block_mean_median_sampling_variance": float(np.median(variances)),
        "block_mean_average_hac_bandwidth": float(np.mean(block_bandwidths)) if block_bandwidths else 0.0,
        "block_mean_method": "der_simonian_laird_random_effects_on_portfolio_time_blocks",
        "block_length_method": "politis_white_automatic_dependent_bootstrap_length",
    }

def _max_log_return_drawdown(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    cumulative = np.cumsum(x, dtype=np.float64)
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = running_peak - cumulative
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    return max(max_drawdown, 0.0) if np.isfinite(max_drawdown) else 0.0

def _continuous_meta_decay_feature_values(x: np.ndarray, se_mu: float, bandwidth: int) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    eps = float(np.finfo(np.float64).tiny)
    if x.size == 0:
        return {name: 0.0 for name in CONTINUOUS_META_DECAY_FEATURE_NAMES}

    n = int(x.size)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if n > 1 else 0.0
    sigma = max(sigma, eps)
    se = float(se_mu) if np.isfinite(se_mu) and se_mu > 0.0 else float(sigma / math.sqrt(float(max(n, 1))))
    t_stat = float(mu / max(se, eps))

    ac1 = 0.0
    if n > 2:
        lag0 = x[:-1] - float(np.mean(x[:-1]))
        lag1 = x[1:] - float(np.mean(x[1:]))
        denom = float(np.sqrt(np.sum(lag0 * lag0) * np.sum(lag1 * lag1)))
        if np.isfinite(denom) and denom > 0.0:
            ac1 = float(np.sum(lag0 * lag1) / denom)
    ac1 = float(np.clip(ac1, -1.0, 1.0)) if np.isfinite(ac1) else 0.0

    centered = x - mu
    variance = float(np.mean(centered * centered)) if n > 0 else 0.0
    if np.isfinite(variance) and variance > eps:
        skewness = float(np.mean(centered**3) / (variance ** 1.5))
        excess_kurtosis = float(np.mean(centered**4) / (variance * variance) - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0
    if not np.isfinite(skewness):
        skewness = 0.0
    if not np.isfinite(excess_kurtosis):
        excess_kurtosis = 0.0

    try:
        block_length = int(_politis_white_block_length(x))
    except Exception:
        block_length = int(round(math.sqrt(float(n))))
    block_length = int(min(max(block_length, 5), max(n, 1)))
    block_means = [
        float(np.mean(x[start : start + block_length]))
        for start in range(0, n - block_length + 1, block_length)
        if x[start : start + block_length].size > 0
    ]
    if len(block_means) >= 2:
        means = np.asarray(block_means, dtype=np.float64)
        block_mean_sd = float(np.std(means, ddof=1))
        sample_sign = 1.0 if mu >= 0.0 else -1.0
        sign_stability = float(np.mean(np.sign(means) == sample_sign))
    else:
        block_mean_sd = 0.0
        sign_stability = 0.5

    feature_values = {
        "log_years": math.log(max(float(n) / 252.0, eps)),
        "log1p_abs_t": math.log1p(abs(t_stat)),
        "sign_t": float(np.sign(t_stat)) if np.isfinite(t_stat) else 0.0,
        "abs_t_cap5_scaled": min(abs(t_stat), 5.0) / 5.0 if np.isfinite(t_stat) else 0.0,
        "log_ann_vol": math.log(max(sigma * math.sqrt(252.0), eps)),
        "ac1": ac1,
        "skew_scaled": float(np.clip(skewness, -3.0, 3.0) / 3.0),
        "exk_scaled": float(np.clip(excess_kurtosis, -10.0, 10.0) / 10.0),
        "sign_stability_minus_half": float(sign_stability - 0.5),
        "log1p_block_mean_sd_over_se": math.log1p(max(block_mean_sd, 0.0) / max(se, eps)),
        "log1p_maxdd": math.log1p(_max_log_return_drawdown(x)),
        "log1p_hac_bandwidth": math.log1p(float(max(int(bandwidth), 0))),
    }
    return {
        name: float(value) if np.isfinite(value) else 0.0
        for name, value in feature_values.items()
    }

def _continuous_meta_decay_scale(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "scale": 1.0,
            "raw_log_scale": 0.0,
            "clipped_log_scale": 0.0,
            "features": {name: 0.0 for name in CONTINUOUS_META_DECAY_FEATURE_NAMES},
            "standardized_features": {name: 0.0 for name in CONTINUOUS_META_DECAY_FEATURE_NAMES},
        }
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se_mu, bandwidth, _ = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = float(max(sigma, np.finfo(np.float64).tiny) / math.sqrt(float(max(x.size, 1))))
        bandwidth = 0

    features = _continuous_meta_decay_feature_values(x, se_mu, int(bandwidth))
    coefs = np.asarray(CONTINUOUS_META_DECAY_COEFFICIENTS, dtype=np.float64)
    means = np.asarray(CONTINUOUS_META_DECAY_FEATURE_MEANS, dtype=np.float64)
    sds = np.asarray(CONTINUOUS_META_DECAY_FEATURE_SDS, dtype=np.float64)
    values = np.asarray([features[name] for name in CONTINUOUS_META_DECAY_FEATURE_NAMES], dtype=np.float64)
    z = (values - means) / np.maximum(sds, np.finfo(np.float64).eps)
    raw_log_scale = float(coefs[0] + np.dot(coefs[1:], z))
    min_log = math.log(CONTINUOUS_META_DECAY_MIN_SCALE)
    max_log = math.log(CONTINUOUS_META_DECAY_MAX_SCALE)
    clipped_log_scale = float(np.clip(raw_log_scale, min_log, max_log))
    scale = float(math.exp(clipped_log_scale))
    return {
        "scale": scale if np.isfinite(scale) and scale > 0.0 else 1.0,
        "raw_log_scale": raw_log_scale if np.isfinite(raw_log_scale) else 0.0,
        "clipped_log_scale": clipped_log_scale if np.isfinite(clipped_log_scale) else 0.0,
        "features": {name: float(value) for name, value in zip(CONTINUOUS_META_DECAY_FEATURE_NAMES, values)},
        "standardized_features": {name: float(value) for name, value in zip(CONTINUOUS_META_DECAY_FEATURE_NAMES, z)},
    }

def _fit_horizon_credibility_hac_drift_uncertainty_sbb(
    train_values: np.ndarray,
    *,
    uncertainty_ratio_scale: float = 1.0,
    robust_location: bool = False,
) -> Optional[Dict[str, Any]]:
    base = (
        _fit_robust_realized_mean_sbb(train_values)
        if bool(robust_location)
        else _fit_historical_realized_sharpe_sbb(train_values)
    )
    if base is None:
        return None
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    sigma = float(base.get("sigma", 0.0))
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = float(sigma / math.sqrt(float(max(x.size, 1))))
        bandwidth = 0
        long_run_var = float(sigma * sigma)
    uncertainty_ratio = float((se_mu / max(sigma, np.finfo(np.float64).tiny)) ** 2)
    posterior_mean = float(base["posterior_mean"])
    scale = float(max(uncertainty_ratio_scale, 0.0))
    scaled_uncertainty_ratio = float(uncertainty_ratio * scale)
    method_suffix = (
        "robust_trimmed_mean"
        if robust_location and np.isclose(scale, 1.0)
        else "robust_trimmed_mean_higher_uncertainty"
        if robust_location and scale > 1.0
        else "robust_trimmed_mean_slow_decay"
        if robust_location and scale < 1.0
        else "higher_uncertainty"
        if scale > 1.0
        else "slow_decay"
        if scale < 1.0
        else ""
    )
    method = (
        "horizon_credibility_signed_hac_mean_shrink_to_zero"
        if not method_suffix
        else f"horizon_credibility_signed_hac_mean_shrink_to_zero_{method_suffix}"
    )
    decay_meta = {
        "method": method,
        "short_posterior_mean": posterior_mean,
        "long_posterior_mean": 0.0,
        "base_mean_uncertainty_to_process_variance_ratio": uncertainty_ratio,
        "mean_uncertainty_ratio_scale": scale,
        "mean_uncertainty_to_process_variance_ratio": scaled_uncertainty_ratio,
        "terminal_signal_to_noise_rule": "weight_h_equals_1_over_1_plus_h_times_hac_mean_variance_over_daily_process_variance",
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": int(bandwidth),
        "hac_long_run_variance": float(long_run_var),
    }
    out = dict(base)
    out["posterior_sd"] = 0.0
    out["posterior_mu_draws"] = False
    out["nonnegative_drift"] = False
    out["mu_cap"] = 0.0
    out["posterior_mean_decay_meta"] = decay_meta
    out["meta"] = {
        **dict(base.get("meta", {}) or {}),
        "method": method,
        "restriction_source": "none_sign_allowed_realized_drift",
        "posterior_uncertainty_source": "newey_west_hac_long_run_mean_standard_error",
        "posterior_mean": posterior_mean,
        "posterior_sd": 0.0,
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": int(bandwidth),
        "hac_long_run_variance": float(long_run_var),
        "mean_uncertainty_to_process_variance_ratio": scaled_uncertainty_ratio,
        "base_mean_uncertainty_to_process_variance_ratio": uncertainty_ratio,
        "mean_uncertainty_ratio_scale": scale,
        "robust_location": bool(robust_location),
        "posterior_mean_decay_meta": decay_meta,
        "cap_method": "horizon_credibility_shrink_to_zero",
    }
    return out

def _fit_horizon_credibility_hac_continuous_meta_decay_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    scale_info = _continuous_meta_decay_scale(x)
    scale = float(scale_info.get("scale", 1.0))
    out = _fit_horizon_credibility_hac_drift_uncertainty_sbb(
        x,
        uncertainty_ratio_scale=scale,
    )
    if out is None:
        return None
    method = "horizon_credibility_signed_hac_mean_shrink_to_zero_continuous_meta_decay"
    decay_meta = dict(out.get("posterior_mean_decay_meta") or {})
    decay_meta.update(
        {
            "method": method,
            "mean_uncertainty_ratio_scale": scale,
            "continuous_meta_decay_raw_log_scale": float(scale_info.get("raw_log_scale", 0.0)),
            "continuous_meta_decay_clipped_log_scale": float(scale_info.get("clipped_log_scale", 0.0)),
            "continuous_meta_decay_min_scale": CONTINUOUS_META_DECAY_MIN_SCALE,
            "continuous_meta_decay_max_scale": CONTINUOUS_META_DECAY_MAX_SCALE,
            "continuous_meta_decay_model": "weighted_ridge_quadratic_surrogate_on_saved_oos_score_corpus",
            "continuous_meta_decay_source_result": CONTINUOUS_META_DECAY_SOURCE_RESULT,
            "continuous_meta_decay_feature_names": list(CONTINUOUS_META_DECAY_FEATURE_NAMES),
            "continuous_meta_decay_features": dict(scale_info.get("features", {}) or {}),
            "continuous_meta_decay_standardized_features": dict(scale_info.get("standardized_features", {}) or {}),
        }
    )
    meta = dict(out.get("meta", {}) or {})
    meta.update(
        {
            "method": method,
            "posterior_uncertainty_source": (
                "newey_west_hac_long_run_mean_standard_error_with_cross_portfolio_oos_meta_decay_scale"
            ),
            "mean_uncertainty_ratio_scale": scale,
            "posterior_mean_decay_meta": decay_meta,
            "continuous_meta_decay_scale": scale,
            "continuous_meta_decay_raw_log_scale": float(scale_info.get("raw_log_scale", 0.0)),
            "continuous_meta_decay_clipped_log_scale": float(scale_info.get("clipped_log_scale", 0.0)),
            "continuous_meta_decay_min_scale": CONTINUOUS_META_DECAY_MIN_SCALE,
            "continuous_meta_decay_max_scale": CONTINUOUS_META_DECAY_MAX_SCALE,
            "continuous_meta_decay_model": "weighted_ridge_quadratic_surrogate_on_saved_oos_score_corpus",
            "continuous_meta_decay_source_result": CONTINUOUS_META_DECAY_SOURCE_RESULT,
            "continuous_meta_decay_features": dict(scale_info.get("features", {}) or {}),
            "continuous_meta_decay_standardized_features": dict(scale_info.get("standardized_features", {}) or {}),
            "live_oos_refit_required": False,
        }
    )
    out["posterior_mean_decay_meta"] = decay_meta
    out["meta"] = meta
    return out

def _standard_normal_cdf(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    return float(0.5 * math.erfc(-float(value) / math.sqrt(2.0)))

def _hac_sign_uncertainty_multiplier(
    values: np.ndarray,
    *,
    max_multiplier: float,
) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    eps = float(np.finfo(np.float64).tiny)
    if x.size == 0:
        return {
            "multiplier": 1.0,
            "max_multiplier": float(max(max_multiplier, 1.0)),
            "hac_t_statistic": 0.0,
            "absolute_hac_t_statistic": 0.0,
            "sign_correct_probability": 0.5,
            "sign_uncertainty_entropy": 1.0,
        }
    sample_mean = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = float(max(sigma, eps) / math.sqrt(float(max(x.size, 1))))
        bandwidth = 0
        long_run_var = float(sigma * sigma)
    t_stat = float(sample_mean / max(se_mu, eps))
    abs_t = abs(t_stat) if np.isfinite(t_stat) else 0.0
    sign_correct_prob = _standard_normal_cdf(abs_t)
    p = float(np.clip(sign_correct_prob, 1e-12, 1.0 - 1e-12))
    normalized_entropy = float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0))
    normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))
    upper = float(max(max_multiplier, 1.0))
    multiplier = float(1.0 + (upper - 1.0) * normalized_entropy)
    multiplier = float(np.clip(multiplier, 1.0, upper))
    return {
        "multiplier": multiplier,
        "max_multiplier": upper,
        "hac_t_statistic": t_stat if np.isfinite(t_stat) else 0.0,
        "absolute_hac_t_statistic": abs_t,
        "sign_correct_probability": sign_correct_prob,
        "sign_uncertainty_entropy": normalized_entropy,
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": int(bandwidth),
        "hac_long_run_variance": float(long_run_var),
        "sample_mean": sample_mean,
        "sample_sigma": sigma,
    }

def _fit_horizon_credibility_hac_sign_uncertainty_decay_sbb(
    train_values: np.ndarray,
    *,
    max_multiplier: float,
) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    scale_info = _hac_sign_uncertainty_multiplier(x, max_multiplier=max_multiplier)
    multiplier = float(scale_info.get("multiplier", 1.0))
    out = _fit_horizon_credibility_hac_drift_uncertainty_sbb(
        x,
        uncertainty_ratio_scale=multiplier,
    )
    if out is None:
        return None
    max_label = f"{float(scale_info.get('max_multiplier', max_multiplier)):.0f}x"
    method = f"horizon_credibility_signed_hac_mean_shrink_to_zero_hac_sign_uncertainty_{max_label}"
    decay_meta = dict(out.get("posterior_mean_decay_meta") or {})
    decay_meta.update(
        {
            "method": method,
            "mean_uncertainty_ratio_scale": multiplier,
            "minimum_mean_uncertainty_ratio_scale": 1.0,
            "maximum_mean_uncertainty_ratio_scale": float(scale_info.get("max_multiplier", max_multiplier)),
            "hac_sign_uncertainty_rule": (
                "scale_equals_1_plus_max_minus_1_times_normalized_binary_entropy_of_"
                "normal_hac_mean_sign_probability"
            ),
            "hac_t_statistic": float(scale_info.get("hac_t_statistic", 0.0)),
            "absolute_hac_t_statistic": float(scale_info.get("absolute_hac_t_statistic", 0.0)),
            "sign_correct_probability": float(scale_info.get("sign_correct_probability", 0.5)),
            "sign_uncertainty_entropy": float(scale_info.get("sign_uncertainty_entropy", 1.0)),
        }
    )
    meta = dict(out.get("meta", {}) or {})
    meta.update(
        {
            "method": method,
            "posterior_uncertainty_source": (
                "newey_west_hac_long_run_mean_standard_error_with_hac_sign_uncertainty_decay_multiplier"
            ),
            "mean_uncertainty_ratio_scale": multiplier,
            "minimum_mean_uncertainty_ratio_scale": 1.0,
            "maximum_mean_uncertainty_ratio_scale": float(scale_info.get("max_multiplier", max_multiplier)),
            "hac_sign_uncertainty_rule": decay_meta["hac_sign_uncertainty_rule"],
            "hac_t_statistic": float(scale_info.get("hac_t_statistic", 0.0)),
            "absolute_hac_t_statistic": float(scale_info.get("absolute_hac_t_statistic", 0.0)),
            "sign_correct_probability": float(scale_info.get("sign_correct_probability", 0.5)),
            "sign_uncertainty_entropy": float(scale_info.get("sign_uncertainty_entropy", 1.0)),
            "posterior_mean_decay_meta": decay_meta,
            "live_oos_refit_required": False,
        }
    )
    out["posterior_mean_decay_meta"] = decay_meta
    out["meta"] = meta
    return out

def _fit_horizon_credibility_hac_predictive_drift_draws_sbb(
    train_values: np.ndarray,
    *,
    uncertainty_ratio_scale: float,
) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = _fit_horizon_credibility_hac_drift_uncertainty_sbb(
        x,
        uncertainty_ratio_scale=uncertainty_ratio_scale,
    )
    if out is None:
        return None
    meta = dict(out.get("meta", {}) or {})
    se_mu = float(meta.get("hac_mean_standard_error", 0.0))
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        sigma = float(out.get("sigma", 0.0))
        se_mu = float(max(sigma, np.finfo(np.float64).tiny) / math.sqrt(float(max(x.size, 1))))
    scale = float(max(uncertainty_ratio_scale, 0.0))
    scale_label = f"{scale:.0f}x" if np.isclose(scale, round(scale)) else f"{scale:g}x"
    method = f"horizon_credibility_signed_hac_mean_shrink_to_zero_bayesian_predictive_drift_draws_{scale_label}"
    decay_meta = dict(out.get("posterior_mean_decay_meta") or {})
    decay_meta.update(
        {
            "method": method,
            "predictive_drift_draws": True,
            "predictive_drift_draw_distribution": "normal_hac_mean_posterior_approximation",
            "posterior_mu_draws_with_horizon_decay": True,
            "posterior_mean_standard_error": se_mu,
        }
    )
    out["posterior_sd"] = float(max(se_mu, 0.0))
    out["posterior_mu_draws"] = True
    out["posterior_mu_draws_with_horizon_decay"] = True
    out["nonnegative_drift"] = False
    out["mu_cap"] = 0.0
    out["posterior_mean_decay_meta"] = decay_meta
    meta.update(
        {
            "method": method,
            "posterior_uncertainty_source": (
                "newey_west_hac_long_run_mean_standard_error_sampled_inside_posterior_predictive_paths"
            ),
            "posterior_sd": float(out["posterior_sd"]),
            "posterior_mu_draws": True,
            "posterior_mu_draws_with_horizon_decay": True,
            "predictive_drift_draws": True,
            "predictive_drift_draw_distribution": "normal_hac_mean_posterior_approximation",
            "mean_uncertainty_ratio_scale": scale,
            "posterior_mean_decay_meta": decay_meta,
            "live_oos_refit_required": False,
        }
    )
    out["meta"] = meta
    return out

def _fit_evidence_estimated_dlm_drift_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 252:
        return _fit_horizon_credibility_hac_drift_uncertainty_sbb(x)
    sample_mu = float(np.mean(x))
    sample_sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sample_sigma) or sample_sigma <= 1.0e-10:
        return None
    anchor_fit = _fit_empirical_bayes_sharpe_sbb(x)
    anchor_mu = float(anchor_fit.get("posterior_mean", 0.0)) if anchor_fit is not None else 0.0
    centered = x - anchor_mu
    obs_var = float(np.var(centered, ddof=1))
    if not np.isfinite(obs_var) or obs_var <= 1.0e-16:
        obs_var = float(sample_sigma * sample_sigma)
    obs_sd = math.sqrt(max(obs_var, np.finfo(np.float64).tiny))
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = obs_sd / math.sqrt(float(max(x.size, 1)))
        bandwidth = 0
        long_run_var = obs_var
    signal_ratio_start = float((se_mu / max(obs_sd, np.finfo(np.float64).tiny)) ** 2)
    params = _estimate_evidence_dlm_drift_params(
        centered,
        obs_var=obs_var,
        signal_ratio_start=signal_ratio_start,
    )
    phi = float(params["phi"])
    state_var_ratio = float(params["state_var_ratio"])
    filtered = _kalman_ar1_drift_filter(
        centered,
        obs_var=obs_var,
        phi=phi,
        state_var_ratio=state_var_ratio,
    )
    if not np.isfinite(float(filtered.get("loglik", -math.inf))):
        return _fit_horizon_credibility_hac_drift_uncertainty_sbb(x)
    predicted_deviation = np.asarray(filtered["predicted_mean"], dtype=np.float64)
    residuals = x - (anchor_mu + predicted_deviation)
    z = residuals / obs_sd
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    z = z - float(np.mean(z))
    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1.0e-12:
        z = z / z_sd
    final_deviation_mean = float(filtered["final_mean"])
    final_deviation_var = float(max(filtered["final_var"], 0.0))
    posterior_mean = float(anchor_mu + final_deviation_mean)
    posterior_sd = float(math.sqrt(final_deviation_var))
    baseline_loglik = float(
        -0.5
        * np.sum(
            np.log(2.0 * math.pi * obs_var)
            + np.square(centered) / max(obs_var, np.finfo(np.float64).tiny)
        )
    )
    method = "evidence_estimated_ar1_latent_drift_dlm_with_empirical_bayes_anchor"
    half_life = _overlay_half_life_days(phi)
    return {
        "sample_mu": sample_mu,
        "sigma": obs_sd,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "mu_cap": 0.0,
        "sample_mu_days": 0,
        "nonnegative_drift": False,
        "posterior_mu_draws": False,
        "dlm_drift_paths": True,
        "dlm_long_run_anchor_mean": anchor_mu,
        "dlm_state_transition_phi": phi,
        "dlm_state_stationary_var": float(filtered["stationary_var"]),
        "dlm_state_stationary_var_ratio": state_var_ratio,
        "dlm_state_noise_var": float(filtered["state_noise_var"]),
        "dlm_state_noise_sd": float(math.sqrt(max(float(filtered["state_noise_var"]), 0.0))),
        "dlm_state_posterior_deviation_mean": final_deviation_mean,
        "dlm_state_posterior_deviation_var": final_deviation_var,
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "meta": {
            "method": method,
            "prior_source": "empirical_bayes_hac_sharpe_long_run_anchor",
            "state_equation": "mu_t_minus_anchor_equals_phi_times_previous_deviation_plus_gaussian_state_shock",
            "observation_equation": "return_t_equals_anchor_plus_latent_drift_deviation_plus_gaussian_observation_shock",
            "parameter_estimation": "continuous_kalman_marginal_likelihood_l_bfgs_b",
            "decay_source": "portfolio_specific_marginal_likelihood_estimated_state_transition",
            "live_oos_refit_required": False,
            "anchor_posterior_mean": anchor_mu,
            "anchor_method": str((anchor_fit or {}).get("meta", {}).get("method", "zero_anchor")),
            "sample_mean": sample_mu,
            "sample_sigma": sample_sigma,
            "observation_sigma": obs_sd,
            "posterior_mean": posterior_mean,
            "posterior_sd": posterior_sd,
            "posterior_mu_draws": False,
            "dlm_drift_paths": True,
            "dlm_state_transition_phi": phi,
            "dlm_state_half_life_days": half_life,
            "dlm_state_stationary_var_ratio": state_var_ratio,
            "dlm_state_noise_var": float(filtered["state_noise_var"]),
            "dlm_state_posterior_deviation_mean": final_deviation_mean,
            "dlm_state_posterior_deviation_var": final_deviation_var,
            "dlm_loglik": float(filtered["loglik"]),
            "anchor_only_gaussian_loglik": baseline_loglik,
            "dlm_loglik_lift_vs_anchor_only": float(filtered["loglik"]) - baseline_loglik,
            "hac_mean_standard_error": float(se_mu),
            "hac_bandwidth": int(bandwidth),
            "hac_long_run_variance": float(long_run_var),
            "signal_ratio_start": signal_ratio_start,
            "phi_bounds": [DLM_DRIFT_PHI_MIN, DLM_DRIFT_PHI_MAX],
            "state_var_ratio_bounds": [DLM_DRIFT_STATE_VAR_RATIO_MIN, DLM_DRIFT_STATE_VAR_RATIO_MAX],
            "max_annual_drift_sharpe_sd": DLM_DRIFT_MAX_ANNUAL_DRIFT_SHARPE_SD,
            "academic_lineage": (
                "discounted_dynamic_linear_models_west_harrison_"
                "bayesian_state_space_forecasting_with_jorion_style_mean_shrinkage"
            ),
        },
    }

def _fit_horizon_credibility_hac_drift_instability_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    base = _fit_historical_realized_sharpe_sbb(train_values)
    if base is None:
        return None
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    sigma = float(base.get("sigma", 0.0))
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = float(sigma / math.sqrt(float(max(x.size, 1))))
        bandwidth = 0
        long_run_var = float(sigma * sigma)
    hac_mean_variance = float(se_mu * se_mu)
    instability = _estimate_random_effects_drift_instability(x, fallback_sigma=sigma)
    drift_instability_variance = float(max(instability.get("drift_instability_variance", 0.0), 0.0))
    total_mean_uncertainty_variance = float(hac_mean_variance + drift_instability_variance)
    process_variance = float(max(sigma * sigma, np.finfo(np.float64).tiny))
    base_uncertainty_ratio = float(hac_mean_variance / process_variance)
    drift_instability_ratio = float(drift_instability_variance / process_variance)
    total_uncertainty_ratio = float(total_mean_uncertainty_variance / process_variance)
    posterior_mean = float(base["posterior_mean"])
    method = "horizon_credibility_signed_hac_mean_plus_random_effects_drift_instability_shrink_to_zero"
    decay_meta = {
        "method": method,
        "short_posterior_mean": posterior_mean,
        "long_posterior_mean": 0.0,
        "base_mean_uncertainty_to_process_variance_ratio": base_uncertainty_ratio,
        "drift_instability_to_process_variance_ratio": drift_instability_ratio,
        "mean_uncertainty_ratio_scale": 1.0,
        "mean_uncertainty_to_process_variance_ratio": total_uncertainty_ratio,
        "terminal_signal_to_noise_rule": (
            "weight_h_equals_1_over_1_plus_h_times_"
            "portfolio_hac_plus_random_effects_drift_instability_variance_over_daily_process_variance"
        ),
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": int(bandwidth),
        "hac_long_run_variance": float(long_run_var),
        "hac_mean_variance": hac_mean_variance,
        "drift_instability_variance": drift_instability_variance,
        "total_mean_uncertainty_variance": total_mean_uncertainty_variance,
        **instability,
    }
    out = dict(base)
    out["posterior_sd"] = 0.0
    out["posterior_mu_draws"] = False
    out["nonnegative_drift"] = False
    out["mu_cap"] = 0.0
    out["posterior_mean_decay_meta"] = decay_meta
    out["meta"] = {
        **dict(base.get("meta", {}) or {}),
        "method": method,
        "restriction_source": "none_sign_allowed_realized_drift",
        "posterior_uncertainty_source": "newey_west_hac_plus_der_simonian_laird_block_mean_drift_instability",
        "posterior_mean": posterior_mean,
        "posterior_sd": 0.0,
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": int(bandwidth),
        "hac_long_run_variance": float(long_run_var),
        "hac_mean_variance": hac_mean_variance,
        "drift_instability_variance": drift_instability_variance,
        "total_mean_uncertainty_variance": total_mean_uncertainty_variance,
        "mean_uncertainty_to_process_variance_ratio": total_uncertainty_ratio,
        "base_mean_uncertainty_to_process_variance_ratio": base_uncertainty_ratio,
        "drift_instability_to_process_variance_ratio": drift_instability_ratio,
        "mean_uncertainty_ratio_scale": 1.0,
        "posterior_mean_decay_meta": decay_meta,
        "cap_method": "portfolio_direct_horizon_credibility_shrink_to_zero",
        **instability,
    }
    return out

def _fit_sbb_mean_model(train_values: np.ndarray, mean_model: str) -> Optional[Dict[str, Any]]:
    model = str(mean_model or "positive_sample_mean")
    if model == "evidence_estimated_sharpe_dlm_historical_cagr_anchor":
        x = np.asarray(train_values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 60:
            return None
        mu, residuals, mean_meta = _evidence_estimated_sharpe_dlm_historical_cagr_anchor_mean(x)
        base_fit_override = mean_meta.get("_base_fit_override") if isinstance(mean_meta, dict) else None
        if isinstance(base_fit_override, dict):
            return dict(base_fit_override)
        return _base_sbb_fit_from_mean_fit(
            x,
            float(mu),
            np.asarray(residuals, dtype=np.float64),
            dict(mean_meta),
        )
    if model == "merton_positive_hac_drift_uncertainty":
        return _fit_merton_positive_hac_drift_uncertainty_sbb(train_values)
    if model == "ar1_constant_ols_mean":
        return _fit_ar1_constant_ols_mean_sbb(train_values)
    if model == "unconstrained_hac_drift_uncertainty":
        return _fit_unconstrained_hac_drift_uncertainty_sbb(train_values)
    if model == "horizon_credibility_hac_drift_uncertainty":
        return _fit_horizon_credibility_hac_drift_uncertainty_sbb(train_values)
    if model == "horizon_credibility_hac_drift_uncertainty_continuous_meta_decay":
        return _fit_horizon_credibility_hac_continuous_meta_decay_sbb(train_values)
    if model == "horizon_credibility_hac_drift_uncertainty_sign_uncertainty_2x":
        return _fit_horizon_credibility_hac_sign_uncertainty_decay_sbb(
            train_values,
            max_multiplier=2.0,
        )
    if model == "horizon_credibility_hac_drift_uncertainty_sign_uncertainty_3x":
        return _fit_horizon_credibility_hac_sign_uncertainty_decay_sbb(
            train_values,
            max_multiplier=3.0,
        )
    if model == "horizon_credibility_hac_drift_uncertainty_predictive_drift_draws_1x":
        return _fit_horizon_credibility_hac_predictive_drift_draws_sbb(
            train_values,
            uncertainty_ratio_scale=1.0,
        )
    if model == "horizon_credibility_hac_drift_uncertainty_predictive_drift_draws_2x":
        return _fit_horizon_credibility_hac_predictive_drift_draws_sbb(
            train_values,
            uncertainty_ratio_scale=2.0,
        )
    if model == "evidence_estimated_dlm_drift":
        return _fit_evidence_estimated_dlm_drift_sbb(train_values)
    if model == "horizon_credibility_hac_drift_uncertainty_slow_decay":
        return _fit_horizon_credibility_hac_drift_uncertainty_sbb(
            train_values,
            uncertainty_ratio_scale=0.5,
        )
    if model == "robust_horizon_credibility_hac_drift_uncertainty":
        return _fit_horizon_credibility_hac_drift_uncertainty_sbb(
            train_values,
            robust_location=True,
        )
    if model == "horizon_credibility_hac_drift_instability":
        return _fit_horizon_credibility_hac_drift_instability_sbb(train_values)
    if model == "empirical_bayes_hac_sharpe":
        return _fit_empirical_bayes_sharpe_sbb(train_values)
    if model == "empirical_bayes_hac_positive_sharpe":
        return _fit_empirical_bayes_sharpe_sbb(train_values, nonnegative_sharpe=True)
    if model == "hierarchical_empirical_bayes_sharpe":
        return _fit_hierarchical_empirical_bayes_sharpe_sbb(train_values)
    if model == "hierarchical_empirical_bayes_sharpe_mu_uncertainty":
        return _fit_hierarchical_empirical_bayes_sharpe_mu_uncertainty_sbb(train_values)
    if model == "prequential_crps_shrinkage":
        return _fit_prequential_crps_shrinkage_sbb(train_values)
    if model == "zero_sharpe":
        return _fit_zero_sharpe_sbb(train_values)
    if model == "historical_realized_sharpe":
        return _fit_historical_realized_sharpe_sbb(train_values)
    return _fit_positive_sample_mean_sbb(train_values)

def _fit_ar1_constant_ols_mean_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 252:
        return None
    y = x[1:]
    lagged = x[:-1]
    design = np.column_stack([np.ones_like(lagged), lagged])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except Exception:
        return None
    intercept = float(beta[0])
    ar_coef = float(beta[1])
    if not (np.isfinite(intercept) and np.isfinite(ar_coef)) or abs(ar_coef) >= 1.0:
        return None
    unconditional_mean = float(intercept / max(1.0 - ar_coef, np.finfo(np.float64).eps))
    fitted = np.empty_like(x, dtype=np.float64)
    fitted[0] = unconditional_mean
    fitted[1:] = intercept + ar_coef * lagged
    residuals = x - fitted
    sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    z = residuals / sigma
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    z = z - float(np.mean(z))
    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1e-12:
        z = z / z_sd
    return {
        "sample_mu": unconditional_mean,
        "sigma": sigma,
        "posterior_mean": unconditional_mean,
        "posterior_sd": 0.0,
        "mu_cap": 0.0,
        "sample_mu_days": 0,
        "nonnegative_drift": False,
        "posterior_mu_draws": False,
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "mean_dynamics": {
            "type": "ar1_constant_ols",
            "intercept": intercept,
            "ar_coef": ar_coef,
            "unconditional_mean": unconditional_mean,
            "last_return": float(x[-1]),
        },
        "meta": {
            "method": "ols_ar1_with_constant_mean",
            "mean_equation": "r_t = intercept + phi * r_{t-1} + epsilon_t",
            "intercept": intercept,
            "ar_coef": ar_coef,
            "unconditional_mean": unconditional_mean,
            "stationarity_condition": "abs(phi) < 1",
            "cap_method": "none",
            "fixed_sharpe_cap": False,
        },
    }

def _fit_zero_sharpe_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    residuals = x - sample_mu
    z = residuals / sigma
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    z = z - float(np.mean(z))
    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1e-12:
        z = z / z_sd
    return {
        "sample_mu": sample_mu,
        "sigma": sigma,
        "posterior_mean": 0.0,
        "posterior_sd": 0.0,
        "mu_cap": 0.0,
        "sample_mu_days": 0,
        "nonnegative_drift": False,
        "posterior_mu_draws": False,
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "meta": {
            "method": "zero_sharpe_stationary_bootstrap_residuals",
            "prior_source": "zero_expected_excess_return",
            "cap_method": "point_zero_sharpe",
            "sample_mean": sample_mu,
            "posterior_mean": 0.0,
            "posterior_sd": 0.0,
        },
    }

def _fit_historical_realized_sharpe_sbb(train_values: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    sample_sr = float(sample_mu / sigma * math.sqrt(252.0))
    residuals = x - sample_mu
    z = residuals / sigma
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    z = z - float(np.mean(z))
    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1e-12:
        z = z / z_sd
    return {
        "sample_mu": sample_mu,
        "sigma": sigma,
        "posterior_mean": sample_mu,
        "posterior_sd": 0.0,
        "mu_cap": 0.0,
        "sample_mu_days": 0,
        "nonnegative_drift": False,
        "posterior_mu_draws": False,
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "meta": {
            "method": "historical_realized_sharpe_sample_mean_stationary_bootstrap_residuals",
            "prior_source": "raw_portfolio_history",
            "cap_method": "none",
            "sample_mean": sample_mu,
            "sample_sharpe_annualized": sample_sr,
            "posterior_mean": sample_mu,
            "posterior_sd": 0.0,
        },
    }

def _overlay_half_life_days(persistence: float) -> Optional[float]:
    p = abs(float(persistence))
    if not np.isfinite(p) or p <= 0.0 or p >= 0.999999:
        return None
    return float(math.log(0.5) / math.log(p))

def _sv_observed_log_variance(
    train_values: np.ndarray,
    mu: float,
    *,
    winsorize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return _sv_observed_log_variance_from_residuals(x - float(mu), winsorize=winsorize)

def _sv_observed_log_variance_from_residuals(
    residual_values: np.ndarray,
    *,
    winsorize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(residual_values, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    eps_x = residuals * 100.0
    squared = eps_x * eps_x
    if winsorize:
        positive = squared[np.isfinite(squared) & (squared > 0.0)]
        floor = float(max(np.quantile(positive, 0.001) * 0.1, 1e-10)) if positive.size else 1e-10
        observed = np.log(np.maximum(squared, floor)) - SV_LOG_CHI_SQUARE_MEAN
        observed = observed[np.isfinite(observed)]
    else:
        # log(0) is undefined in the SV measurement equation. Treat exact zero
        # squared residuals as missing volatility measurements instead of
        # winsorizing them to an arbitrary floor.
        mask = np.isfinite(squared) & (squared > 0.0)
        observed = np.log(squared[mask]) - SV_LOG_CHI_SQUARE_MEAN
        eps_x = eps_x[mask]
        observed = observed[np.isfinite(observed)]
        eps_x = eps_x[: observed.size]
    if winsorize and observed.size >= 80:
        lo, hi = np.quantile(observed, [0.005, 0.995])
        observed = np.clip(observed, lo, hi)
    return observed.astype(np.float64), eps_x[: observed.size].astype(np.float64)

def _sv_kalman_filter(
    observed_log_var: np.ndarray,
    level: float,
    phi: float,
    eta: float,
    *,
    return_path: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    y = np.asarray(observed_log_var, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return -math.inf, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if not (np.isfinite(level) and np.isfinite(phi) and np.isfinite(eta)):
        return -math.inf, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    phi = float(phi)
    eta = float(eta)
    if phi < 0.0 or phi >= 0.999 or eta <= 1e-6:
        return -math.inf, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    measurement_var = SV_LOG_CHI_SQUARE_VAR
    q = eta * eta
    mean = float(level)
    variance = float(max(q / max(1.0 - phi * phi, 1e-4), 1e-6))
    loglik = 0.0
    filtered = np.empty(y.size, dtype=np.float64) if return_path else np.empty(1, dtype=np.float64)
    filtered_var = np.empty(y.size, dtype=np.float64) if return_path else np.empty(1, dtype=np.float64)
    last_mean = mean
    last_var = variance
    for idx, obs in enumerate(y):
        forecast_var = float(max(variance + measurement_var, 1e-8))
        innovation = float(obs - mean)
        loglik += -0.5 * (math.log(2.0 * math.pi * forecast_var) + (innovation * innovation) / forecast_var)
        gain = variance / forecast_var
        updated_mean = mean + gain * innovation
        updated_var = float(max((1.0 - gain) * variance, 1e-8))
        last_mean = float(updated_mean)
        last_var = updated_var
        if return_path:
            filtered[idx] = last_mean
            filtered_var[idx] = last_var
        mean = float(level + phi * (updated_mean - level))
        variance = float(phi * phi * updated_var + q)
    if not return_path:
        filtered[0] = last_mean
        filtered_var[0] = last_var
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

def _fit_sv_state_space_params(
    observed_log_var: np.ndarray,
    *,
    start_count: int = 4,
    maxiter: int = 120,
) -> Tuple[float, float, float]:
    y = np.asarray(observed_log_var, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 30:
        return _initial_sv_state_space_params(y)
    init_level, init_phi, init_eta = _initial_sv_state_space_params(y)
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])
    bounds = [(float(y_lo - 4.0), float(y_hi + 4.0)), (0.001, 0.995), (math.log(0.02), math.log(2.5))]

    def objective(params: np.ndarray) -> float:
        level = float(params[0])
        phi = float(params[1])
        eta = float(math.exp(params[2]))
        loglik, _, _ = _sv_kalman_filter(y, level, phi, eta)
        if not np.isfinite(loglik):
            return 1e100
        return -float(loglik)

    starts = [
        (init_level, init_phi, init_eta),
        (init_level, 0.90, max(init_eta, 0.10)),
        (init_level, 0.97, max(init_eta * 0.5, 0.08)),
        (float(np.median(y)), 0.985, 0.12),
    ]
    starts = starts[: int(np.clip(int(start_count), 1, len(starts)))]
    best: Optional[Tuple[float, np.ndarray]] = None
    try:
        from scipy import optimize

        for level, phi, eta in starts:
            x0 = np.asarray([level, phi, math.log(float(np.clip(eta, 0.02, 2.5)))], dtype=np.float64)
            result = optimize.minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(max(1, maxiter))},
            )
            value = float(result.fun) if np.isfinite(result.fun) else math.inf
            if result.success and (best is None or value < best[0]):
                best = (value, np.asarray(result.x, dtype=np.float64))
    except Exception:
        best = None
    if best is None:
        return init_level, init_phi, init_eta
    params = best[1]
    return float(params[0]), float(params[1]), float(math.exp(params[2]))

def _sv_mcmc_log_posterior(y: np.ndarray, level: float, phi: float, eta: float) -> float:
    if not (np.isfinite(level) and np.isfinite(phi) and np.isfinite(eta)):
        return -math.inf
    if phi < 0.0 or phi >= 0.999 or eta <= 1e-4:
        return -math.inf
    ll, _, _ = _sv_kalman_filter(y, level, phi, eta)
    if not np.isfinite(ll):
        return -math.inf
    prior_level = -0.5 * ((level - float(np.mean(y))) / 4.0) ** 2
    prior_phi = -0.5 * ((phi - 0.94) / 0.20) ** 2
    prior_eta = -0.5 * ((math.log(eta) - math.log(0.35)) / 1.0) ** 2
    return float(ll + prior_level + prior_phi + prior_eta)

def _logit_unit_interval(value: float) -> float:
    eps = np.finfo(np.float64).eps
    p = float(min(max(value, eps), 1.0 - eps))
    return float(math.log(p / (1.0 - p)))

def _inv_logit_unit_interval(value: float) -> float:
    x = float(value)
    if x >= 0.0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))

def _sv_transformed_log_posterior(y: np.ndarray, level: float, phi_logit: float, log_eta: float) -> float:
    if not (np.isfinite(level) and np.isfinite(phi_logit) and np.isfinite(log_eta)):
        return -math.inf
    phi = _inv_logit_unit_interval(float(phi_logit))
    eta = float(math.exp(float(log_eta)))
    lp = _sv_mcmc_log_posterior(y, level, phi, eta)
    if not np.isfinite(lp):
        return -math.inf
    # Symmetric proposals are made in unconstrained coordinates; the Jacobian
    # gives the implied posterior density over those coordinates.
    return float(lp + math.log(max(phi * (1.0 - phi), 1e-300)) + log_eta)

def _standardized_empirical_innovation_pool(
    values: np.ndarray,
    *,
    clip: Optional[float] = None,
    method: str = "mean_std",
) -> Optional[np.ndarray]:
    z = np.asarray(values, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    if str(method) == "median_mad":
        z = z - float(np.median(z))
        mad = float(np.median(np.abs(z)))
        sd = 1.4826 * mad
    else:
        z = z - float(np.mean(z))
        sd = float(np.std(z, ddof=1)) if z.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    z = z / sd
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z.astype(np.float64)

def _finite_correlation(x_values: np.ndarray, y_values: np.ndarray, *, method: str = "pearson") -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    n = min(int(x.size), int(y.size))
    if n < 3:
        return math.nan
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return math.nan
    if str(method) == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    x_sd = float(np.std(x, ddof=1))
    y_sd = float(np.std(y, ddof=1))
    if not (np.isfinite(x_sd) and np.isfinite(y_sd)) or x_sd <= 1e-12 or y_sd <= 1e-12:
        return math.nan
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else math.nan

def _circular_block_bootstrap_indices(
    *,
    n: int,
    block_length: int,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(max(n, 1))
    block_length = int(max(block_length, 1))
    total_days = int(max(total_days, 0))
    n_paths = int(max(n_paths, 0))
    if total_days <= 0 or n_paths <= 0:
        return np.empty((n_paths, total_days), dtype=np.int64)
    block_count = int(math.ceil(float(total_days) / float(block_length)))
    starts = rng.integers(0, n, size=(n_paths, block_count), endpoint=False)
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(n_paths, block_count * block_length)[:, :total_days].astype(np.int64)

def _ffbs_sv_log_variance_path(
    observed_log_var: np.ndarray,
    level: float,
    phi: float,
    eta: float,
    rng: np.random.Generator,
    *,
    clip_path: bool = True,
) -> Optional[np.ndarray]:
    y = np.asarray(observed_log_var, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n <= 0:
        return None
    phi = float(phi)
    eta = float(eta)
    if not (np.isfinite(level) and np.isfinite(phi) and np.isfinite(eta)):
        return None
    if phi < 0.0 or phi >= 0.999 or eta <= 1e-6:
        return None
    obs_var = float(SV_LOG_CHI_SQUARE_VAR)
    state_var = float(max(eta * eta, 1e-10))
    init_var = float(state_var / max(1.0 - phi * phi, 1e-4))

    pred_mean = np.empty(n, dtype=np.float64)
    pred_var = np.empty(n, dtype=np.float64)
    filt_mean = np.empty(n, dtype=np.float64)
    filt_var = np.empty(n, dtype=np.float64)

    mean_prev = float(level)
    var_prev = float(init_var)
    for idx in range(n):
        if idx == 0:
            mean_pred = float(level)
            var_pred = float(init_var)
        else:
            mean_pred = float(level + phi * (mean_prev - level))
            var_pred = float(phi * phi * var_prev + state_var)
        var_pred = float(max(var_pred, 1e-10))
        gain = float(var_pred / max(var_pred + obs_var, 1e-10))
        mean_filt = float(mean_pred + gain * (float(y[idx]) - mean_pred))
        var_filt = float(max((1.0 - gain) * var_pred, 1e-10))
        pred_mean[idx] = mean_pred
        pred_var[idx] = var_pred
        filt_mean[idx] = mean_filt
        filt_var[idx] = var_filt
        mean_prev = mean_filt
        var_prev = var_filt

    path = np.empty(n, dtype=np.float64)
    path[-1] = float(rng.normal(filt_mean[-1], math.sqrt(max(filt_var[-1], 1e-10))))
    for idx in range(n - 2, -1, -1):
        next_pred_var = float(max(pred_var[idx + 1], 1e-10))
        smoother_gain = float(filt_var[idx] * phi / next_pred_var)
        mean = float(filt_mean[idx] + smoother_gain * (path[idx + 1] - pred_mean[idx + 1]))
        var = float(max(filt_var[idx] - smoother_gain * smoother_gain * next_pred_var, 1e-10))
        path[idx] = float(rng.normal(mean, math.sqrt(var)))
    path = path.astype(np.float64)
    if bool(clip_path):
        path = np.clip(path, -18.0, 18.0)
    return path

def _full_mcmc_sv_overlay_fit_signature(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    defaults = {
        "mean_model": "positive_sample_mean",
        "leverage": False,
        "leverage_alignment": "lagged_return",
        "leverage_correlation_method": "pearson",
        "leverage_correlation_scope": "per_state_path",
        "state_innovation_distribution": "gaussian",
        "state_innovation_source": "posterior_ffbs_path",
        "state_innovation_standardization": "mean_std",
        "state_innovation_resampling": "iid",
        "state_innovation_coupling": "correlation_mixture",
        "sv_sigma_scale_method": "none",
        "sv_measurement_bias": "log_chi_square_theoretical",
        "unclipped_sv_measurement": False,
        "transformed_parameter_mcmc": False,
        "unclipped_empirical_innovations": False,
        "innovation_standardization": "mean_std",
        "innovation_resampling": "stationary_bootstrap",
        "innovation_conditioning": "none",
        "innovation_pool_source": "sample_standardized",
        "mcmc_iterations": 160,
        "mcmc_burn": 60,
        "mcmc_thin": 10,
        "mcmc_stopping": "fixed",
        "mcmc_min_iterations": 100,
        "mcmc_check_interval": 40,
        "latent_vol_persistence": "ar1",
    }
    signature = []
    for key, default in defaults.items():
        value = candidate.get(key, default)
        if key == "innovation_resampling" and _full_mcmc_sv_parametric_innovation_meta(str(value)) is None:
            value = "empirical_nonparametric_fit_agnostic"
        signature.append((key, value))
    if str(candidate.get("mcmc_stopping", "fixed")) == "dynamic_multichain_rhat_ess_forecast_stability":
        for key, default in (
            ("mcmc_initial_iterations", candidate.get("mcmc_iterations", 160)),
            ("mcmc_extend_iterations", candidate.get("mcmc_check_interval", 40)),
            ("mcmc_max_iterations", candidate.get("mcmc_iterations", 160)),
            ("mcmc_chains", 4),
            ("mcmc_rhat_threshold", 1.01),
            ("mcmc_min_ess", 100.0),
            ("mcmc_forecast_stability_tolerance", 0.05),
            ("mcmc_proposal_adaptation", "fixed"),
            ("mcmc_adaptation_iterations", candidate.get("mcmc_burn", 60)),
            ("mcmc_adaptation_target_acceptance", 0.234),
        ):
            signature.append((key, candidate.get(key, default)))
    elif str(candidate.get("mcmc_proposal_adaptation", "fixed")) != "fixed":
        for key, default in (
            ("mcmc_proposal_adaptation", "fixed"),
            ("mcmc_adaptation_iterations", candidate.get("mcmc_burn", 60)),
            ("mcmc_adaptation_target_acceptance", 0.234),
        ):
            signature.append((key, candidate.get(key, default)))
    if str(candidate.get("mcmc_seed_salt", "")):
        signature.append(("mcmc_seed_salt", str(candidate.get("mcmc_seed_salt"))))
    return tuple(signature)

def _evt_tail_splice_standardized_draws(z_pool: np.ndarray, z_draws: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pool = np.asarray(z_pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return z_draws
    if pool.size < 20:
        return _standardize_generated_innovations(z_draws)
    tail_draws = _evt_standardized_draws(pool, z_draws.shape, rng)
    threshold = float(np.quantile(np.abs(pool), 0.90))
    extreme = np.abs(z_draws) >= threshold
    spliced = np.where(extreme, tail_draws, z_draws)
    return _standardize_generated_innovations(spliced)

def _mcmc_effective_sample_size(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return float(n)
    centered = arr - float(np.mean(arr))
    var = float(np.dot(centered, centered) / n)
    if not np.isfinite(var) or var <= 1e-16:
        return float(n)
    max_lag = int(min(n - 1, max(1, math.floor(math.sqrt(float(n))))))
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / max((n - lag) * var, 1e-16))
        if not np.isfinite(rho) or rho <= 0.0:
            break
        rho_sum += rho
    return float(max(1.0, n / max(1.0 + 2.0 * rho_sum, 1e-12)))

def _split_rhat_from_chains(chains: np.ndarray) -> float:
    arr = np.asarray(chains, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    finite_cols = np.all(np.isfinite(arr), axis=0)
    arr = arr[:, finite_cols]
    chain_count, draw_count = arr.shape if arr.ndim == 2 else (0, 0)
    if chain_count < 2 or draw_count < 4:
        return math.inf
    half = draw_count // 2
    if half < 2:
        return math.inf
    split = np.concatenate([arr[:, :half], arr[:, -half:]], axis=0)
    m, n = split.shape
    chain_means = np.mean(split, axis=1)
    chain_vars = np.var(split, axis=1, ddof=1)
    w = float(np.mean(chain_vars))
    b = float(n * np.var(chain_means, ddof=1))
    if not np.isfinite(w) or w <= 1e-16:
        return 1.0 if np.isfinite(b) and b <= 1e-16 else math.inf
    var_hat = float(((n - 1.0) / n) * w + b / n)
    if not np.isfinite(var_hat) or var_hat < 0.0:
        return math.inf
    return float(math.sqrt(max(var_hat / w, 0.0)))

def _rank_normalized_chains(chains: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(chains, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    try:
        from scipy import stats

        flat = arr.reshape(-1)
        ranks = pd.Series(flat).rank(method="average").to_numpy(dtype=np.float64)
        u = (ranks - 0.375) / (float(flat.size) + 0.25)
        z = stats.norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))
        return z.reshape(arr.shape).astype(np.float64)
    except Exception:
        return None

def _mcmc_multichain_diagnostics(chains: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(chains, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    finite_cols = np.all(np.isfinite(arr), axis=0)
    arr = arr[:, finite_cols]
    chain_count, draw_count = arr.shape if arr.ndim == 2 else (0, 0)
    if chain_count < 2 or draw_count < 4:
        return {
            "rhat": math.inf,
            "ess": 0.0,
            "chains": int(chain_count),
            "draws_per_chain": int(draw_count),
            "rank_normalized": False,
        }
    raw_rhat = _split_rhat_from_chains(arr)
    rank_arr = _rank_normalized_chains(arr)
    rank_rhat = _split_rhat_from_chains(rank_arr) if rank_arr is not None else raw_rhat
    rhat = float(max(raw_rhat, rank_rhat))
    chain_ess = float(sum(_mcmc_effective_sample_size(arr[idx, :]) for idx in range(chain_count)))
    total_draws = float(chain_count * draw_count)
    rhat_adjusted_ess = total_draws / max(rhat * rhat, 1.0) if np.isfinite(rhat) else 0.0
    ess = float(max(1.0, min(chain_ess, rhat_adjusted_ess))) if total_draws > 0 else 0.0
    return {
        "rhat": rhat,
        "raw_split_rhat": float(raw_rhat),
        "rank_split_rhat": float(rank_rhat),
        "ess": ess,
        "chains": int(chain_count),
        "draws_per_chain": int(draw_count),
        "rank_normalized": rank_arr is not None,
    }

def _dynamic_mcmc_path_sample_interval(observed_log_var: Sequence[float]) -> int:
    arr = np.asarray(observed_log_var, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n <= 0:
        return 1
    size_stride = int(math.ceil(math.sqrt(float(n)) / 32.0))
    dependence_stride = 1
    if n >= 20:
        try:
            centered = arr - float(np.mean(arr))
            dependence_span = max(
                _politis_white_block_length(centered),
                _politis_white_block_length(centered * centered),
            )
            dependence_stride = int(math.ceil(float(dependence_span) / 8.0))
        except Exception:
            dependence_stride = 1
    return int(max(1, min(8, max(size_stride, dependence_stride))))

def _fit_bayesian_sbb_full_mcmc_sv_overlay(
    train_values: np.ndarray,
    candidate: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < FULL_MCMC_SV_MIN_OBS:
        return None
    mean_model = str(candidate.get("mean_model", "positive_sample_mean"))
    base_fit = _fit_sbb_mean_model(x, mean_model)
    if base_fit is None:
        return None

    residual_center = float(base_fit["sample_mu"])
    mean_dynamics = dict(base_fit.get("mean_dynamics", {}) or {})
    unclipped_measurement = bool(candidate.get("unclipped_sv_measurement", False))
    sv_measurement_bias = str(candidate.get("sv_measurement_bias", "log_chi_square_theoretical"))
    transformed_sampler = bool(candidate.get("transformed_parameter_mcmc", False))
    raw_empirical_innovations = bool(candidate.get("unclipped_empirical_innovations", False))
    innovation_standardization = str(candidate.get("innovation_standardization", "mean_std"))
    innovation_resampling = str(candidate.get("innovation_resampling", "stationary_bootstrap"))
    innovation_conditioning = str(candidate.get("innovation_conditioning", "none"))
    innovation_pool_source = str(candidate.get("innovation_pool_source", "sample_standardized"))
    leverage_alignment = str(candidate.get("leverage_alignment", "lagged_return"))
    leverage_correlation_method = str(candidate.get("leverage_correlation_method", "pearson"))
    leverage_correlation_scope = str(candidate.get("leverage_correlation_scope", "per_state_path"))
    state_innovation_distribution = str(candidate.get("state_innovation_distribution", "gaussian"))
    state_innovation_source = str(candidate.get("state_innovation_source", "posterior_ffbs_path"))
    state_innovation_standardization = str(candidate.get("state_innovation_standardization", "mean_std"))
    state_innovation_resampling = str(candidate.get("state_innovation_resampling", "iid"))
    state_innovation_coupling = str(candidate.get("state_innovation_coupling", "correlation_mixture"))
    sv_sigma_scale_method = str(candidate.get("sv_sigma_scale_method", "none"))
    innovation_tail_splice = str(candidate.get("innovation_tail_splice", "none"))
    latent_vol_persistence = str(candidate.get("latent_vol_persistence", "ar1"))
    no_latent_ar = latent_vol_persistence in {"none", "iid", "no_ar"}
    mcmc_stopping = str(candidate.get("mcmc_stopping", "fixed"))
    if str(mean_dynamics.get("type", "")) == "ar1_constant_ols":
        intercept = float(mean_dynamics.get("intercept", 0.0))
        ar_coef = float(mean_dynamics.get("ar_coef", 0.0))
        unconditional_mean = float(mean_dynamics.get("unconditional_mean", residual_center))
        fitted_mean = np.empty_like(x, dtype=np.float64)
        fitted_mean[0] = unconditional_mean
        fitted_mean[1:] = intercept + ar_coef * x[:-1]
        observed_log_var, eps_x = _sv_observed_log_variance_from_residuals(
            x - fitted_mean,
            winsorize=not unclipped_measurement,
        )
    else:
        observed_log_var, eps_x = _sv_observed_log_variance(
            x,
            residual_center,
            winsorize=not unclipped_measurement,
        )
    if observed_log_var.size < FULL_MCMC_SV_MIN_OBS or eps_x.size < FULL_MCMC_SV_MIN_OBS:
        return None
    sv_measurement_bias_value = float(SV_LOG_CHI_SQUARE_MEAN)
    if sv_measurement_bias == "empirical_standardized_residuals":
        z_for_bias = _standardized_empirical_innovation_pool(eps_x, clip=None, method=innovation_standardization)
        if z_for_bias is None or z_for_bias.size < 30:
            return None
        empirical_bias = float(np.mean(np.log(np.maximum(z_for_bias * z_for_bias, np.finfo(np.float64).tiny))))
        if not np.isfinite(empirical_bias):
            return None
        observed_log_var = observed_log_var + float(SV_LOG_CHI_SQUARE_MEAN - empirical_bias)
        sv_measurement_bias_value = empirical_bias
    level, phi, eta = _fit_sv_state_space_params(observed_log_var)
    if no_latent_ar:
        phi = 0.0
    rng = np.random.default_rng(
        deterministic_seed(
            "full_mcmc_sv_overlay_fit",
            repr(_full_mcmc_sv_overlay_fit_signature(candidate)),
            len(x),
            round(float(np.mean(x)), 10),
        )
    )
    if transformed_sampler and not no_latent_ar:
        phi_coord = _logit_unit_interval(phi)
        eta_coord = math.log(max(float(eta), np.finfo(np.float64).tiny))
        current_lp = _sv_transformed_log_posterior(observed_log_var, level, phi_coord, eta_coord)
    else:
        phi_coord = 0.0
        eta_coord = 0.0
        current_lp = _sv_mcmc_log_posterior(observed_log_var, level, phi, eta)
    iterations_limit = int(max(candidate.get("mcmc_iterations", 160), 40))
    burn = int(min(max(candidate.get("mcmc_burn", 60), 0), iterations_limit - 1))
    thin = int(max(candidate.get("mcmc_thin", 10), 1))
    if mcmc_stopping == "adaptive_ess_forecast_stability":
        min_iterations = int(min(max(candidate.get("mcmc_min_iterations", 120), burn + thin), iterations_limit))
        check_interval = int(max(candidate.get("mcmc_check_interval", 40), thin))
    else:
        min_iterations = iterations_limit
        check_interval = iterations_limit
    leverage = bool(candidate.get("leverage", False))
    samples: List[Tuple[float, float, float, float, float]] = []
    state_innovation_values: List[np.ndarray] = []
    paired_innovation_values: List[np.ndarray] = []
    accepted = 0
    iterations_run = 0
    mcmc_stop_reason = "fixed_iteration_budget"
    previous_forecast_quantiles: Optional[np.ndarray] = None
    final_ess_min: Optional[float] = None
    final_rhat_max: Optional[float] = None
    mcmc_diagnostic_checks: List[Dict[str, Any]] = []
    mcmc_chain_count = 1
    proposal_adaptation = str(candidate.get("mcmc_proposal_adaptation", "fixed"))
    proposal_adaptation_meta: Dict[str, Any] = {
        "method": proposal_adaptation,
        "active": False,
    }

    if innovation_pool_source == "latent_filtered_standardized":
        loglik, filtered_log_var, _ = _sv_kalman_filter(observed_log_var, level, phi, eta, return_path=True)
        if not np.isfinite(loglik) or filtered_log_var.size == 0:
            return None
        sigma_x = np.exp(0.5 * np.clip(filtered_log_var, -18.0, 18.0))
        latent_standardized = eps_x[: sigma_x.size] / np.maximum(sigma_x, 1e-8)
        z_pool = _standardized_empirical_innovation_pool(
            latent_standardized,
            clip=None if raw_empirical_innovations else 12.0,
            method=innovation_standardization,
        )
        if z_pool is None:
            return None
    elif raw_empirical_innovations:
        z_pool = _standardized_empirical_innovation_pool(eps_x, clip=None, method=innovation_standardization)
        if z_pool is None:
            return None
    else:
        z_pool = np.asarray(base_fit.get("standardized_residuals"), dtype=np.float64)
        z_pool = z_pool[np.isfinite(z_pool)]
        z_pool = z_pool[: observed_log_var.size]
    if z_pool.size < observed_log_var.size:
        return None
    innovation_distribution_params: Optional[Tuple[float, ...]] = None
    if _full_mcmc_sv_parametric_innovation_meta(innovation_resampling) is not None:
        innovation_distribution_params = _fit_full_mcmc_sv_parametric_innovation_distribution(
            z_pool,
            innovation_resampling,
        )
        if innovation_distribution_params is None:
            return None

    if mcmc_stopping == "dynamic_multichain_rhat_ess_forecast_stability":
        mcmc_chain_count = int(max(candidate.get("mcmc_chains", 4), 2))
        initial_iterations = int(max(candidate.get("mcmc_initial_iterations", 160), burn + thin, 40))
        extend_iterations = int(max(candidate.get("mcmc_extend_iterations", 80), thin, 1))
        iterations_limit = int(max(candidate.get("mcmc_max_iterations", candidate.get("mcmc_iterations", 320)), initial_iterations))
        rhat_threshold = float(candidate.get("mcmc_rhat_threshold", 1.01))
        min_ess_config = float(candidate.get("mcmc_min_ess", 100.0))
        ess_target = float(max(min_ess_config, min(400.0, math.sqrt(float(observed_log_var.size)))))
        stability_tolerance = float(candidate.get("mcmc_forecast_stability_tolerance", 0.05))
        path_sample_interval = _dynamic_mcmc_path_sample_interval(observed_log_var)
        proposal_adaptation = str(candidate.get("mcmc_proposal_adaptation", "fixed"))
        adaptive_proposal = bool(
            proposal_adaptation == "warmup_empirical_covariance" and transformed_sampler and not no_latent_ar
        )
        adaptation_iterations = int(
            min(
                max(int(candidate.get("mcmc_adaptation_iterations", burn)), 20),
                max(iterations_limit - 1, 20),
            )
        )
        target_acceptance = float(candidate.get("mcmc_adaptation_target_acceptance", 0.234))
        proposal_chol: Optional[np.ndarray] = None
        proposal_adaptation_meta: Dict[str, Any] = {
            "method": proposal_adaptation,
            "active": adaptive_proposal,
            "adaptation_iterations": int(adaptation_iterations),
            "target_acceptance": target_acceptance,
        }
        chain_states: List[Dict[str, Any]] = []
        for chain_idx in range(mcmc_chain_count):
            chain_rng = np.random.default_rng(
                deterministic_seed(
                    "full_mcmc_sv_overlay_dynamic_chain",
                    repr(_full_mcmc_sv_overlay_fit_signature(candidate)),
                    len(x),
                    round(float(np.mean(x)), 10),
                    int(chain_idx),
                )
            )
            chain_level = float(level + chain_rng.normal(0.0, 0.02))
            if no_latent_ar:
                chain_phi = 0.0
            else:
                chain_phi = float(np.clip(phi + chain_rng.normal(0.0, 0.01), 0.001, 0.994))
            chain_eta = float(np.clip(eta * math.exp(chain_rng.normal(0.0, 0.05)), 1e-4, 5.0))
            if transformed_sampler and not no_latent_ar:
                chain_phi_coord = _logit_unit_interval(chain_phi)
                chain_eta_coord = math.log(max(float(chain_eta), np.finfo(np.float64).tiny))
                chain_lp = _sv_transformed_log_posterior(observed_log_var, chain_level, chain_phi_coord, chain_eta_coord)
            else:
                chain_phi_coord = 0.0
                chain_eta_coord = 0.0
                chain_lp = _sv_mcmc_log_posterior(observed_log_var, chain_level, chain_phi, chain_eta)
            chain_states.append(
                {
                    "level": chain_level,
                    "phi": chain_phi,
                    "eta": chain_eta,
                    "phi_coord": chain_phi_coord,
                    "eta_coord": chain_eta_coord,
                    "current_lp": chain_lp,
                    "rng": chain_rng,
                    "param_values": [],
                    "samples": [],
                    "state_innovation_values": [],
                    "paired_innovation_values": [],
                    "accepted": 0,
                    "adapt_values": [],
                    "accepted_during_adaptation": 0,
                    "proposal_sq_jump_sum": 0.0,
                }
            )
        mcmc_stop_reason = "dynamic_max_iteration_budget"
        for step in range(iterations_limit):
            iterations_run = int(step + 1)
            if adaptive_proposal and proposal_chol is None and step >= adaptation_iterations:
                adapted_vectors = [
                    np.asarray(chain_state["adapt_values"], dtype=np.float64)
                    for chain_state in chain_states
                    if len(chain_state["adapt_values"]) >= 4
                ]
                if adapted_vectors:
                    adapted = np.concatenate(adapted_vectors, axis=0)
                    finite = np.all(np.isfinite(adapted), axis=1)
                    adapted = adapted[finite]
                    if adapted.ndim == 2 and adapted.shape[0] >= max(12, adapted.shape[1] + 2):
                        dim = int(adapted.shape[1])
                        cov = np.cov(adapted, rowvar=False)
                        if np.ndim(cov) == 0:
                            cov = np.asarray([[float(cov)]], dtype=np.float64)
                        cov = np.asarray(cov, dtype=np.float64)
                        if cov.shape == (dim, dim) and np.all(np.isfinite(cov)):
                            diag_floor = np.asarray([0.03, 0.06, 0.05], dtype=np.float64)[:dim] ** 2 * 0.01
                            total_proposals = max(adaptation_iterations * mcmc_chain_count, 1)
                            adaptation_acceptance = float(
                                sum(int(chain_state["accepted_during_adaptation"]) for chain_state in chain_states)
                                / total_proposals
                            )
                            scale = float((2.38 * 2.38) / max(dim, 1))
                            if adaptation_acceptance < 0.5 * target_acceptance:
                                scale *= 0.5
                            elif adaptation_acceptance > max(0.40, 1.75 * target_acceptance):
                                scale *= 1.5
                            regularized = scale * cov + np.diag(diag_floor)
                            try:
                                proposal_chol = np.linalg.cholesky(regularized)
                            except np.linalg.LinAlgError:
                                proposal_chol = np.linalg.cholesky(np.diag(np.maximum(np.diag(regularized), diag_floor)))
                            proposal_adaptation_meta.update(
                                {
                                    "accepted_during_adaptation": int(
                                        sum(int(chain_state["accepted_during_adaptation"]) for chain_state in chain_states)
                                    ),
                                    "adaptation_acceptance_rate": adaptation_acceptance,
                                    "adaptation_esjd": float(
                                        sum(float(chain_state["proposal_sq_jump_sum"]) for chain_state in chain_states)
                                        / max(
                                            sum(int(chain_state["accepted_during_adaptation"]) for chain_state in chain_states),
                                            1,
                                        )
                                    ),
                                    "proposal_dimension": dim,
                                    "proposal_scale": scale,
                                    "proposal_covariance_diag": [float(value) for value in np.diag(regularized)],
                                }
                            )
            for chain_state in chain_states:
                chain_rng = chain_state["rng"]
                chain_level = float(chain_state["level"])
                chain_phi = float(chain_state["phi"])
                chain_eta = float(chain_state["eta"])
                chain_phi_coord = float(chain_state["phi_coord"])
                chain_eta_coord = float(chain_state["eta_coord"])
                chain_lp = float(chain_state["current_lp"])
                current_vector = np.asarray([chain_level, chain_phi_coord, chain_eta_coord], dtype=np.float64)
                proposal_vector = current_vector.copy()
                if adaptive_proposal and proposal_chol is not None:
                    proposal_vector = current_vector + proposal_chol @ chain_rng.normal(size=current_vector.size)
                    proposal_level = float(proposal_vector[0])
                    proposal_phi_coord = float(proposal_vector[1])
                    proposal_eta_coord = float(proposal_vector[2])
                    proposal_phi = _inv_logit_unit_interval(proposal_phi_coord)
                    proposal_eta = float(math.exp(proposal_eta_coord)) if np.isfinite(proposal_eta_coord) else math.inf
                    proposal_lp = (
                        _sv_transformed_log_posterior(
                            observed_log_var,
                            proposal_level,
                            proposal_phi_coord,
                            proposal_eta_coord,
                        )
                        if np.isfinite(proposal_level) and np.isfinite(proposal_phi_coord) and np.isfinite(proposal_eta)
                        else -math.inf
                    )
                else:
                    proposal_level = float(chain_level + chain_rng.normal(0.0, 0.03))
                    if no_latent_ar:
                        proposal_phi = 0.0
                        proposal_eta = float(np.clip(chain_eta * math.exp(chain_rng.normal(0.0, 0.05)), 1e-4, 5.0))
                        proposal_lp = _sv_mcmc_log_posterior(observed_log_var, proposal_level, proposal_phi, proposal_eta)
                        proposal_phi_coord = 0.0
                        proposal_eta_coord = 0.0
                        proposal_vector = np.asarray(
                            [proposal_level, proposal_phi_coord, math.log(max(proposal_eta, np.finfo(np.float64).tiny))],
                            dtype=np.float64,
                        )
                    elif transformed_sampler:
                        proposal_phi_coord = float(chain_phi_coord + chain_rng.normal(0.0, 0.06))
                        proposal_eta_coord = float(chain_eta_coord + chain_rng.normal(0.0, 0.05))
                        proposal_phi = _inv_logit_unit_interval(proposal_phi_coord)
                        proposal_eta = float(math.exp(proposal_eta_coord))
                        proposal_lp = _sv_transformed_log_posterior(
                            observed_log_var,
                            proposal_level,
                            proposal_phi_coord,
                            proposal_eta_coord,
                        )
                        proposal_vector = np.asarray([proposal_level, proposal_phi_coord, proposal_eta_coord], dtype=np.float64)
                    else:
                        proposal_phi = float(np.clip(chain_phi + chain_rng.normal(0.0, 0.012), 0.001, 0.994))
                        proposal_eta = float(np.clip(chain_eta * math.exp(chain_rng.normal(0.0, 0.05)), 1e-4, 5.0))
                        proposal_lp = _sv_mcmc_log_posterior(observed_log_var, proposal_level, proposal_phi, proposal_eta)
                        proposal_phi_coord = 0.0
                        proposal_eta_coord = 0.0
                        proposal_vector = np.asarray(
                            [proposal_level, proposal_phi, math.log(max(proposal_eta, np.finfo(np.float64).tiny))],
                            dtype=np.float64,
                        )
                accepted_move = False
                if math.log(max(float(chain_rng.random()), 1e-300)) < proposal_lp - chain_lp:
                    chain_state["level"] = proposal_level
                    chain_state["phi"] = proposal_phi
                    chain_state["eta"] = proposal_eta
                    if transformed_sampler and not no_latent_ar:
                        chain_state["phi_coord"] = proposal_phi_coord
                        chain_state["eta_coord"] = proposal_eta_coord
                    chain_state["current_lp"] = proposal_lp
                    chain_state["accepted"] = int(chain_state["accepted"]) + 1
                    accepted_move = True
                if adaptive_proposal and step < adaptation_iterations:
                    adapted_vector = np.asarray(
                        [
                            float(chain_state["level"]),
                            float(chain_state["phi_coord"]),
                            float(chain_state["eta_coord"]),
                        ],
                        dtype=np.float64,
                    )
                    if np.all(np.isfinite(adapted_vector)):
                        chain_state["adapt_values"].append(adapted_vector)
                    if accepted_move:
                        chain_state["accepted_during_adaptation"] = int(chain_state["accepted_during_adaptation"]) + 1
                        if np.all(np.isfinite(proposal_vector)) and np.all(np.isfinite(current_vector)):
                            chain_state["proposal_sq_jump_sum"] = float(chain_state["proposal_sq_jump_sum"]) + float(
                                np.sum((proposal_vector - current_vector) ** 2)
                            )
                if step >= burn:
                    chain_state["param_values"].append(
                        (
                            float(chain_state["level"]),
                            float(chain_state["phi"]),
                            float(chain_state["eta"]),
                        )
                    )
                if step >= burn and (step - burn) % path_sample_interval == 0:
                    state_path = _ffbs_sv_log_variance_path(
                        observed_log_var,
                        float(chain_state["level"]),
                        float(chain_state["phi"]),
                        float(chain_state["eta"]),
                        chain_rng,
                        clip_path=not unclipped_measurement,
                    )
                    if state_path is None or state_path.size < 2:
                        continue
                    rho = 0.0
                    if leverage:
                        state_innov = (
                            state_path[1:]
                            - float(chain_state["level"])
                            - float(chain_state["phi"]) * (state_path[:-1] - float(chain_state["level"]))
                        ) / max(float(chain_state["eta"]), 1e-8)
                        if state_innovation_distribution == "empirical_centered_standardized":
                            if state_innovation_source == "kalman_filtered_path":
                                loglik, filtered_path, _ = _sv_kalman_filter(
                                    observed_log_var,
                                    float(chain_state["level"]),
                                    float(chain_state["phi"]),
                                    float(chain_state["eta"]),
                                    return_path=True,
                                )
                                if np.isfinite(loglik) and filtered_path.size >= 2:
                                    filtered_innov = (
                                        filtered_path[1:]
                                        - float(chain_state["level"])
                                        - float(chain_state["phi"]) * (filtered_path[:-1] - float(chain_state["level"]))
                                    ) / max(float(chain_state["eta"]), 1e-8)
                                    chain_state["state_innovation_values"].append(filtered_innov.astype(np.float64))
                            else:
                                chain_state["state_innovation_values"].append(state_innov.astype(np.float64))
                        if leverage_alignment == "same_period_return":
                            z_for_rho = z_pool[1 : state_innov.size + 1]
                        else:
                            z_for_rho = z_pool[: state_innov.size]
                        if state_innov.size == z_for_rho.size and state_innov.size >= 30:
                            if (
                                state_innovation_resampling == "paired_stationary_bootstrap"
                                or leverage_correlation_scope == "pooled_posterior"
                            ):
                                chain_state["paired_innovation_values"].append(
                                    np.column_stack(
                                        [
                                            z_for_rho.astype(np.float64),
                                            state_innov.astype(np.float64),
                                        ]
                                    )
                                )
                            corr = _finite_correlation(z_for_rho, state_innov, method=leverage_correlation_method)
                            if np.isfinite(corr):
                                if raw_empirical_innovations and abs(corr) < 1.0:
                                    rho = float(corr)
                                else:
                                    rho = float(np.clip(corr, -0.95, 0.95))
                    chain_state["samples"].append(
                        (
                            float(chain_state["level"]),
                            float(chain_state["phi"]),
                            float(chain_state["eta"]),
                            float(state_path[-1]),
                            float(rho),
                        )
                    )
            should_check = iterations_run >= initial_iterations and (
                iterations_run == initial_iterations or (iterations_run - initial_iterations) % extend_iterations == 0
            )
            if should_check:
                retained_param_counts = [len(chain_state["param_values"]) for chain_state in chain_states]
                min_param_retained = int(min(retained_param_counts)) if retained_param_counts else 0
                retained_path_counts = [len(chain_state["samples"]) for chain_state in chain_states]
                min_path_retained = int(min(retained_path_counts)) if retained_path_counts else 0
                if min_param_retained >= 8 and min_path_retained >= 8:
                    param_arrays = [
                        np.asarray(chain_state["param_values"][-min_param_retained:], dtype=np.float64)
                        for chain_state in chain_states
                    ]
                    sample_arrays = [
                        np.asarray(chain_state["samples"][-min_path_retained:], dtype=np.float64)
                        for chain_state in chain_states
                    ]
                    param_diags: Dict[str, Dict[str, Any]] = {}
                    for col_idx, name in enumerate(("level", "phi", "eta")):
                        chain_values = np.vstack([arr[:, col_idx] for arr in param_arrays])
                        param_diags[name] = _mcmc_multichain_diagnostics(chain_values)
                    chain_last_log_variance = np.vstack([arr[:, 3] for arr in sample_arrays])
                    param_diags["last_log_variance"] = _mcmc_multichain_diagnostics(chain_last_log_variance)
                    final_rhat_max = float(max(diag["rhat"] for diag in param_diags.values()))
                    final_ess_min = float(min(diag["ess"] for diag in param_diags.values()))
                    pooled_last_log_var = np.concatenate([arr[:, 3] for arr in sample_arrays])
                    forecast_quantiles = np.quantile(pooled_last_log_var, [0.05, 0.50, 0.95])
                    forecast_scale = float(max(np.std(pooled_last_log_var, ddof=1), 1e-6))
                    stable = False
                    if previous_forecast_quantiles is not None:
                        stable = bool(
                            np.max(np.abs(forecast_quantiles - previous_forecast_quantiles))
                            <= stability_tolerance * forecast_scale
                        )
                    previous_forecast_quantiles = forecast_quantiles
                    mcmc_diagnostic_checks.append(
                        {
                            "iterations": int(iterations_run),
                            "retained_parameter_draws_per_chain": int(min_param_retained),
                            "retained_path_draws_per_chain": int(min_path_retained),
                            "path_sample_interval": int(path_sample_interval),
                            "rhat_max": final_rhat_max,
                            "ess_min": final_ess_min,
                            "ess_target": ess_target,
                            "forecast_quantiles_stable": bool(stable),
                            "forecast_stability_tolerance": stability_tolerance,
                            "parameter_diagnostics": param_diags,
                        }
                    )
                    if final_rhat_max <= rhat_threshold and final_ess_min >= ess_target and stable:
                        mcmc_stop_reason = "dynamic_rhat_ess_and_forecast_quantile_stability"
                        break
        for chain_state in chain_states:
            samples.extend([tuple(sample) for sample in chain_state["samples"]])
            state_innovation_values.extend(
                value for value in chain_state["state_innovation_values"] if isinstance(value, np.ndarray)
            )
            paired_innovation_values.extend(
                value for value in chain_state["paired_innovation_values"] if isinstance(value, np.ndarray)
            )
            accepted += int(chain_state["accepted"])
    else:
        adaptive_proposal = bool(
            proposal_adaptation in {"adaptive_metropolis_warmup_covariance", "warmup_empirical_covariance"}
            and transformed_sampler
            and not no_latent_ar
        )
        adaptation_iterations = int(
            min(
                max(int(candidate.get("mcmc_adaptation_iterations", burn)), 20),
                max(iterations_limit - 1, 20),
            )
        )
        target_acceptance = float(candidate.get("mcmc_adaptation_target_acceptance", 0.234))
        proposal_chol: Optional[np.ndarray] = None
        adapt_values: List[np.ndarray] = []
        accepted_during_adaptation = 0
        proposal_sq_jump_sum = 0.0
        if adaptive_proposal:
            proposal_adaptation_meta = {
                "method": proposal_adaptation,
                "active": True,
                "adaptation_iterations": int(adaptation_iterations),
                "target_acceptance": target_acceptance,
                "adaptation_scope": "single_chain_warmup_then_frozen_for_retained_samples",
            }
        for step in range(iterations_limit):
            iterations_run = int(step + 1)
            if adaptive_proposal and proposal_chol is None and step >= adaptation_iterations:
                adapted = np.asarray(adapt_values, dtype=np.float64)
                finite = np.all(np.isfinite(adapted), axis=1) if adapted.ndim == 2 else np.asarray([], dtype=bool)
                adapted = adapted[finite] if adapted.ndim == 2 else np.empty((0, 3), dtype=np.float64)
                if adapted.ndim == 2 and adapted.shape[0] >= max(12, adapted.shape[1] + 2):
                    dim = int(adapted.shape[1])
                    cov = np.cov(adapted, rowvar=False)
                    if np.ndim(cov) == 0:
                        cov = np.asarray([[float(cov)]], dtype=np.float64)
                    cov = np.asarray(cov, dtype=np.float64)
                    if cov.shape == (dim, dim) and np.all(np.isfinite(cov)):
                        diag_floor = np.asarray([0.03, 0.06, 0.05], dtype=np.float64)[:dim] ** 2 * 0.01
                        adaptation_acceptance = float(accepted_during_adaptation / max(adaptation_iterations, 1))
                        scale = float((2.38 * 2.38) / max(dim, 1))
                        if adaptation_acceptance < 0.5 * target_acceptance:
                            scale *= 0.5
                        elif adaptation_acceptance > max(0.40, 1.75 * target_acceptance):
                            scale *= 1.5
                        regularized = scale * cov + np.diag(diag_floor)
                        try:
                            proposal_chol = np.linalg.cholesky(regularized)
                        except np.linalg.LinAlgError:
                            proposal_chol = np.linalg.cholesky(np.diag(np.maximum(np.diag(regularized), diag_floor)))
                        proposal_adaptation_meta.update(
                            {
                                "accepted_during_adaptation": int(accepted_during_adaptation),
                                "adaptation_acceptance_rate": adaptation_acceptance,
                                "adaptation_esjd": float(proposal_sq_jump_sum / max(accepted_during_adaptation, 1)),
                                "proposal_dimension": dim,
                                "proposal_scale": scale,
                                "proposal_covariance_diag": [float(value) for value in np.diag(regularized)],
                            }
                        )
                if proposal_chol is None:
                    fallback_diag = np.asarray([0.03, 0.06, 0.05], dtype=np.float64)
                    proposal_chol = np.diag(fallback_diag)
                    proposal_adaptation_meta.update(
                        {
                            "fallback_reason": "insufficient_or_nonfinite_warmup_covariance",
                            "proposal_dimension": int(fallback_diag.size),
                            "proposal_covariance_diag": [float(value * value) for value in fallback_diag],
                        }
                    )
            current_vector = np.asarray([level, phi_coord, eta_coord], dtype=np.float64)
            proposal_vector = current_vector.copy()
            if adaptive_proposal and proposal_chol is not None:
                proposal_vector = current_vector + proposal_chol @ rng.normal(size=current_vector.size)
                proposal_level = float(proposal_vector[0])
                proposal_phi_coord = float(proposal_vector[1])
                proposal_eta_coord = float(proposal_vector[2])
                if np.isfinite(proposal_level) and np.isfinite(proposal_phi_coord) and np.isfinite(proposal_eta_coord):
                    proposal_phi = _inv_logit_unit_interval(proposal_phi_coord)
                    proposal_eta = float(math.exp(proposal_eta_coord)) if -20.0 <= proposal_eta_coord <= 20.0 else math.inf
                    proposal_lp = (
                        _sv_transformed_log_posterior(
                            observed_log_var,
                            proposal_level,
                            proposal_phi_coord,
                            proposal_eta_coord,
                        )
                        if np.isfinite(proposal_eta)
                        else -math.inf
                    )
                else:
                    proposal_phi = phi
                    proposal_eta = eta
                    proposal_lp = -math.inf
            else:
                proposal_level = float(level + rng.normal(0.0, 0.03))
                if no_latent_ar:
                    proposal_phi = 0.0
                    proposal_eta = float(np.clip(eta * math.exp(rng.normal(0.0, 0.05)), 1e-4, 5.0))
                    proposal_lp = _sv_mcmc_log_posterior(observed_log_var, proposal_level, proposal_phi, proposal_eta)
                    proposal_vector = np.asarray(
                        [proposal_level, 0.0, math.log(max(proposal_eta, np.finfo(np.float64).tiny))],
                        dtype=np.float64,
                    )
                elif transformed_sampler:
                    proposal_phi_coord = float(phi_coord + rng.normal(0.0, 0.06))
                    proposal_eta_coord = float(eta_coord + rng.normal(0.0, 0.05))
                    proposal_phi = _inv_logit_unit_interval(proposal_phi_coord)
                    proposal_eta = float(math.exp(proposal_eta_coord))
                    proposal_lp = _sv_transformed_log_posterior(
                        observed_log_var,
                        proposal_level,
                        proposal_phi_coord,
                        proposal_eta_coord,
                    )
                    proposal_vector = np.asarray([proposal_level, proposal_phi_coord, proposal_eta_coord], dtype=np.float64)
                else:
                    proposal_phi = float(np.clip(phi + rng.normal(0.0, 0.012), 0.001, 0.994))
                    proposal_eta = float(np.clip(eta * math.exp(rng.normal(0.0, 0.05)), 1e-4, 5.0))
                    proposal_lp = _sv_mcmc_log_posterior(observed_log_var, proposal_level, proposal_phi, proposal_eta)
                    proposal_vector = np.asarray(
                        [proposal_level, proposal_phi, math.log(max(proposal_eta, np.finfo(np.float64).tiny))],
                        dtype=np.float64,
                    )
            accepted_move = False
            if math.log(max(float(rng.random()), 1e-300)) < proposal_lp - current_lp:
                level, phi, eta = proposal_level, proposal_phi, proposal_eta
                if transformed_sampler and not no_latent_ar:
                    phi_coord, eta_coord = proposal_phi_coord, proposal_eta_coord
                current_lp = proposal_lp
                accepted += 1
                accepted_move = True
            if adaptive_proposal and step < adaptation_iterations:
                adapted_vector = np.asarray([level, phi_coord, eta_coord], dtype=np.float64)
                if np.all(np.isfinite(adapted_vector)):
                    adapt_values.append(adapted_vector)
                if accepted_move:
                    accepted_during_adaptation += 1
                    if np.all(np.isfinite(proposal_vector)) and np.all(np.isfinite(current_vector)):
                        proposal_sq_jump_sum += float(np.sum((proposal_vector - current_vector) ** 2))
            if step >= burn and (step - burn) % thin == 0:
                state_path = _ffbs_sv_log_variance_path(
                    observed_log_var,
                    level,
                    phi,
                    eta,
                    rng,
                    clip_path=not unclipped_measurement,
                )
                if state_path is None or state_path.size < 2:
                    continue
                rho = 0.0
                if leverage:
                    state_innov = (state_path[1:] - level - phi * (state_path[:-1] - level)) / max(eta, 1e-8)
                    if state_innovation_distribution == "empirical_centered_standardized":
                        if state_innovation_source == "kalman_filtered_path":
                            loglik, filtered_path, _ = _sv_kalman_filter(observed_log_var, level, phi, eta, return_path=True)
                            if np.isfinite(loglik) and filtered_path.size >= 2:
                                filtered_innov = (filtered_path[1:] - level - phi * (filtered_path[:-1] - level)) / max(eta, 1e-8)
                                state_innovation_values.append(filtered_innov.astype(np.float64))
                        else:
                            state_innovation_values.append(state_innov.astype(np.float64))
                    if leverage_alignment == "same_period_return":
                        z_for_rho = z_pool[1 : state_innov.size + 1]
                    else:
                        z_for_rho = z_pool[: state_innov.size]
                    if state_innov.size == z_for_rho.size and state_innov.size >= 30:
                        if (
                            state_innovation_resampling == "paired_stationary_bootstrap"
                            or leverage_correlation_scope == "pooled_posterior"
                        ):
                            paired_innovation_values.append(
                                np.column_stack(
                                    [
                                        z_for_rho.astype(np.float64),
                                        state_innov.astype(np.float64),
                                    ]
                                )
                            )
                        corr = _finite_correlation(z_for_rho, state_innov, method=leverage_correlation_method)
                        if np.isfinite(corr):
                            if raw_empirical_innovations and abs(corr) < 1.0:
                                rho = float(corr)
                            else:
                                rho = float(np.clip(corr, -0.95, 0.95))
                samples.append((float(level), float(phi), float(eta), float(state_path[-1]), float(rho)))
            if (
                mcmc_stopping == "adaptive_ess_forecast_stability"
                and iterations_run >= min_iterations
                and iterations_run % check_interval == 0
                and len(samples) >= 8
            ):
                arr = np.asarray(samples, dtype=np.float64)
                ess_values = [
                    _mcmc_effective_sample_size(arr[:, 0]),
                    _mcmc_effective_sample_size(arr[:, 1]),
                    _mcmc_effective_sample_size(arr[:, 2]),
                    _mcmc_effective_sample_size(arr[:, 3]),
                ]
                final_ess_min = float(min(ess_values))
                forecast_quantiles = np.quantile(arr[:, 3], [0.05, 0.50, 0.95])
                forecast_scale = float(max(np.std(arr[:, 3], ddof=1), 1e-6))
                ess_target = float(max(8.0, min(24.0, math.sqrt(float(observed_log_var.size)) / 2.0)))
                stable = False
                if previous_forecast_quantiles is not None:
                    stable = bool(np.max(np.abs(forecast_quantiles - previous_forecast_quantiles)) <= 0.10 * forecast_scale)
                previous_forecast_quantiles = forecast_quantiles
                if final_ess_min >= ess_target and stable:
                    mcmc_stop_reason = "adaptive_ess_and_forecast_quantile_stability"
                    break
    if not samples:
        state_path = _ffbs_sv_log_variance_path(
            observed_log_var,
            level,
            phi,
            eta,
            rng,
            clip_path=not unclipped_measurement,
        )
        if state_path is None:
            return None
        samples = [(float(level), float(phi), float(eta), float(state_path[-1]), 0.0)]

    if leverage_correlation_scope == "pooled_posterior" and paired_innovation_values:
        pooled = np.concatenate(paired_innovation_values, axis=0)
        pooled_rho = _finite_correlation(pooled[:, 0], pooled[:, 1], method=leverage_correlation_method)
        if np.isfinite(pooled_rho):
            if raw_empirical_innovations and abs(float(pooled_rho)) < 1.0:
                rho_value = float(pooled_rho)
            else:
                rho_value = float(np.clip(pooled_rho, -0.95, 0.95))
            samples = [(a, b, c, d, rho_value) for a, b, c, d, _ in samples]
    elif leverage_correlation_scope == "posterior_median" and samples:
        rho_arr = np.asarray([sample[4] for sample in samples], dtype=np.float64)
        rho_arr = rho_arr[np.isfinite(rho_arr)]
        if rho_arr.size:
            rho_value = float(np.median(rho_arr))
            samples = [(a, b, c, d, rho_value) for a, b, c, d, _ in samples]

    state_innovation_pool: Optional[np.ndarray] = None
    state_innovation_t_params: Optional[Tuple[float, float, float]] = None
    if state_innovation_distribution == "empirical_centered_standardized" and state_innovation_values:
        state_innovation_pool = _standardized_empirical_innovation_pool(
            np.concatenate(state_innovation_values),
            clip=None,
            method=state_innovation_standardization,
        )
        if state_innovation_pool is None:
            return None
        if state_innovation_resampling == "student_t_mle":
            try:
                from scipy import stats

                t_df, t_loc, t_scale = stats.t.fit(state_innovation_pool, floc=0.0)
                if not (np.isfinite(t_df) and np.isfinite(t_loc) and np.isfinite(t_scale)) or t_df <= 0.0 or t_scale <= 0.0:
                    return None
                state_innovation_t_params = (float(t_df), float(t_loc), float(t_scale))
            except Exception:
                return None
    joint_innovation_pool: Optional[np.ndarray] = None
    if state_innovation_resampling == "paired_stationary_bootstrap" and paired_innovation_values:
        joint = np.concatenate(paired_innovation_values, axis=0)
        finite = np.isfinite(joint[:, 0]) & np.isfinite(joint[:, 1])
        joint = joint[finite]
        z_joint = _standardized_empirical_innovation_pool(joint[:, 0], clip=None, method=innovation_standardization)
        state_joint = _standardized_empirical_innovation_pool(
            joint[:, 1],
            clip=None,
            method=state_innovation_standardization,
        )
        if z_joint is None or state_joint is None or z_joint.size != state_joint.size:
            return None
        joint_innovation_pool = np.column_stack([z_joint, state_joint]).astype(np.float64)

    conditioned_innovation_low_pool: Optional[np.ndarray] = None
    conditioned_innovation_high_pool: Optional[np.ndarray] = None
    innovation_condition_threshold: Optional[float] = None
    if innovation_conditioning == "filtered_log_volatility_median":
        loglik, filtered_for_conditioning, _ = _sv_kalman_filter(observed_log_var, level, phi, eta, return_path=True)
        if not np.isfinite(loglik) or filtered_for_conditioning.size < 60:
            return None
        conditioning_len = min(int(filtered_for_conditioning.size), int(z_pool.size))
        threshold = float(np.median(filtered_for_conditioning[:conditioning_len]))
        low_mask = filtered_for_conditioning[:conditioning_len] < threshold
        high_mask = ~low_mask
        conditioned_innovation_low_pool = z_pool[:conditioning_len][low_mask]
        conditioned_innovation_high_pool = z_pool[:conditioning_len][high_mask]
        conditioned_innovation_low_pool = conditioned_innovation_low_pool[np.isfinite(conditioned_innovation_low_pool)]
        conditioned_innovation_high_pool = conditioned_innovation_high_pool[np.isfinite(conditioned_innovation_high_pool)]
        if conditioned_innovation_low_pool.size < 30 or conditioned_innovation_high_pool.size < 30:
            return None
        innovation_condition_threshold = threshold

    sigma_scale = 1.0
    if sv_sigma_scale_method != "none":
        target_var = float(np.var(eps_x, ddof=1)) if eps_x.size > 1 else math.nan
        model_var = math.nan
        if sv_sigma_scale_method == "filtered_mean_variance_target":
            loglik, filtered_log_var, _ = _sv_kalman_filter(observed_log_var, level, phi, eta, return_path=True)
            if np.isfinite(loglik) and filtered_log_var.size:
                model_var = float(np.mean(np.exp(np.clip(filtered_log_var, -60.0, 60.0))))
        elif sv_sigma_scale_method == "posterior_stationary_variance_target" and samples:
            stationary_vars: List[float] = []
            for sample_level, sample_phi, sample_eta, _, _ in samples:
                denom = max(1.0 - float(sample_phi) * float(sample_phi), 1e-8)
                stationary_log_var = float(sample_level) + 0.5 * float(sample_eta) * float(sample_eta) / denom
                stationary_vars.append(float(math.exp(float(np.clip(stationary_log_var, -60.0, 60.0)))))
            if stationary_vars:
                model_var = float(np.mean(stationary_vars))
        if not (np.isfinite(target_var) and np.isfinite(model_var)) or target_var <= 0.0 or model_var <= 0.0:
            return None
        sigma_scale = float(math.sqrt(target_var / model_var))
        if not np.isfinite(sigma_scale) or sigma_scale <= 0.0:
            return None

    if mcmc_stopping == "adaptive_ess_forecast_stability" and mcmc_stop_reason == "fixed_iteration_budget":
        mcmc_stop_reason = "adaptive_max_iteration_budget"
    if final_ess_min is None and samples:
        sample_arr = np.asarray(samples, dtype=np.float64)
        final_ess_min = float(
            min(
                _mcmc_effective_sample_size(sample_arr[:, 0]),
                _mcmc_effective_sample_size(sample_arr[:, 1]),
                _mcmc_effective_sample_size(sample_arr[:, 2]),
                _mcmc_effective_sample_size(sample_arr[:, 3]),
            )
        )
    if (
        mcmc_stopping == "dynamic_multichain_rhat_ess_forecast_stability"
        and final_rhat_max is None
        and mcmc_diagnostic_checks
    ):
        final_rhat_max = float(mcmc_diagnostic_checks[-1].get("rhat_max", math.inf))
    mcmc_iteration_count = int(iterations_run or iterations_limit)
    mcmc_total_chain_iterations = int(max(1, mcmc_iteration_count * max(int(mcmc_chain_count), 1)))
    parametric_innovation_meta = _full_mcmc_sv_parametric_innovation_meta(innovation_resampling)
    fit_tail_method = (
        parametric_innovation_meta[0]
        if parametric_innovation_meta is not None
        else "raw_empirical_untruncated"
        if raw_empirical_innovations
        else "filtered_empirical_tail"
    )
    fit_path_generator = (
        parametric_innovation_meta[1]
        if parametric_innovation_meta is not None
        else "paired_stationary_bootstrap_return_state_shocks_scaled_by_latent_sv_paths"
        if state_innovation_resampling == "paired_stationary_bootstrap"
        else "iid_standardized_residuals_scaled_by_latent_sv_paths"
        if innovation_resampling == "iid"
        else "stationary_bootstrap_standardized_residuals_scaled_by_latent_sv_paths"
    )

    return {
        "base_fit": base_fit,
        "innovation_pool": z_pool.astype(float) if raw_empirical_innovations or innovation_pool_source != "sample_standardized" else None,
        "innovation_distribution_params": list(innovation_distribution_params) if innovation_distribution_params is not None else None,
        "innovation_standardization": innovation_standardization,
        "innovation_resampling": innovation_resampling,
        "innovation_tail_splice": innovation_tail_splice,
        "innovation_conditioning": innovation_conditioning,
        "innovation_condition_threshold": innovation_condition_threshold,
        "conditioned_innovation_low_pool": (
            conditioned_innovation_low_pool.astype(float) if conditioned_innovation_low_pool is not None else None
        ),
        "conditioned_innovation_high_pool": (
            conditioned_innovation_high_pool.astype(float) if conditioned_innovation_high_pool is not None else None
        ),
        "innovation_pool_source": innovation_pool_source,
        "state_innovation_pool": state_innovation_pool.astype(float) if state_innovation_pool is not None else None,
        "state_innovation_distribution": state_innovation_distribution,
        "state_innovation_source": state_innovation_source,
        "state_innovation_standardization": state_innovation_standardization,
        "state_innovation_resampling": state_innovation_resampling,
        "state_innovation_t_params": list(state_innovation_t_params) if state_innovation_t_params is not None else None,
        "state_innovation_coupling": state_innovation_coupling,
        "joint_innovation_pool": joint_innovation_pool.astype(float) if joint_innovation_pool is not None else None,
        "sv_sigma_scale_method": sv_sigma_scale_method,
        "sv_sigma_scale": float(sigma_scale),
        "latent_vol_persistence": latent_vol_persistence,
        "observed_log_var_count": int(observed_log_var.size),
        "posterior_samples": samples,
        "leverage": leverage,
        "leverage_alignment": leverage_alignment,
        "leverage_correlation_method": leverage_correlation_method,
        "leverage_correlation_scope": leverage_correlation_scope,
        "unclipped_sv_measurement": unclipped_measurement,
        "sv_measurement_bias": sv_measurement_bias,
        "sv_measurement_bias_value": sv_measurement_bias_value,
        "transformed_parameter_mcmc": transformed_sampler,
        "unclipped_empirical_innovations": raw_empirical_innovations,
        "clip_simulated_returns": bool(candidate.get("clip_simulated_returns", True)),
        "mcmc_iterations": int(iterations_run or iterations_limit),
        "mcmc_iteration_limit": int(iterations_limit),
        "mcmc_burn": burn,
        "mcmc_thin": thin,
        "mcmc_chains": int(mcmc_chain_count),
        "mcmc_path_sample_interval": int(path_sample_interval) if "path_sample_interval" in locals() else int(thin),
        "mcmc_stopping": mcmc_stopping,
        "mcmc_stop_reason": mcmc_stop_reason,
        "mcmc_rhat_max": final_rhat_max,
        "mcmc_effective_sample_size_min": final_ess_min,
        "mcmc_ess_target": float(ess_target) if "ess_target" in locals() else None,
        "mcmc_rhat_threshold": (
            float(rhat_threshold)
            if mcmc_stopping == "dynamic_multichain_rhat_ess_forecast_stability" and "rhat_threshold" in locals()
            else None
        ),
        "mcmc_forecast_stability_tolerance": (
            float(stability_tolerance)
            if mcmc_stopping == "dynamic_multichain_rhat_ess_forecast_stability" and "stability_tolerance" in locals()
            else None
        ),
        "mcmc_proposal_adaptation": str(proposal_adaptation),
        "mcmc_proposal_adaptation_meta": dict(proposal_adaptation_meta),
        "mcmc_diagnostic_checks": mcmc_diagnostic_checks,
        "mcmc_retained_samples": int(len(samples)),
        "mcmc_acceptance_rate": float(accepted / mcmc_total_chain_iterations),
        "meta": {
            "method": "bayesian_sbb_full_latent_mcmc_stochastic_volatility_overlay",
            "state_sampler": "forward_filtering_backward_sampling_latent_log_variance",
            "parameter_sampler": (
                "level_eta_metropolis_hastings_no_latent_ar"
                if no_latent_ar
                else
                "stationary_transformed_metropolis_hastings"
                if transformed_sampler
                else "collapsed_marginal_metropolis_hastings"
            ),
            "latent_vol_persistence": latent_vol_persistence,
            "observation_equation": (
                "empirical_log_squared_standardized_residual_bias_unwinsorized"
                if sv_measurement_bias == "empirical_standardized_residuals"
                else "log_chi_square_mean_corrected_unwinsorized"
                if unclipped_measurement
                else "bias_corrected_log_squared_return_gaussian_state_space"
            ),
            "sv_measurement_bias": sv_measurement_bias,
            "sv_measurement_bias_value": sv_measurement_bias_value,
            "mean_model": mean_model,
            "mean_method": str((base_fit.get("meta") or {}).get("method", mean_model)),
            "tail_method": fit_tail_method,
            "innovation_standardization": innovation_standardization,
            "innovation_resampling": innovation_resampling,
            "innovation_distribution_params": list(innovation_distribution_params) if innovation_distribution_params is not None else None,
            "innovation_tail_splice": innovation_tail_splice,
            "innovation_conditioning": innovation_conditioning,
            "innovation_condition_threshold": innovation_condition_threshold,
            "innovation_pool_source": innovation_pool_source,
            "state_innovation_distribution": state_innovation_distribution,
            "state_innovation_source": state_innovation_source,
            "state_innovation_standardization": state_innovation_standardization,
            "state_innovation_resampling": state_innovation_resampling,
            "state_innovation_t_params": list(state_innovation_t_params) if state_innovation_t_params is not None else None,
            "state_innovation_coupling": state_innovation_coupling,
            "sv_sigma_scale_method": sv_sigma_scale_method,
            "sv_sigma_scale": float(sigma_scale),
            "mcmc_stopping": mcmc_stopping,
            "mcmc_stop_reason": mcmc_stop_reason,
            "mcmc_chains": int(mcmc_chain_count),
            "mcmc_path_sample_interval": int(path_sample_interval) if "path_sample_interval" in locals() else int(thin),
            "mcmc_rhat_max": final_rhat_max,
            "mcmc_effective_sample_size_min": final_ess_min,
            "mcmc_ess_target": float(ess_target) if "ess_target" in locals() else None,
            "mcmc_proposal_adaptation": str(proposal_adaptation),
            "mcmc_proposal_adaptation_meta": dict(proposal_adaptation_meta),
            "mcmc_diagnostic_checks": mcmc_diagnostic_checks,
            "path_generator": fit_path_generator,
            "leverage": leverage,
            "leverage_alignment": leverage_alignment,
            "leverage_correlation_method": leverage_correlation_method,
            "leverage_correlation_scope": leverage_correlation_scope,
            "simulated_return_clip": bool(candidate.get("clip_simulated_returns", True)),
        },
    }

def _simulate_bayesian_sbb_full_mcmc_sv_overlay(
    fit: Dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base_fit = fit.get("base_fit", {}) or {}
    samples = list(fit.get("posterior_samples") or [])
    if not samples:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    if bool(fit.get("unclipped_empirical_innovations", False)):
        z_pool = np.asarray(fit.get("innovation_pool"), dtype=np.float64)
    elif str(fit.get("innovation_pool_source", "sample_standardized")) != "sample_standardized" and fit.get("innovation_pool") is not None:
        z_pool = np.asarray(fit.get("innovation_pool"), dtype=np.float64)
    else:
        z_pool = np.asarray(base_fit.get("standardized_residuals"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size == 0:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    state_innovation_distribution = str(fit.get("state_innovation_distribution", "gaussian"))
    state_innovation_resampling = str(fit.get("state_innovation_resampling", "iid"))
    state_innovation_coupling = str(fit.get("state_innovation_coupling", "correlation_mixture"))
    state_shock_draws: Optional[np.ndarray] = None
    raw_joint_innovation_pool = fit.get("joint_innovation_pool")
    joint_innovation_pool = np.asarray(
        raw_joint_innovation_pool if raw_joint_innovation_pool is not None else [],
        dtype=np.float64,
    )
    if joint_innovation_pool.ndim == 2 and joint_innovation_pool.shape[1] >= 2:
        joint_innovation_pool = joint_innovation_pool[np.all(np.isfinite(joint_innovation_pool[:, :2]), axis=1), :2]
    else:
        joint_innovation_pool = np.empty((0, 2), dtype=np.float64)
    if state_innovation_resampling == "paired_stationary_bootstrap" and joint_innovation_pool.size:
        try:
            

            joint_block_length = int(
                max(
                    _politis_white_block_length(joint_innovation_pool[:, 0]),
                    _politis_white_block_length(joint_innovation_pool[:, 1]),
                    _politis_white_block_length(joint_innovation_pool[:, 0] * joint_innovation_pool[:, 0]),
                    _politis_white_block_length(joint_innovation_pool[:, 1] * joint_innovation_pool[:, 1]),
                )
            )
            joint_indices = _stationary_bootstrap_indices(
                n=len(joint_innovation_pool),
                block_length=int(max(1, joint_block_length)),
                total_days=int(total_days),
                n_paths=int(n_paths),
                rng=rng,
            )
            z_draws = joint_innovation_pool[joint_indices, 0]
            state_shock_draws = joint_innovation_pool[joint_indices, 1]
        except Exception:
            sampled = rng.integers(0, len(joint_innovation_pool), size=(int(n_paths), int(total_days)))
            z_draws = joint_innovation_pool[sampled, 0]
            state_shock_draws = joint_innovation_pool[sampled, 1]
    else:
        block_length = max(_politis_white_block_length(z_pool), _politis_white_block_length(z_pool * z_pool))
        innovation_resampling = str(fit.get("innovation_resampling", "stationary_bootstrap"))
        if innovation_resampling == "jones_faddy_skew_t_iid":
            raw_params = fit.get("innovation_distribution_params")
            if raw_params is None:
                return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
            z_draws = _draw_full_mcmc_sv_parametric_innovations(
                raw_params,
                innovation_resampling,
                (int(n_paths), int(total_days)),
                rng,
            )
            if z_draws is None:
                return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
        elif _full_mcmc_sv_parametric_innovation_meta(innovation_resampling) is not None:
            raw_params = fit.get("innovation_distribution_params")
            if raw_params is None:
                return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
            z_draws = _draw_full_mcmc_sv_parametric_innovations(
                raw_params,
                innovation_resampling,
                (int(n_paths), int(total_days)),
                rng,
            )
            if z_draws is None:
                return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
        elif innovation_resampling == "iid":
            z_draws = rng.choice(z_pool, size=(int(n_paths), int(total_days)), replace=True).astype(np.float64)
        elif innovation_resampling == "circular_block_bootstrap":
            z_indices = _circular_block_bootstrap_indices(
                n=len(z_pool),
                block_length=int(max(1, block_length)),
                total_days=int(total_days),
                n_paths=int(n_paths),
                rng=rng,
            )
            z_draws = z_pool[z_indices]
        else:
            try:
                

                z_indices = _stationary_bootstrap_indices(
                    n=len(z_pool),
                    block_length=int(max(1, block_length)),
                    total_days=int(total_days),
                    n_paths=int(n_paths),
                    rng=rng,
                )
                z_draws = z_pool[z_indices]
            except Exception:
                z_draws = rng.choice(z_pool, size=(int(n_paths), int(total_days)), replace=True)
    if str(fit.get("innovation_tail_splice", "none")) == "automated_evt_pot_gpd_tail":
        z_draws = _evt_tail_splice_standardized_draws(z_pool, z_draws, rng)

    conditioned_z_low_draws: Optional[np.ndarray] = None
    conditioned_z_high_draws: Optional[np.ndarray] = None
    innovation_condition_threshold = fit.get("innovation_condition_threshold")
    if str(fit.get("innovation_conditioning", "none")) == "filtered_log_volatility_median":
        low_pool = np.asarray(fit.get("conditioned_innovation_low_pool"), dtype=np.float64)
        high_pool = np.asarray(fit.get("conditioned_innovation_high_pool"), dtype=np.float64)
        low_pool = low_pool[np.isfinite(low_pool)]
        high_pool = high_pool[np.isfinite(high_pool)]
        if low_pool.size == 0 or high_pool.size == 0 or innovation_condition_threshold is None:
            return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
        try:
            

            low_block_length = max(_politis_white_block_length(low_pool), _politis_white_block_length(low_pool * low_pool))
            high_block_length = max(_politis_white_block_length(high_pool), _politis_white_block_length(high_pool * high_pool))
            low_indices = _stationary_bootstrap_indices(
                n=len(low_pool),
                block_length=int(max(1, low_block_length)),
                total_days=int(total_days),
                n_paths=int(n_paths),
                rng=rng,
            )
            high_indices = _stationary_bootstrap_indices(
                n=len(high_pool),
                block_length=int(max(1, high_block_length)),
                total_days=int(total_days),
                n_paths=int(n_paths),
                rng=rng,
            )
            conditioned_z_low_draws = low_pool[low_indices]
            conditioned_z_high_draws = high_pool[high_indices]
        except Exception:
            conditioned_z_low_draws = rng.choice(low_pool, size=(int(n_paths), int(total_days)), replace=True)
            conditioned_z_high_draws = rng.choice(high_pool, size=(int(n_paths), int(total_days)), replace=True)

    sample_idx = rng.integers(0, len(samples), size=int(n_paths))
    levels = np.asarray([samples[int(idx)][0] for idx in sample_idx], dtype=np.float64)
    phis = np.asarray([samples[int(idx)][1] for idx in sample_idx], dtype=np.float64)
    etas = np.asarray([samples[int(idx)][2] for idx in sample_idx], dtype=np.float64)
    log_var = np.asarray([samples[int(idx)][3] for idx in sample_idx], dtype=np.float64)
    rhos = np.asarray([samples[int(idx)][4] for idx in sample_idx], dtype=np.float64)
    rhos = np.where(np.isfinite(rhos), rhos, 0.0)
    if bool(fit.get("unclipped_empirical_innovations", False)):
        rhos = np.where(np.abs(rhos) < 1.0, rhos, np.sign(rhos) * (1.0 - np.finfo(np.float64).eps))
    else:
        rhos = np.clip(rhos, -0.95, 0.95)
    use_leverage = bool(fit.get("leverage", False))
    leverage_alignment = str(fit.get("leverage_alignment", "lagged_return"))
    raw_state_innovation_pool = fit.get("state_innovation_pool")
    state_innovation_pool = np.asarray(
        raw_state_innovation_pool if raw_state_innovation_pool is not None else [],
        dtype=np.float64,
    )
    state_innovation_pool = state_innovation_pool[np.isfinite(state_innovation_pool)]
    raw_state_innovation_t_params = fit.get("state_innovation_t_params")
    state_innovation_t_params = (
        tuple(float(value) for value in raw_state_innovation_t_params)
        if raw_state_innovation_t_params is not None
        else None
    )
    clip_simulated_returns = bool(fit.get("clip_simulated_returns", True))
    sigma_scale = float(fit.get("sv_sigma_scale", 1.0))
    if not np.isfinite(sigma_scale) or sigma_scale <= 0.0:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)

    bdes_multiscale = (
        fit.get("bdes_multiscale_vol")
        if str(fit.get("vol_path_model", "none")) == "bdes_multiscale_log_vol"
        else None
    )
    bdes_multiscale_shock_coupling = str(fit.get("bdes_multiscale_shock_coupling", "convex_state_independent_mix"))
    vol_path_blend = str(fit.get("vol_path_blend", "none"))
    q_state: Optional[np.ndarray] = None
    q_phis: Optional[np.ndarray] = None
    q_b: Optional[np.ndarray] = None
    q_var: Optional[np.ndarray] = None
    q_ell = 0.0
    q_resid_sd = 0.0
    q_low = -18.0
    q_high = 18.0
    q_component_low = -18.0
    q_component_high = 18.0
    q_reliability_weight = 1.0
    q_inverse_mse_weight = 1.0
    q_blend_horizon_days = 252.0
    q_long_scale_horizon_days = 252.0
    if bdes_multiscale is not None:
        q_phis = np.asarray(bdes_multiscale.get("phis"), dtype=np.float64)
        q_b = np.asarray(bdes_multiscale.get("b"), dtype=np.float64)
        q_var = np.asarray(bdes_multiscale.get("q_var"), dtype=np.float64)
        q_last = np.asarray(bdes_multiscale.get("q_last"), dtype=np.float64)
        if (
            q_phis.ndim != 1
            or q_b.shape != q_phis.shape
            or q_var.shape != q_phis.shape
            or q_last.shape != q_phis.shape
            or q_phis.size == 0
        ):
            return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
        q_state = np.tile(q_last.reshape(1, -1), (int(n_paths), 1)).astype(np.float64)
        q_ell = float(bdes_multiscale.get("ell", 0.0))
        q_resid_sd = math.sqrt(max(float(bdes_multiscale.get("resid_var", 0.0)), 0.0))
        q_low = float(bdes_multiscale.get("h_low", -18.0))
        q_high = float(bdes_multiscale.get("h_high", 18.0))
        q_component_low = float(bdes_multiscale.get("component_low", q_low - q_ell))
        q_component_high = float(bdes_multiscale.get("component_high", q_high - q_ell))
        q_reliability_weight = float(np.clip(float(bdes_multiscale.get("reliability_weight", 1.0)), 0.0, 1.0))
        q_inverse_mse_weight = float(np.clip(float(bdes_multiscale.get("inverse_mse_weight", q_reliability_weight)), 0.0, 1.0))
        q_blend_horizon_days = float(bdes_multiscale.get("dominant_half_life_days", 252.0))
        q_blend_horizon_days = max(q_blend_horizon_days, 1.0) if np.isfinite(q_blend_horizon_days) else 252.0
        q_long_scale_horizon_days = float(bdes_multiscale.get("max_half_life_days", q_blend_horizon_days))
        q_long_scale_horizon_days = max(q_long_scale_horizon_days, 1.0) if np.isfinite(q_long_scale_horizon_days) else q_blend_horizon_days
    mean_state_scaling = str(fit.get("mean_state_scaling", "none"))
    base_mean_sigma = float(base_fit.get("sigma", 0.0))
    if mean_state_scaling in {"conditional_sv_sharpe", "conditional_sv_variance_premium"} and (
        not np.isfinite(base_mean_sigma) or base_mean_sigma <= 0.0
    ):
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    sv_in_mean = fit.get("bdes_sv_in_mean") if mean_state_scaling == "bdes_sv_in_mean" else None
    sv_in_mean_lambda = 0.0
    sv_in_mean_vbar: Optional[np.ndarray] = None
    sv_in_mean_phi = 0.0
    sv_in_mean_cap = 0.0
    if sv_in_mean is not None:
        sv_in_mean_lambda = float(sv_in_mean.get("lambda", 0.0))
        sv_in_mean_phi = float(sv_in_mean.get("vbar_phi", 0.0))
        sv_in_mean_cap = float(sv_in_mean.get("premium_cap", 0.0))
        vbar_last = float(sv_in_mean.get("vbar_last", 0.0))
        if (
            not np.isfinite(sv_in_mean_lambda)
            or not np.isfinite(sv_in_mean_phi)
            or not np.isfinite(sv_in_mean_cap)
            or not np.isfinite(vbar_last)
            or sv_in_mean_cap <= 0.0
        ):
            return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
        sv_in_mean_phi = float(np.clip(sv_in_mean_phi, 0.0, 0.9999))
        sv_in_mean_vbar = np.full(int(n_paths), max(vbar_last, 1e-12), dtype=np.float64)

    out = np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    mu = float(base_fit.get("posterior_mean", 0.0))
    mean_dynamics = dict(base_fit.get("mean_dynamics", {}) or {})
    dynamic_ar1_mean = str(mean_dynamics.get("type", "")) == "ar1_constant_ols"
    mean_schedule = None if bool(base_fit.get("posterior_mu_draws", False)) or dynamic_ar1_mean else _mean_schedule_from_fit(base_fit, int(total_days))
    path_mu_is_schedule = False
    path_mu_is_path_matrix = False
    if dynamic_ar1_mean:
        ar1_intercept = float(mean_dynamics.get("intercept", 0.0))
        ar1_coef = float(mean_dynamics.get("ar_coef", 0.0))
        prev_return_path = np.full(int(n_paths), float(mean_dynamics.get("last_return", 0.0)), dtype=np.float64)
        path_mu = 0.0
    elif (mu_draw_path := _dlm_mu_draw_paths(base_fit, int(total_days), int(n_paths), rng)) is not None:
        path_mu = mu_draw_path
        path_mu_is_path_matrix = True
    elif (mu_draw_path := _posterior_decay_mu_draw_paths(base_fit, int(total_days), int(n_paths), rng)) is not None:
        path_mu = mu_draw_path
        path_mu_is_path_matrix = True
    elif bool(base_fit.get("posterior_mu_draws", False)):
        mu_draws = rng.normal(mu, float(max(base_fit.get("posterior_sd", 0.0), 0.0)), size=int(n_paths))
        cap = float(max(base_fit.get("mu_cap", 0.0), 0.0))
        if cap > 0.0:
            mu_draws = np.clip(mu_draws, -cap, cap)
        if bool(base_fit.get("nonnegative_drift", False)):
            mu_draws = np.maximum(mu_draws, 0.0)
        path_mu: Any = mu_draws
    elif mean_schedule is not None:
        path_mu = mean_schedule
        path_mu_is_schedule = True
    else:
        path_mu = mu
    prev_z = rng.choice(z_pool, size=int(n_paths), replace=True)
    if (
        state_innovation_distribution == "empirical_centered_standardized"
        and state_innovation_resampling in {"stationary_bootstrap", "circular_block_bootstrap"}
        and state_innovation_pool.size
    ):
        if state_innovation_resampling == "circular_block_bootstrap":
            state_block_length = max(
                _politis_white_block_length(state_innovation_pool),
                _politis_white_block_length(state_innovation_pool * state_innovation_pool),
            )
            state_indices = _circular_block_bootstrap_indices(
                n=len(state_innovation_pool),
                block_length=int(max(1, state_block_length)),
                total_days=int(total_days),
                n_paths=int(n_paths),
                rng=rng,
            )
            state_shock_draws = state_innovation_pool[state_indices]
        else:
            try:
                

                state_block_length = max(
                    _politis_white_block_length(state_innovation_pool),
                    _politis_white_block_length(state_innovation_pool * state_innovation_pool),
                )
                state_indices = _stationary_bootstrap_indices(
                    n=len(state_innovation_pool),
                    block_length=int(max(1, state_block_length)),
                    total_days=int(total_days),
                    n_paths=int(n_paths),
                    rng=rng,
                )
                state_shock_draws = state_innovation_pool[state_indices]
            except Exception:
                state_shock_draws = None
    for day in range(int(total_days)):
        if conditioned_z_low_draws is not None and conditioned_z_high_draws is not None:
            high_state = log_var >= float(innovation_condition_threshold)
            z = np.where(high_state, conditioned_z_high_draws[:, day], conditioned_z_low_draws[:, day])
        else:
            z = z_draws[:, day]
        if state_shock_draws is not None:
            base_state_shock = state_shock_draws[:, day]
        elif state_innovation_distribution == "empirical_centered_standardized" and state_innovation_pool.size:
            if state_innovation_resampling == "student_t_mle" and state_innovation_t_params is not None:
                t_df, t_loc, t_scale = state_innovation_t_params
                base_state_shock = t_loc + t_scale * rng.standard_t(t_df, size=int(n_paths))
            else:
                base_state_shock = rng.choice(state_innovation_pool, size=int(n_paths), replace=True)
        else:
            base_state_shock = rng.normal(0.0, 1.0, size=int(n_paths))
        if use_leverage:
            leverage_driver = z if leverage_alignment == "same_period_return" else prev_z
            if state_innovation_coupling == "empirical_direct" or (
                state_innovation_coupling == "empirical_pair" and state_shock_draws is not None
            ):
                state_shock = base_state_shock
            else:
                state_shock = rhos * leverage_driver + np.sqrt(np.maximum(1.0 - rhos * rhos, 1e-8)) * base_state_shock
        else:
            state_shock = base_state_shock
        ar1_log_var = levels + phis * (log_var - levels) + etas * state_shock
        if q_state is not None and q_phis is not None and q_b is not None and q_var is not None:
            if bdes_multiscale_shock_coupling == "state_innovation":
                local_shock = state_shock[:, None]
            else:
                local_shock = 0.70 * state_shock[:, None] + 0.30 * rng.normal(size=q_state.shape)
            q_state = q_phis.reshape(1, -1) * q_state + np.sqrt(np.maximum(q_var, 1e-10)).reshape(1, -1) * local_shock
            multiscale_log_var = q_ell + q_state @ q_b + 0.15 * q_resid_sd * state_shock
            multiscale_log_var = np.clip(multiscale_log_var, q_low, q_high)
            if vol_path_blend == "annual_horizon_ramp":
                blend_weight = float((day + 1) / ((day + 1) + 252.0))
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "data_reliability":
                log_var = (1.0 - q_reliability_weight) * ar1_log_var + q_reliability_weight * multiscale_log_var
            elif vol_path_blend == "inverse_mse":
                log_var = (1.0 - q_inverse_mse_weight) * ar1_log_var + q_inverse_mse_weight * multiscale_log_var
            elif vol_path_blend == "reliability_inverse_mse":
                blend_weight = q_reliability_weight * q_inverse_mse_weight
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "inverse_mse_horizon_ramp":
                blend_weight = q_inverse_mse_weight * float((day + 1) / ((day + 1) + q_blend_horizon_days))
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "inverse_mse_long_horizon_ramp":
                if float(total_days) >= q_blend_horizon_days:
                    blend_weight = q_inverse_mse_weight * float((day + 1) / ((day + 1) + q_blend_horizon_days))
                else:
                    blend_weight = 0.0
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "inverse_mse_long_scale_ramp":
                if float(total_days) >= q_long_scale_horizon_days:
                    blend_weight = q_inverse_mse_weight * float((day + 1) / ((day + 1) + q_blend_horizon_days))
                else:
                    blend_weight = 0.0
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "inverse_mse_maturity_ramp":
                maturity_weight = float(total_days) / (float(total_days) + q_long_scale_horizon_days)
                day_weight = float((day + 1) / ((day + 1) + q_blend_horizon_days))
                blend_weight = q_inverse_mse_weight * maturity_weight * day_weight
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "inverse_mse_horizon_gate":
                blend_weight = q_inverse_mse_weight if float(total_days) >= q_blend_horizon_days else 0.0
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "inverse_mse_long_scale_gate":
                blend_weight = q_inverse_mse_weight if float(total_days) >= q_long_scale_horizon_days else 0.0
                log_var = (1.0 - blend_weight) * ar1_log_var + blend_weight * multiscale_log_var
            elif vol_path_blend == "additive_residual_overlay":
                multiscale_residual = np.clip(multiscale_log_var - q_ell, q_component_low, q_component_high)
                log_var = ar1_log_var + multiscale_residual
            else:
                log_var = multiscale_log_var
        else:
            log_var = ar1_log_var
        if not bool(fit.get("unclipped_sv_measurement", False)):
            log_var = np.clip(log_var, -18.0, 18.0)
        sigma = (sigma_scale * np.exp(0.5 * log_var)) / 100.0
        day_mu = (
            ar1_intercept + ar1_coef * prev_return_path
            if dynamic_ar1_mean
            else path_mu[:, day]
            if path_mu_is_path_matrix
            else path_mu[day]
            if path_mu_is_schedule
            else path_mu
        )
        if mean_state_scaling == "conditional_sv_sharpe" and not dynamic_ar1_mean:
            day_mu = day_mu * sigma / max(base_mean_sigma, np.finfo(np.float64).tiny)
        elif mean_state_scaling == "conditional_sv_variance_premium" and not dynamic_ar1_mean:
            sigma_ratio = sigma / max(base_mean_sigma, np.finfo(np.float64).tiny)
            day_mu = day_mu * sigma_ratio * sigma_ratio
        elif sv_in_mean_vbar is not None:
            day_var = sigma * sigma
            sv_in_mean_vbar = sv_in_mean_phi * sv_in_mean_vbar + (1.0 - sv_in_mean_phi) * day_var
            premium = sv_in_mean_lambda * (day_var - sv_in_mean_vbar)
            day_mu = day_mu + np.clip(premium, -sv_in_mean_cap, sv_in_mean_cap)
        out[:, day] = day_mu + sigma * z
        if dynamic_ar1_mean:
            prev_return_path = out[:, day]
        prev_z = z
    if not np.all(np.isfinite(out)):
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    if clip_simulated_returns:
        out = np.clip(out, -1.0, 1.0)
    return out

def _legacy_stationary_bootstrap_indices(n: int, rng: np.random.Generator, p: float = 0.25) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    out[0] = int(rng.integers(0, n))
    for i in range(1, n):
        if float(rng.random()) < p:
            out[i] = int(rng.integers(0, n))
        else:
            out[i] = (out[i - 1] + 1) % n
    return out

def _politis_white_block_length(values: np.ndarray) -> int:
    """Pinned SimfolioEngine Politis-White automatic block length."""
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
    if not np.isfinite(gamma0) or gamma0 <= 0:
        return 1
    gamma = np.empty(max_lag + 1, dtype=np.float64)
    gamma[0] = gamma0
    for lag in range(1, max_lag + 1):
        gamma[lag] = float(np.dot(arr[lag:], arr[:-lag]) / n)
    weights = 1.0 - (np.arange(1, max_lag + 1, dtype=np.float64) / (max_lag + 1.0))
    g_hat = float(np.sum(2.0 * weights * np.arange(1, max_lag + 1, dtype=np.float64) * gamma[1:]))
    d_hat = float(gamma0 + 2.0 * np.sum(weights * gamma[1:]))
    if not np.isfinite(g_hat) or not np.isfinite(d_hat) or abs(d_hat) < 1e-12:
        return 1
    b_opt = ((2.0 * g_hat * g_hat) / (d_hat * d_hat) * n) ** (1.0 / 3.0)
    if not np.isfinite(b_opt) or b_opt < 1.0:
        return 1
    return int(max(1, round(float(b_opt))))


def _stationary_bootstrap_indices(
    n: int, block_length: int, total_days: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Pinned engine stationary bootstrap index matrix (including RNG draw order)."""
    n = int(n); block_length = max(int(block_length), 1)
    total_days = int(total_days); n_paths = int(n_paths)
    if n_paths <= 0 or total_days <= 0:
        return np.empty((max(n_paths, 0), max(total_days, 0)), dtype=np.int32)
    if n <= 0:
        raise ValueError("Stationary bootstrap requires at least one observation.")
    initial = rng.integers(0, n, size=n_paths, dtype=np.int32)
    if total_days == 1:
        return initial.reshape(n_paths, 1)
    flags = rng.random((n_paths, total_days - 1)) < (1.0 / block_length)
    points = rng.integers(0, n, size=(n_paths, total_days - 1), dtype=np.int32)
    idx = np.empty((n_paths, total_days), dtype=np.int32)
    idx[:, 0] = initial
    for t in range(1, total_days):
        idx[:, t] = np.where(flags[:, t - 1], points[:, t - 1], (idx[:, t - 1] + 1) % n)
    return idx


def _standardize_generated_innovations(draws: np.ndarray) -> np.ndarray:
    arr = np.array(draws, dtype=np.float64, copy=True)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
    np.clip(arr, -20.0, 20.0, out=arr)
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr)); std = float(np.std(arr))
    arr -= mean
    if std > 1e-12 and np.isfinite(std):
        arr /= std
    np.clip(arr, -20.0, 20.0, out=arr)
    return arr


def _evt_standardized_draws(z: np.ndarray, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Pinned engine EVT/POT standardized innovation draw helper."""
    from scipy import stats
    clean = np.asarray(z, dtype=np.float64); clean = clean[np.isfinite(clean)]
    if clean.size < 100:
        return rng.choice(clean if clean.size else np.array([0.0]), size=size, replace=True)
    exceedance_share = float(np.clip(math.sqrt(clean.size) / clean.size, 0.02, 0.10))
    lower_q = float(np.quantile(clean, exceedance_share)); upper_q = float(np.quantile(clean, 1.0-exceedance_share))
    central = clean[(clean >= lower_q) & (clean <= upper_q)]
    if central.size == 0: central = clean
    out = rng.choice(central, size=size, replace=True).astype(np.float64)
    uniforms = rng.random(size); lower_mask = uniforms < exceedance_share; upper_mask = uniforms > (1.0-exceedance_share)
    try:
        lower_excess = lower_q - clean[clean < lower_q]; upper_excess = clean[clean > upper_q] - upper_q
        if lower_excess.size >= 10:
            c, _, scale = stats.genpareto.fit(lower_excess, floc=0.0); c = float(np.clip(c, -0.45, 0.45)) if np.isfinite(c) else 0.0
            out[lower_mask] = lower_q - stats.genpareto.rvs(c, loc=0.0, scale=max(scale,1e-8), size=int(np.sum(lower_mask)), random_state=rng)
        if upper_excess.size >= 10:
            c, _, scale = stats.genpareto.fit(upper_excess, floc=0.0); c = float(np.clip(c, -0.45, 0.45)) if np.isfinite(c) else 0.0
            out[upper_mask] = upper_q + stats.genpareto.rvs(c, loc=0.0, scale=max(scale,1e-8), size=int(np.sum(upper_mask)), random_state=rng)
    except Exception:
        out[lower_mask | upper_mask] = rng.choice(clean, size=int(np.sum(lower_mask | upper_mask)), replace=True)
    return np.clip(out, -20.0, 20.0)


def _base_sbb_fit_from_mean_fit(
    train_values: np.ndarray,
    posterior_mean: float,
    residuals: np.ndarray,
    mean_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build the source SBB fit payload from a mean-model fit."""
    x = np.asarray(train_values, dtype=np.float64); x = x[np.isfinite(x)]
    eps = np.asarray(residuals, dtype=np.float64); eps = eps[np.isfinite(eps)]
    if x.size < FULL_MCMC_SV_MIN_OBS or eps.size < FULL_MCMC_SV_MIN_OBS:
        return None
    sample_mu = float(mean_meta.get("sample_mean", np.mean(x)))
    sigma = float(np.std(eps, ddof=1)) if eps.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    z = eps / sigma; z = z[np.isfinite(z)]
    if z.size == 0: return None
    z = z - float(np.mean(z)); z_sd = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if np.isfinite(z_sd) and z_sd > 1e-12: z = z / z_sd
    out: Dict[str, Any] = {
        "sample_mu": sample_mu, "sigma": sigma, "posterior_mean": float(posterior_mean),
        "posterior_sd": float(mean_meta.get("posterior_sd", 0.0) or 0.0),
        "mu_cap": float(mean_meta.get("mu_cap", 0.0) or 0.0), "sample_mu_days": 0,
        "nonnegative_drift": bool(mean_meta.get("nonnegative_drift", False)),
        "posterior_mu_draws": bool(mean_meta.get("posterior_mu_draws", False)),
        "standardized_residuals": np.clip(z, -20.0, 20.0), "meta": copy.deepcopy(mean_meta),
    }
    for key in ("posterior_mean_decay_meta", "posterior_mean_schedule_segments", "mean_dynamics"):
        if key in mean_meta: out[key] = copy.deepcopy(mean_meta[key])
    return out

def _evidence_estimated_sharpe_dlm_historical_cagr_anchor_mean(log_returns: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Standalone equivalent of the pinned engine CAGR anchored Sharpe DLM mean."""
    arr = np.asarray(log_returns, dtype=np.float64); arr = arr[np.isfinite(arr)]
    if arr.size == 0: return 0.0, np.array([], dtype=np.float64), {"method":"empty_zero"}
    if arr.size < 5:
        return float(np.mean(arr)), arr - float(np.mean(arr)), {"method":"historical_sample_mean_short_history"}
    sample_mu = float(np.mean(arr)); sample_sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sample_sigma) or sample_sigma <= 1e-10:
        return sample_mu, arr-sample_mu, {"method":"historical_cagr_anchor_degenerate_sample","sample_mean":sample_mu}
    daily_to_annual_sharpe = math.sqrt(252.0)/sample_sigma; annual_sharpe_to_daily_mu = sample_sigma/math.sqrt(252.0)
    anchor_sharpe = sample_mu*daily_to_annual_sharpe; observed_sharpe = arr*daily_to_annual_sharpe
    centered_sharpe = observed_sharpe-anchor_sharpe; obs_var=float(np.var(centered_sharpe,ddof=1))
    if not np.isfinite(obs_var) or obs_var <= 1e-16: obs_var=252.0
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(arr)
    if not np.isfinite(se_mu) or se_mu <= 0: se_mu=sample_sigma/math.sqrt(max(arr.size,1)); bandwidth=0; long_run_var=sample_sigma*sample_sigma
    se_sharpe=se_mu*daily_to_annual_sharpe; signal_ratio_start=(se_sharpe/max(math.sqrt(obs_var),np.finfo(np.float64).tiny))**2
    params=_estimate_evidence_dlm_drift_params(centered_sharpe,obs_var=obs_var,signal_ratio_start=signal_ratio_start)
    filtered=_kalman_ar1_drift_filter(centered_sharpe,obs_var=obs_var,phi=float(params['phi']),state_var_ratio=float(params['state_var_ratio']))
    if not np.isfinite(float(filtered.get('loglik',-math.inf))):
        return sample_mu, arr-sample_mu, {"method":"historical_cagr_anchor_filter_failed"}
    predicted=(anchor_sharpe+np.asarray(filtered['predicted_mean'],dtype=np.float64))*annual_sharpe_to_daily_mu
    residuals=arr-predicted; z=residuals/sample_sigma; z=z[np.isfinite(z)]
    if z.size == 0: z= (arr-sample_mu)/sample_sigma
    z=z-float(np.mean(z)); zsd=float(np.std(z,ddof=1)) if z.size>1 else 1.0
    if np.isfinite(zsd) and zsd>1e-12: z=z/zsd
    final_dev=float(filtered['final_mean'])*annual_sharpe_to_daily_mu; final_var=float(max(filtered['final_var'],0.0))*(annual_sharpe_to_daily_mu**2)
    posterior_mean=sample_mu+final_dev; posterior_sd=math.sqrt(max(final_var,0.0))
    meta={"method":"evidence_estimated_ar1_latent_sharpe_dlm_with_historical_cagr_anchor","historical_cagr_anchor":True,"anchor_posterior_mean":sample_mu,"anchor_sharpe_annualized":anchor_sharpe,"sample_mean":sample_mu,"sample_sigma":sample_sigma,"posterior_mean":posterior_mean,"posterior_sd":posterior_sd,"dlm_drift_paths":True,"dlm_long_run_anchor_mean":sample_mu,"dlm_state_transition_phi":float(params['phi']),"dlm_state_stationary_var":float(filtered['stationary_var'])*(annual_sharpe_to_daily_mu**2),"dlm_state_noise_var":float(filtered['state_noise_var'])*(annual_sharpe_to_daily_mu**2),"dlm_state_posterior_deviation_mean":final_dev,"dlm_state_posterior_deviation_var":final_var,"dlm_loglik":float(filtered['loglik']),"hac_mean_standard_error":float(se_mu),"hac_bandwidth":int(bandwidth),"hac_long_run_variance":float(long_run_var),"signal_ratio_start":float(signal_ratio_start)}
    base={"sample_mu":sample_mu,"sigma":sample_sigma,"posterior_mean":posterior_mean,"posterior_sd":posterior_sd,"mu_cap":0.0,"sample_mu_days":0,"nonnegative_drift":False,"posterior_mu_draws":False,"dlm_drift_paths":True,"dlm_long_run_anchor_mean":sample_mu,"dlm_state_transition_phi":float(params['phi']),"dlm_state_stationary_var":float(filtered['stationary_var'])*(annual_sharpe_to_daily_mu**2),"dlm_state_stationary_var_ratio":float(params['state_var_ratio']),"dlm_state_noise_var":float(filtered['state_noise_var'])*(annual_sharpe_to_daily_mu**2),"dlm_state_noise_sd":float(math.sqrt(max(float(filtered['state_noise_var'])*(annual_sharpe_to_daily_mu**2),0.0))),"dlm_state_posterior_deviation_mean":final_dev,"dlm_state_posterior_deviation_var":final_var,"standardized_residuals":np.clip(z,-20.0,20.0),"meta":copy.deepcopy(meta)}
    meta['_base_fit_override']=base
    return posterior_mean,residuals,meta


def _fit_harx_ff6_current_log_variance_anchor(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """The retained no-AR HARX factor anchor requires source factor inputs.

    The standalone package intentionally fails closed until a caller supplies
    an independently verified factor panel and fit, preserving source rights
    and preventing a silent proxy.
    """
    return None


def _dlm_ar1_loglik(values: np.ndarray, obs_var: float, phi: float, state_var_ratio: float) -> float:
    """Pure Python form of the pinned engine DLM likelihood kernel."""
    values = np.asarray(values, dtype=np.float64)
    obs = float(obs_var); phi_value = float(phi); ratio = float(state_var_ratio)
    if values.size < 2 or not np.isfinite(obs) or obs <= 0.0:
        return -math.inf
    if not np.isfinite(phi_value) or phi_value < 0.0 or phi_value >= 1.0:
        return -math.inf
    if not np.isfinite(ratio) or ratio <= 0.0:
        return -math.inf
    stationary_var = max(obs * ratio, obs * 1.0e-14)
    state_noise_var = max((1.0 - phi_value * phi_value) * stationary_var, obs * 1.0e-14)
    m = 0.0; p = stationary_var; loglik = 0.0; log_two_pi = math.log(2.0 * math.pi)
    for value in values:
        a = phi_value * m; r = phi_value * phi_value * p + state_noise_var; f = r + obs
        if not np.isfinite(f) or f <= 0.0:
            return -math.inf
        innovation = float(value) - a
        loglik += -0.5 * (log_two_pi + math.log(f) + (innovation * innovation) / f)
        k = r / f; m = a + k * innovation; p = max((1.0 - k) * r, obs * 1.0e-14)
    return float(loglik) if np.isfinite(loglik) else -math.inf


def _apply_full_mcmc_sv_simulation_options(fit: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Retain the source harness's post-fit simulation option application."""
    model = dict(fit)
    model["innovation_resampling"] = str(candidate.get("innovation_resampling", fit.get("innovation_resampling", "stationary_bootstrap")))
    model["innovation_tail_splice"] = str(candidate.get("innovation_tail_splice", fit.get("innovation_tail_splice", "none")))
    model["clip_simulated_returns"] = bool(candidate.get("clip_simulated_returns", fit.get("clip_simulated_returns", True)))
    meta = dict(model.get("meta", {}) or {})
    meta["innovation_resampling"] = model["innovation_resampling"]
    meta["innovation_tail_splice"] = model["innovation_tail_splice"]
    meta["simulated_return_clip"] = model["clip_simulated_returns"]
    parametric_meta = _full_mcmc_sv_parametric_innovation_meta(model["innovation_resampling"])
    meta["tail_method"] = (
        parametric_meta[0] if parametric_meta is not None
        else "automated_evt_pot_gpd_tail" if model["innovation_tail_splice"] == "automated_evt_pot_gpd_tail"
        else meta.get("tail_method", "filtered_empirical_tail")
    )
    meta["path_generator"] = (
        parametric_meta[1] if parametric_meta is not None
        else "iid_standardized_residuals_scaled_by_latent_sv_paths"
        if model["innovation_resampling"] == "iid"
        else "stationary_bootstrap_standardized_residuals_scaled_by_latent_sv_paths"
    )
    model["meta"] = meta
    return model


# Public canonical catalogue boundary -------------------------------------------------
CANONICAL_FULL_MCMC_SV_TYPE = "bayesian_sbb_full_mcmc_sv_overlay"
_CANONICAL_SV_RESOURCE = "resources/catalogs/canonical_40_full_mcmc_sv_specs.json"
_CANONICAL_SV_SOURCE_SCRIPT_SHA256 = (
    "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997"
)


def load_canonical_full_mcmc_sv_specs() -> tuple[dict[str, Any], ...]:
    """Load the immutable source-derived manifest for the owned 40 entries."""
    resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "catalogs", "canonical_40_full_mcmc_sv_specs.json"
    )
    raw = resource.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("full MCMC SV manifest schema drifted")
    if payload.get("source_script_sha256") != _CANONICAL_SV_SOURCE_SCRIPT_SHA256:
        raise ValueError("full MCMC SV source harness identity drifted")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 40:
        raise ValueError("full MCMC SV manifest must contain exactly 40 entries")
    seen: set[str] = set()
    for expected_rank, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError("full MCMC SV manifest entry is not an object")
        model_id = str(entry.get("id", ""))
        if not model_id or model_id in seen:
            raise ValueError("full MCMC SV manifest contains duplicate or empty IDs")
        seen.add(model_id)
        if entry.get("type") != CANONICAL_FULL_MCMC_SV_TYPE:
            raise ValueError(f"full MCMC SV entry {model_id} has the wrong type")
        if int(entry.get("owned_rank", 0)) != expected_rank:
            raise ValueError("full MCMC SV owned ranks drifted")
    return tuple(dict(entry) for entry in entries)


def canonical_full_mcmc_sv_ids() -> tuple[str, ...]:
    return tuple(str(entry["id"]) for entry in load_canonical_full_mcmc_sv_specs())


def spec_for_full_mcmc_sv(model_id: str) -> dict[str, Any]:
    for entry in load_canonical_full_mcmc_sv_specs():
        if entry["id"] == model_id:
            return entry
    raise KeyError(f"model is outside the owned canonical full MCMC SV set: {model_id}")


def fit_full_mcmc_sv(
    train_values: np.ndarray,
    model_id: str | Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Fit one of the 40 pinned full latent MCMC SV overlays.

    ``model_id`` may be an owned ID or its source-derived manifest entry.  The
    source fit uses a deterministic candidate signature seed, so fit results do
    not depend on process or call order.
    """
    candidate = (
        dict(model_id)
        if isinstance(model_id, Mapping)
        else spec_for_full_mcmc_sv(str(model_id))
    )
    if candidate.get("type") != CANONICAL_FULL_MCMC_SV_TYPE:
        raise ValueError("fit_full_mcmc_sv received a candidate outside the owned type")
    fitted = _fit_bayesian_sbb_full_mcmc_sv_overlay(np.asarray(train_values, dtype=np.float64), candidate)
    if fitted is not None:
        # The research harness applies these options immediately before
        # simulation.  Store them at fit time too so direct callers cannot
        # accidentally drop a candidate's innovation policy.
        fitted = _apply_full_mcmc_sv_simulation_options(fitted, candidate)
        fitted["candidate_id"] = str(candidate["id"])
        fitted["candidate_type"] = CANONICAL_FULL_MCMC_SV_TYPE
        fitted["source_script_sha256"] = _CANONICAL_SV_SOURCE_SCRIPT_SHA256
    return fitted


def simulate_full_mcmc_sv(
    fit: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate daily log-return paths using the retained SV overlay fit."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if int(total_days) < 0 or int(n_paths) < 0:
        raise ValueError("total_days and n_paths must be nonnegative")
    return _simulate_bayesian_sbb_full_mcmc_sv_overlay(
        dict(fit), int(total_days), int(n_paths), rng
    )
