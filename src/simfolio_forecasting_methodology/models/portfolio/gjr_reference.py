"""Source-backed legacy GJR-GARCH portfolio extension families.

The two rows in this module are the recovered Model 023/024 whitepaper
definitions.  Their source implementation fits the empirical-Bayes mean and a
GJR-GARCH volatility overlay, then resamples standardized residuals with
either Kunsch moving blocks or the source stationary bootstrap.  The adapter
returns daily log-return paths; :mod:`simfolio_forecasting_methodology.runner`
performs the cumulative conversion used for forecast scoring.

The retained score artifact was produced by a historical terminal adapter
whose cumulative conversion is not independently established here.  Model
path parity therefore does not mark the retained scalar score as reproduced.
"""

from __future__ import annotations

import hashlib
import logging
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ...runner import ForecastContext, TrainingData

SOURCE_ARTIFACTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "research_gate": MappingProxyType(
            {
                "path": "source-research/scripts/forecast_oos_research_gate.py",
                "sha256": "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997",
            }
        ),
        "engine": MappingProxyType(
            {
                "path": "source-research/app/engine.py",
                "sha256": "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665",
            }
        ),
        "recovery_wrapper": MappingProxyType(
            {
                "path": "evidence/recovery/oos_all_daily_exact_crps.py",
                "sha256": None,
                "status": "read-only recovery source; artifact digest unresolved",
            }
        ),
    }
)

# AST source fragments are identified in their original source order.  The
# recovery wrapper remains a read-only source reference; its full private
# file digest is intentionally not copied into the public package.
SOURCE_FUNCTION_NAMES: tuple[str, ...] = (
    "SimfolioEngine._deterministic_seed",
    "_hac_mean_standard_error",
    "_fit_empirical_bayes_sharpe_sbb",
    "_simulate_bayesian_constrained_sbb",
    "_overlay_vol_clip_bounds",
    "_fit_bayesian_sbb_vol_overlay[GJR branch]",
    "_overlay_vol_multiplier_curve[variance_decay branch]",
    "_simulate_bayesian_sbb_vol_overlay",
    "_politis_white_block_length",
    "_stationary_bootstrap_indices",
    "SimfolioEngine._fit_arch_volatility",
    "_moving_block_indices",
    "_moving_block_bayesian_sbb_paths",
)

# SHA-256 of the exact AST fragments from the pinned source files, in the
# order documented above.  The local implementation is the explicit GJR
# branch of the source dispatcher, with source-engine object references
# replaced by local functions so the public package has no private imports.
SOURCE_FUNCTIONS_SHA256 = "ccfd7a138df23e4668f50a82cb0cd87d2ae71089229849eaee82c3105d144677"

SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, public_model_id, simulations)"
)

REFERENCE_MODEL_IDS: tuple[str, ...] = (
    "portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal",
    "portfolio_gjr_garch_eb_sampler_moving_block_optimal",
)

SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal": MappingProxyType(
            {
                "id": "portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal",
                "type": "bayesian_sbb_vol_overlay",
                "overlay_model": "gjr_garch_1_1",
                "vol_model": "gjr_garch_1_1_volatility_overlay",
                "mean_model": "empirical_bayes_hac_sharpe",
                "innovation_method": "empirical_standardized_residuals",
                "tail_method": "filtered_empirical_tail",
                "path_generator": "stationary_bootstrap_standardized_residuals",
                "residual_resampling": "stationary_bootstrap",
                "prior_source": "data_driven_empirical_bayes_hac_sharpe",
            }
        ),
        "portfolio_gjr_garch_eb_sampler_moving_block_optimal": MappingProxyType(
            {
                "id": "portfolio_gjr_garch_eb_sampler_moving_block_optimal",
                "type": "bayesian_sbb_vol_overlay",
                "overlay_model": "gjr_garch_1_1",
                "vol_model": "gjr_garch_1_1_volatility_overlay",
                "mean_model": "empirical_bayes_hac_sharpe",
                "innovation_method": "empirical_standardized_residuals",
                "tail_method": "filtered_empirical_tail",
                "path_generator": "moving_block_bootstrap_standardized_residuals",
                "residual_resampling": "moving_block_bootstrap",
                "prior_source": "data_driven_empirical_bayes_hac_sharpe",
            }
        ),
    }
)


def _deterministic_seed(*parts: Any) -> int:
    """Pinned ``SimfolioEngine._deterministic_seed``."""

    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


def _politis_white_block_length(values: np.ndarray) -> int:
    """Pinned engine Politis-White block-length calculation."""

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


def _hac_mean_standard_error(values: np.ndarray) -> tuple[float, int, float]:
    """Pinned research-gate HAC mean standard error."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0, 0, 0.0
    centered = x - float(np.mean(x))
    sample_var = float(np.var(centered, ddof=1))
    if not np.isfinite(sample_var) or sample_var <= 0.0:
        return 0.0, 0, 0.0
    try:
        bandwidth = int(
            max(
                _politis_white_block_length(centered),
                _politis_white_block_length(centered * centered),
            )
        )
    except Exception:
        logging.getLogger(__name__).warning("Using retained source bandwidth fallback", exc_info=True)
        bandwidth = round(float(x.size) ** (1.0 / 3.0))
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
) -> dict[str, Any] | None:
    """Pinned source empirical-Bayes HAC-Sharpe base fit."""

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


def _arch_dist_name(innovation_method: str) -> str:
    if innovation_method == "gaussian_iid_standardized_innovations":
        return "normal"
    if innovation_method == "student_t_standardized_innovations":
        return "t"
    if innovation_method == "skew_t_standardized_innovations":
        return "skewt"
    return "normal"


def _fit_arch_volatility(
    residuals: np.ndarray,
    vol_model: str,
    innovation_method: str,
) -> dict[str, Any] | None:
    """Pinned source engine ARCH fit; missing ``arch`` fails closed."""

    eps = np.asarray(residuals, dtype=np.float64)
    eps = eps[np.isfinite(eps)]
    if eps.size < 80:
        return None
    x = eps[-min(len(eps), 1260) :] * 100.0
    if np.std(x) <= 1e-8:
        return None
    try:
        from arch import arch_model

        if vol_model == "garch_1_1_volatility":
            kwargs = {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "power": 2.0}
        elif vol_model == "gjr_tarch_1_1_volatility":
            kwargs = {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "power": 2.0}
        elif vol_model == "egarch_1_1_volatility":
            kwargs = {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}
        else:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = arch_model(
                x,
                mean="Zero",
                dist=_arch_dist_name(innovation_method),
                rescale=False,
                **kwargs,
            ).fit(disp="off", options={"maxiter": 80, "ftol": 1e-6})
        params = {str(key): float(value) for key, value in result.params.items() if np.isfinite(value)}
        sigma = np.asarray(result.conditional_volatility, dtype=np.float64)
        sigma = np.maximum(sigma[np.isfinite(sigma)], 1e-6)
        if sigma.size < 20:
            return None
        return {
            "vol_model": vol_model,
            "sigma_x": sigma,
            "last_sigma_x": float(sigma[-1]),
            "params": params,
            "fit_status": (
                "complete"
                if int(getattr(result, "convergence_flag", 0) or 0) == 0
                else "optimizer_warning"
            ),
        }
    except Exception:
        logging.getLogger(__name__).warning("Source GJR fit failed", exc_info=True)
        return None


def _overlay_vol_clip_bounds(vol_proxy_x: np.ndarray, long_sigma_x: float) -> tuple[float, float]:
    proxy = np.asarray(vol_proxy_x, dtype=np.float64)
    proxy = proxy[np.isfinite(proxy)]
    proxy = proxy[proxy > 0.0]
    base = float(max(long_sigma_x, 1e-6))
    if proxy.size < 20:
        return 1e-6, float(max(base * 100.0, 1e-6))
    floor = float(np.quantile(proxy, 0.01))
    ceiling = float(np.quantile(proxy, 0.99))
    floor = float(max(min(floor, base), 1e-6))
    ceiling = float(max(ceiling, base, floor * 1.01))
    return floor, ceiling


def _overlay_half_life_days(persistence: float) -> float | None:
    p = abs(float(persistence))
    if not np.isfinite(p) or p <= 0.0 or p >= 0.999999:
        return None
    return float(math.log(0.5) / math.log(p))


def _fit_gjr_bayesian_sbb_vol_overlay(
    train_values: np.ndarray,
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Exact GJR branch of the source ``_fit_bayesian_sbb_vol_overlay``."""

    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    if str(candidate.get("type")) != "bayesian_sbb_vol_overlay":
        raise ValueError("GJR adapter received a non-GJR source descriptor")
    if str(candidate.get("mean_model")) != "empirical_bayes_hac_sharpe":
        raise ValueError("GJR adapter requires empirical_bayes_hac_sharpe")
    if str(candidate.get("overlay_model")) != "gjr_garch_1_1":
        raise ValueError("GJR adapter requires gjr_garch_1_1")
    base_fit = _fit_empirical_bayes_sharpe_sbb(x)
    if base_fit is None:
        return None
    residuals = x - float(np.mean(x))
    long_sigma_x = float(max(float(base_fit["sigma"]) * 100.0, 1e-6))
    long_variance_x = float(max(long_sigma_x * long_sigma_x, 1e-8))
    vol_fit = _fit_arch_volatility(
        residuals,
        "gjr_tarch_1_1_volatility",
        "gaussian_iid_standardized_innovations",
    )
    if vol_fit is None:
        return None
    params = vol_fit.get("params", {}) or {}
    sigma_x = np.asarray(vol_fit.get("sigma_x", []), dtype=np.float64)
    if sigma_x.size < 20:
        return None
    alpha = float(np.clip(params.get("alpha[1]", 0.05), 0.0, 0.80))
    beta = float(np.clip(params.get("beta[1]", 0.90), 0.0, 0.999))
    gamma = float(np.clip(params.get("gamma[1]", 0.0), -0.80, 1.20))
    persistence = float(np.clip(alpha + beta + 0.5 * gamma, 0.0, 0.999))
    curve_fit = {
        "curve_type": "variance_decay",
        "persistence": persistence,
        "last_variance_x": float(max(float(vol_fit.get("last_sigma_x", long_sigma_x)) ** 2, 1e-8)),
        "long_variance_x": long_variance_x,
        "sigma_clip_bounds_x": _overlay_vol_clip_bounds(sigma_x, long_sigma_x),
        "fit_status": str(vol_fit.get("fit_status", "complete")),
        "arch_params": params,
    }
    return {
        "base_fit": base_fit,
        "curve_fit": curve_fit,
        "meta": {
            "method": "bayesian_constrained_sbb_with_data_driven_short_horizon_volatility_overlay",
            "overlay_model": "gjr_garch_1_1",
            "long_run_sigma_source": "base_bayesian_constrained_sbb_sample_sigma",
            "long_sigma_x": long_sigma_x,
            "overlay_persistence": persistence,
            "overlay_half_life_days": _overlay_half_life_days(persistence),
            "overlay_fit_status": str(curve_fit.get("fit_status", "complete")),
        },
    }


def _mean_schedule_from_fit(fit: Mapping[str, Any], total_days: int) -> np.ndarray | None:
    """Source helper retained for exact simulator dispatch."""

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
        end = total_days if through_day is None else int(max(start, min(total_days, int(through_day))))
        if end > start:
            schedule[start:end] = mu
            start = end
        last_mu = mu
        if start >= total_days:
            break
    if start < total_days:
        schedule[start:] = last_mu
    return schedule if np.all(np.isfinite(schedule)) else None


DLM_DRIFT_PHI_MIN = 0.0
DLM_DRIFT_PHI_MAX = 0.9995


def _dlm_mu_draw_paths(*args: Any, **kwargs: Any) -> np.ndarray | None:
    """The exact GJR base fit does not enable DLM drift paths."""

    fit = args[0] if args else kwargs.get("fit", {})
    if fit.get("dlm_drift_paths", False):
        raise ValueError("DLM drift paths are outside the verified GJR specification")
    return None


def _posterior_decay_mu_draw_paths(*args: Any, **kwargs: Any) -> np.ndarray | None:
    """The exact GJR base fit does not enable decaying posterior draws."""

    fit = args[0] if args else kwargs.get("fit", {})
    if fit.get("posterior_mu_draws_with_horizon_decay", False):
        raise ValueError("Posterior decay is outside the verified GJR specification")
    return None


STATIONARY_BOOTSTRAP_ROW_ASSEMBLY_MIN_DAYS = 128


def _stationary_bootstrap_indices(
    n: int,
    block_length: int,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pinned source-engine stationary bootstrap index kernel."""

    if int(n_paths) <= 0 or int(total_days) <= 0:
        return np.empty((max(int(n_paths), 0), max(int(total_days), 0)), dtype=np.int32)
    if n <= 0:
        raise ValueError("Stationary bootstrap requires at least one observation.")
    b = max(int(block_length), 1)
    p = 1.0 / b
    initial_indices = rng.integers(0, n, size=n_paths, dtype=np.int32)
    if total_days == 1:
        return initial_indices.reshape(n_paths, 1)
    restart_flags = rng.random((n_paths, total_days - 1)) < p
    restart_points = rng.integers(0, n, size=(n_paths, total_days - 1), dtype=np.int32)
    if total_days < STATIONARY_BOOTSTRAP_ROW_ASSEMBLY_MIN_DAYS:
        idx = np.empty((n_paths, total_days), dtype=np.int32)
        idx[:, 0] = initial_indices
        for t in range(1, total_days):
            idx[:, t] = np.where(
                restart_flags[:, t - 1],
                restart_points[:, t - 1],
                (idx[:, t - 1] + 1) % n,
            )
        return idx
    idx = np.empty((n_paths, total_days), dtype=np.int32)
    idx[:, 0] = initial_indices
    idx[:, 1:] = restart_points
    positions = np.arange(total_days, dtype=np.int32)
    start_positions = np.empty(total_days, dtype=np.int32)
    for row_idx in range(n_paths):
        start_positions[0] = 0
        start_positions[1:] = np.where(restart_flags[row_idx], positions[1:], 0)
        np.maximum.accumulate(start_positions, out=start_positions)
        idx[row_idx] = (idx[row_idx, start_positions] + positions - start_positions) % int(n)
    return idx


def _simulate_bayesian_constrained_sbb(
    fit: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pinned source simulator for the stationary-resampling branch."""

    total_days = int(total_days)
    n_paths = int(n_paths)
    z_pool = np.asarray(fit.get("standardized_residuals"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size == 0:
        return np.empty((n_paths, total_days), dtype=np.float64)
    fixed_block_length = fit.get("fixed_block_length")
    if fixed_block_length is not None:
        block_length = int(max(1, min(int(fixed_block_length), int(z_pool.size))))
    else:
        block_length = max(_politis_white_block_length(z_pool), _politis_white_block_length(z_pool * z_pool))
    z_indices = _stationary_bootstrap_indices(len(z_pool), block_length, total_days, n_paths, rng)
    z_draws = z_pool[z_indices]
    mu = float(fit.get("posterior_mean", 0.0))
    mu_schedule = _mean_schedule_from_fit(fit, total_days)
    mu_draw_path = _dlm_mu_draw_paths(fit, total_days, n_paths, rng)
    if mu_draw_path is None:
        mu_draw_path = _posterior_decay_mu_draw_paths(fit, total_days, n_paths, rng)
    if mu_draw_path is not None:
        mu_path: Any = mu_draw_path
    elif bool(fit.get("posterior_mu_draws", True)):
        mu_draws = rng.normal(mu, float(max(fit.get("posterior_sd", 0.0), 0.0)), size=n_paths)
        cap = float(max(fit.get("mu_cap", 0.0), 0.0))
        if cap > 0.0:
            mu_draws = np.clip(mu_draws, -cap, cap)
        if bool(fit.get("nonnegative_drift", False)):
            mu_draws = np.maximum(mu_draws, 0.0)
        mu_path = mu_draws[:, None]
    elif mu_schedule is not None:
        mu_path = mu_schedule[None, :]
    else:
        mu_path = mu
    sample_mu_days = int(max(fit.get("sample_mu_days", 0), 0))
    if sample_mu_days > 0:
        day_index = np.arange(total_days, dtype=np.int64)
        use_sample_mu = day_index < sample_mu_days
        if np.ndim(mu_path) == 0:
            mu_path = np.where(use_sample_mu, float(fit["sample_mu"]), float(mu_path))[None, :]
        else:
            mu_path = np.where(
                use_sample_mu[None, :], float(fit["sample_mu"]), np.asarray(mu_path, dtype=np.float64)
            )
    return np.clip(mu_path + float(fit["sigma"]) * z_draws, -1.0, 1.0)


def _overlay_vol_multiplier_curve(fit: Mapping[str, Any], total_days: int) -> np.ndarray:
    """Pinned source variance-decay overlay branch."""

    total_days = int(total_days)
    if total_days <= 0:
        return np.empty(0, dtype=np.float64)
    base_fit = fit.get("base_fit", {}) or {}
    curve_fit = fit.get("curve_fit", {}) or {}
    long_sigma_x = float(max(float(base_fit.get("sigma", 0.0)) * 100.0, 1e-6))
    steps = np.arange(1, total_days + 1, dtype=np.float64)
    p = float(curve_fit.get("persistence", 0.0))
    long_h = float(max(curve_fit.get("long_variance_x", long_sigma_x * long_sigma_x), 1e-8))
    last_h = float(max(curve_fit.get("last_variance_x", long_h), 1e-8))
    sigma_x = np.sqrt(np.maximum(long_h + np.power(p, steps) * (last_h - long_h), 1e-8))
    bounds = curve_fit.get("sigma_clip_bounds_x")
    if bounds is not None and len(bounds) == 2:
        floor, ceiling = float(bounds[0]), float(bounds[1])
        if np.isfinite(floor) and np.isfinite(ceiling) and ceiling > floor > 0.0:
            sigma_x = np.clip(sigma_x, floor, ceiling)
    multiplier = np.asarray(sigma_x, dtype=np.float64) / max(long_sigma_x, 1e-8)
    if multiplier.size != total_days:
        return np.ones(total_days, dtype=np.float64)
    return np.clip(multiplier, 0.01, 100.0)


def _simulate_bayesian_sbb_vol_overlay(
    fit: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base_fit = fit.get("base_fit", {}) or {}
    base_paths = _simulate_bayesian_constrained_sbb(base_fit, int(total_days), int(n_paths), rng)
    if base_paths.size == 0:
        return base_paths
    multipliers = _overlay_vol_multiplier_curve(fit, int(total_days))
    if multipliers.size != base_paths.shape[1]:
        return base_paths
    center_schedule = _mean_schedule_from_fit(base_fit, int(total_days))
    center: Any = (
        center_schedule[None, :]
        if center_schedule is not None
        else float(base_fit.get("posterior_mean", 0.0))
    )
    paths = center + (base_paths - center) * multipliers[None, :]
    return np.clip(paths, -1.0, 1.0)


def _moving_block_indices(
    n_observations: int,
    block_length: int,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pinned recovery-wrapper Kunsch moving-block indices."""

    n = int(n_observations)
    length = int(max(1, min(int(block_length), n)))
    total = int(total_days)
    paths = int(n_paths)
    if n <= 0 or total <= 0 or paths <= 0:
        return np.empty((paths, total), dtype=np.int64)
    starts = rng.integers(0, max(1, n - length + 1), size=(paths, math.ceil(total / length)))
    offsets = np.arange(length, dtype=np.int64)[None, None, :]
    blocks = starts[:, :, None] + offsets
    return blocks.reshape(paths, -1)[:, :total]


def _moving_block_bayesian_sbb_paths(
    fit: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pinned recovery-wrapper moving-block simulator."""

    base_fit = fit.get("base_fit", {}) or {}
    z_pool = np.asarray(base_fit.get("standardized_residuals"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size == 0:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    fixed_block_length = base_fit.get("fixed_block_length")
    if fixed_block_length is not None:
        block_length = int(max(1, min(int(fixed_block_length), int(z_pool.size))))
    else:
        block_length = max(_politis_white_block_length(z_pool), _politis_white_block_length(z_pool * z_pool))
    z_draws = z_pool[
        _moving_block_indices(
            int(z_pool.size), block_length, int(total_days), int(n_paths), rng
        )
    ]
    mu = float(base_fit.get("posterior_mean", 0.0))
    mu_schedule = _mean_schedule_from_fit(base_fit, int(total_days))
    mu_draw_path = _dlm_mu_draw_paths(base_fit, int(total_days), int(n_paths), rng)
    if mu_draw_path is None:
        mu_draw_path = _posterior_decay_mu_draw_paths(base_fit, int(total_days), int(n_paths), rng)
    if mu_draw_path is not None:
        mu_path: Any = mu_draw_path
    elif bool(base_fit.get("posterior_mu_draws", True)):
        mu_draws = rng.normal(mu, float(max(base_fit.get("posterior_sd", 0.0), 0.0)), size=int(n_paths))
        cap = float(max(base_fit.get("mu_cap", 0.0), 0.0))
        if cap > 0.0:
            mu_draws = np.clip(mu_draws, -cap, cap)
        if bool(base_fit.get("nonnegative_drift", False)):
            mu_draws = np.maximum(mu_draws, 0.0)
        mu_path = mu_draws[:, None]
    elif mu_schedule is not None:
        mu_path = mu_schedule[None, :]
    else:
        mu_path = mu
    sample_mu_days = int(max(base_fit.get("sample_mu_days", 0), 0))
    if sample_mu_days > 0:
        day_index = np.arange(int(total_days), dtype=np.int64)
        use_sample_mu = day_index < sample_mu_days
        if np.ndim(mu_path) == 0:
            mu_path = np.where(use_sample_mu, float(base_fit["sample_mu"]), float(mu_path))[None, :]
        else:
            mu_path = np.where(
                use_sample_mu[None, :], float(base_fit["sample_mu"]), np.asarray(mu_path, dtype=np.float64)
            )
    base_paths = np.clip(mu_path + float(base_fit["sigma"]) * z_draws, -1.0, 1.0)
    multipliers = _overlay_vol_multiplier_curve(fit, int(total_days))
    if multipliers.size != base_paths.shape[1]:
        return base_paths
    center_schedule = _mean_schedule_from_fit(base_fit, int(total_days))
    center: Any = (
        center_schedule[None, :]
        if center_schedule is not None
        else float(base_fit.get("posterior_mean", 0.0))
    )
    return np.clip(center + (base_paths - center) * multipliers[None, :], -1.0, 1.0)


@dataclass(frozen=True)
class PortfolioGJRGARCHModel:
    """Exact-ID daily-path adapter for the recovered GJR rows."""

    model_id: str
    dependency_identity: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {
                "source_imports": ("hashlib", "math", "numpy", "arch.arch_model"),
                "runtime_constraints": ("numpy>=2.0,<3", "arch==8.0.0 unresolved source pin"),
            }
        )
    )

    def __post_init__(self) -> None:
        if self.model_id not in REFERENCE_MODEL_IDS:
            raise ValueError(f"unknown GJR reference model: {self.model_id!r}")

    @property
    def family(self) -> str:
        return str(SOURCE_CANDIDATE_SPECS[self.model_id]["type"])

    @property
    def source_specification(self) -> Mapping[str, str]:
        return SOURCE_CANDIDATE_SPECS[self.model_id]

    @property
    def source_function_names(self) -> tuple[str, ...]:
        return SOURCE_FUNCTION_NAMES

    @property
    def source_fragment_digest(self) -> str:
        return SOURCE_FUNCTIONS_SHA256

    def _source_rng(self, context: ForecastContext) -> np.random.Generator:
        source_origin = (
            str(context.origin_date)
            if context.origin_date is not None
            else str(context.origin_label)
        )
        dense_horizon_tuple = tuple(range(1, int(context.horizon_days) + 1))
        seed = _deterministic_seed(
            "forecast_oos_candidate",
            source_origin,
            dense_horizon_tuple,
            self.model_id,
            int(context.simulations),
        )
        return np.random.default_rng(seed)

    def simulate_daily_log_returns(self, training: TrainingData, context: ForecastContext) -> np.ndarray:
        training.validate()
        fit = _fit_gjr_bayesian_sbb_vol_overlay(
            np.asarray(training.portfolio_log_returns, dtype=np.float64),
            SOURCE_CANDIDATE_SPECS[self.model_id],
        )
        if fit is None:
            raise RuntimeError(
                f"{self.model_id} cannot execute: exact GJR/ARCH closure returned no fit; "
                "the arch dependency and at least 60 finite observations are required"
            )
        rng = self._source_rng(context)
        if SOURCE_CANDIDATE_SPECS[self.model_id]["residual_resampling"] == "moving_block_bootstrap":
            paths = _moving_block_bayesian_sbb_paths(
                fit, int(context.horizon_days), int(context.simulations), rng
            )
        else:
            paths = _simulate_bayesian_sbb_vol_overlay(
                fit, int(context.horizon_days), int(context.simulations), rng
            )
        expected_shape = (int(context.simulations), int(context.horizon_days))
        if paths.shape != expected_shape or not np.all(np.isfinite(paths)):
            raise ValueError(f"{self.model_id} produced invalid daily paths: {paths.shape}")
        return np.asarray(paths, dtype=np.float64)


REFERENCE_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {
        model_id: (lambda model_id=model_id: PortfolioGJRGARCHModel(model_id))
        for model_id in REFERENCE_MODEL_IDS
    }
)


def make_gjr_reference_model(model_id: str) -> PortfolioGJRGARCHModel:
    """Construct one exact-ID GJR model or reject the requested ID."""

    factory = REFERENCE_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"no source-backed GJR factory for canonical model {model_id!r}")
    return factory()


__all__ = [
    "REFERENCE_FACTORIES",
    "REFERENCE_MODEL_IDS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "PortfolioGJRGARCHModel",
    "_fit_gjr_bayesian_sbb_vol_overlay",
    "_moving_block_bayesian_sbb_paths",
    "_simulate_bayesian_sbb_vol_overlay",
    "make_gjr_reference_model",
]
