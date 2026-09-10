"""Source-derived numerical closure for the canonical base family.

The public canonical-175 base rows are the Cartesian product of three mean
specifications, four volatility specifications, and seven innovation/path
specifications.  This module is a narrow extraction of the numerical methods
used by the pinned Simfolio research engine (revision label
``773bc1c325559e6bf57a567f1d8bf473a3427fbc``), rather than a replacement
forecast heuristic.  The extracted source file was
``app/engine.py`` (SHA-256
``702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665``).

The source repository did not include a destination license.  The research
code is user-authorized; this attribution records provenance without inventing
a license.  Operational caches, workers, service imports, and deployment
configuration are intentionally outside this closure.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from ...runner import ForecastContext, TrainingData
from ...seeds import deterministic_seed
from ...specifications import CompositionalSpec, parse_compositional_spec

STATIONARY_BOOTSTRAP_ROW_ASSEMBLY_MIN_DAYS = 128
_SOURCE_FAILURES = (
    ArithmeticError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


try:  # The source uses Numba for the long path loop when available.
    from numba import njit, prange
except (ImportError, ModuleNotFoundError, OSError):  # pragma: no cover
    njit = None
    prange = range


if njit is not None:

    @njit(cache=False, parallel=True)
    def _simulate_auto_vol_log_paths_numba(
        draws: np.ndarray,
        mu: float,
        vol_code: int,
        sigma0: float,
        lam: float,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
    ) -> np.ndarray:
        """Pinned source daily path kernel (``engine.py:2207-2258``)."""

        n_paths, total_days = draws.shape
        out = np.empty((n_paths, total_days), dtype=np.float64)
        sqrt_two_over_pi = math.sqrt(2.0 / math.pi)
        sigma0 = max(sigma0, 1e-6)
        for i in prange(n_paths):
            sigma_x = sigma0
            h_x = max(sigma_x * sigma_x, 1e-8)
            log_h = math.log(h_x)
            for t in range(total_days):
                z_t = draws[i, t]
                eps_x = sigma_x * z_t
                value = mu + eps_x / 100.0
                if value < -1.0:
                    value = -1.0
                elif value > 1.0:
                    value = 1.0
                out[i, t] = value
                if vol_code == 1:
                    h_x = lam * h_x + (1.0 - lam) * eps_x * eps_x
                    if h_x < 1e-8:
                        h_x = 1e-8
                    elif h_x > 1e8:
                        h_x = 1e8
                    sigma_x = math.sqrt(h_x)
                elif vol_code == 2:
                    log_h = omega + alpha * (abs(z_t) - sqrt_two_over_pi) + gamma * z_t + beta * log_h
                    if log_h < -18.0:
                        log_h = -18.0
                    elif log_h > 18.0:
                        log_h = 18.0
                    h_x = math.exp(log_h)
                    sigma_x = math.sqrt(h_x)
                else:
                    asym = gamma * eps_x * eps_x if vol_code == 4 and eps_x < 0.0 else 0.0
                    h_x = omega + alpha * eps_x * eps_x + asym + beta * h_x
                    if h_x < 1e-8:
                        h_x = 1e-8
                    elif h_x > 1e8:
                        h_x = 1e8
                    sigma_x = math.sqrt(h_x)
        return out

else:
    _simulate_auto_vol_log_paths_numba = None


def _robust_shrinkage_mean(log_returns: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Pinned positive-part t-statistic shrinkage to zero."""

    arr = np.asarray(log_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, np.array([], dtype=np.float64), {"method": "empty_zero"}
    center = float(stats.trim_mean(arr, 0.10)) if arr.size >= 20 else float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = sigma / math.sqrt(float(arr.size)) if sigma > 0.0 else 0.0
    t_stat = abs(center) / se if se > 0.0 else 0.0
    shrink_weight = float((t_stat * t_stat) / (1.0 + t_stat * t_stat)) if np.isfinite(t_stat) else 0.0
    mu = float(center * shrink_weight)
    return mu, arr - mu, {
        "method": "positive_part_t_stat_shrinkage_to_zero",
        "raw_robust_mean": center,
        "shrink_weight": shrink_weight,
    }


def _sample_mean_near_zero_shrinkage(log_returns: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Retained source helper; the base dispatch does not call this method."""

    arr = np.asarray(log_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, {"method": "empty_zero"}
    sample_mean = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = sigma / math.sqrt(float(arr.size)) if sigma > 0.0 else 0.0
    t_stat = abs(sample_mean) / se if se > 0.0 else 0.0
    shrink_weight = float((t_stat * t_stat) / (1.0 + t_stat * t_stat)) if np.isfinite(t_stat) else 0.0
    return float(sample_mean * shrink_weight), {
        "method": "sample_mean_positive_part_t_stat_shrinkage_to_zero",
        "sample_mean": sample_mean,
        "shrink_weight": shrink_weight,
        "standard_error": se,
    }


def _expanding_sample_mean(log_returns: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
    arr = np.asarray(log_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, np.array([], dtype=np.float64), {"method": "empty_zero"}
    mu = float(np.mean(arr))
    return mu, arr - mu, {"method": "expanding_sample_daily_log_mean"}


def _auto_arma_mean(log_returns: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
    arr = np.asarray(log_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 80:
        mu, residuals, meta = _robust_shrinkage_mean(arr)
        meta["fallback_reason"] = "insufficient_observations_for_auto_arma"
        return mu, residuals, meta
    fit_arr = arr[-min(len(arr), 756) :]
    best = None
    try:
        from statsmodels.tsa.arima.model import ARIMA

        for order in ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)):
            result = _fit_arima_order(ARIMA, fit_arr, order)
            if result is None:
                continue
            bic = float(result.bic)
            if np.isfinite(bic) and (best is None or bic < best[0]):
                best = (bic, order, result)
        if best is not None:
            _, order, result = best
            forecast = np.asarray(result.forecast(steps=1), dtype=np.float64)
            mu = float(forecast[0]) if forecast.size and np.isfinite(forecast[0]) else float(np.mean(fit_arr))
            residuals = np.asarray(result.resid, dtype=np.float64)
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size >= 30:
                return mu, residuals, {"method": "bic_selected_arima", "order": tuple(int(v) for v in order)}
    except _SOURCE_FAILURES:
        best = None
    mu, residuals, meta = _robust_shrinkage_mean(arr)
    meta["fallback_reason"] = "auto_arma_fit_failed"
    return mu, residuals, meta


def _fit_arima_order(arima: Any, values: np.ndarray, order: tuple[int, int, int]) -> Any | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return arima(values, order=order, trend="c").fit()
    except _SOURCE_FAILURES:
        return None


def fit_mean(log_returns: np.ndarray, mean_model: str) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Dispatch exactly the three canonical base mean names."""

    if mean_model == "expanding_sample_mean":
        return _expanding_sample_mean(log_returns)
    if mean_model == "bic_auto_arma_mean":
        return _auto_arma_mean(log_returns)
    if mean_model == "factor_premium_near_zero_alpha_shrinkage":
        # This public name falls through the pinned source dispatcher to its
        # robust-shrinkage default.  The similarly named sample helper above
        # is not invoked by the source base path.
        return _robust_shrinkage_mean(log_returns)
    raise ValueError(f"unknown canonical base mean model: {mean_model}")


def _fit_constant_sample_volatility(residuals: np.ndarray) -> dict[str, Any]:
    eps = np.asarray(residuals, dtype=np.float64)
    eps = eps[np.isfinite(eps)]
    x = eps * 100.0
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    sigma = float(max(sigma if np.isfinite(sigma) else 1.0, 1e-6))
    return {
        "vol_model": "constant_sample_volatility",
        "sigma_x": np.full(max(len(x), 1), sigma, dtype=np.float64),
        "last_sigma_x": sigma,
        "params": {"sigma_x": sigma},
        "fit_status": "complete",
    }


def _ewma_volatility_nll(x: np.ndarray, base_var: float, lam: float) -> float:
    arr = np.asarray(x, dtype=np.float64)
    h = float(base_var)
    ll = 0.0
    for value in arr:
        h = float(lam * h + (1.0 - lam) * value * value)
        h = max(h, 1e-8)
        ll += math.log(h) + (value * value / h)
    return float(ll)


def _ewma_volatility_sigma_path(x: np.ndarray, base_var: float, lam: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    sigma = np.empty_like(arr, dtype=np.float64)
    h = float(base_var)
    for idx, value in enumerate(arr):
        h = float(lam * h + (1.0 - lam) * value * value)
        h = max(h, 1e-8)
        sigma[idx] = math.sqrt(h)
    return sigma


def _fit_ewma_volatility(residuals: np.ndarray) -> dict[str, Any]:
    eps = np.asarray(residuals, dtype=np.float64)
    eps = eps[np.isfinite(eps)]
    x = eps * 100.0
    if x.size < 30:
        sigma = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
        sigma = float(max(sigma, 1e-6))
        return {
            "vol_model": "estimated_decay_ewma_volatility",
            "lambda": 0.94,
            "sigma_x": np.full(max(len(x), 1), sigma, dtype=np.float64),
            "last_sigma_x": sigma,
            "params": {"lambda": 0.94},
            "fit_status": "fallback_short_history",
        }
    base_var = float(np.var(x, ddof=1))
    base_var = max(base_var, 1e-8)
    try:
        opt = minimize_scalar(
            lambda lam: _ewma_volatility_nll(x, base_var, float(lam)),
            bounds=(0.80, 0.995),
            method="bounded",
            options={"xatol": 1e-4},
        )
        lam = float(opt.x) if opt.success and np.isfinite(opt.x) else 0.94
    except _SOURCE_FAILURES:
        lam = 0.94
    sigma = _ewma_volatility_sigma_path(x, base_var, lam)
    return {
        "vol_model": "estimated_decay_ewma_volatility",
        "lambda": lam,
        "sigma_x": sigma,
        "last_sigma_x": float(sigma[-1]),
        "params": {"lambda": lam},
        "fit_status": "complete",
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
            "fit_status": "complete" if int(getattr(result, "convergence_flag", 0) or 0) == 0 else "optimizer_warning",
        }
    except _SOURCE_FAILURES:
        return None


def _standardize_residuals(residuals: np.ndarray, sigma_x: np.ndarray) -> np.ndarray:
    x = np.asarray(residuals, dtype=np.float64) * 100.0
    sigma = np.asarray(sigma_x, dtype=np.float64)
    n = int(min(len(x), len(sigma)))
    if n <= 0:
        return np.array([0.0], dtype=np.float64)
    z = x[-n:] / np.maximum(sigma[-n:], 1e-8)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.array([0.0], dtype=np.float64)
    z = z - float(np.mean(z))
    std = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if std > 1e-8:
        z = z / std
    return np.clip(z, -12.0, 12.0)


def _politis_white_block_length(values: np.ndarray) -> int:
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


def _stationary_bootstrap_indices_from_draws_fast(
    *, n: int, initial_indices: np.ndarray, restart_flags: np.ndarray, restart_points: np.ndarray
) -> np.ndarray:
    initial = np.asarray(initial_indices, dtype=np.int32).reshape(-1)
    flags = np.asarray(restart_flags, dtype=bool)
    points = np.asarray(restart_points, dtype=np.int32)
    n_paths = len(initial)
    total_days = int(flags.shape[1] + 1)
    idx = np.empty((n_paths, total_days), dtype=np.int32)
    if n_paths == 0:
        return idx
    idx[:, 0] = initial
    if total_days == 1:
        return idx
    idx[:, 1:] = points
    positions = np.arange(total_days, dtype=np.int32)
    start_positions = np.empty(total_days, dtype=np.int32)
    for row_idx in range(n_paths):
        start_positions[0] = 0
        start_positions[1:] = np.where(flags[row_idx], positions[1:], 0)
        np.maximum.accumulate(start_positions, out=start_positions)
        idx[row_idx] = (idx[row_idx, start_positions] + positions - start_positions) % int(n)
    return idx


def _stationary_bootstrap_indices_from_draws(
    *, n: int, initial_indices: np.ndarray, restart_flags: np.ndarray, restart_points: np.ndarray
) -> np.ndarray:
    initial = np.asarray(initial_indices, dtype=np.int32).reshape(-1)
    flags = np.asarray(restart_flags, dtype=bool)
    points = np.asarray(restart_points, dtype=np.int32)
    if flags.ndim != 2 or points.ndim != 2 or flags.shape != points.shape or flags.shape[0] != len(initial):
        raise ValueError("Bootstrap restart draws must align with initial indices.")
    return _stationary_bootstrap_indices_from_draws_fast(
        n=n, initial_indices=initial, restart_flags=flags, restart_points=points
    )


def _stationary_bootstrap_indices(
    n: int, block_length: int, total_days: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
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
                restart_flags[:, t - 1], restart_points[:, t - 1], (idx[:, t - 1] + 1) % n
            )
        return idx
    return _stationary_bootstrap_indices_from_draws(
        n=n,
        initial_indices=initial_indices,
        restart_flags=restart_flags,
        restart_points=restart_points,
    )


def _evt_standardized_draws(
    z: np.ndarray, size: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    clean = np.asarray(z, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 100:
        return rng.choice(clean if clean.size else np.array([0.0]), size=size, replace=True)
    exceedance_share = float(np.clip(math.sqrt(clean.size) / clean.size, 0.02, 0.10))
    lower_q = float(np.quantile(clean, exceedance_share))
    upper_q = float(np.quantile(clean, 1.0 - exceedance_share))
    central = clean[(clean >= lower_q) & (clean <= upper_q)]
    if central.size == 0:
        central = clean
    out = rng.choice(central, size=size, replace=True).astype(np.float64)
    uniforms = rng.random(size)
    lower_mask = uniforms < exceedance_share
    upper_mask = uniforms > (1.0 - exceedance_share)
    try:
        lower_excess = lower_q - clean[clean < lower_q]
        upper_excess = clean[clean > upper_q] - upper_q
        if lower_excess.size >= 10:
            c, _, scale = stats.genpareto.fit(lower_excess, floc=0.0)
            c = float(np.clip(c, -0.45, 0.45)) if np.isfinite(c) else 0.0
            out[lower_mask] = lower_q - stats.genpareto.rvs(
                c,
                loc=0.0,
                scale=max(scale, 1e-8),
                size=int(np.sum(lower_mask)),
                random_state=rng,
            )
        if upper_excess.size >= 10:
            c, _, scale = stats.genpareto.fit(upper_excess, floc=0.0)
            c = float(np.clip(c, -0.45, 0.45)) if np.isfinite(c) else 0.0
            out[upper_mask] = upper_q + stats.genpareto.rvs(
                c,
                loc=0.0,
                scale=max(scale, 1e-8),
                size=int(np.sum(upper_mask)),
                random_state=rng,
            )
    except _SOURCE_FAILURES:
        out[lower_mask | upper_mask] = rng.choice(
            clean, size=int(np.sum(lower_mask | upper_mask)), replace=True
        )
    return np.clip(out, -20.0, 20.0)


def _standardize_generated_innovations_inplace(draws: np.ndarray) -> np.ndarray:
    arr = np.asarray(draws, dtype=np.float64)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
    np.clip(arr, -20.0, 20.0, out=arr)
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    arr -= mean
    if std > 1e-12 and np.isfinite(std):
        arr /= std
    np.clip(arr, -20.0, 20.0, out=arr)
    return arr


def draw_standardized_innovations(
    base: dict[str, Any],
    tail_method: str,
    path_generator: str,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pinned source innovation dispatch and tail handling."""

    z = np.asarray(base.get("standardized_residuals"), dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        z = np.array([0.0], dtype=np.float64)
    size = (int(n_paths), int(total_days))
    if path_generator == "stationary_bootstrap_standardized_residuals":
        block = max(_politis_white_block_length(z), _politis_white_block_length(z * z))
        idx = _stationary_bootstrap_indices(len(z), block, int(total_days), int(n_paths), rng)
        draws = z[idx]
        if tail_method == "automated_evt_pot_gpd_tail":
            tail_draws = _evt_standardized_draws(z, size, rng)
            extreme = (
                np.abs(draws) >= float(np.quantile(np.abs(z), 0.90))
                if z.size >= 20
                else np.zeros_like(draws, dtype=bool)
            )
            draws = np.where(extreme, tail_draws, draws)
        return _standardize_generated_innovations_inplace(draws)
    if tail_method == "automated_evt_pot_gpd_tail":
        return _standardize_generated_innovations_inplace(_evt_standardized_draws(z, size, rng))
    if (
        path_generator == "filtered_historical_simulation"
        or tail_method == "filtered_empirical_tail"
        or base.get("innovation_method") == "empirical_standardized_residuals"
    ):
        return _standardize_generated_innovations_inplace(
            rng.choice(z, size=size, replace=True).astype(np.float64)
        )
    if base.get("innovation_method") == "gaussian_iid_standardized_innovations":
        return _standardize_generated_innovations_inplace(
            rng.standard_normal(size=size).astype(np.float64)
        )
    try:
        if base.get("innovation_method") == "skew_t_standardized_innovations":
            from arch.univariate import SkewStudent

            return _standardize_generated_innovations_inplace(
                SkewStudent(seed=rng).simulate(
                    [base.get("eta", 8.0), base.get("skew_lambda", 0.0)]
                )(size)
            )
        from arch.univariate import StudentsT

        return _standardize_generated_innovations_inplace(
            StudentsT(seed=rng).simulate([base.get("nu", 8.0)])(size)
        )
    except _SOURCE_FAILURES:
        return _standardize_generated_innovations_inplace(
            rng.choice(z, size=size, replace=True).astype(np.float64)
        )


def _simulate_candidate_log_paths(
    base: dict[str, Any],
    tail_method: str,
    path_generator: str,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_days = int(total_days)
    n_paths = int(n_paths)
    if total_days <= 0 or n_paths <= 0:
        return np.empty((max(n_paths, 0), max(total_days, 0)), dtype=np.float64)
    draws = draw_standardized_innovations(
        base, tail_method, path_generator, total_days, n_paths, rng
    )
    mu = float(base.get("mu", 0.0))
    vol_fit = base.get("vol_fit", {}) or {}
    params = vol_fit.get("params", {}) or {}
    vol_model = str(base.get("vol_model", "estimated_decay_ewma_volatility"))
    sigma_x = np.full(n_paths, float(max(vol_fit.get("last_sigma_x", 1.0), 1e-6)), dtype=np.float64)
    if vol_model == "constant_sample_volatility":
        return np.clip(mu + (float(sigma_x[0]) / 100.0) * draws, -1.0, 1.0)
    if _simulate_auto_vol_log_paths_numba is not None:
        if vol_model == "estimated_decay_ewma_volatility":
            vol_code = 1
            lam = float(vol_fit.get("lambda", params.get("lambda", 0.94)))
            omega = alpha = gamma = 0.0
            beta = 0.0
        elif vol_model == "egarch_1_1_volatility":
            vol_code = 2
            lam = 0.94
            omega = float(params.get("omega", 0.0))
            alpha = float(params.get("alpha[1]", 0.05))
            gamma = float(params.get("gamma[1]", 0.0))
            beta = float(np.clip(params.get("beta[1]", 0.94), -0.98, 0.995))
        else:
            vol_code = 4 if vol_model == "gjr_tarch_1_1_volatility" else 3
            lam = 0.94
            omega = float(max(params.get("omega", 1e-8), 1e-8))
            alpha = float(np.clip(params.get("alpha[1]", 0.05), 0.0, 0.50))
            gamma = float(np.clip(params.get("gamma[1]", 0.0), -0.50, 0.80))
            beta = float(np.clip(params.get("beta[1]", 0.90), 0.0, 0.995))
        return _simulate_auto_vol_log_paths_numba(
            np.asarray(draws, dtype=np.float64),
            mu,
            int(vol_code),
            float(sigma_x[0]),
            float(lam),
            float(omega),
            float(alpha),
            float(gamma),
            float(beta),
        )
    out = np.empty((n_paths, total_days), dtype=np.float64)
    h_x = np.maximum(sigma_x * sigma_x, 1e-8)
    log_h = np.log(h_x)
    sqrt_two_over_pi = math.sqrt(2.0 / math.pi)
    for t in range(total_days):
        z_t = np.asarray(draws[:, t], dtype=np.float64)
        eps_x = sigma_x * z_t
        out[:, t] = mu + eps_x / 100.0
        if vol_model == "estimated_decay_ewma_volatility":
            lam = float(vol_fit.get("lambda", params.get("lambda", 0.94)))
            h_x = lam * h_x + (1.0 - lam) * eps_x * eps_x
            h_x = np.clip(h_x, 1e-8, 1e8)
            sigma_x = np.sqrt(h_x)
        elif vol_model == "egarch_1_1_volatility":
            omega = float(params.get("omega", 0.0))
            alpha = float(params.get("alpha[1]", 0.05))
            gamma = float(params.get("gamma[1]", 0.0))
            beta = float(np.clip(params.get("beta[1]", 0.94), -0.98, 0.995))
            log_h = omega + alpha * (np.abs(z_t) - sqrt_two_over_pi) + gamma * z_t + beta * log_h
            log_h = np.clip(log_h, -18.0, 18.0)
            sigma_x = np.sqrt(np.exp(log_h))
            h_x = sigma_x * sigma_x
        else:
            omega = float(max(params.get("omega", 1e-8), 1e-8))
            alpha = float(np.clip(params.get("alpha[1]", 0.05), 0.0, 0.50))
            gamma = float(np.clip(params.get("gamma[1]", 0.0), -0.50, 0.80))
            beta = float(np.clip(params.get("beta[1]", 0.90), 0.0, 0.995))
            asym = (
                gamma * (eps_x < 0.0) * eps_x * eps_x
                if vol_model == "gjr_tarch_1_1_volatility"
                else 0.0
            )
            h_x = omega + alpha * eps_x * eps_x + asym + beta * h_x
            h_x = np.clip(h_x, 1e-8, 1e8)
            sigma_x = np.sqrt(h_x)
    return np.clip(out, -1.0, 1.0)


def fit_base_model(
    log_returns: np.ndarray,
    spec: CompositionalSpec,
    *,
    allow_source_arch_fallback: bool = False,
) -> dict[str, Any]:
    """Fit one canonical base specification using the extracted source path."""

    spec.validate()
    arr = np.asarray(log_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 30:
        raise ValueError("canonical base model requires at least 30 observations")
    mu, residuals, mean_meta = fit_mean(arr, spec.mean_model)
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 30:
        raise ValueError("canonical base mean fit returned fewer than 30 residuals")
    innovation_method = (
        "empirical_standardized_residuals"
        if spec.innovation_model == "empirical"
        else spec.innovation_model
    )
    if spec.volatility_model == "constant_sample_volatility":
        vol_fit = _fit_constant_sample_volatility(residuals)
    else:
        vol_fit = _fit_arch_volatility(
            residuals, spec.volatility_model, innovation_method
        )
        if vol_fit is None:
            if not allow_source_arch_fallback:
                raise RuntimeError(
                    f"exact {spec.volatility_model} fit unavailable; install pinned arch dependency"
                )
            # This is the pinned source engine's explicit failure path.  It is
            # retained only as an opt-in source replay mode and is recorded in
            # vol_fit; canonical model adapters stay strict by default.
            vol_fit = _fit_ewma_volatility(residuals)
            vol_fit["requested_vol_model"] = spec.volatility_model
            vol_fit["vol_model"] = spec.volatility_model
            vol_fit["fit_status"] = "fallback_ewma_after_arch_failure"
    z = _standardize_residuals(
        residuals, np.asarray(vol_fit["sigma_x"], dtype=np.float64)
    )
    if z.size < 20:
        raise ValueError("canonical base fit returned fewer than 20 standardized residuals")
    nu = float(vol_fit.get("params", {}).get("nu", 8.0))
    eta = float(vol_fit.get("params", {}).get("eta", nu if np.isfinite(nu) else 8.0))
    skew_lambda = float(vol_fit.get("params", {}).get("lambda", 0.0))
    if not np.isfinite(nu) or nu < 2.05:
        nu = 8.0
    if not np.isfinite(eta) or eta < 2.05:
        eta = max(2.1, nu)
    if not np.isfinite(skew_lambda):
        skew_lambda = 0.0
    return {
        "id": spec.source_id,
        "simulation_method": spec.source_id,
        "mean_model": spec.mean_model,
        "vol_model": spec.volatility_model,
        "innovation_method": innovation_method,
        "tail_method": _source_tail_method(spec),
        "path_generator": _source_path_generator(spec),
        "innovation_resampling": spec.resampling,
        "innovation_pool_source": "sample_standardized",
        "mu": float(mu),
        "residuals": residuals.astype(np.float64),
        "standardized_residuals": z.astype(np.float64),
        "mean_meta": mean_meta,
        "vol_fit": vol_fit,
        "block_length": int(max(_politis_white_block_length(z), _politis_white_block_length(z * z))),
        "nu": float(np.clip(nu, 2.05, 80.0)),
        "eta": float(np.clip(eta, 2.05, 80.0)),
        "skew_lambda": float(np.clip(skew_lambda, -0.95, 0.95)),
        "n_obs": len(arr),
        "canonical_spec": {
            "mean_model": spec.mean_model,
            "vol_model": spec.volatility_model,
            "innovation_model": spec.innovation_model,
            "resampling": spec.resampling,
            "tail_method": spec.tail_method,
        },
    }


def _source_tail_method(spec: CompositionalSpec) -> str:
    if spec.tail_method == "parametric":
        return "native_distribution_tail"
    return spec.tail_method


def _source_path_generator(spec: CompositionalSpec) -> str:
    if spec.innovation_model != "empirical":
        return "parametric_monte_carlo"
    if spec.resampling == "stationary_bootstrap":
        return "stationary_bootstrap_standardized_residuals"
    if spec.resampling == "iid":
        return "filtered_historical_simulation"
    raise ValueError(f"unknown canonical base resampling method: {spec.resampling}")


def historical_base_seed(
    model_id: str, origin_date_or_label: str, horizon_days: int, simulations: int
) -> int:
    """Reproduce source ``forecast_oos_candidate`` seed identity."""

    return deterministic_seed(
        "forecast_oos_candidate",
        str(origin_date_or_label),
        tuple(range(1, int(horizon_days) + 1)),
        str(model_id),
        int(simulations),
    )


def simulate_base_paths(
    fit: dict[str, Any], *, horizon_days: int, simulations: int, rng: np.random.Generator
) -> np.ndarray:
    return _simulate_candidate_log_paths(
        fit,
        str(fit["tail_method"]),
        str(fit["path_generator"]),
        int(horizon_days),
        int(simulations),
        rng,
    )


@dataclass(frozen=True)
class CanonicalBaseModel:
    """Concrete model adapter for exactly the canonical base ID domain."""

    model_id: str

    def _spec(self) -> CompositionalSpec:
        spec = parse_compositional_spec(self.model_id)
        spec.validate()
        return spec

    def fit(self, training: TrainingData) -> dict[str, Any]:
        training.validate()
        return fit_base_model(training.portfolio_log_returns, self._spec())

    def simulate_daily_log_returns(
        self, training: TrainingData, context: ForecastContext
    ) -> np.ndarray:
        spec = self._spec()
        training.validate()
        fit = fit_base_model(training.portfolio_log_returns, spec)
        origin = context.origin_date or context.origin_label
        seed = historical_base_seed(
            self.model_id, str(origin), context.horizon_days, context.simulations
        )
        rng = np.random.default_rng(seed)
        return simulate_base_paths(
            fit,
            horizon_days=context.horizon_days,
            simulations=context.simulations,
            rng=rng,
        )


def build_canonical_base_model(model_id: str) -> CanonicalBaseModel:
    """Build only a validated canonical base model; unknown IDs raise."""

    model = CanonicalBaseModel(model_id=model_id)
    model._spec()
    return model


__all__ = [
    "CanonicalBaseModel",
    "_fit_arch_volatility",
    "_fit_constant_sample_volatility",
    "_politis_white_block_length",
    "_simulate_auto_vol_log_paths_numba",
    "_standardize_residuals",
    "build_canonical_base_model",
    "draw_standardized_innovations",
    "fit_base_model",
    "fit_mean",
    "historical_base_seed",
    "simulate_base_paths",
]
