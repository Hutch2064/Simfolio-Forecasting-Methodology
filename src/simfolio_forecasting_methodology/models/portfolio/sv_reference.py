"""Source-backed canonical stochastic-volatility reference families.

This module contains only the three small SV candidates retained in the
research gate.  The fit and simulation closure is copied from the source
research gate and its engine dependency.  The public adapters expose the
daily path-derived terminal ensemble used by that gate; no terminal samples
are turned back into artificial daily increments.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import optimize, stats

from ...runner import TERMINAL_FORECAST_SEMANTICS, ForecastContext, TrainingData

SOURCE_ARTIFACTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
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
    }
)

SOURCE_FUNCTION_NAMES: tuple[str, ...] = (
    "_sv_observed_log_variance",
    "_sv_observed_log_variance_from_residuals",
    "_sv_kalman_filter",
    "_initial_sv_state_space_params",
    "_fit_sv_state_space_params",
    "_fit_sv_ar1_with_mean",
    "_fit_sv_ar1",
    "_sv_initial_log_var",
    "_simulate_sv_ar1",
    "_simulate_sv_ar1_sbb",
    "_politis_white_block_length",
)

# SHA-256 over the exact source function segments above: the first ten come
# from the research gate and the last comes from its imported engine module;
# segments are joined with one blank line and a trailing newline.
SOURCE_FUNCTIONS_SHA256 = "63035755f9fa6f9097c099ab481667f9fc97effc0d5364b3fe6ae0282e20944f"

SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, public_model_id, simulations)"
)

REFERENCE_MODEL_IDS: tuple[str, ...] = (
    "stochastic_volatility_ar1_empirical",
    "stochastic_volatility_ar1_empirical_sbb",
    "stochastic_volatility_ar1_student_t",
)

SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "stochastic_volatility_ar1_empirical": MappingProxyType(
            {
                "id": "stochastic_volatility_ar1_empirical",
                "type": "sv",
                "innovation": "empirical",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "sample_mean_near_zero_shrinkage",
                "vol_model": "stochastic_volatility_ar1_log_variance",
                "innovation_method": "empirical_standardized_residuals_iid",
                "tail_method": "empirical_standardized_residual_tail",
                "path_generator": "stochastic_volatility_ar1",
                "candidate_role": "stochastic_volatility_reference",
            }
        ),
        "stochastic_volatility_ar1_empirical_sbb": MappingProxyType(
            {
                "id": "stochastic_volatility_ar1_empirical_sbb",
                "type": "sv_sbb",
                "innovation": "empirical",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "sample_mean_near_zero_shrinkage",
                "vol_model": "stochastic_volatility_ar1_log_variance",
                "innovation_method": "empirical_standardized_residuals_stationary_bootstrap",
                "tail_method": "empirical_standardized_residual_tail",
                "path_generator": "stochastic_volatility_ar1_stationary_bootstrap",
                "candidate_role": "stochastic_volatility_stationary_bootstrap_reference",
            }
        ),
        "stochastic_volatility_ar1_student_t": MappingProxyType(
            {
                "id": "stochastic_volatility_ar1_student_t",
                "type": "sv",
                "innovation": "student_t",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "sample_mean_near_zero_shrinkage",
                "vol_model": "stochastic_volatility_ar1_log_variance",
                "innovation_method": "student_t_standardized_innovations",
                "tail_method": "student_t_standardized_residual_tail",
                "path_generator": "stochastic_volatility_ar1",
                "candidate_role": "stochastic_volatility_student_t_reference",
            }
        ),
    }
)

SV_LOG_CHI_SQUARE_MEAN = -1.2703628454614782
SV_LOG_CHI_SQUARE_VAR = (math.pi * math.pi) / 2.0


def _sv_observed_log_variance(
    train_values: np.ndarray,
    mu: float,
    *,
    winsorize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Source gate `_sv_observed_log_variance`."""
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return _sv_observed_log_variance_from_residuals(x - float(mu), winsorize=winsorize)


def _sv_observed_log_variance_from_residuals(
    residual_values: np.ndarray,
    *,
    winsorize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Source gate `_sv_observed_log_variance_from_residuals`."""
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
        # log(0) is undefined in the SV measurement equation. Treat exact
        # zero squared residuals as missing volatility measurements.
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
) -> tuple[float, np.ndarray, np.ndarray]:
    """Source gate `_sv_kalman_filter` without optional numba dispatch."""
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


def _initial_sv_state_space_params(observed_log_var: np.ndarray) -> tuple[float, float, float]:
    """Source gate `_initial_sv_state_space_params`."""
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
) -> tuple[float, float, float]:
    """Source gate `_fit_sv_state_space_params`."""
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
    best: tuple[float, np.ndarray] | None = None
    try:
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
    except Exception:  # noqa: BLE001 - preserve source fallback when the fit fails
        best = None
    if best is None:
        return init_level, init_phi, init_eta
    params = best[1]
    return float(params[0]), float(params[1]), float(math.exp(params[2]))


def _sample_mean_near_zero_shrinkage(log_returns: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Source engine `SimfolioEngine._sample_mean_near_zero_shrinkage`."""
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


def _fit_sv_ar1_with_mean(
    train_values: np.ndarray,
    *,
    forecast_mu: float,
    residual_center: float,
    mean_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Source gate `_fit_sv_ar1_with_mean`."""
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 252:
        return None
    observed_log_var, eps_x = _sv_observed_log_variance(x, float(residual_center))
    if observed_log_var.size < 252 or eps_x.size < 252:
        return None
    level, phi, eta_sd = _fit_sv_state_space_params(observed_log_var)
    loglik, filtered, filtered_var = _sv_kalman_filter(
        observed_log_var, level, phi, eta_sd, return_path=True
    )
    if not np.isfinite(loglik) or filtered.size == 0:
        return None
    sigma_x = np.exp(0.5 * np.clip(filtered, -18.0, 18.0))
    standardized = eps_x[: sigma_x.size] / np.maximum(sigma_x, 1e-8)
    standardized = standardized[np.isfinite(standardized)]
    standardized = standardized - float(np.mean(standardized))
    z_sd = float(np.std(standardized, ddof=1)) if standardized.size > 1 else 1.0
    if z_sd > 1e-12 and np.isfinite(z_sd):
        standardized = standardized / z_sd
    kurt = 0.0
    try:
        kurt = float(stats.kurtosis(standardized, fisher=True, bias=False))
    except Exception:  # noqa: BLE001 - preserve source fallback for scipy statistics
        kurt = 0.0
    df = float(np.clip(4.0 + 6.0 / max(kurt, 0.1), 4.0, 30.0)) if np.isfinite(kurt) and kurt > 0.1 else 30.0
    return {
        "mu": float(forecast_mu),
        "residual_center": float(residual_center),
        "mean_meta": mean_meta,
        "level": level,
        "phi": phi,
        "eta_sd": eta_sd,
        "last_log_var": float(filtered[-1]),
        "last_state_mean": float(filtered[-1]),
        "last_state_var": float(filtered_var[-1]),
        "filtered_log_var": filtered.astype(float),
        "filtered_log_var_var": filtered_var.astype(float),
        "observed_log_var": observed_log_var.astype(float),
        "standardized_residuals": np.clip(standardized, -12.0, 12.0),
        "df": df,
        "state_space_loglikelihood": float(loglik),
        "observation_model": "log_chi_square_bias_corrected_kalman",
    }


def _fit_sv_ar1(train_values: np.ndarray) -> dict[str, Any] | None:
    """Source gate `_fit_sv_ar1`."""
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 252:
        return None
    mu, meta = _sample_mean_near_zero_shrinkage(x)
    return _fit_sv_ar1_with_mean(
        x,
        forecast_mu=float(mu),
        residual_center=float(mu),
        mean_meta=meta,
    )


def _sv_initial_log_var(fit: dict[str, Any], n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Source gate `_sv_initial_log_var`."""
    mean = float(fit.get("last_state_mean", fit.get("last_log_var", fit.get("level", 0.0))))
    var = float(max(fit.get("last_state_var", 0.0), 0.0))
    if var <= 1e-10:
        return np.full(int(n_paths), mean, dtype=np.float64)
    return rng.normal(mean, math.sqrt(var), size=int(n_paths)).astype(np.float64)


def _simulate_sv_ar1(
    fit: dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
    innovation: str,
) -> np.ndarray:
    """Source gate `_simulate_sv_ar1`."""
    total_days = int(total_days)
    n_paths = int(n_paths)
    out = np.empty((n_paths, total_days), dtype=np.float64)
    log_var = _sv_initial_log_var(fit, n_paths, rng)
    level = float(fit["level"])
    phi = float(fit["phi"])
    eta_sd = float(fit["eta_sd"])
    mu = float(fit["mu"])
    z_pool = np.asarray(fit.get("standardized_residuals"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size == 0:
        z_pool = np.array([0.0], dtype=np.float64)
    df = float(max(fit.get("df", 8.0), 2.05))
    t_scale = math.sqrt((df - 2.0) / df) if df > 2.0 else 1.0
    for day in range(total_days):
        if eta_sd > 0.0:
            log_var = level + phi * (log_var - level) + rng.normal(0.0, eta_sd, size=n_paths)
        else:
            log_var = level + phi * (log_var - level)
        log_var = np.clip(log_var, -18.0, 18.0)
        sigma = np.exp(0.5 * log_var) / 100.0
        if innovation == "student_t":
            z = rng.standard_t(df, size=n_paths) * t_scale
        else:
            z = rng.choice(z_pool, size=n_paths, replace=True)
        out[:, day] = mu + sigma * z
    return np.clip(out, -1.0, 1.0)


def _politis_white_block_length(values: np.ndarray) -> int:
    """Source engine `_politis_white_block_length`."""
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
    n: int,
    block_length: int,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source engine stationary bootstrap index kernel."""
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


def _simulate_sv_ar1_sbb(
    fit: dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source gate `_simulate_sv_ar1_sbb`."""
    total_days = int(total_days)
    n_paths = int(n_paths)
    out = np.empty((n_paths, total_days), dtype=np.float64)
    log_var = _sv_initial_log_var(fit, n_paths, rng)
    level = float(fit["level"])
    phi = float(fit["phi"])
    eta_sd = float(fit["eta_sd"])
    mu = float(fit["mu"])
    z_pool = np.asarray(fit.get("standardized_residuals"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size == 0:
        z_pool = np.array([0.0], dtype=np.float64)
    block_length = max(1, _politis_white_block_length(z_pool))
    z_indices = _stationary_bootstrap_indices(
        n=len(z_pool),
        block_length=int(block_length),
        total_days=total_days,
        n_paths=n_paths,
        rng=rng,
    )
    z_draws = z_pool[z_indices]
    for day in range(total_days):
        if eta_sd > 0.0:
            log_var = level + phi * (log_var - level) + rng.normal(0.0, eta_sd, size=n_paths)
        else:
            log_var = level + phi * (log_var - level)
        log_var = np.clip(log_var, -18.0, 18.0)
        sigma = np.exp(0.5 * log_var) / 100.0
        out[:, day] = mu + sigma * z_draws[:, day]
    return np.clip(out, -1.0, 1.0)


@dataclass(frozen=True)
class SVReferenceModel:
    """Explicit factory result for one of the three source SV candidates."""

    model_id: str
    forecast_output_semantics: str = TERMINAL_FORECAST_SEMANTICS
    seed_contract: str = SOURCE_SEED_CONTRACT
    dependency_identity: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {
                "source_imports": ("hashlib", "math", "numpy", "scipy.optimize", "scipy.stats"),
                "runtime_constraints": ("numpy>=2.0,<3", "scipy>=1.13,<2"),
            }
        )
    )

    def __post_init__(self) -> None:
        if self.model_id not in REFERENCE_MODEL_IDS:
            raise ValueError(f"unknown SV reference model: {self.model_id!r}")

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
        source_origin = str(context.origin_date) if context.origin_date is not None else str(context.origin_label)
        dense_horizon_tuple = tuple(range(1, int(context.horizon_days) + 1))
        seed = _deterministic_seed(
            "forecast_oos_candidate",
            source_origin,
            dense_horizon_tuple,
            self.model_id,
            int(context.simulations),
        )
        return np.random.default_rng(seed)

    def simulate_terminal_log_returns(
        self,
        training: TrainingData,
        context: ForecastContext,
    ) -> np.ndarray:
        training.validate()
        values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
        fit = _fit_sv_ar1(values)
        if fit is None:
            raise ValueError(f"{self.model_id} requires at least 252 finite observations")
        rng = self._source_rng(context)
        if self.model_id == "stochastic_volatility_ar1_empirical_sbb":
            paths = _simulate_sv_ar1_sbb(fit, context.horizon_days, context.simulations, rng)
        else:
            innovation = str(SOURCE_CANDIDATE_SPECS[self.model_id]["innovation"])
            paths = _simulate_sv_ar1(fit, context.horizon_days, context.simulations, rng, innovation)
        terminals = np.cumsum(paths, axis=1, dtype=np.float64)
        expected_shape = (int(context.simulations), int(context.horizon_days))
        if terminals.shape != expected_shape or not np.all(np.isfinite(terminals)):
            raise ValueError(f"{self.model_id} produced an invalid terminal ensemble")
        return terminals


def _deterministic_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


REFERENCE_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {model_id: (lambda model_id=model_id: SVReferenceModel(model_id)) for model_id in REFERENCE_MODEL_IDS}
)


def make_sv_reference_model(model_id: str) -> SVReferenceModel:
    """Construct one exact-ID SV reference model; unknown IDs fail closed."""
    factory = REFERENCE_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"no source-backed SV factory for canonical model {model_id!r}")
    return factory()


__all__ = [
    "REFERENCE_FACTORIES",
    "REFERENCE_MODEL_IDS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "SVReferenceModel",
    "make_sv_reference_model",
]
