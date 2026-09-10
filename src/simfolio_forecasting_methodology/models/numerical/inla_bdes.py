"""Historical full-INLA BDES marginal closure.

The descriptor and numerical steps are recovered from the pinned source engine
candidate ``_bdes_cagr_candidate``.  This module deliberately accepts exactly
that descriptor; it never derives settings from a public model id or silently
substitutes the current catalogue.  The shared state-space and BDES helpers
are the source-parity closures already extracted in :mod:`bdes_fastmap` and
:meth:`mcmc_sv`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .bdes_fastmap import (
    BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC,
    FULL_MCMC_SV_MIN_OBS,
    SV_LOG_CHI_SQUARE_MEAN,
    _finite_correlation,
    _fit_sv_map_state_space_params,
    _standardized_empirical_innovation_pool,
    _sv_kalman_filter,
    _sv_kalman_rts_smoother_mean,
    _sv_observed_log_variance,
    _sv_transformed_log_posterior,
    dlm_mu_draw_paths,
    politis_white_block_length,
    simulate_full_mcmc_sv_bdes_log_paths_serial_numba,
    stationary_bootstrap_indices,
)
from .mcmc_sv import (
    _base_sbb_fit_from_mean_fit,
    _evidence_estimated_sharpe_dlm_historical_cagr_anchor_mean,
)

INLA_MODEL_ID = (
    "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_"
    "conditional_sharpe_full_inla_laplace_quadrature_centered_multiscale"
)
SOURCE_ENGINE_PATH = "source-research/app/engine.py"
SOURCE_ENGINE_SHA256 = "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
SOURCE_RESEARCH_GATE_PATH = "source-research/scripts/forecast_oos_research_gate.py"
SOURCE_RESEARCH_GATE_SHA256 = "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997"

# Exact result of SimfolioEngine._bdes_cagr_candidate() reconstructed from
# _bdes_cagr_mcmc_candidate -> _bdes_cagr_fastmap_candidate -> _bdes_cagr_candidate.
# Keep omitted fields omitted: the raw descriptor is part of the fit seed
# identity and default insertion would change historical results.
INLA_CANDIDATE: dict[str, Any] = {
    "ablation_family": "current_production_bdes_cagr_inla_laplace",
    "bdes_multiscale_component_count": "history_gated",
    "bdes_multiscale_scale_grid": BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC,
    "bdes_multiscale_shock_coupling": "shared_state_estimated_residual",
    "clip_simulated_returns": True,
    "comparison_target": "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_inla_laplace_sigma_points",
    "external_factors": False,
    "factor_model": "not_applicable",
    "feature_family": "current_production_bdes_cagr_non_mcmc_state_inference",
    "forecast_level": "portfolio_return",
    "id": INLA_MODEL_ID,
    "inla_hyperparameter_quadrature_order": 3,
    "inla_latent_terminal_quadrature_order": 3,
    "innovation_component": "bdes_multiscale_latent_log_volatility_with_conditional_sharpe_mean_scaling",
    "innovation_conditioning": "none",
    "innovation_method": "empirical_standardized_residuals",
    "innovation_pool_source": "sample_standardized",
    "innovation_resampling": "stationary_bootstrap",
    "innovation_standardization": "mean_std",
    "innovation_tail_splice": "none",
    "latent_vol_persistence": "ar1",
    "leverage": True,
    "leverage_alignment": "lagged_return",
    "leverage_correlation_method": "pearson",
    "leverage_correlation_scope": "deterministic_selected_state_path",
    "map_maxiter": 100,
    "map_start_count": 1,
    "mcmc_adaptation_iterations": 0,
    "mcmc_adaptation_target_acceptance": None,
    "mcmc_burn": 0,
    "mcmc_chains": 0,
    "mcmc_check_interval": 0,
    "mcmc_extend_iterations": 0,
    "mcmc_forecast_stability_tolerance": None,
    "mcmc_initial_iterations": 0,
    "mcmc_iterations": 0,
    "mcmc_max_iterations": 0,
    "mcmc_min_ess": None,
    "mcmc_min_iterations": 0,
    "mcmc_proposal_adaptation": "none",
    "mcmc_rhat_threshold": None,
    "mcmc_stopping": "none",
    "mcmc_thin": 0,
    "mean_component": "portfolio_likelihood_estimated_latent_sharpe_dlm_to_historical_cagr_anchor",
    "mean_model": "evidence_estimated_sharpe_dlm_historical_cagr_anchor",
    "mean_model_ablation_of": "bayesian_sbb_overlay_gjr_garch_1_1_empirical_bayes_sharpe",
    "mean_state_scaling": "conditional_sv_sharpe",
    "measurement_equation": "log_chi_square_mean_corrected_winsorized",
    "overlay_model": INLA_MODEL_ID,
    "parameter_sampler": "deterministic_full_inla_laplace_quadrature",
    "path_generator": "stationary_bootstrap_standardized_residuals_scaled_by_bdes_multiscale_latent_sv_paths",
    "previous_incumbent_id": "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_inla_laplace_sigma_points",
    "production_status": "promoted_to_live_engine_20260713",
    "regime_model": "not_applicable",
    "research_rationale": "Posterior-weighted integrated nested Laplace quadrature perturbs the central multiscale log-volatility forecast by the sampled AR(1) path's deviation from its central-mode counterpart under common innovations. The central multiscale basis uses history-supported calendar half-lives through one year and geometric doubling thereafter. Its residual state and common-state loading are estimated from information available at the forecast origin.",
    "residual_tail_method": "raw_empirical_untruncated",
    "return_target": "portfolio_daily_log_return",
    "rollback_env_var": "SIMFOLIO_FORECAST_MODEL=inla_sigma_points",
    "selection_method": "pre_specified_bdes_cagr_mode_centered_full_inla_laplace_quadrature_sv_validation",
    "simulation_method": INLA_MODEL_ID,
    "state_inference": "full_inla_laplace_quadrature",
    "state_inference_model": "full_inla_laplace_quadrature",
    "state_innovation_coupling": "correlation_mixture",
    "state_innovation_distribution": "gaussian",
    "state_innovation_resampling": "iid",
    "state_innovation_source": "mode_centered_common_innovation_full_inla_quadrature_state_path",
    "state_innovation_standardization": "mean_std",
    "sv_measurement_bias": "log_chi_square_theoretical",
    "sv_research_baseline": False,
    "sv_sigma_scale_method": "none",
    "tail_method": "filtered_empirical_tail",
    "transformed_parameter_mcmc": False,
    "type": "bdes_non_mcmc_sv_overlay",
    "unclipped_empirical_innovations": True,
    "unclipped_sv_measurement": False,
    "validation_status": "paired_80_portfolio_calendar_geometric_mode_centered_full_inla_33_origin_15_horizon_300_simulation_validation",
    "vol_anchor_model": "none",
    "vol_anchor_target": "none",
    "vol_model": "latent_stochastic_log_volatility_overlay",
    "vol_overlay_model": "latent_stochastic_log_volatility_overlay",
    "vol_path_blend": "inla_mode_centered_multiscale",
    "vol_path_model": "bdes_multiscale_log_vol",
}


def source_candidate_digest() -> str:
    payload = json.dumps(INLA_CANDIDATE, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def historical_inla_candidate() -> dict[str, Any]:
    """Return a defensive copy of the exact raw source candidate."""
    return copy.deepcopy(INLA_CANDIDATE)


def _validate_candidate(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = historical_inla_candidate() if candidate is None else dict(candidate)
    if settings != INLA_CANDIDATE:
        unknown = sorted(set(settings) - set(INLA_CANDIDATE))
        mismatched = sorted(
            key for key in INLA_CANDIDATE if settings.get(key) != INLA_CANDIDATE[key]
        )
        raise ValueError(f"inla_unknown_or_mismatched_source_candidate:{unknown + mismatched}")
    return settings


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


def _sv_transformed_vector_from_params(level: float, phi: float, eta: float) -> np.ndarray:
    return np.asarray(
        [
            float(level),
            _logit_unit_interval(float(np.clip(phi, 0.001, 0.995))),
            math.log(float(np.clip(eta, 0.02, 2.5))),
        ],
        dtype=np.float64,
    )


def _sv_params_from_transformed_vector(params: np.ndarray) -> tuple[float, float, float] | None:
    values = np.asarray(params, dtype=np.float64)
    if values.size < 3 or not np.all(np.isfinite(values[:3])):
        return None
    level = float(np.clip(values[0], -18.0, 18.0))
    phi = float(_inv_logit_unit_interval(float(np.clip(values[1], -9.0, 9.0))))
    eta = float(math.exp(float(np.clip(values[2], math.log(0.02), math.log(2.5)))))
    if not all(np.isfinite(value) for value in (level, phi, eta)):
        return None
    return level, float(np.clip(phi, 0.001, 0.995)), float(np.clip(eta, 0.02, 2.5))


def _sv_transformed_negative_log_posterior(
    observed_log_var: np.ndarray, params: np.ndarray
) -> float:
    natural = _sv_params_from_transformed_vector(params)
    if natural is None:
        return 1e100
    level, phi, eta = natural
    value = _sv_transformed_log_posterior(
        observed_log_var, level, _logit_unit_interval(phi), math.log(eta)
    )
    return -float(value) if np.isfinite(value) else 1e100


def _finite_difference_hessian(
    objective: Callable[[np.ndarray], float], center: np.ndarray, steps: np.ndarray
) -> np.ndarray | None:
    x0 = np.asarray(center, dtype=np.float64)
    h = np.asarray(steps, dtype=np.float64)
    dim = int(x0.size)
    if dim == 0 or h.size != dim or not (np.all(np.isfinite(x0)) and np.all(np.isfinite(h))):
        return None
    f0 = float(objective(x0))
    if not np.isfinite(f0):
        return None
    hessian = np.empty((dim, dim), dtype=np.float64)
    for i in range(dim):
        step_i = np.zeros(dim, dtype=np.float64)
        step_i[i] = float(max(abs(h[i]), 1e-4))
        f_plus, f_minus = float(objective(x0 + step_i)), float(objective(x0 - step_i))
        if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
            return None
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step_i[i] * step_i[i])
        for j in range(i + 1, dim):
            step_j = np.zeros(dim, dtype=np.float64)
            step_j[j] = float(max(abs(h[j]), 1e-4))
            values = (
                float(objective(x0 + step_i + step_j)),
                float(objective(x0 + step_i - step_j)),
                float(objective(x0 - step_i + step_j)),
                float(objective(x0 - step_i - step_j)),
            )
            if not all(np.isfinite(value) for value in values):
                return None
            value = (values[0] - values[1] - values[2] + values[3]) / (4.0 * step_i[i] * step_j[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return 0.5 * (hessian + hessian.T)


def _inverse_hessian_covariance(hessian: np.ndarray | None) -> np.ndarray | None:
    if hessian is None:
        return None
    matrix = np.asarray(hessian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        return None
    try:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        return None
    if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
        return None
    eigvals = np.clip(eigvals, 1e-4, 1e6)
    covariance = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance if np.all(np.isfinite(covariance)) else None


def _sv_sigma_samples_from_transformed_vectors(
    observed_log_var: np.ndarray, vectors: Sequence[np.ndarray], rho: float
) -> list[tuple[float, float, float, float, float]]:
    samples: list[tuple[float, float, float, float, float]] = []
    seen: set[tuple[float, float, float, float, float]] = set()
    for vector in vectors:
        natural = _sv_params_from_transformed_vector(np.asarray(vector, dtype=np.float64))
        if natural is None:
            continue
        level, phi, eta = natural
        _, terminal_state, _ = _sv_kalman_filter(
            observed_log_var, level, phi, eta, return_path=False
        )
        if terminal_state.size == 0:
            continue
        sample = (
            float(np.clip(level, -18.0, 18.0)),
            float(np.clip(phi, 0.0, 0.995)),
            float(np.clip(eta, 0.02, 2.50)),
            float(np.clip(terminal_state[-1], -18.0, 18.0)),
            float(np.clip(rho, -0.95, 0.95)),
        )
        if not all(np.isfinite(value) for value in sample):
            continue
        key = tuple(round(value, 12) for value in sample)
        if key not in seen:
            seen.add(key)
            samples.append(sample)
    return samples


def _sv_inla_laplace_sigma_samples(
    observed_log_var: np.ndarray, *, level: float, phi: float, eta: float, rho: float
) -> tuple[list[tuple[float, float, float, float, float]], dict[str, Any]]:
    center = _sv_transformed_vector_from_params(level, phi, eta)
    objective = lambda vector: _sv_transformed_negative_log_posterior(observed_log_var, vector)
    hessian = _finite_difference_hessian(
        objective, center, np.asarray([0.03, 0.08, 0.06], dtype=np.float64)
    )
    covariance = _inverse_hessian_covariance(hessian)
    vectors: list[np.ndarray] = [center]
    radius = math.sqrt(float(center.size))
    if covariance is not None:
        try:
            chol = np.linalg.cholesky(covariance + np.eye(center.size, dtype=np.float64) * 1e-10)
            for idx in range(center.size):
                offset = radius * chol[:, idx]
                vectors.extend((center + offset, center - offset))
        except np.linalg.LinAlgError:
            covariance = None
    if covariance is None:
        std = np.asarray([0.05, 0.12, 0.10], dtype=np.float64)
        for idx in range(center.size):
            offset = np.zeros(center.size, dtype=np.float64)
            offset[idx] = std[idx]
            vectors.extend((center + offset, center - offset))
    samples = _sv_sigma_samples_from_transformed_vectors(observed_log_var, vectors, rho)
    return samples, {
        "approximation": "inla_style_nested_laplace",
        "hyperparameter_dimension": int(center.size),
        "sigma_point_count": len(samples),
        "hessian_available": bool(hessian is not None),
        "covariance_available": bool(covariance is not None),
    }


def _normalize_positive_weights(raw_weights: Sequence[float]) -> np.ndarray | None:
    weights = np.asarray(raw_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or not np.all(np.isfinite(weights)):
        return None
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    return weights / total if np.isfinite(total) and total > 0.0 else None


def _sv_full_inla_laplace_quadrature_samples(
    observed_log_var: np.ndarray,
    *,
    level: float,
    phi: float,
    eta: float,
    rho: float,
    hyperparameter_order: int = 3,
    latent_order: int = 3,
) -> tuple[list[tuple[float, float, float, float, float]], dict[str, Any], np.ndarray | None]:
    """Source ``_sv_full_inla_laplace_quadrature_samples`` closure."""
    center = _sv_transformed_vector_from_params(level, phi, eta)
    objective = lambda vector: _sv_transformed_negative_log_posterior(observed_log_var, vector)
    hessian = _finite_difference_hessian(
        objective, center, np.asarray([0.03, 0.08, 0.06], dtype=np.float64)
    )
    covariance = _inverse_hessian_covariance(hessian)
    if covariance is None:
        samples, meta = _sv_inla_laplace_sigma_samples(
            observed_log_var, level=level, phi=phi, eta=eta, rho=rho
        )
        weights = (
            np.full(len(samples), 1.0 / max(len(samples), 1), dtype=np.float64) if samples else None
        )
        return (
            samples,
            {**meta, "approximation": "full_inla_laplace_quadrature_fallback_to_sigma_points"},
            weights,
        )
    dim = int(center.size)
    order = int(np.clip(int(hyperparameter_order), 3, 5))
    latent_order = int(np.clip(int(latent_order), 1, 3))
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(order)
    gh_nodes = np.asarray(gh_nodes, dtype=np.float64)
    gh_weights = np.asarray(gh_weights, dtype=np.float64) / math.sqrt(math.pi)
    try:
        chol = np.linalg.cholesky(covariance + np.eye(dim, dtype=np.float64) * 1e-10)
    except np.linalg.LinAlgError:
        samples, meta = _sv_inla_laplace_sigma_samples(
            observed_log_var, level=level, phi=phi, eta=eta, rho=rho
        )
        weights = (
            np.full(len(samples), 1.0 / max(len(samples), 1), dtype=np.float64) if samples else None
        )
        return (
            samples,
            {**meta, "approximation": "full_inla_laplace_quadrature_fallback_to_sigma_points"},
            weights,
        )
    center_log_post = -float(objective(center))
    if not np.isfinite(center_log_post):
        center_log_post = -math.inf
    latent_nodes = np.asarray([0.0], dtype=np.float64)
    latent_weights = np.asarray([1.0], dtype=np.float64)
    if latent_order >= 3:
        latent_nodes = np.asarray([-math.sqrt(3.0), 0.0, math.sqrt(3.0)], dtype=np.float64)
        latent_weights = np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
    raw_samples: list[tuple[float, float, float, float, float]] = []
    raw_log_weights: list[float] = []
    for i in range(order):
        for j in range(order):
            for k in range(order):
                z = math.sqrt(2.0) * np.asarray(
                    [gh_nodes[i], gh_nodes[j], gh_nodes[k]], dtype=np.float64
                )
                vector = center + chol @ z
                natural = _sv_params_from_transformed_vector(vector)
                if natural is None:
                    continue
                node_log_post = -float(objective(vector))
                if not np.isfinite(node_log_post):
                    continue
                log_weight = (
                    math.log(
                        float(
                            max(
                                gh_weights[i] * gh_weights[j] * gh_weights[k],
                                np.finfo(np.float64).tiny,
                            )
                        )
                    )
                    + node_log_post
                    - center_log_post
                    + 0.5 * float(np.dot(z, z))
                )
                if not np.isfinite(log_weight):
                    continue
                node_level, node_phi, node_eta = natural
                _, terminal_mean, terminal_var = _sv_kalman_filter(
                    observed_log_var, node_level, node_phi, node_eta, return_path=False
                )
                if terminal_mean.size == 0 or terminal_var.size == 0:
                    continue
                terminal_sd = math.sqrt(max(float(terminal_var[-1]), 0.0))
                for latent_node, latent_weight in zip(latent_nodes, latent_weights):
                    sample = (
                        float(np.clip(node_level, -18.0, 18.0)),
                        float(np.clip(node_phi, 0.0, 0.995)),
                        float(np.clip(node_eta, 0.02, 2.50)),
                        float(
                            np.clip(
                                terminal_mean[-1] + terminal_sd * float(latent_node), -18.0, 18.0
                            )
                        ),
                        float(np.clip(rho, -0.95, 0.95)),
                    )
                    if all(np.isfinite(value) for value in sample):
                        raw_samples.append(sample)
                        raw_log_weights.append(
                            log_weight
                            + math.log(float(max(latent_weight, np.finfo(np.float64).tiny)))
                        )
    if not raw_samples:
        samples, meta = _sv_inla_laplace_sigma_samples(
            observed_log_var, level=level, phi=phi, eta=eta, rho=rho
        )
        weights = (
            np.full(len(samples), 1.0 / max(len(samples), 1), dtype=np.float64) if samples else None
        )
        return (
            samples,
            {**meta, "approximation": "full_inla_laplace_quadrature_fallback_to_sigma_points"},
            weights,
        )
    max_log_weight = float(np.max(raw_log_weights))
    weights = _normalize_positive_weights(
        np.exp(np.asarray(raw_log_weights, dtype=np.float64) - max_log_weight)
    )
    if weights is None:
        weights = np.full(len(raw_samples), 1.0 / len(raw_samples), dtype=np.float64)
    return (
        raw_samples,
        {
            "approximation": "full_integrated_nested_laplace_quadrature",
            "latent_gaussian_model": "bias_corrected_log_squared_return_ar1_log_volatility",
            "hyperparameter_dimension": dim,
            "hyperparameter_quadrature": "tensor_gauss_hermite_with_laplace_posterior_ratio_correction",
            "hyperparameter_quadrature_order": order,
            "latent_terminal_quadrature": "gaussian_terminal_marginal_three_point"
            if latent_order >= 3
            else "terminal_mean_only",
            "latent_terminal_quadrature_order": latent_order,
            "quadrature_node_count": int(order**dim),
            "posterior_sample_count": len(raw_samples),
            "effective_quadrature_sample_count": float(1.0 / np.sum(weights * weights)),
            "hessian_available": True,
            "covariance_available": True,
        },
        weights,
    )


def _bdes_origin_initialized_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """Source calendar-grid EWMA, initialized at the first observation."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    level = float(values[0])
    out[0] = level
    for index in range(1, values.size):
        level = alpha * float(values[index]) + (1.0 - alpha) * level
        out[index] = level
    return out


def _bdes_multiscale_half_lives(history_length: int, k_star: int, scale_grid: str) -> np.ndarray:
    n = int(history_length)
    if scale_grid == BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC:
        requested = [5.0, 21.0, 63.0, 126.0, 252.0]
        next_scale = 504.0
        while next_scale <= float(n):
            requested.append(next_scale)
            next_scale *= 2.0
        half_lives = np.asarray([scale for scale in requested if scale <= float(n)], dtype=float)
        if half_lives.size == 0:
            raise ValueError(
                "calendar-geometric multiscale fit requires at least five observations"
            )
        return half_lives
    half_lives = np.array([5.0, 21.0, 63.0, 252.0, 756.0, 1512.0], dtype=float)[: int(k_star)]
    return np.minimum(half_lives, max(float(n) / 2.0, 5.0))


def _bdes_multiscale_components(
    h_path: np.ndarray,
    k_star: int,
    scale_grid: str = "fixed_four_scale",
) -> dict[str, Any]:
    """Source ``_bdes_multiscale_components`` with calendar initialization."""
    h_path = np.asarray(h_path, dtype=float)
    finite_h = h_path[np.isfinite(h_path)]
    fill_value = float(np.nanmedian(finite_h)) if finite_h.size else 0.0
    h_path = np.nan_to_num(h_path, nan=fill_value, posinf=fill_value, neginf=fill_value)
    h_path = np.clip(h_path, -18.0, 18.0)
    n = h_path.size
    ell = float(np.nanmean(h_path))
    centered = h_path - ell
    half_lives = _bdes_multiscale_half_lives(n, k_star, scale_grid)
    phis = np.exp(-np.log(2.0) / np.maximum(half_lives, 2.0))
    q = np.empty((n, phis.size), dtype=float)
    for k, phi in enumerate(phis):
        alpha = 1.0 - phi
        if scale_grid == BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC:
            q[:, k] = _bdes_origin_initialized_ewma(centered, alpha)
        else:
            finite = centered[np.isfinite(centered)]
            level = float(np.nanmedian(finite)) if finite.size else 0.0
            for index, value in enumerate(centered):
                if np.isfinite(value):
                    level = alpha * float(value) + (1.0 - alpha) * level
                q[index, k] = level
        q[:, k] -= float(np.nanmean(q[:, k]))
    ridge = 0.05 * max(float(np.nanvar(centered)), 1e-8) * np.eye(phis.size)
    try:
        lhs = q.T @ q + ridge
        rhs = q.T @ centered
        if not (np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs))):
            raise np.linalg.LinAlgError("non-finite multiscale normal equation")
        b = np.linalg.solve(lhs, rhs)
        if not np.all(np.isfinite(b)) or float(np.max(np.abs(b))) > 1.0e6:
            raise np.linalg.LinAlgError("non-finite multiscale coefficients")
    except np.linalg.LinAlgError:
        b = np.zeros(phis.size, dtype=float)
    with np.errstate(all="ignore"):
        component = q @ b
    if not np.all(np.isfinite(component)):
        b = np.zeros(phis.size, dtype=float)
        component = np.zeros_like(centered, dtype=float)
    resid = centered - component
    signal = np.abs(b) * np.nanstd(q, axis=0)
    if np.sum(signal) > 0.0:
        shrink = signal / (signal + np.nanmedian(signal[signal > 0.0]) + 1e-8)
        b = b * shrink
        with np.errstate(all="ignore"):
            component = q @ b
        if not np.all(np.isfinite(component)):
            b = np.zeros(phis.size, dtype=float)
            component = np.zeros_like(centered, dtype=float)
        resid = centered - component
    component_var = max(float(np.nanvar(component)), 0.0)
    resid_var = max(float(np.nanvar(resid)), 1e-8)
    reliability_weight = float(component_var / max(component_var + resid_var, 1e-8))
    post_signal = np.abs(b) * np.nanstd(q, axis=0)
    dominant_half_life = (
        float(np.sum(post_signal * half_lives) / np.sum(post_signal))
        if np.sum(post_signal) > 0.0
        else float(np.median(half_lives))
    )
    inverse_mse_weight = reliability_weight
    if n > 8:
        lagged, actual = centered[:-1], centered[1:]
        denom = float(np.dot(lagged, lagged))
        ar_phi = (
            float(np.clip(float(np.dot(lagged, actual)) / denom, -0.999, 0.999))
            if denom > 1e-12
            else 0.0
        )
        ar_pred = ar_phi * lagged
        with np.errstate(all="ignore"):
            multiscale_pred = (q[:-1, :] * phis.reshape(1, -1)) @ b
        finite = np.isfinite(actual) & np.isfinite(ar_pred) & np.isfinite(multiscale_pred)
        if np.count_nonzero(finite) >= 8:
            ar_mse = max(float(np.mean((actual[finite] - ar_pred[finite]) ** 2)), 1e-8)
            multiscale_mse = max(
                float(np.mean((actual[finite] - multiscale_pred[finite]) ** 2)), 1e-8
            )
            inverse_mse_weight = float(ar_mse / (ar_mse + multiscale_mse))
    q_var = np.empty(phis.size, dtype=float)
    q_innovations = np.empty((max(n - 1, 0), phis.size), dtype=float)
    for k, phi in enumerate(phis):
        innovations = q[1:, k] - phi * q[:-1, k]
        q_innovations[:, k] = innovations
        q_var[k] = max(float(np.nanvar(innovations)), 1e-8)
    residual = resid - float(np.nanmean(resid))
    residual_lag, residual_next = residual[:-1], residual[1:]
    residual_denom = float(np.dot(residual_lag, residual_lag))
    residual_phi = (
        float(
            np.clip(
                float(np.dot(residual_lag, residual_next)) / residual_denom, -0.999999, 0.999999
            )
        )
        if residual_denom > np.finfo(np.float64).tiny
        else 0.0
    )
    residual_innovations = residual_next - residual_phi * residual_lag
    residual_innovation_sd = max(
        float(np.nanstd(residual_innovations, ddof=1)), np.finfo(np.float64).tiny
    )
    q_innovation_sd = np.maximum(
        np.nanstd(q_innovations, axis=0, ddof=1), np.finfo(np.float64).tiny
    )
    standardized_q = q_innovations / q_innovation_sd.reshape(1, -1)
    common_innovation = np.nanmean(standardized_q, axis=1)
    common_innovation -= float(np.nanmean(common_innovation))
    common_sd = float(np.nanstd(common_innovation, ddof=1))
    if common_sd > np.finfo(np.float64).tiny:
        common_innovation /= common_sd
    else:
        common_innovation = np.zeros_like(common_innovation)
    standardized_residual = residual_innovations / residual_innovation_sd
    common_denom = float(np.dot(common_innovation, common_innovation))
    residual_common_loading = (
        float(
            np.clip(
                float(np.dot(common_innovation, standardized_residual)) / common_denom, -1.0, 1.0
            )
        )
        if common_denom > np.finfo(np.float64).tiny
        else 0.0
    )
    h_q005, h_q25, h_q75, h_q995 = np.quantile(h_path, [0.005, 0.25, 0.75, 0.995])
    h_iqr = max(float(h_q75 - h_q25), 1e-6)
    return {
        "ell": ell,
        "phis": phis,
        "b": b,
        "q_last": q[-1, :].copy(),
        "q_var": q_var,
        "half_lives": half_lives.copy(),
        "scale_count": int(half_lives.size),
        "hbar": float(np.nanmean(h_path)),
        "h_low": float(h_q005 - h_iqr),
        "h_high": float(h_q995 + h_iqr),
        "component_low": float(np.quantile(component, 0.01)),
        "component_high": float(np.quantile(component, 0.99)),
        "resid_var": resid_var,
        "residual_last": float(residual[-1]),
        "residual_phi": residual_phi,
        "residual_innovation_sd": residual_innovation_sd,
        "residual_common_loading": residual_common_loading,
        "component_var": component_var,
        "reliability_weight": reliability_weight,
        "inverse_mse_weight": inverse_mse_weight,
        "dominant_half_life_days": dominant_half_life,
        "max_half_life_days": float(np.max(half_lives)),
    }


def fit_bdes_full_inla(
    log_returns: np.ndarray, candidate: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Fit the exact historical full-INLA BDES candidate for one series."""
    settings = _validate_candidate(candidate)
    x = np.asarray(log_returns, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < FULL_MCMC_SV_MIN_OBS:
        raise ValueError("inla_requires_five_observations")
    posterior_mean, residuals, mean_meta = (
        _evidence_estimated_sharpe_dlm_historical_cagr_anchor_mean(x)
    )
    base_override = mean_meta.get("_base_fit_override") if isinstance(mean_meta, dict) else None
    base_fit = (
        copy.deepcopy(base_override)
        if isinstance(base_override, dict)
        else _base_sbb_fit_from_mean_fit(x, posterior_mean, residuals, mean_meta)
    )
    if base_fit is None:
        raise ValueError("inla_mean_fit_failed")
    residual_center = float(base_fit["sample_mu"])
    observed_log_var, eps_x = _sv_observed_log_variance(x, residual_center, winsorize=True)
    if observed_log_var.size < FULL_MCMC_SV_MIN_OBS or eps_x.size < FULL_MCMC_SV_MIN_OBS:
        raise ValueError("inla_invalid_sv_observations")
    level, phi, eta = _fit_sv_map_state_space_params(observed_log_var, start_count=1, maxiter=100)
    filtered_loglik, filtered_path, filtered_var = _sv_kalman_filter(
        observed_log_var, level, phi, eta, return_path=True
    )
    if not np.isfinite(filtered_loglik) or filtered_path.size < FULL_MCMC_SV_MIN_OBS:
        raise ValueError("inla_sv_fit_failed")
    smoother_loglik, smoother_path, smoother_var = _sv_kalman_rts_smoother_mean(
        observed_log_var, level, phi, eta
    )
    if np.isfinite(smoother_loglik) and smoother_path.size >= FULL_MCMC_SV_MIN_OBS:
        state_path, state_var_path, state_loglik = smoother_path, smoother_var, smoother_loglik
    else:
        state_path, state_var_path, state_loglik = filtered_path, filtered_var, filtered_loglik
    z_pool = _standardized_empirical_innovation_pool(eps_x, clip=None, method="mean_std")
    if z_pool is None or z_pool.size < FULL_MCMC_SV_MIN_OBS:
        raise ValueError("inla_invalid_innovation_pool")
    state_innov = (
        state_path[1:] - float(level) - float(phi) * (state_path[:-1] - float(level))
    ) / max(float(eta), 1e-8)
    corr = (
        _finite_correlation(z_pool[: state_innov.size], state_innov)
        if state_innov.size >= 4
        else 0.0
    )
    rho = float(
        corr
        if np.isfinite(corr) and abs(corr) < 1.0
        else np.clip(corr if np.isfinite(corr) else 0.0, -0.95, 0.95)
    )
    posterior_samples, approximation_meta, posterior_sample_weights = (
        _sv_full_inla_laplace_quadrature_samples(
            observed_log_var,
            level=float(level),
            phi=float(phi),
            eta=float(eta),
            rho=float(rho),
            hyperparameter_order=3,
            latent_order=3,
        )
    )
    if not posterior_samples:
        raise ValueError("inla_posterior_empty")
    bdes = _bdes_multiscale_components(state_path, 4, BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC)
    sigma_daily = float(np.std(x, ddof=0)) if x.size > 1 else 0.0
    fit = {
        **settings,
        "mu": float(base_fit.get("posterior_mean", posterior_mean)),
        "residuals": (x - residual_center).astype(np.float64),
        "standardized_residuals": np.asarray(
            base_fit.get("standardized_residuals"), dtype=np.float64
        ),
        "base_fit": base_fit,
        "innovation_pool": z_pool.astype(np.float64),
        "innovation_distribution_params": None,
        "innovation_standardization": "mean_std",
        "innovation_resampling": "stationary_bootstrap",
        "innovation_tail_splice": "none",
        "innovation_conditioning": "none",
        "innovation_pool_source": "sample_standardized",
        "state_innovation_distribution": "gaussian",
        "state_innovation_source": "mode_centered_common_innovation_full_inla_quadrature_state_path",
        "state_innovation_standardization": "mean_std",
        "state_innovation_resampling": "iid",
        "state_innovation_coupling": "correlation_mixture",
        "sv_sigma_scale_method": "none",
        "sv_sigma_scale": 1.0,
        "latent_vol_persistence": "ar1",
        "mean_state_scaling": "conditional_sv_sharpe",
        "vol_path_model": "bdes_multiscale_log_vol",
        "vol_path_blend": "inla_mode_centered_multiscale",
        "bdes_multiscale_shock_coupling": "shared_state_estimated_residual",
        "bdes_multiscale_vol": bdes,
        "observed_log_var_count": int(observed_log_var.size),
        "posterior_samples": posterior_samples,
        "posterior_sample_weights": posterior_sample_weights.astype(np.float64)
        if posterior_sample_weights is not None
        else None,
        "posterior_center": (
            float(level),
            float(phi),
            float(eta),
            float(state_path[-1]),
            float(rho),
        ),
        "leverage": True,
        "leverage_alignment": "lagged_return",
        "leverage_correlation_method": "pearson",
        "leverage_correlation_scope": "deterministic_selected_state_path",
        "unclipped_sv_measurement": False,
        "sv_measurement_bias": "log_chi_square_theoretical",
        "sv_measurement_bias_value": float(SV_LOG_CHI_SQUARE_MEAN),
        "unclipped_empirical_innovations": True,
        "clip_simulated_returns": True,
        "state_loglikelihood": float(state_loglik),
        "state_path_variance_last": float(state_var_path[-1]) if state_var_path.size else None,
        "approximation_meta": approximation_meta,
        "mean_meta": copy.deepcopy(base_fit.get("meta", mean_meta)),
        "return_pool_size": int(x.size),
        "historical_daily_log_volatility": sigma_daily,
        "historical_volatility_annualized": sigma_daily * math.sqrt(252.0),
        "history_years": float(x.size / 252.0),
        "n_obs": int(x.size),
        "meta": {
            "method": "bdes_non_mcmc_full_inla_laplace_quadrature",
            "source_engine_path": SOURCE_ENGINE_PATH,
            "source_engine_sha256": SOURCE_ENGINE_SHA256,
            "source_candidate_digest": source_candidate_digest(),
            "state_sampler": "rts_full_inla_laplace_quadrature_smoother_mean",
            "parameter_sampler": "deterministic_full_inla_laplace_quadrature",
            "uses_mcmc": False,
            "bdes_multiscale_scale_grid": BDES_MULTISCALE_GRID_CALENDAR_GEOMETRIC,
            "bdes_multiscale_half_lives_days": [
                float(value) for value in np.asarray(bdes["half_lives"], dtype=float)
            ],
            "bdes_multiscale_scale_count": int(bdes["scale_count"]),
            "observation_equation": "bias_corrected_log_squared_return_gaussian_state_space",
            "mean_model": settings["mean_model"],
            "mean_method": str((base_fit.get("meta") or {}).get("method", settings["mean_model"])),
            "tail_method": "raw_empirical_untruncated",
            "innovation_standardization": "mean_std",
            "innovation_resampling": "stationary_bootstrap",
            "innovation_pool_source": "sample_standardized",
            "state_inference": "full_inla_laplace_quadrature",
            "path_generator": settings["path_generator"],
            "leverage": True,
            "leverage_alignment": "lagged_return",
            "leverage_correlation_method": "pearson",
            "leverage_correlation_scope": "deterministic_selected_state_path",
            "leverage_correlation": float(rho),
            "approximation_meta": approximation_meta,
        },
    }
    return fit


def simulate_bdes_full_inla(
    fit: Mapping[str, Any], simulations: int, horizon: int, seed: int
) -> np.ndarray:
    """Simulate source-order daily log-return paths from a fitted INLA model."""
    n_paths, total_days = int(simulations), int(horizon)
    if n_paths <= 0 or total_days <= 0:
        return np.empty((max(n_paths, 0), max(total_days, 0)), dtype=np.float64)
    samples = list(fit.get("posterior_samples") or [])
    if not samples:
        raise ValueError("inla_missing_posterior_samples")
    z_pool = np.asarray(fit.get("innovation_pool"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size < FULL_MCMC_SV_MIN_OBS:
        raise ValueError("inla_missing_innovation_pool")
    rng = np.random.default_rng(int(seed))
    block_length = max(
        politis_white_block_length(z_pool), politis_white_block_length(z_pool * z_pool)
    )
    z_indices = stationary_bootstrap_indices(z_pool.size, block_length, total_days, n_paths, rng)
    z_draws = z_pool[z_indices]
    weights = _normalize_positive_weights(fit.get("posterior_sample_weights", []))
    sample_idx = (
        rng.choice(len(samples), size=n_paths, replace=True, p=weights)
        if weights is not None and weights.size == len(samples)
        else rng.integers(0, len(samples), size=n_paths)
    )
    levels = np.asarray([samples[int(idx)][0] for idx in sample_idx], dtype=np.float64)
    phis = np.asarray([samples[int(idx)][1] for idx in sample_idx], dtype=np.float64)
    etas = np.asarray([samples[int(idx)][2] for idx in sample_idx], dtype=np.float64)
    log_var = np.asarray([samples[int(idx)][3] for idx in sample_idx], dtype=np.float64)
    rhos = np.asarray([samples[int(idx)][4] for idx in sample_idx], dtype=np.float64)
    rhos = np.where(np.isfinite(rhos), rhos, 0.0)
    rhos = np.where(np.abs(rhos) < 1.0, rhos, np.sign(rhos) * (1.0 - np.finfo(np.float64).eps))
    base_fit = fit.get("base_fit", {}) or {}
    path_mu = dlm_mu_draw_paths(base_fit, total_days, n_paths, rng)
    if path_mu is None:
        raise ValueError("inla_missing_dlm_mean_paths")
    q = fit.get("bdes_multiscale_vol")
    if not isinstance(q, Mapping):
        raise TypeError("inla_missing_multiscale_state")
    prev_z = rng.choice(z_pool, size=n_paths, replace=True)
    base_state_shocks = rng.normal(0.0, 1.0, size=(n_paths, total_days)).astype(
        np.float64, copy=False
    )
    residual_shocks = rng.normal(0.0, 1.0, size=(n_paths, total_days)).astype(
        np.float64, copy=False
    )
    out = simulate_full_mcmc_sv_bdes_log_paths_serial_numba(
        z_draws.astype(np.float64),
        path_mu.astype(np.float64),
        levels,
        phis,
        etas,
        log_var,
        rhos,
        prev_z.astype(np.float64),
        base_state_shocks,
        residual_shocks,
        np.asarray(q["q_last"], dtype=np.float64),
        np.asarray(q["phis"], dtype=np.float64),
        np.asarray(q["b"], dtype=np.float64),
        np.sqrt(np.maximum(np.asarray(q["q_var"], dtype=np.float64), 1e-10)),
        float(q["ell"]),
        float(q["residual_last"]),
        float(q["residual_phi"]),
        float(q["residual_innovation_sd"]),
        float(np.clip(q["residual_common_loading"], -1.0, 1.0)),
        float(q["h_low"]),
        float(q["h_high"]),
        float(fit["posterior_center"][0]),
        float(fit["posterior_center"][1]),
        float(fit["posterior_center"][2]),
        float(fit["posterior_center"][3]),
        float(fit["posterior_center"][4]),
        1.0,
        max(float(base_fit.get("sigma", 0.0)), np.finfo(np.float64).tiny),
        True,
        False,
        True,
        1,
        False,
        True,
    )
    if out.shape != (n_paths, total_days) or not np.all(np.isfinite(out)):
        raise ValueError("inla_nonfinite_paths")
    return np.asarray(out, dtype=np.float64)


__all__ = [
    "INLA_CANDIDATE",
    "INLA_MODEL_ID",
    "SOURCE_ENGINE_PATH",
    "SOURCE_ENGINE_SHA256",
    "SOURCE_RESEARCH_GATE_PATH",
    "SOURCE_RESEARCH_GATE_SHA256",
    "fit_bdes_full_inla",
    "historical_inla_candidate",
    "simulate_bdes_full_inla",
    "source_candidate_digest",
]
