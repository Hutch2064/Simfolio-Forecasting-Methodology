"""Source-backed Bayesian stationary-bootstrap volatility overlays.

This module is a narrow executable extraction of the pinned research gate's
Bayesian SBB volatility closure.  The function bodies below are retained from
``forecast_oos_research_gate.py`` at source revision
``773bc1c325559e6bf57a567f1d8bf473a3427fbc``.  The destination adapter adds
only the package boundary, exact candidate dictionaries, dated factor loading,
and fail-closed model construction.  No live provider, source checkout,
operational cache, or deployment configuration is imported at runtime.

The research code is user-authorized and had no destination license.  This
attribution records provenance without inventing a license.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ...runner import ForecastContext, TrainingData
from ...seeds import deterministic_seed, forecast_oos_candidate_seed
from ..numerical.base_models import (
    _fit_arch_volatility,
    _fit_ewma_volatility,
    _politis_white_block_length,
    _stationary_bootstrap_indices,
)
from ..numerical.factors import load_packaged_factor_frame
from ..numerical.mcmc_sv import (
    _dlm_mu_draw_paths,
    _fit_empirical_bayes_sharpe_sbb,
    _fit_hierarchical_empirical_bayes_sharpe_sbb,
    _fit_historical_realized_sharpe_sbb,
    _fit_merton_positive_hac_drift_uncertainty_sbb,
    _fit_positive_sample_mean_sbb,
    _fit_prequential_crps_shrinkage_sbb,
    _fit_zero_sharpe_sbb,
    _hac_mean_standard_error,
    _posterior_decay_mu_draw_paths,
)
from .sv_reference import _fit_sv_ar1

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
        "source_revision": MappingProxyType(
            {
                "revision": "773bc1c325559e6bf57a567f1d8bf473a3427fbc",
            }
        ),
    }
)

SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, public_model_id, simulations)"
)

# These are the exact raw descriptors recovered from the retained wrapper/catalog
# binding.  No model-name suffix or later current-catalog default is inferred.
_SOURCE_CANDIDATE_SPECS: dict[str, dict[str, Any]] = {
    "bayesian_sbb_overlay_sv_ar1_logvol_bias_corrected": {
        "id": "bayesian_sbb_overlay_sv_ar1_logvol_bias_corrected",
        "overlay_model": "sv_ar1_logvol_bias_corrected",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_ml_vol_overlay_rf_harx_ff6": {
        "id": "bayesian_sbb_ml_vol_overlay_rf_harx_ff6",
        "type": "bayesian_sbb_ml_vol_overlay",
    },
    "bayesian_sbb_overlay_harch_1_5_22": {
        "id": "bayesian_sbb_overlay_harch_1_5_22",
        "overlay_model": "harch_1_5_22",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_ewma_absolute": {
        "id": "bayesian_sbb_overlay_ewma_absolute",
        "overlay_model": "ewma_absolute",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_gjr_garch_1_1": {
        "id": "bayesian_sbb_overlay_gjr_garch_1_1",
        "overlay_model": "gjr_garch_1_1",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_figarch_1_d_1": {
        "id": "bayesian_sbb_overlay_figarch_1_d_1",
        "overlay_model": "figarch_1_d_1",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_ewma_absolute_historical_realized_sharpe": {
        "id": "bayesian_sbb_overlay_ewma_absolute_historical_realized_sharpe",
        "mean_model": "historical_realized_sharpe",
        "overlay_model": "ewma_absolute",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_ewma_absolute_merton_positive_sample_mean": {
        "id": "bayesian_sbb_overlay_ewma_absolute_merton_positive_sample_mean",
        "mean_model": "positive_sample_mean",
        "overlay_model": "ewma_absolute",
        "path_generator": "stationary_bootstrap_standardized_residuals",
        "tail_method": "filtered_empirical_tail",
        "type": "bayesian_sbb_vol_overlay",
        "vol_model": "absolute_return_ewma_volatility_overlay",
    },
    "bayesian_sbb_overlay_garch_1_1_merton_positive_sample_mean": {
        "id": "bayesian_sbb_overlay_garch_1_1_merton_positive_sample_mean",
        "mean_model": "positive_sample_mean",
        "overlay_model": "garch_1_1",
        "path_generator": "stationary_bootstrap_standardized_residuals",
        "tail_method": "filtered_empirical_tail",
        "type": "bayesian_sbb_vol_overlay",
        "validation_status": "previous_live_garch_incumbent_promoted_from_strict_rank10_baseline",
        "vol_model": "garch_1_1_volatility_overlay",
    },
    "bayesian_sbb_overlay_gjr_garch_1_1_merton_positive_sample_mean": {
        "id": "bayesian_sbb_overlay_gjr_garch_1_1_merton_positive_sample_mean",
        "mean_model": "positive_sample_mean",
        "overlay_model": "gjr_garch_1_1",
        "path_generator": "stationary_bootstrap_standardized_residuals",
        "tail_method": "filtered_empirical_tail",
        "type": "bayesian_sbb_vol_overlay",
        "validation_status": "focused_80_portfolio_validation_candidate",
        "vol_model": "gjr_garch_1_1_volatility_overlay",
    },
    "bayesian_sbb_overlay_sv_ar1_logvol_bias_corrected_hac_drift_uncertainty_harx_ff6_vol_anchor": {
        "id": "bayesian_sbb_overlay_sv_ar1_logvol_bias_corrected_hac_drift_uncertainty_harx_ff6_vol_anchor",
        "mean_model": "merton_positive_hac_drift_uncertainty",
        "overlay_model": "sv_ar1_logvol_bias_corrected",
        "path_generator": "stationary_bootstrap_standardized_residuals",
        "tail_method": "filtered_empirical_tail",
        "type": "bayesian_sbb_vol_overlay",
        "validation_status": "clean_rank33_sv_overlay_with_hac_drift_uncertainty_and_harx_ff6_current_vol_anchor_candidate",
        "vol_model": "sv_ar1_logvol_bias_corrected_volatility_overlay_with_harx_ff6_current_vol_anchor",
    },
    "bayesian_sbb_overlay_ewma_absolute_empirical_bayes_positive_sharpe": {
        "id": "bayesian_sbb_overlay_ewma_absolute_empirical_bayes_positive_sharpe",
        "mean_model": "empirical_bayes_hac_positive_sharpe",
        "overlay_model": "ewma_absolute",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_ewma_absolute_empirical_bayes_positive_sharpe_mu_uncertainty": {
        "id": "bayesian_sbb_overlay_ewma_absolute_empirical_bayes_positive_sharpe_mu_uncertainty",
        "mean_model": "empirical_bayes_hac_positive_sharpe",
        "overlay_model": "ewma_absolute",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_garch_1_1_empirical_bayes_sharpe": {
        "id": "bayesian_sbb_overlay_garch_1_1_empirical_bayes_sharpe",
        "mean_model": "empirical_bayes_hac_sharpe",
        "overlay_model": "garch_1_1",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_gjr_garch_1_1_empirical_bayes_sharpe": {
        "id": "bayesian_sbb_overlay_gjr_garch_1_1_empirical_bayes_sharpe",
        "type": "bayesian_sbb_vol_overlay",
        "overlay_model": "gjr_garch_1_1",
        "vol_model": "gjr_garch_1_1_volatility_overlay",
        "mean_model": "empirical_bayes_hac_sharpe",
        "innovation_method": "empirical_standardized_residuals",
        "tail_method": "filtered_empirical_tail",
        "path_generator": "stationary_bootstrap_standardized_residuals",
        "residual_resampling": "stationary_bootstrap",
        "prior_source": "data_driven_sharpe_shrinkage",
    },
    "bayesian_sbb_overlay_egarch_1_1_empirical_bayes_sharpe": {
        "id": "bayesian_sbb_overlay_egarch_1_1_empirical_bayes_sharpe",
        "type": "bayesian_sbb_vol_overlay",
        "overlay_model": "egarch_1_1",
        "vol_model": "egarch_1_1_volatility_overlay",
        "mean_model": "empirical_bayes_hac_sharpe",
        "innovation_method": "empirical_standardized_residuals",
        "tail_method": "filtered_empirical_tail",
        "path_generator": "stationary_bootstrap_standardized_residuals",
        "residual_resampling": "stationary_bootstrap",
        "prior_source": "data_driven_sharpe_shrinkage",
    },
    "bayesian_sbb_overlay_garch_1_1_hierarchical_empirical_bayes_sharpe": {
        "id": "bayesian_sbb_overlay_garch_1_1_hierarchical_empirical_bayes_sharpe",
        "mean_model": "hierarchical_empirical_bayes_sharpe",
        "overlay_model": "garch_1_1",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_ewma_absolute_empirical_bayes_sharpe": {
        "id": "bayesian_sbb_overlay_ewma_absolute_empirical_bayes_sharpe",
        "mean_model": "empirical_bayes_hac_sharpe",
        "overlay_model": "ewma_absolute",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_garch_1_1_prequential_crps_shrinkage": {
        "id": "bayesian_sbb_overlay_garch_1_1_prequential_crps_shrinkage",
        "mean_model": "prequential_crps_shrinkage",
        "overlay_model": "garch_1_1",
        "type": "bayesian_sbb_vol_overlay",
    },
    "bayesian_sbb_overlay_ewma_absolute_zero_sharpe": {
        "id": "bayesian_sbb_overlay_ewma_absolute_zero_sharpe",
        "mean_model": "zero_sharpe",
        "overlay_model": "ewma_absolute",
        "type": "bayesian_sbb_vol_overlay",
    },
}
SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {key: MappingProxyType(value) for key, value in _SOURCE_CANDIDATE_SPECS.items()}
)
BAYESIAN_VOL_MODEL_IDS: tuple[str, ...] = tuple(_SOURCE_CANDIDATE_SPECS)

# Raw descriptors are intentionally identical to the retained source rows.
RAW_CANDIDATE_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {key: MappingProxyType(dict(value)) for key, value in _SOURCE_CANDIDATE_SPECS.items()}
)

RAW_SEED_DESCRIPTORS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        model_id: MappingProxyType(
            {
                "source_model_key": model_id,
                "panel_seed": 20260528,
                "per_model_rng_schedule": None,
                "status": "source-declared panel seed; per-model RNG schedule and exact call identity unresolved",
            }
        )
        for model_id in BAYESIAN_VOL_MODEL_IDS
    }
)


def _resolved_statistical_spec(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve source defaults without adding parameters to a raw descriptor."""
    model_type = str(candidate["type"])
    mean_model = str(candidate.get("mean_model", ""))
    if model_type == "bayesian_sbb_ml_vol_overlay":
        ml_model = str(candidate.get("ml_model", "ridge_harx"))
        mean_dispatch = {
            "fit_function": "_fit_bayesian_sbb_vol_overlay",
            "branch": "source ML wrapper base fit",
            "base_candidate": {"overlay_model": "ewma_absolute"},
            "wrapper_defaults": {
                "prior_sr": 0.75,
                "sr_cap": 0.75,
                "prior_source": "zero",
                "nonnegative_drift": False,
                "posterior_mu_draws": False,
                "posterior_sd_scale": 0.0,
                "sample_mean_blend": 0.0,
                "sample_mu_days": 0,
            },
        }
        minimum_observations = 180
        overlay_dispatch = {
            "fit_function": "_fit_bayesian_sbb_ml_vol_overlay",
            "model": ml_model,
            "factor_model": "ff6" if "ff6" in ml_model else "none",
            "factor_parameter_source": (
                "raw ml_model descriptor"
                if "ml_model" in candidate
                else "source function default because raw descriptor omits ml_model"
            ),
            "random_forest": {
                "n_estimators": 64,
                "max_depth": 5,
                "min_samples_leaf": 25,
                "max_features": 0.75,
                "bootstrap": True,
                "n_jobs": 1,
            },
            "recent_weight_half_life_days": 504.0,
        }
    else:
        mean_branches = {
            "": "default_bayesian_constrained_sbb",
            "empirical_bayes_hac_sharpe": "_fit_empirical_bayes_sharpe_sbb",
            "empirical_bayes_hac_positive_sharpe": "_fit_empirical_bayes_sharpe_sbb",
            "hierarchical_empirical_bayes_sharpe": "_fit_hierarchical_empirical_bayes_sharpe_sbb",
            "prequential_crps_shrinkage": "_fit_prequential_crps_shrinkage_sbb",
            "positive_sample_mean": "_fit_positive_sample_mean_sbb",
            "merton_positive_hac_drift_uncertainty": "_fit_merton_positive_hac_drift_uncertainty_sbb",
            "historical_realized_sharpe": "_fit_historical_realized_sharpe_sbb",
            "zero_sharpe": "_fit_zero_sharpe_sbb",
        }
        if mean_model not in mean_branches:
            raise ValueError(f"unresolved Bayesian volatility mean branch: {mean_model!r}")
        mean_dispatch = {
            "fit_function": mean_branches[mean_model],
            "branch": mean_model or "source else branch",
        }
        if not mean_model:
            mean_dispatch["wrapper_defaults"] = {
                "prior_sr": float(candidate.get("prior_sr", 0.75)),
                "sr_cap": float(candidate.get("sr_cap", 0.75)),
                "prior_source": "zero",
                "nonnegative_drift": False,
                "posterior_mu_draws": False,
                "posterior_sd_scale": 0.0,
                "sample_mean_blend": 0.0,
                "sample_mu_days": 0,
            }
        if mean_model == "empirical_bayes_hac_positive_sharpe":
            mean_dispatch["arguments"] = {
                "nonnegative_sharpe": True,
                "posterior_mu_draws": bool(candidate.get("posterior_mu_draws", False)),
            }
        minimum_observations = 60
        overlay_model = str(candidate.get("overlay_model", ""))
        overlay_dispatch = {"fit_function": "_fit_bayesian_sbb_vol_overlay", "model": overlay_model}
        if overlay_model == "ewma_absolute":
            overlay_dispatch["fit_function"] = "_fit_absolute_ewma_volatility"
            overlay_dispatch["minimum_residual_observations"] = 60
            overlay_dispatch["lambda_grid"] = [0.80, 0.995, 80]
            overlay_dispatch["objective"] = "one_step_log_absolute_return_volatility_mse"
        elif overlay_model in {"garch_1_1", "gjr_garch_1_1", "egarch_1_1"}:
            overlay_dispatch["fit_function"] = "_fit_arch_volatility"
            overlay_dispatch["vol_model"] = {
                "garch_1_1": "garch_1_1_volatility",
                "gjr_garch_1_1": "gjr_tarch_1_1_volatility",
                "egarch_1_1": "egarch_1_1_volatility",
            }[overlay_model]
            overlay_dispatch["innovation_method"] = "gaussian_iid_standardized_innovations"
            overlay_dispatch["max_fit_observations"] = 1260
            overlay_dispatch["optimizer"] = {"maxiter": 80, "ftol": 1e-6}
        elif overlay_model in {"harch_1_5_22", "figarch_1_d_1"}:
            overlay_dispatch["fit_function"] = "_fit_arch_forecast_overlay"
            overlay_dispatch["max_fit_observations"] = 1260
            overlay_dispatch["optimizer"] = {"maxiter": 80, "ftol": 1e-6}
            overlay_dispatch["arch_specification"] = (
                {"vol": "HARCH", "p": [1, 5, 22]}
                if overlay_model == "harch_1_5_22"
                else {"vol": "FIGARCH", "p": 1, "q": 1}
            )
        elif overlay_model == "sv_ar1_logvol_bias_corrected":
            overlay_dispatch["fit_function"] = "_fit_sv_variant"
            overlay_dispatch["variant"] = {
                "bias_correct": True,
                "eta_scale": 0.0,
                "last_blend": 0.0,
                "phi_cap": 0.995,
            }
    resolved = {
        "family": model_type,
        "mean_dispatch": mean_dispatch,
        "minimum_finite_training_observations": minimum_observations,
        "innovation_and_simulation": {
            "residual_scale": 100.0,
            "standardized_residual_clip": [-20.0, 20.0],
            "block_length": "max(_politis_white_block_length(z), _politis_white_block_length(z*z))",
            "block_length_floor": 1,
            "resampling": "stationary_bootstrap",
            "index_kernel": "_stationary_bootstrap_indices",
            "simulated_return_clip": [-1.0, 1.0],
            "overlay_multiplier_clip": [0.01, 100.0],
        },
        "overlay_dispatch": overlay_dispatch,
    }
    if candidate.get("vol_anchor_model"):
        resolved["vol_anchor"] = {
            "fit_function": "_fit_harx_ff6_current_log_variance_anchor",
            "model": str(candidate["vol_anchor_model"]),
            "target": str(candidate.get("vol_anchor_target", "")),
            "blend": str(candidate.get("vol_anchor_blend", "")),
            "factor_model": "ff6",
            "parameter_source": "explicit raw descriptor field vol_anchor_model",
        }
    if model_type == "bayesian_sbb_ml_vol_overlay":
        resolved["ml_descriptor"] = {
            "model": ml_model,
            "factor_model": "ff6" if "ff6" in ml_model else "none",
            "parameter_source": (
                "explicit raw descriptor field ml_model"
                if "ml_model" in candidate
                else "source function default because raw descriptor omits ml_model"
            ),
        }
    return resolved


RESOLVED_STATISTICAL_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        model_id: MappingProxyType(_resolved_statistical_spec(RAW_CANDIDATE_SPECS[model_id]))
        for model_id in BAYESIAN_VOL_MODEL_IDS
    }
)


def _load_packaged_factor_frame(model: str) -> pd.DataFrame:
    """Compatibility wrapper for the shared offline factor loader."""
    return load_packaged_factor_frame(model)


class _SourceEngine:
    """Minimal source-engine facade required by the extracted source functions."""

    _fit_ewma_volatility = staticmethod(_fit_ewma_volatility)
    _fit_arch_volatility = staticmethod(_fit_arch_volatility)
    _deterministic_seed = staticmethod(deterministic_seed)

    @staticmethod
    def _load_french_factor_frame() -> pd.DataFrame:
        return _load_packaged_factor_frame("ff6")

    @staticmethod
    def _load_q5_factor_frame() -> pd.DataFrame:
        return _load_packaged_factor_frame("q5")


SimfolioEngine = _SourceEngine

_SOURCE_FALLBACK_ERRORS = (
    ArithmeticError,
    AttributeError,
    FloatingPointError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _fit_bayesian_constrained_sbb(
    train_values: np.ndarray, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None

    prior_sr = float(max(candidate.get("prior_sr", 0.50), 1e-6))
    sr_cap = float(max(candidate.get("sr_cap", 1.00), 1e-6))
    prior_source = str(candidate.get("prior_source", "zero"))
    prior_mu = 0.0
    prior_meta: dict[str, Any] = {"prior_source": prior_source}
    if prior_source == "technical_combo" and x.size >= 252:
        trailing = []
        for lookback, weight in ((63, 0.25), (126, 0.25), (252, 0.50)):
            if x.size >= lookback:
                window_mu = float(np.mean(x[-lookback:]))
                window_mu = float(
                    np.clip(
                        window_mu,
                        -sr_cap * sigma / math.sqrt(252.0),
                        sr_cap * sigma / math.sqrt(252.0),
                    )
                )
                trailing.append((weight, window_mu))
        if trailing:
            total_weight = float(sum(weight for weight, _ in trailing))
            technical_mu = float(
                sum(weight * value for weight, value in trailing) / max(total_weight, 1e-12)
            )
            prior_mu = 0.25 * technical_mu
            prior_meta["technical_prior_daily_mu"] = technical_mu

    prior_sd = prior_sr * sigma / math.sqrt(252.0)
    likelihood_se = sigma / math.sqrt(float(x.size))
    prior_var = max(prior_sd * prior_sd, 1e-16)
    likelihood_var = max(likelihood_se * likelihood_se, 1e-16)
    posterior_var = 1.0 / ((1.0 / prior_var) + (1.0 / likelihood_var))
    posterior_mean = posterior_var * ((prior_mu / prior_var) + (sample_mu / likelihood_var))
    posterior_sd = math.sqrt(max(posterior_var, 0.0))
    mu_cap = sr_cap * sigma / math.sqrt(252.0)
    sample_blend = float(np.clip(candidate.get("sample_mean_blend", 0.0), 0.0, 1.0))
    posterior_mean = (sample_blend * sample_mu) + ((1.0 - sample_blend) * posterior_mean)
    posterior_mean = float(np.clip(posterior_mean, -mu_cap, mu_cap))
    if bool(candidate.get("nonnegative_drift", False)):
        posterior_mean = float(max(0.0, posterior_mean))
    posterior_sd *= float(np.clip(candidate.get("posterior_sd_scale", 1.0), 0.0, 4.0))

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
        "sample_mu_days": int(max(candidate.get("sample_mu_days", 0), 0)),
        "nonnegative_drift": bool(candidate.get("nonnegative_drift", False)),
        "posterior_mu_draws": bool(candidate.get("posterior_mu_draws", True)),
        "standardized_residuals": np.clip(z, -20.0, 20.0),
        "meta": {
            **prior_meta,
            "method": "normal_normal_constrained_predictive_mean_with_stationary_bootstrap_residuals",
            "prior_sr": prior_sr,
            "sr_cap": sr_cap,
            "sample_mean_blend": sample_blend,
            "sample_mean": sample_mu,
            "sample_mu_days": int(max(candidate.get("sample_mu_days", 0), 0)),
            "posterior_mean": posterior_mean,
            "posterior_sd": posterior_sd,
        },
    }


def _mean_schedule_from_fit(fit: dict[str, Any], total_days: int) -> np.ndarray | None:
    total_days = int(total_days)
    if total_days <= 0:
        return None
    decay_meta = fit.get("posterior_mean_decay_meta")
    if isinstance(decay_meta, dict):
        short_mu = float(decay_meta.get("short_posterior_mean", fit.get("posterior_mean", 0.0)))
        long_mu = float(decay_meta.get("long_posterior_mean", short_mu))
        uncertainty_ratio = float(decay_meta.get("mean_uncertainty_to_process_variance_ratio", 0.0))
        if (
            not np.isfinite(short_mu)
            or not np.isfinite(long_mu)
            or not np.isfinite(uncertainty_ratio)
        ):
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


def _simulate_bayesian_constrained_sbb(
    fit: dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
        block_length = max(
            _politis_white_block_length(z_pool), _politis_white_block_length(z_pool * z_pool)
        )
    z_indices = _stationary_bootstrap_indices(
        n=len(z_pool),
        block_length=int(max(1, block_length)),
        total_days=total_days,
        n_paths=n_paths,
        rng=rng,
    )
    z_draws = z_pool[z_indices]
    mu = float(fit.get("posterior_mean", 0.0))
    mu_schedule = _mean_schedule_from_fit(fit, total_days)
    mu_draw_path = _dlm_mu_draw_paths(fit, total_days, n_paths, rng)
    if mu_draw_path is None:
        mu_draw_path = _posterior_decay_mu_draw_paths(fit, total_days, n_paths, rng)
    if mu_draw_path is not None:
        mu_path = mu_draw_path
    elif bool(fit.get("posterior_mu_draws", True)):
        mu_draws = rng.normal(mu, float(max(fit.get("posterior_sd", 0.0), 0.0)), size=n_paths)
        cap = float(max(fit.get("mu_cap", 0.0), 0.0))
        if cap > 0.0:
            mu_draws = np.clip(mu_draws, -cap, cap)
        if bool(fit.get("nonnegative_drift", False)):
            mu_draws = np.maximum(mu_draws, 0.0)
        mu_path: Any = mu_draws[:, None]
    elif mu_schedule is not None:
        mu_path = mu_schedule[None, :]
    else:
        mu_path = mu
    sample_mu_days = int(max(fit.get("sample_mu_days", 0), 0))
    if sample_mu_days > 0:
        day_index = np.arange(total_days, dtype=np.int64)
        use_sample_mu = day_index < sample_mu_days
        if np.ndim(mu_path) == 0:
            mu_schedule = np.where(use_sample_mu, float(fit["sample_mu"]), float(mu_path))[None, :]
        else:
            mu_schedule = np.where(
                use_sample_mu[None, :],
                float(fit["sample_mu"]),
                np.asarray(mu_path, dtype=np.float64),
            )
        mu_path = mu_schedule
    return np.clip(mu_path + float(fit["sigma"]) * z_draws, -1.0, 1.0)


def _historical_sharpe_stats(values: np.ndarray) -> dict[str, float] | None:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    sample_mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    sample_sr = float(sample_mu / sigma * math.sqrt(252.0))
    se_mu, bandwidth, long_run_var = _hac_mean_standard_error(x)
    if not np.isfinite(se_mu) or se_mu <= 0.0:
        se_mu = sigma / math.sqrt(float(x.size))
        bandwidth = 0
        long_run_var = sigma * sigma
    se_sr = float(se_mu / sigma * math.sqrt(252.0))
    return {
        "sample_mean": sample_mu,
        "sigma": sigma,
        "historical_sharpe": sample_sr,
        "historical_abs_sharpe": float(max(abs(sample_sr), 1e-6)),
        "hac_mean_standard_error": float(se_mu),
        "hac_bandwidth": float(bandwidth),
        "hac_long_run_variance": float(long_run_var),
        "sharpe_standard_error": se_sr,
        "mean_uncertainty_to_process_variance_ratio": float((se_mu / sigma) ** 2),
    }


def _overlay_half_life_days(persistence: float) -> float | None:
    p = abs(float(persistence))
    if not np.isfinite(p) or p <= 0.0 or p >= 0.999999:
        return None
    return float(math.log(0.5) / math.log(p))


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


def _fit_absolute_ewma_volatility(residuals: np.ndarray) -> dict[str, Any] | None:
    eps = np.asarray(residuals, dtype=np.float64)
    eps = eps[np.isfinite(eps)]
    if eps.size < 60:
        return None
    abs_vol_x = np.abs(eps * 100.0) * math.sqrt(math.pi / 2.0)
    abs_vol_x = abs_vol_x[np.isfinite(abs_vol_x)]
    if abs_vol_x.size < 60:
        return None
    level = float(np.mean(abs_vol_x))
    if not np.isfinite(level) or level <= 1e-8:
        return None
    lambdas = np.linspace(0.80, 0.995, 80, dtype=np.float64)
    target = np.log(np.maximum(abs_vol_x[1:], 1e-8))
    states = np.full(lambdas.shape, level, dtype=np.float64)
    losses = np.zeros(lambdas.shape, dtype=np.float64)
    for idx in range(abs_vol_x.size - 1):
        states = lambdas * states + (1.0 - lambdas) * float(abs_vol_x[idx])
        err = float(target[idx]) - np.log(np.maximum(states, 1e-8))
        losses += err * err
    best_idx = (
        int(np.argmin(losses))
        if np.all(np.isfinite(losses))
        else int(np.argmin(np.nan_to_num(losses, nan=np.inf)))
    )
    best_lambda = float(lambdas[best_idx])
    state = level
    sigma_path = np.empty(abs_vol_x.size, dtype=np.float64)
    for idx, value in enumerate(abs_vol_x):
        state = float(best_lambda * state + (1.0 - best_lambda) * value)
        sigma_path[idx] = state
    return {
        "lambda": best_lambda,
        "sigma_x": np.maximum(sigma_path, 1e-8),
        "last_sigma_x": float(max(sigma_path[-1], 1e-8)),
        "fit_status": "complete",
        "objective": "one_step_log_absolute_return_volatility_mse",
    }


def _fit_har_overlay(
    residuals: np.ndarray, mode: str, long_sigma_x: float
) -> dict[str, Any] | None:
    eps = np.asarray(residuals, dtype=np.float64)
    eps = eps[np.isfinite(eps)]
    if eps.size < 252:
        return None
    eps_x = eps * 100.0
    long_sigma_x = float(max(long_sigma_x, 1e-6))
    if mode == "log_variance":
        y = np.log(np.maximum(eps_x * eps_x, 1e-8))
        long_level = math.log(max(long_sigma_x * long_sigma_x, 1e-8))
        vol_proxy_x = np.sqrt(np.maximum(eps_x * eps_x, 1e-8))
    elif mode == "absolute_volatility":
        vol_proxy_x = np.abs(eps_x) * math.sqrt(math.pi / 2.0)
        y = np.log(np.maximum(vol_proxy_x, 1e-8))
        long_level = math.log(long_sigma_x)
    else:
        return None
    y = y[np.isfinite(y)]
    if y.size < 252:
        return None
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    d = y - float(long_level)
    x_rows = []
    y_rows = []
    for idx in range(22, len(d)):
        x_rows.append(
            [float(d[idx - 1]), float(np.mean(d[idx - 5 : idx])), float(np.mean(d[idx - 22 : idx]))]
        )
        y_rows.append(float(d[idx]))
    if len(y_rows) < 80:
        return None
    x_mat = np.asarray(x_rows, dtype=np.float64)
    y_vec = np.asarray(y_rows, dtype=np.float64)
    try:
        coef, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    except _SOURCE_FALLBACK_ERRORS:
        return None
    coef = np.asarray(coef, dtype=np.float64)
    if coef.shape != (3,) or not np.all(np.isfinite(coef)):
        return None
    persistence = float(np.sum(coef))
    if not np.isfinite(persistence) or abs(persistence) >= 0.995:
        scale = 0.995 / max(abs(persistence), 1e-12)
        coef = coef * scale
        persistence = float(np.sum(coef))
    return {
        "mode": mode,
        "coefficients": coef.astype(float).tolist(),
        "history": y.astype(float),
        "long_level": float(long_level),
        "persistence": persistence,
        "sigma_clip_bounds_x": _overlay_vol_clip_bounds(vol_proxy_x, long_sigma_x),
        "fit_status": "complete",
    }


def _fit_arch_forecast_overlay(residuals: np.ndarray, overlay_model: str) -> dict[str, Any] | None:
    eps = np.asarray(residuals, dtype=np.float64)
    eps = eps[np.isfinite(eps)]
    if eps.size < 252:
        return None
    x = eps[-min(len(eps), 1260) :] * 100.0
    if x.size < 252 or float(np.std(x, ddof=1)) <= 1e-8:
        return None
    try:
        from arch import arch_model

        if overlay_model == "harch_1_5_22":
            kwargs = {"vol": "HARCH", "p": [1, 5, 22]}
        elif overlay_model == "figarch_1_d_1":
            kwargs = {"vol": "FIGARCH", "p": 1, "q": 1}
        else:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = arch_model(
                x,
                mean="Zero",
                dist="normal",
                rescale=False,
                **kwargs,
            ).fit(disp="off", options={"maxiter": 80, "ftol": 1e-6})
        sigma = np.asarray(result.conditional_volatility, dtype=np.float64)
        sigma = sigma[np.isfinite(sigma)]
        if sigma.size < 20:
            return None
        params = {
            str(key): float(value) for key, value in result.params.items() if np.isfinite(value)
        }
        return {
            "result": result,
            "params": params,
            "sigma_x": np.maximum(sigma, 1e-8),
            "last_sigma_x": float(max(sigma[-1], 1e-8)),
            "fit_status": "complete"
            if int(getattr(result, "convergence_flag", 0) or 0) == 0
            else "optimizer_warning",
        }
    except _SOURCE_FALLBACK_ERRORS:
        return None


def _fit_bayesian_sbb_vol_overlay(
    train_values: np.ndarray, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 60:
        return None
    base_fit: dict[str, Any] | None = None
    if str(candidate.get("mean_model", "")) == "empirical_bayes_hac_sharpe":
        base_fit = _fit_empirical_bayes_sharpe_sbb(x)
    elif str(candidate.get("mean_model", "")) == "empirical_bayes_hac_positive_sharpe":
        base_fit = _fit_empirical_bayes_sharpe_sbb(
            x,
            nonnegative_sharpe=True,
            posterior_mu_draws=bool(candidate.get("posterior_mu_draws", False)),
        )
    elif str(candidate.get("mean_model", "")) == "hierarchical_empirical_bayes_sharpe":
        base_fit = _fit_hierarchical_empirical_bayes_sharpe_sbb(x)
    elif str(candidate.get("mean_model", "")) == "prequential_crps_shrinkage":
        base_fit = _fit_prequential_crps_shrinkage_sbb(x)
    elif str(candidate.get("mean_model", "")) == "positive_sample_mean":
        base_fit = _fit_positive_sample_mean_sbb(x)
    elif str(candidate.get("mean_model", "")) == "merton_positive_hac_drift_uncertainty":
        base_fit = _fit_merton_positive_hac_drift_uncertainty_sbb(x)
    elif (
        str(candidate.get("mean_model", ""))
        == "evidence_estimated_sharpe_dlm_historical_cagr_anchor"
    ):
        mu, residuals, mean_meta = (
            SimfolioEngine._evidence_estimated_sharpe_dlm_historical_cagr_anchor_mean(x)
        )
        base_fit_override = (
            mean_meta.get("_base_fit_override") if isinstance(mean_meta, dict) else None
        )
        if isinstance(base_fit_override, dict):
            base_fit = dict(base_fit_override)
        else:
            base_fit = SimfolioEngine._base_sbb_fit_from_mean_fit(
                x,
                float(mu),
                np.asarray(residuals, dtype=np.float64),
                dict(mean_meta),
            )
    elif str(candidate.get("mean_model", "")) == "zero_sharpe":
        base_fit = _fit_zero_sharpe_sbb(x)
    elif str(candidate.get("mean_model", "")) == "historical_realized_sharpe":
        base_fit = _fit_historical_realized_sharpe_sbb(x)
    elif str(candidate.get("mean_model", "")) == "historical_realized_sharpe_prior_cap":
        sharpe_stats = _historical_sharpe_stats(x)
        if sharpe_stats is None:
            base_fit = None
        else:
            realized_sr = float(sharpe_stats["historical_sharpe"])
            realized_abs_sr = float(sharpe_stats["historical_abs_sharpe"])
            base_fit = _fit_bayesian_constrained_sbb(
                x,
                {
                    "prior_sr": realized_abs_sr,
                    "sr_cap": realized_abs_sr,
                    "prior_source": "zero",
                    "nonnegative_drift": False,
                    "posterior_mu_draws": False,
                    "posterior_sd_scale": 0.0,
                },
            )
            if base_fit is not None:
                base_fit["meta"]["prior_source"] = "portfolio_historical_realized_sharpe"
                base_fit["meta"]["sample_sharpe_annualized"] = realized_sr
                base_fit["meta"]["historical_realized_abs_sharpe_cap"] = realized_abs_sr
    elif str(candidate.get("mean_model", "")) in {
        "horizon_sharpe_cap_schedule",
        "historical_sharpe_horizon_schedule",
    }:
        sharpe_stats = _historical_sharpe_stats(x)
        schedule_specs = candidate.get("sharpe_schedule") or []
        segment_fits: list[dict[str, Any]] = []
        for raw_segment in schedule_specs:
            if not isinstance(raw_segment, dict):
                continue
            if raw_segment.get("prior_sr_source") == "historical_abs_sharpe":
                if sharpe_stats is None:
                    continue
                prior_sr = float(sharpe_stats["historical_abs_sharpe"])
                sr_cap = float(raw_segment.get("sr_cap", prior_sr))
            else:
                prior_sr = float(raw_segment.get("prior_sr", 0.4375))
                sr_cap = float(raw_segment.get("sr_cap", prior_sr))
            segment_fit = _fit_bayesian_constrained_sbb(
                x,
                {
                    "prior_sr": prior_sr,
                    "sr_cap": sr_cap,
                    "prior_source": "zero",
                    "nonnegative_drift": False,
                    "posterior_mu_draws": False,
                    "posterior_sd_scale": 0.0,
                },
            )
            if segment_fit is None:
                continue
            segment_fits.append(
                {
                    "through_day": raw_segment.get("through_day"),
                    "prior_sr": prior_sr,
                    "sr_cap": sr_cap,
                    "prior_sr_source": str(raw_segment.get("prior_sr_source") or "fixed"),
                    "posterior_mean": float(segment_fit["posterior_mean"]),
                    "mu_cap": float(segment_fit.get("mu_cap", 0.0)),
                }
            )
            if base_fit is None:
                base_fit = segment_fit
        if base_fit is not None and segment_fits:
            base_fit = dict(base_fit)
            base_fit["posterior_mean_schedule_segments"] = segment_fits
            base_fit["posterior_mu_draws"] = False
            base_fit["meta"] = {
                **dict(base_fit.get("meta", {}) or {}),
                "mean_schedule_method": str(candidate.get("mean_model")),
                "historical_sharpe_stats": sharpe_stats,
                "posterior_mean_schedule_segments": segment_fits,
            }
    elif str(candidate.get("mean_model", "")) == "historical_sharpe_uncertainty_decay":
        sharpe_stats = _historical_sharpe_stats(x)
        if sharpe_stats is None:
            base_fit = None
        else:
            historical_abs_sr = float(sharpe_stats["historical_abs_sharpe"])
            long_run_sr = float(max(candidate.get("long_run_sr", 0.4375), 1e-6))
            short_fit = _fit_bayesian_constrained_sbb(
                x,
                {
                    "prior_sr": historical_abs_sr,
                    "sr_cap": historical_abs_sr,
                    "prior_source": "zero",
                    "nonnegative_drift": False,
                    "posterior_mu_draws": False,
                    "posterior_sd_scale": 0.0,
                },
            )
            long_fit = _fit_bayesian_constrained_sbb(
                x,
                {
                    "prior_sr": long_run_sr,
                    "sr_cap": long_run_sr,
                    "prior_source": "zero",
                    "nonnegative_drift": False,
                    "posterior_mu_draws": False,
                    "posterior_sd_scale": 0.0,
                },
            )
            if short_fit is not None and long_fit is not None:
                base_fit = dict(short_fit)
                decay_meta = {
                    **sharpe_stats,
                    "method": "mean_parameter_uncertainty_decay_to_long_run_sharpe",
                    "long_run_sr": long_run_sr,
                    "short_posterior_mean": float(short_fit["posterior_mean"]),
                    "long_posterior_mean": float(long_fit["posterior_mean"]),
                    "short_prior_sr": historical_abs_sr,
                    "short_sr_cap": historical_abs_sr,
                    "long_prior_sr": long_run_sr,
                    "long_sr_cap": long_run_sr,
                }
                base_fit["posterior_mean"] = float(short_fit["posterior_mean"])
                base_fit["posterior_mean_decay_meta"] = decay_meta
                base_fit["posterior_mu_draws"] = False
                base_fit["meta"] = {
                    **dict(base_fit.get("meta", {}) or {}),
                    "mean_schedule_method": "historical_sharpe_uncertainty_decay",
                    "posterior_mean_decay_meta": decay_meta,
                }
    else:
        prior_sr = float(candidate.get("prior_sr", 0.75))
        sr_cap = float(candidate.get("sr_cap", 0.75))
        base_fit = _fit_bayesian_constrained_sbb(
            x,
            {
                "prior_sr": prior_sr,
                "sr_cap": sr_cap,
                "prior_source": "zero",
                "nonnegative_drift": False,
                "posterior_mu_draws": False,
                "posterior_sd_scale": 0.0,
            },
        )
    if base_fit is None:
        return None
    if candidate.get("fixed_block_length") is not None:
        base_fit = dict(base_fit)
        base_fit["fixed_block_length"] = int(max(1, int(candidate["fixed_block_length"])))
        base_fit["meta"] = {
            **dict(base_fit.get("meta", {}) or {}),
            "block_length_method": "fixed_trading_days",
            "fixed_block_length": int(base_fit["fixed_block_length"]),
        }
    residuals = x - float(np.mean(x))
    long_sigma_x = float(max(float(base_fit["sigma"]) * 100.0, 1e-6))
    long_variance_x = float(max(long_sigma_x * long_sigma_x, 1e-8))
    overlay_model = str(candidate.get("overlay_model", "ewma_qmle"))
    meta: dict[str, Any] = {
        "method": "bayesian_constrained_sbb_with_data_driven_short_horizon_volatility_overlay",
        "overlay_model": overlay_model,
        "long_run_sigma_source": "base_bayesian_constrained_sbb_sample_sigma",
        "long_sigma_x": long_sigma_x,
    }
    curve_fit: dict[str, Any]
    if overlay_model == "ewma_qmle":
        vol_fit = SimfolioEngine._fit_ewma_volatility(residuals)
        persistence = float(vol_fit.get("lambda", 0.94))
        curve_fit = {
            "curve_type": "variance_decay",
            "persistence": float(np.clip(persistence, 0.0, 0.999)),
            "last_variance_x": float(
                max(float(vol_fit.get("last_sigma_x", long_sigma_x)) ** 2, 1e-8)
            ),
            "long_variance_x": long_variance_x,
            "sigma_clip_bounds_x": _overlay_vol_clip_bounds(
                np.asarray(vol_fit.get("sigma_x", []), dtype=np.float64), long_sigma_x
            ),
            "fit_status": str(vol_fit.get("fit_status", "complete")),
        }
    elif overlay_model == "ewma_absolute":
        vol_fit = _fit_absolute_ewma_volatility(residuals)
        if vol_fit is None:
            return None
        persistence = float(vol_fit.get("lambda", 0.94))
        curve_fit = {
            "curve_type": "variance_decay",
            "persistence": float(np.clip(persistence, 0.0, 0.999)),
            "last_variance_x": float(
                max(float(vol_fit.get("last_sigma_x", long_sigma_x)) ** 2, 1e-8)
            ),
            "long_variance_x": long_variance_x,
            "sigma_clip_bounds_x": _overlay_vol_clip_bounds(
                np.asarray(vol_fit.get("sigma_x", []), dtype=np.float64), long_sigma_x
            ),
            "fit_status": str(vol_fit.get("fit_status", "complete")),
            "objective": str(vol_fit.get("objective", "")),
        }
    elif overlay_model in {"garch_1_1", "gjr_garch_1_1", "egarch_1_1"}:
        vol_name = {
            "garch_1_1": "garch_1_1_volatility",
            "gjr_garch_1_1": "gjr_tarch_1_1_volatility",
            "egarch_1_1": "egarch_1_1_volatility",
        }[overlay_model]
        vol_fit = SimfolioEngine._fit_arch_volatility(
            residuals, vol_name, "gaussian_iid_standardized_innovations"
        )
        if vol_fit is None:
            return None
        params = vol_fit.get("params", {}) or {}
        sigma_x = np.asarray(vol_fit.get("sigma_x", []), dtype=np.float64)
        if overlay_model == "egarch_1_1":
            persistence = float(np.clip(params.get("beta[1]", 0.94), -0.98, 0.995))
            curve_fit = {
                "curve_type": "log_variance_decay",
                "persistence": persistence,
                "last_log_variance_x": float(
                    math.log(max(float(vol_fit.get("last_sigma_x", long_sigma_x)) ** 2, 1e-8))
                ),
                "long_log_variance_x": float(math.log(long_variance_x)),
                "sigma_clip_bounds_x": _overlay_vol_clip_bounds(sigma_x, long_sigma_x),
                "fit_status": str(vol_fit.get("fit_status", "complete")),
            }
        else:
            alpha = float(np.clip(params.get("alpha[1]", 0.05), 0.0, 0.80))
            beta = float(np.clip(params.get("beta[1]", 0.90), 0.0, 0.999))
            gamma = (
                float(np.clip(params.get("gamma[1]", 0.0), -0.80, 1.20))
                if overlay_model == "gjr_garch_1_1"
                else 0.0
            )
            persistence = float(np.clip(alpha + beta + 0.5 * gamma, 0.0, 0.999))
            curve_fit = {
                "curve_type": "variance_decay",
                "persistence": persistence,
                "last_variance_x": float(
                    max(float(vol_fit.get("last_sigma_x", long_sigma_x)) ** 2, 1e-8)
                ),
                "long_variance_x": long_variance_x,
                "sigma_clip_bounds_x": _overlay_vol_clip_bounds(sigma_x, long_sigma_x),
                "fit_status": str(vol_fit.get("fit_status", "complete")),
                "arch_params": params,
            }
    elif overlay_model in {
        "sv_ar1_logvol",
        "sv_ar1_logvol_bias_corrected",
        "sv_no_ar_logvol_bias_corrected",
    }:
        if overlay_model == "sv_no_ar_logvol_bias_corrected":
            sv_fit = _fit_sv_variant(
                x,
                {"bias_correct": True, "eta_scale": 0.0, "last_blend": 0.0, "phi_cap": 0.0},
            )
        elif overlay_model == "sv_ar1_logvol_bias_corrected":
            sv_fit = _fit_sv_variant(
                x,
                {"bias_correct": True, "eta_scale": 0.0, "last_blend": 0.0, "phi_cap": 0.995},
            )
        else:
            sv_fit = _fit_sv_ar1(x)
        if sv_fit is None:
            return None
        persistence = (
            0.0
            if overlay_model == "sv_no_ar_logvol_bias_corrected"
            else float(np.clip(sv_fit.get("phi", 0.94), 0.0, 0.995))
        )
        curve_fit = {
            "curve_type": "log_variance_decay",
            "persistence": persistence,
            "last_log_variance_x": float(sv_fit.get("last_log_var", math.log(long_variance_x))),
            "long_log_variance_x": float(math.log(long_variance_x)),
            "sigma_clip_bounds_x": _overlay_vol_clip_bounds(
                np.sqrt(np.maximum((residuals * 100.0) ** 2, 1e-8)), long_sigma_x
            ),
            "fit_status": "complete",
        }
        filtered_var = np.asarray(sv_fit.get("filtered_log_var_var", []), dtype=np.float64)
        filtered_var = filtered_var[np.isfinite(filtered_var)]
        if filtered_var.size:
            curve_fit["last_log_variance_error_variance"] = float(max(filtered_var[-1], 1e-10))
        if "sv_measurement_bias_correction" in sv_fit:
            curve_fit["sv_measurement_bias_correction"] = float(
                sv_fit["sv_measurement_bias_correction"]
            )
    elif overlay_model in {"har_log_variance", "har_absolute_volatility"}:
        mode = "log_variance" if overlay_model == "har_log_variance" else "absolute_volatility"
        har_fit = _fit_har_overlay(residuals, mode, long_sigma_x)
        if har_fit is None:
            return None
        curve_fit = {
            "curve_type": "har_recursive",
            **har_fit,
        }
        persistence = float(har_fit.get("persistence", 0.0))
    elif overlay_model in {"harch_1_5_22", "figarch_1_d_1"}:
        arch_fit = _fit_arch_forecast_overlay(residuals, overlay_model)
        if arch_fit is None:
            return None
        params = arch_fit.get("params", {}) or {}
        if overlay_model == "harch_1_5_22":
            persistence = float(
                sum(value for key, value in params.items() if str(key).startswith("alpha["))
            )
        else:
            persistence = float(params.get("beta", 0.0) + params.get("d", 0.0))
        persistence = float(np.clip(persistence, 0.0, 0.999))
        curve_fit = {
            "curve_type": "arch_forecast",
            "arch_result": arch_fit["result"],
            "arch_params": params,
            "persistence": persistence,
            "sigma_clip_bounds_x": _overlay_vol_clip_bounds(
                np.asarray(arch_fit.get("sigma_x", []), dtype=np.float64), long_sigma_x
            ),
            "fit_status": str(arch_fit.get("fit_status", "complete")),
        }
    else:
        return None
    meta["overlay_persistence"] = float(persistence)
    meta["overlay_half_life_days"] = _overlay_half_life_days(float(persistence))
    meta["overlay_fit_status"] = str(curve_fit.get("fit_status", "complete"))
    return {
        "base_fit": base_fit,
        "curve_fit": curve_fit,
        "meta": meta,
    }


def _overlay_vol_multiplier_curve(fit: dict[str, Any], total_days: int) -> np.ndarray:
    total_days = int(total_days)
    if total_days <= 0:
        return np.empty(0, dtype=np.float64)
    base_fit = fit.get("base_fit", {}) or {}
    curve_fit = fit.get("curve_fit", {}) or {}
    long_sigma_x = float(max(float(base_fit.get("sigma", 0.0)) * 100.0, 1e-6))
    steps = np.arange(1, total_days + 1, dtype=np.float64)
    curve_type = str(curve_fit.get("curve_type", "variance_decay"))
    if curve_type == "variance_decay":
        p = float(curve_fit.get("persistence", 0.0))
        long_h = float(max(curve_fit.get("long_variance_x", long_sigma_x * long_sigma_x), 1e-8))
        last_h = float(max(curve_fit.get("last_variance_x", long_h), 1e-8))
        sigma_x = np.sqrt(np.maximum(long_h + np.power(p, steps) * (last_h - long_h), 1e-8))
    elif curve_type == "log_variance_decay":
        p = float(curve_fit.get("persistence", 0.0))
        long_log_h = float(
            curve_fit.get("long_log_variance_x", math.log(max(long_sigma_x * long_sigma_x, 1e-8)))
        )
        last_log_h = float(curve_fit.get("last_log_variance_x", long_log_h))
        log_h = long_log_h + np.power(p, steps) * (last_log_h - long_log_h)
        sigma_x = np.exp(0.5 * np.clip(log_h, -18.0, 18.0))
    elif curve_type == "har_recursive":
        coef = np.asarray(curve_fit.get("coefficients", [0.0, 0.0, 0.0]), dtype=np.float64)
        history = list(np.asarray(curve_fit.get("history", []), dtype=np.float64))
        long_level = float(curve_fit.get("long_level", math.log(max(long_sigma_x, 1e-8))))
        mode = str(curve_fit.get("mode", "absolute_volatility"))
        values = np.empty(total_days, dtype=np.float64)
        if len(history) < 22 or coef.shape != (3,) or not np.all(np.isfinite(coef)):
            values.fill(long_sigma_x)
        else:
            for idx in range(total_days):
                hist = np.asarray(history, dtype=np.float64)
                dev = hist - long_level
                pred_dev = float(
                    coef[0] * dev[-1]
                    + coef[1] * float(np.mean(dev[-5:]))
                    + coef[2] * float(np.mean(dev[-22:]))
                )
                pred_level = float(long_level + pred_dev)
                if mode == "log_variance":
                    values[idx] = math.exp(0.5 * float(np.clip(pred_level, -18.0, 18.0)))
                else:
                    values[idx] = math.exp(float(np.clip(pred_level, -18.0, 18.0)))
                history.append(pred_level)
            sigma_x = values
    elif curve_type == "arch_forecast":
        result = curve_fit.get("arch_result")
        try:
            forecast = result.forecast(horizon=total_days, reindex=False, method="analytic")
            values = np.asarray(forecast.variance.values[-1], dtype=np.float64)
            sigma_x = np.sqrt(np.maximum(values, 1e-8))
        except _SOURCE_FALLBACK_ERRORS:
            sigma_x = np.full(total_days, long_sigma_x, dtype=np.float64)
    else:
        sigma_x = np.full(total_days, long_sigma_x, dtype=np.float64)
    bounds = curve_fit.get("sigma_clip_bounds_x")
    if bounds is not None and len(bounds) == 2:
        floor, ceiling = float(bounds[0]), float(bounds[1])
        if np.isfinite(floor) and np.isfinite(ceiling) and ceiling > floor > 0.0:
            sigma_x = np.clip(sigma_x, floor, ceiling)
    multiplier = np.asarray(sigma_x, dtype=np.float64) / max(long_sigma_x, 1e-8)
    multiplier = multiplier[np.isfinite(multiplier)]
    if multiplier.size != total_days:
        return np.ones(total_days, dtype=np.float64)
    return np.clip(multiplier, 0.01, 100.0)


def _simulate_bayesian_sbb_vol_overlay(
    fit: dict[str, Any],
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
    if center_schedule is not None:
        center: Any = center_schedule[None, :]
    else:
        center = float(base_fit.get("posterior_mean", 0.0))
    paths = center + (base_paths - center) * multipliers[None, :]
    return np.clip(paths, -1.0, 1.0)


def _factor_model_frame_config(
    engine: SimfolioEngine,
    factor_model: str,
    factor_frame: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    model = str(factor_model or "none").lower()
    if model in {"none", "not_applicable", ""}:
        return None
    if model == "ff6":
        frame = (
            factor_frame.copy() if factor_frame is not None else engine._load_french_factor_frame()
        )
        return {
            "name": "ff6",
            "frame": frame,
            "rf_col": "RF",
            "factor_cols": ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"],
        }
    if model == "q5":
        frame = factor_frame.copy() if factor_frame is not None else engine._load_q5_factor_frame()
        return {
            "name": "q5",
            "frame": frame,
            "rf_col": "R_F",
            "factor_cols": ["R_MKT", "R_ME", "R_IA", "R_ROE", "R_EG"],
        }
    return None


def _ml_forecast_feature_panel(
    train: pd.Series,
    *,
    factor_frame: pd.DataFrame | None = None,
    factor_model: str = "none",
) -> dict[str, Any] | None:
    series = pd.Series(train).dropna().astype(float)
    if len(series) < 180:
        return None
    series.index = pd.to_datetime(series.index).normalize()
    r = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 180:
        return None

    features = pd.DataFrame(index=r.index)
    features["ret_1d"] = r
    features["abs_ret_1d_x"] = np.abs(r) * math.sqrt(math.pi / 2.0) * 100.0
    for window in (5, 21, 63, 126, 252):
        min_periods = max(3, min(window, window // 2))
        features[f"ret_mean_{window}d"] = r.rolling(window, min_periods=min_periods).mean()
        features[f"abs_vol_{window}d_x"] = (
            np.abs(r).rolling(window, min_periods=min_periods).mean()
            * math.sqrt(math.pi / 2.0)
            * 100.0
        )
        features[f"rv_{window}d_x"] = r.rolling(window, min_periods=min_periods).std(ddof=0) * 100.0
    for span in (10, 21, 63):
        features[f"abs_vol_ewm_{span}d_x"] = (
            np.abs(r).ewm(span=span, adjust=False, min_periods=max(5, span // 2)).mean()
            * math.sqrt(math.pi / 2.0)
            * 100.0
        )
    wealth = np.exp(r.cumsum())
    drawdown = wealth / wealth.cummax() - 1.0
    features["drawdown"] = drawdown
    features["drawdown_min_63d"] = drawdown.rolling(63, min_periods=21).min()

    factor_prefix = str(factor_model or "none").lower()
    factor_cols: list[str] = []
    rf_col: str | None = None
    if factor_frame is not None and factor_prefix in {"ff6", "q5"}:
        cfg = {
            "ff6": {"rf_col": "RF", "factor_cols": ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"]},
            "q5": {"rf_col": "R_F", "factor_cols": ["R_MKT", "R_ME", "R_IA", "R_ROE", "R_EG"]},
        }[factor_prefix]
        rf_col = str(cfg["rf_col"])
        factor_cols = list(cfg["factor_cols"])
        factors = factor_frame[[rf_col, *factor_cols]].copy()
        factors.index = pd.to_datetime(factors.index).normalize()
        factors = factors.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        joined = factors.reindex(r.index)
        features[f"{factor_prefix}_rf"] = joined[rf_col]
        features[f"{factor_prefix}_rf_change_21d"] = joined[rf_col] - joined[rf_col].shift(21)
        excess = r - joined[rf_col]
        for col in factor_cols:
            safe_col = str(col).lower()
            f = joined[col]
            features[f"{factor_prefix}_{safe_col}_ret_1d"] = f
            features[f"{factor_prefix}_{safe_col}_mean_21d"] = f.rolling(21, min_periods=10).mean()
            features[f"{factor_prefix}_{safe_col}_mean_63d"] = f.rolling(63, min_periods=21).mean()
            features[f"{factor_prefix}_{safe_col}_vol_21d"] = f.rolling(21, min_periods=10).std(
                ddof=0
            )
            cov = excess.rolling(252, min_periods=126).cov(f)
            var = f.rolling(252, min_periods=126).var(ddof=0)
            features[f"{factor_prefix}_beta_{safe_col}"] = cov / var.replace(0.0, np.nan)

    next_return = r.shift(-1)
    next_abs_vol = np.abs(next_return) * math.sqrt(math.pi / 2.0) * 100.0
    target_log_abs_vol = np.log(np.maximum(next_abs_vol, 1e-8))
    target_dates = pd.Series(pd.NaT, index=r.index, dtype="datetime64[ns]")
    if len(r.index) > 1:
        target_dates.iloc[:-1] = r.index[1:]

    feature_names = list(features.columns)
    latest = features[feature_names].iloc[-1].to_numpy(dtype=np.float64)
    if latest.size == 0 or not np.all(np.isfinite(latest)):
        return None
    data = features.copy()
    data["y_log_abs_vol"] = target_log_abs_vol
    data["y_return"] = next_return
    data["target_date"] = target_dates
    data = data.dropna(subset=[*feature_names, "y_log_abs_vol", "y_return", "target_date"])
    if len(data) < 120:
        return None
    X = data[feature_names].to_numpy(dtype=np.float64)
    y_vol = data["y_log_abs_vol"].to_numpy(dtype=np.float64)
    y_return = data["y_return"].to_numpy(dtype=np.float64)
    if not (
        np.all(np.isfinite(X)) and np.all(np.isfinite(y_vol)) and np.all(np.isfinite(y_return))
    ):
        return None
    return {
        "X": X,
        "y_log_abs_vol": y_vol,
        "y_return": y_return,
        "latest_x": latest.astype(np.float64),
        "feature_names": feature_names,
        "feature_dates": pd.DatetimeIndex(data.index),
        "target_dates": pd.DatetimeIndex(data["target_date"]),
        "factor_model": factor_prefix,
        "factor_cols": factor_cols,
        "rf_col": rf_col,
    }


def _recent_exponential_weights(n_obs: int, half_life: float = 504.0) -> np.ndarray:
    n = int(n_obs)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = np.power(0.5, ages / max(float(half_life), 1.0))
    return weights / max(float(np.mean(weights)), 1e-12)


def _fit_sklearn_regressor(model_name: str, X: np.ndarray, y: np.ndarray, seed: int) -> Any | None:
    name = str(model_name or "ridge").lower()
    weights = _recent_exponential_weights(len(y))
    try:
        if name.startswith("ridge"):
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=None))
            model.fit(X, y, ridge__sample_weight=weights)
            return model
        if name.startswith("hgbt"):
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_iter=80,
                max_leaf_nodes=8,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=int(seed),
            )
            model.fit(X, y, sample_weight=weights)
            return model
        if name.startswith("rf"):
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(
                n_estimators=64,
                max_depth=5,
                min_samples_leaf=25,
                max_features=0.75,
                bootstrap=True,
                n_jobs=1,
                random_state=int(seed),
            )
            model.fit(X, y, sample_weight=weights)
            return model
    except _SOURCE_FALLBACK_ERRORS:
        return None
    return None


def _target_persistence(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 30:
        return 0.0
    left = arr[:-1]
    right = arr[1:]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, 0.0, 0.995))


def _fit_bayesian_sbb_ml_vol_overlay(
    train: pd.Series,
    candidate: dict[str, Any],
    engine: SimfolioEngine,
    *,
    factor_frame: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    clean = pd.Series(train).dropna().astype(float)
    if len(clean) < 180:
        return None
    ml_model = str(candidate.get("ml_model", "ridge_harx"))
    factor_model = "ff6" if "ff6" in ml_model else ("q5" if "q5" in ml_model else "none")
    cfg = _factor_model_frame_config(engine, factor_model, factor_frame=factor_frame)
    factors = cfg.get("frame") if cfg else None
    panel = _ml_forecast_feature_panel(clean, factor_frame=factors, factor_model=factor_model)
    if panel is None:
        return None
    X = np.asarray(panel["X"], dtype=np.float64)
    y = np.asarray(panel["y_log_abs_vol"], dtype=np.float64)
    if X.shape[0] < 120:
        return None
    base_fit = _fit_bayesian_sbb_vol_overlay(
        clean.to_numpy(dtype=np.float64), {"overlay_model": "ewma_absolute"}
    )
    if base_fit is None:
        return None
    seed = SimfolioEngine._deterministic_seed("ml_vol_overlay", ml_model, len(clean), X.shape[1])
    model = _fit_sklearn_regressor(ml_model, X, y, seed)
    if model is None:
        return None
    latest = np.asarray(panel["latest_x"], dtype=np.float64).reshape(1, -1)
    pred_log_sigma = float(model.predict(latest)[0])
    base_long_sigma_x = float(
        max(float(base_fit.get("base_fit", {}).get("sigma", 0.0)) * 100.0, 1e-6)
    )
    predicted_sigma_x = float(np.exp(np.clip(pred_log_sigma, -18.0, 18.0)))
    sigma_proxy = np.exp(np.clip(y, -18.0, 18.0))
    floor, ceiling = _overlay_vol_clip_bounds(sigma_proxy, base_long_sigma_x)
    predicted_sigma_x = float(np.clip(predicted_sigma_x, floor, ceiling))
    persistence = _target_persistence(y)
    fitted = {
        **base_fit,
        "curve_fit": {
            "curve_type": "variance_decay",
            "persistence": persistence,
            "last_variance_x": float(max(predicted_sigma_x * predicted_sigma_x, 1e-8)),
            "long_variance_x": float(max(base_long_sigma_x * base_long_sigma_x, 1e-8)),
            "sigma_clip_bounds_x": (float(floor), float(ceiling)),
            "fit_status": "complete",
        },
        "ml_model": model,
        "meta": {
            **base_fit.get("meta", {}),
            "method": "bayesian_sbb_with_machine_learning_conditional_volatility_overlay",
            "ml_model": ml_model,
            "feature_set": "harx_ff6" if factor_model == "ff6" else "harx",
            "feature_count": int(X.shape[1]),
            "training_rows": int(X.shape[0]),
            "predicted_sigma_x": predicted_sigma_x,
            "overlay_persistence": persistence,
            "overlay_half_life_days": _overlay_half_life_days(persistence),
        },
    }
    return fitted


def _simulate_bayesian_sbb_ml_vol_overlay(
    fit: dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return _simulate_bayesian_sbb_vol_overlay(fit, int(total_days), int(n_paths), rng)


def _fit_harx_ff6_current_log_variance_anchor(
    train: pd.Series | None,
    engine: SimfolioEngine | None,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    if engine is None or train is None:
        return None
    clean = pd.Series(train).dropna().astype(float)
    if len(clean) < 180:
        return None
    cfg = _factor_model_frame_config(engine, "ff6")
    factors = cfg.get("frame") if cfg else None
    panel = _ml_forecast_feature_panel(clean, factor_frame=factors, factor_model="ff6")
    if panel is None:
        return None
    X = np.asarray(panel["X"], dtype=np.float64)
    y = np.asarray(panel["y_log_abs_vol"], dtype=np.float64)
    if X.shape[0] < 120:
        return None
    seed = SimfolioEngine._deterministic_seed(
        "full_mcmc_sv_harx_ff6_vol_anchor",
        str(candidate.get("id", "")),
        len(clean),
        X.shape[1],
    )
    model = _fit_sklearn_regressor("ridge_harx_ff6", X, y, seed)
    if model is None:
        return None
    latest = np.asarray(panel["latest_x"], dtype=np.float64).reshape(1, -1)
    predicted_log_sigma = float(model.predict(latest)[0])
    fitted = np.asarray(model.predict(X), dtype=np.float64)
    residual = y - fitted
    residual = residual[np.isfinite(residual)]
    if residual.size < 30:
        return None
    sigma_proxy = np.exp(np.clip(y, -18.0, 18.0))
    long_sigma_x = float(max(np.std(clean.to_numpy(dtype=np.float64), ddof=1) * 100.0, 1e-6))
    floor, ceiling = _overlay_vol_clip_bounds(sigma_proxy, long_sigma_x)
    predicted_sigma_x = float(
        np.clip(np.exp(np.clip(predicted_log_sigma, -18.0, 18.0)), floor, ceiling)
    )
    anchor_log_variance = float(2.0 * math.log(max(predicted_sigma_x, 1e-8)))
    anchor_var = float(4.0 * np.var(residual, ddof=1))
    if not np.isfinite(anchor_var) or anchor_var <= 1e-10:
        anchor_var = float(4.0 * np.var(y, ddof=1)) if y.size > 1 else 1.0
    return {
        "model": "ridge_harx_ff6",
        "target": "current_latent_log_variance",
        "anchor_log_variance_x": anchor_log_variance,
        "anchor_log_variance_error_variance": float(max(anchor_var, 1e-10)),
        "predicted_sigma_x": predicted_sigma_x,
        "training_rows": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "factor_model": "ff6",
        "factor_cols": list(panel.get("factor_cols", [])),
    }


def _apply_bayesian_sbb_vol_overlay_vol_anchor(
    fit: dict[str, Any],
    candidate: dict[str, Any],
    *,
    train: pd.Series | None,
    engine: SimfolioEngine | None,
) -> dict[str, Any] | None:
    if str(candidate.get("vol_anchor_model", "none")) != "ridge_harx_ff6":
        return fit
    anchor = _fit_harx_ff6_current_log_variance_anchor(train, engine, candidate)
    if anchor is None:
        return None
    curve = dict(fit.get("curve_fit", {}) or {})
    curve_type = str(curve.get("curve_type", ""))
    anchor_value = float(anchor["anchor_log_variance_x"])
    anchor_var = float(max(anchor["anchor_log_variance_error_variance"], 1e-10))
    if curve_type == "log_variance_decay":
        current_value = float(curve.get("last_log_variance_x", anchor_value))
        state_var = float(curve.get("last_log_variance_error_variance", anchor_var))
        state_var = float(max(state_var, 1e-10)) if np.isfinite(state_var) else anchor_var
        anchor_weight = float(state_var / max(state_var + anchor_var, 1e-10))
        curve["last_log_variance_x"] = float(
            (1.0 - anchor_weight) * current_value + anchor_weight * anchor_value
        )
        curve["last_log_variance_error_variance"] = float(
            1.0 / (1.0 / state_var + 1.0 / anchor_var)
        )
    elif curve_type == "variance_decay":
        current_value = float(
            math.log(max(float(curve.get("last_variance_x", math.exp(anchor_value))), 1e-8))
        )
        state_var = float(curve.get("last_log_variance_error_variance", anchor_var))
        state_var = float(max(state_var, 1e-10)) if np.isfinite(state_var) else anchor_var
        anchor_weight = float(state_var / max(state_var + anchor_var, 1e-10))
        curve["last_variance_x"] = float(
            math.exp(float((1.0 - anchor_weight) * current_value + anchor_weight * anchor_value))
        )
        curve["last_log_variance_error_variance"] = float(
            1.0 / (1.0 / state_var + 1.0 / anchor_var)
        )
    else:
        return None
    model = dict(fit)
    model["curve_fit"] = curve
    model["vol_anchor"] = {
        **anchor,
        "anchor_weight": anchor_weight,
        "pre_anchor_log_variance_x": current_value,
        "pre_anchor_log_variance_error_variance": state_var,
    }
    meta = dict(model.get("meta", {}) or {})
    meta["vol_anchor"] = model["vol_anchor"]
    model["meta"] = meta
    return model


def _sv_log_chi_square_bias(standardized_residuals: np.ndarray) -> float:
    z = np.asarray(standardized_residuals, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size < 30:
        return -1.2703628454614782
    value = float(np.mean(np.log(np.maximum(z * z, 1e-8))))
    if not np.isfinite(value):
        return -1.2703628454614782
    return float(np.clip(value, -3.0, 1.0))


def _fit_sv_variant(train_values: np.ndarray, candidate: dict[str, Any]) -> dict[str, Any] | None:
    base = _fit_sv_ar1(train_values)
    if base is None:
        return None
    fit = dict(base)
    if (
        bool(candidate.get("bias_correct", False))
        and str(fit.get("observation_model")) != "log_chi_square_bias_corrected_kalman"
    ):
        bias = _sv_log_chi_square_bias(
            np.asarray(fit.get("standardized_residuals"), dtype=np.float64)
        )
        fit["level"] = float(fit["level"] - bias)
        fit["last_log_var"] = float(fit["last_log_var"] - bias)
        fit["last_state_mean"] = float(fit.get("last_state_mean", fit["last_log_var"]) - bias)
        fit["sv_measurement_bias_correction"] = float(bias)
    phi_cap = float(candidate.get("phi_cap", 0.995))
    fit["phi"] = float(np.clip(float(fit.get("phi", 0.94)), 0.0, phi_cap))
    fit["eta_sd"] = float(
        np.clip(float(fit.get("eta_sd", 0.0)) * float(candidate.get("eta_scale", 1.0)), 0.0, 2.0)
    )
    last_blend = float(np.clip(candidate.get("last_blend", 0.0), 0.0, 1.0))
    fit["last_log_var"] = float(
        (1.0 - last_blend) * float(fit["last_log_var"]) + last_blend * float(fit["level"])
    )
    fit["last_state_mean"] = float(
        (1.0 - last_blend) * float(fit.get("last_state_mean", fit["last_log_var"]))
        + last_blend * float(fit["level"])
    )
    return fit


# Recovered historical wrappers for the two moving-block legacy cases.  They
# remain private because neither wrapper is one of the canonical 20 target IDs.
def _moving_block_indices(
    n_observations: int,
    block_length: int,
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Künsch moving-block bootstrap indices without circular wrapping."""
    n = int(n_observations)
    length = int(max(1, min(int(block_length), n)))
    total = int(total_days)
    paths = int(n_paths)
    if n <= 0 or total <= 0 or paths <= 0:
        return np.empty((paths, total), dtype=np.int64)
    starts = rng.integers(
        0,
        max(1, n - length + 1),
        size=(paths, math.ceil(total / length)),
    )
    offsets = np.arange(length, dtype=np.int64)[None, None, :]
    blocks = starts[:, :, None] + offsets
    return blocks.reshape(paths, -1)[:, :total]


def _moving_block_bayesian_sbb_paths(
    fit: dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Research-only moving-block analogue of the generic SBB simulator.

    The legacy Model 023 specification explicitly uses moving blocks.  The
    current generic gate only exposes stationary blocks, so this keeps the
    recovered whitepaper specification exact rather than silently substituting
    the stationary sampler.
    """
    base_fit = fit.get("base_fit", {}) or {}
    z_pool = np.asarray(base_fit.get("standardized_residuals"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size == 0:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    fixed_block_length = base_fit.get("fixed_block_length")
    if fixed_block_length is not None:
        block_length = int(max(1, min(int(fixed_block_length), int(z_pool.size))))
    else:
        block_length = max(
            int(_politis_white_block_length(z_pool)),
            int(_politis_white_block_length(z_pool * z_pool)),
        )
    z_draws = z_pool[
        _moving_block_indices(
            int(z_pool.size),
            block_length,
            int(total_days),
            int(n_paths),
            rng,
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
        mu_draws = rng.normal(
            mu,
            float(max(base_fit.get("posterior_sd", 0.0), 0.0)),
            size=int(n_paths),
        )
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
                use_sample_mu[None, :],
                float(base_fit["sample_mu"]),
                np.asarray(mu_path, dtype=np.float64),
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


LEGACY_MOVING_BLOCK_SOURCE = MappingProxyType(
    {
        "path": "simfolio/tmp/asset_level_full_panel_20260823/asset_level_full_exact_crps.py",
        "sha256": "5beb318b918ea367f7048d71a4e53e1bcc9b343faccda2ef493ddcf81bf3d465",
        "function_names": ("_moving_block_indices", "_moving_block_bayesian_sbb_paths"),
        "line_ranges": {
            "_moving_block_indices": (641, 662),
            "_moving_block_bayesian_sbb_paths": (665, 741),
        },
        "function_sha256": {
            "_moving_block_indices": "e193f8ebf78fe52212a099e162d67f8e0e4ebd103f2aebd0c0b5383edec2ce98",
            "_moving_block_bayesian_sbb_paths": "9d3a8972682118fb876e8d1483f23ca57fd2d116caf77329037304997bb54779",
        },
    }
)

SOURCE_FUNCTION_NAMES: tuple[str, ...] = (
    "_fit_bayesian_constrained_sbb",
    "_mean_schedule_from_fit",
    "_simulate_bayesian_constrained_sbb",
    "_historical_sharpe_stats",
    "_overlay_half_life_days",
    "_overlay_vol_clip_bounds",
    "_fit_absolute_ewma_volatility",
    "_fit_har_overlay",
    "_fit_arch_forecast_overlay",
    "_fit_bayesian_sbb_vol_overlay",
    "_overlay_vol_multiplier_curve",
    "_simulate_bayesian_sbb_vol_overlay",
    "_factor_model_frame_config",
    "_ml_forecast_feature_panel",
    "_recent_exponential_weights",
    "_fit_sklearn_regressor",
    "_target_persistence",
    "_fit_bayesian_sbb_ml_vol_overlay",
    "_simulate_bayesian_sbb_ml_vol_overlay",
    "_fit_harx_ff6_current_log_variance_anchor",
    "_apply_bayesian_sbb_vol_overlay_vol_anchor",
    "_sv_log_chi_square_bias",
    "_fit_sv_variant",
)

SOURCE_FUNCTIONS_SHA256 = "13951693868ba16eb6877e415c0b047b8c94f694dc409f209947cb56ef4e70dc"
SOURCE_FUNCTION_LINE_RANGES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "_fit_bayesian_constrained_sbb": (3564, 3637),
        "_mean_schedule_from_fit": (3640, 3681),
        "_simulate_bayesian_constrained_sbb": (5388, 5446),
        "_historical_sharpe_stats": (3921, 3947),
        "_overlay_half_life_days": (5449, 5453),
        "_overlay_vol_clip_bounds": (5456, 5467),
        "_fit_absolute_ewma_volatility": (5470, 5503),
        "_fit_har_overlay": (5506, 5558),
        "_fit_arch_forecast_overlay": (5561, 5602),
        "_fit_bayesian_sbb_vol_overlay": (5605, 5927),
        "_overlay_vol_multiplier_curve": (5930, 5993),
        "_simulate_bayesian_sbb_vol_overlay": (5996, 6015),
        "_factor_model_frame_config": (6018, 6042),
        "_ml_forecast_feature_panel": (6045, 6136),
        "_recent_exponential_weights": (6139, 6145),
        "_fit_sklearn_regressor": (6148, 6190),
        "_target_persistence": (7220, 7232),
        "_fit_bayesian_sbb_ml_vol_overlay": (7235, 7294),
        "_simulate_bayesian_sbb_ml_vol_overlay": (7297, 7303),
        "_fit_harx_ff6_current_log_variance_anchor": (13777, 13830),
        "_apply_bayesian_sbb_vol_overlay_vol_anchor": (13866, 13909),
        "_sv_log_chi_square_bias": (13185, 13193),
        "_fit_sv_variant": (13196, 13213),
    }
)


def _training_series(training: TrainingData) -> pd.Series:
    training.validate()
    if training.training_dates is None:
        raise ValueError("this Bayesian volatility model requires training_dates")
    dates = pd.DatetimeIndex(
        np.asarray(training.training_dates, dtype="datetime64[ns]")
    ).normalize()
    values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
    if dates.size != values.size or dates.duplicated().any():
        raise ValueError("training_dates must be unique and aligned with returns")
    return pd.Series(values, index=dates, dtype=np.float64)


@dataclass(frozen=True)
class BayesianVolOverlayModel:
    """Concrete source-backed adapter for one exact Bayesian SBB model ID."""

    model_id: str
    dependency_identity: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {
                "source_imports": (
                    "math",
                    "numpy",
                    "pandas",
                    "scipy",
                    "arch",
                    "scikit-learn",
                ),
                "source_revision": "773bc1c325559e6bf57a567f1d8bf473a3427fbc",
                "factor_snapshot": "canonical_snapshot/app/factor_data/french_daily.csv.gz",
            }
        )
    )

    def __post_init__(self) -> None:
        spec = SOURCE_CANDIDATE_SPECS.get(str(self.model_id))
        if spec is None:
            raise ValueError(f"unknown Bayesian volatility model: {self.model_id!r}")
        expected_type = (
            "bayesian_sbb_ml_vol_overlay"
            if "ml_vol_overlay" in str(spec["type"])
            else "bayesian_sbb_vol_overlay"
        )
        if str(spec["type"]) != expected_type:
            raise ValueError(f"invalid Bayesian volatility source type for {self.model_id!r}")

    @property
    def family(self) -> str:
        return str(SOURCE_CANDIDATE_SPECS[self.model_id]["type"])

    @property
    def source_specification(self) -> Mapping[str, Any]:
        return SOURCE_CANDIDATE_SPECS[self.model_id]

    @property
    def raw_source_specification(self) -> Mapping[str, Any]:
        return RAW_CANDIDATE_SPECS[self.model_id]

    @property
    def resolved_statistical_specification(self) -> Mapping[str, Any]:
        return RESOLVED_STATISTICAL_SPECS[self.model_id]

    @property
    def raw_seed_descriptor(self) -> Mapping[str, Any]:
        return RAW_SEED_DESCRIPTORS[self.model_id]

    @property
    def source_function_names(self) -> tuple[str, ...]:
        return SOURCE_FUNCTION_NAMES

    @property
    def source_fragment_digest(self) -> str:
        return SOURCE_FUNCTIONS_SHA256

    @property
    def seed_contract(self) -> str:
        return SOURCE_SEED_CONTRACT

    def _source_rng(self, context: ForecastContext) -> np.random.Generator:
        origin = (
            str(context.origin_date)
            if context.origin_date is not None
            else str(context.origin_label)
        )
        available_horizons = tuple(range(1, int(context.horizon_days) + 1))
        seed = forecast_oos_candidate_seed(
            origin,
            available_horizons,
            self.model_id,
            int(context.simulations),
        )
        return np.random.default_rng(seed)

    def _fit(self, training: TrainingData) -> dict[str, Any]:
        spec = dict(SOURCE_CANDIDATE_SPECS[self.model_id])
        if spec["type"] == "bayesian_sbb_ml_vol_overlay":
            series = _training_series(training)
            fit = _fit_bayesian_sbb_ml_vol_overlay(series, spec, SimfolioEngine())
        else:
            values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
            fit = _fit_bayesian_sbb_vol_overlay(values, spec)
            if fit is not None and spec.get("vol_anchor_model") == "ridge_harx_ff6":
                series = _training_series(training)
                fit = _apply_bayesian_sbb_vol_overlay_vol_anchor(
                    fit,
                    spec,
                    train=series,
                    engine=SimfolioEngine(),
                )
        if fit is None:
            raise ValueError(
                f"{self.model_id} source fit failed closed; check history, dependencies, and dates"
            )
        return fit

    def fit(self, training: TrainingData) -> dict[str, Any]:
        """Fit the exact source overlay and expose intermediate fit state."""
        training.validate()
        return self._fit(training)

    def simulate_daily_log_returns(
        self,
        training: TrainingData,
        context: ForecastContext,
    ) -> np.ndarray:
        training.validate()
        fit = self._fit(training)
        rng = self._source_rng(context)
        if self.family == "bayesian_sbb_ml_vol_overlay":
            paths = _simulate_bayesian_sbb_ml_vol_overlay(
                fit,
                int(context.horizon_days),
                int(context.simulations),
                rng,
            )
        else:
            paths = _simulate_bayesian_sbb_vol_overlay(
                fit,
                int(context.horizon_days),
                int(context.simulations),
                rng,
            )
        expected = (int(context.simulations), int(context.horizon_days))
        paths = np.asarray(paths, dtype=np.float64)
        if paths.shape != expected or not np.all(np.isfinite(paths)):
            raise ValueError(
                f"{self.model_id} returned invalid daily paths {paths.shape}; expected {expected}"
            )
        return paths


BAYESIAN_VOL_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {
        model_id: (lambda model_id=model_id: BayesianVolOverlayModel(model_id))
        for model_id in BAYESIAN_VOL_MODEL_IDS
    }
)


def make_bayesian_vol_model(model_id: str) -> BayesianVolOverlayModel:
    """Construct one exact-ID Bayesian volatility model; unknown IDs fail closed."""
    factory = BAYESIAN_VOL_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"no source-backed Bayesian volatility factory for {model_id!r}")
    return factory()


__all__ = [
    "BAYESIAN_VOL_FACTORIES",
    "BAYESIAN_VOL_MODEL_IDS",
    "RAW_CANDIDATE_SPECS",
    "RAW_SEED_DESCRIPTORS",
    "RESOLVED_STATISTICAL_SPECS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_LINE_RANGES",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "BayesianVolOverlayModel",
    "make_bayesian_vol_model",
]
