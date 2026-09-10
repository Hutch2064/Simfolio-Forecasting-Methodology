"""Generate the resolved statistical-definition resource and ledger patches.

The canonical score catalogue remains the membership authority.  This utility
resolves numerical defaults for every one of the 175 canonical rows, including
the source-backed portfolio and asset-level extensions.  It never infers
parameters from a model ID: each raw seed-bearing descriptor is kept beside a
separately resolved object.

The source manifests and fragments are intentionally kept separate from this
resource.  They are row-level historical candidate identities; this file adds
defaults and failure semantics needed to execute a candidate without changing
its ID.  The generated ledger patch contains only source-backed extension
metadata and never mutates the canonical ledger in place.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simfolio_forecasting_methodology.models.numerical.bdes_fastmap import (
    FRONTIER_CANDIDATE,
)
from simfolio_forecasting_methodology.models.portfolio import (
    bayesian_vol as _bayesian_vol,
)
from simfolio_forecasting_methodology.models.portfolio import (
    canonical_stack_reference as _canonical_stack_reference,
)
from simfolio_forecasting_methodology.models.portfolio import (
    factor_residual as _factor_residual,
)
from simfolio_forecasting_methodology.models.portfolio import (
    gas_reference as _gas_reference,
)
from simfolio_forecasting_methodology.models.portfolio import (
    gjr_reference as _gjr_reference,
)
from simfolio_forecasting_methodology.models.portfolio import (
    reference_families as _reference_families,
)
from simfolio_forecasting_methodology.models.portfolio import (
    sv_extensions as _sv_extensions,
)
from simfolio_forecasting_methodology.models.portfolio import (
    sv_mcmc_reference as _sv_mcmc_reference,
)
from simfolio_forecasting_methodology.models.portfolio import (
    sv_reference as _sv_reference,
)
from simfolio_forecasting_methodology.specifications import parse_compositional_spec

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "src/simfolio_forecasting_methodology/resources/canonical_175/ledger.json"
MCMC_MANIFEST_PATH = (
    ROOT
    / "src/simfolio_forecasting_methodology/resources/catalogs/canonical_40_full_mcmc_sv_specs.json"
)
RESOURCE_PATH = (
    ROOT
    / "src/simfolio_forecasting_methodology/resources/specifications/canonical_statistical_specifications.json"
)
READABLE_PATH = ROOT / "docs/canonical-statistical-specifications.md"
PORTFOLIO_LEDGER_PATCH_PATH = ROOT / "audit/statistical-specifications-portfolio-ledger-patch.json"
INLA_ASSET_FRAGMENT_PATH = (
    ROOT
    / "src/simfolio_forecasting_methodology/resources/specifications/canonical_inla_asset_spec_fragment.json"
)

BASE_SOURCE_REVISION = "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
BASE_SOURCE_SHA256 = "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
MCMC_SOURCE_REVISION = "511fb82c0be43564b79df3694ee570677f3137ed"
MCMC_SOURCE_SHA256 = "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997"

# These are the numerical RNG calls made by the extracted factories.  The
# runner checkpoint contract remains separately recorded in ``seed_identity``;
# it must not replace the source factory seed identity in a resolved row.
BASE_FACTORY_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, model_id, simulations)"
)
FULL_MCMC_FACTORY_SEED_CONTRACT = (
    "fit: deterministic_seed('full_mcmc_sv_overlay_fit', "
    "repr(full_mcmc_sv_overlay_fit_signature(candidate)), finite_observation_count, "
    "round(finite_observation_mean, 10)); forecast: blake2b-64-little-mod-2^32-1; "
    "args=('forecast_oos_candidate', origin_date, dense_horizon_tuple, model_id, simulations)"
)
FRONTIER_FACTORY_SEED_CONTRACT = (
    "deterministic_seed('asset_level_current_engine', ticker, origin_date, "
    "selected_model_id, horizon_days, simulations); deterministic_seed('copula_alternatives', "
    "asset_model_id, origin_date, horizon_days, simulations)"
)

PORTFOLIO_SOURCE_SEED_CONTEXT_REF = "portfolio.seed_identity"

# These modules expose exact source candidate dictionaries and explicit ID
# factories.  The two-row INLA/asset fragment is merged separately because it
# has a different shared-component namespace and source wrapper contract.
_PORTFOLIO_SOURCES: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("reference_families", _reference_families, _reference_families.REFERENCE_MODEL_IDS),
    (
        "sv_reference",
        _sv_reference,
        (
            "stochastic_volatility_ar1_empirical",
            "stochastic_volatility_ar1_empirical_sbb",
            "stochastic_volatility_ar1_student_t",
        ),
    ),
    ("sv_extensions", _sv_extensions, _sv_extensions.REFERENCE_MODEL_IDS),
    ("sv_mcmc_reference", _sv_mcmc_reference, _sv_mcmc_reference.REFERENCE_MODEL_IDS),
    ("gas_reference", _gas_reference, _gas_reference.REFERENCE_MODEL_IDS),
    ("gjr_reference", _gjr_reference, _gjr_reference.REFERENCE_MODEL_IDS),
    ("bayesian_vol", _bayesian_vol, _bayesian_vol.BAYESIAN_VOL_MODEL_IDS),
    ("factor_residual", _factor_residual, _factor_residual.FACTOR_RESIDUAL_MODEL_IDS),
    (
        "canonical_stack",
        _canonical_stack_reference,
        _canonical_stack_reference.REFERENCE_MODEL_IDS,
    ),
)

_PORTFOLIO_SOURCE_BY_ID: dict[str, tuple[str, Any]] = {
    model_id: (module_name, module)
    for module_name, module, model_ids in _PORTFOLIO_SOURCES
    for model_id in model_ids
}

_PORTFOLIO_MODEL_IDS: tuple[str, ...] = tuple(
    model_id
    for module_name, module, model_ids in _PORTFOLIO_SOURCES
    for model_id in model_ids
)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shared_component(shared_components: dict[str, Any], reference: str) -> Any:
    """Resolve one dotted shared-component reference."""

    value: Any = shared_components
    for part in reference.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"unknown shared-component reference: {reference}")
        value = value[part]
    return value


def _shared_component_refs(definition: dict[str, Any]) -> list[str]:
    """Return every shared component that can change this definition's behavior."""

    family = str(definition.get("family", ""))
    refs: list[str] = []
    if family == "base":
        refs.extend(
            [
                "base.contract",
                str(definition["mean"]["ref"]),
                str(definition["volatility"]["ref"]),
                "base.innovations.standardization",
            ]
        )
        innovation = definition.get("innovation", {})
        if innovation.get("model") == "empirical_standardized_residuals":
            if innovation.get("resampling") == "stationary_bootstrap":
                refs.append(f"base.innovations.{innovation['resampling']}")
            if innovation.get("tail") == "automated_evt_pot_gpd_tail":
                refs.append(f"base.innovations.{innovation['tail']}")
        else:
            refs.append(f"base.innovations.{innovation['model']}")
    elif family == "frontier":
        refs.extend(
            [
                "frontier.marginal_parameterization",
                "frontier.dependence_parameterization",
                "frontier.portfolio_rejoin",
            ]
        )
    elif family == "full_mcmc_sv":
        refs.extend(["full_mcmc_sv.contract", "full_mcmc_sv.contract.failure"])
    elif family in {
        "bayesian_sbb_vol_overlay",
        "bayesian_sbb_ml_vol_overlay",
        "factor_residual_sbb",
        "canonical",
        "stack",
        "gaussian",
        "student_t",
        "naive_iid_historical_portfolio_bootstrap",
        "zero_gaussian",
        "sv",
        "sv_sbb",
        "sv_extension",
        "sv_mcmc_sbb",
        "gas_score_driven_skewt",
        "portfolio_volatility_extension",
    }:
        refs.extend(
            [
                "portfolio.contract",
                "portfolio.seed_identity",
                str(definition["source_component_ref"]),
            ]
        )
    elif family in {"asset_level_extension", "bdes_non_mcmc_sv_overlay"}:
        # The INLA/asset fragment has its own exact dependency graph.  Keep
        # those references verbatim so merging the fragment does not silently
        # replace its source dispatch or seed aliases.
        refs.extend(str(reference) for reference in definition["shared_component_refs"])
    else:
        raise ValueError(f"cannot bind shared components for family {family!r}")

    seed_context = str(definition.get("source_seed_context_ref", ""))
    if seed_context:
        refs.append(seed_context)
    return sorted(set(refs))


def _bind_shared_component_digests(
    definition: dict[str, Any], shared_components: dict[str, Any]
) -> dict[str, Any]:
    """Add immutable dependency references and values to a definition copy.

    A string ``ref`` alone does not participate in the definition digest.  The
    digest map makes a change to a shared mean, volatility, sampler, or seed
    contract change every affected model fingerprint while keeping the full
    shared objects in one authoritative resource section.
    """

    bound = copy.deepcopy(definition)
    refs = _shared_component_refs(bound)
    bound["shared_component_refs"] = refs
    bound["shared_component_digests"] = {
        reference: _digest(_shared_component(shared_components, reference)) for reference in refs
    }
    return bound


def _base_components() -> dict[str, Any]:
    return {
        "contract": {
            "input_unit": "daily_log_return",
            "finite_observation_policy": "drop nonfinite values before fitting",
            "minimum_input_observations": 30,
            "minimum_residual_observations": 30,
            "minimum_standardized_residuals": 20,
            "degenerate_scale_failure": "constant or nonfinite scale fails the requested ARCH family",
            "arch_failure_policy": "fail closed unless allow_source_arch_fallback is explicitly true",
        },
        "mean_models": {
            "expanding_sample_mean": {
                "estimator": "arithmetic mean of finite observations",
                "residual": "observation minus fitted mean",
                "failure": "empty finite input returns zero only in helper; canonical fit rejects fewer than 30 observations",
            },
            "factor_premium_near_zero_alpha_shrinkage": {
                "estimator": "trimmed mean then positive-part t-statistic shrinkage to zero",
                "trim_fraction": 0.10,
                "trim_activation_min_observations": 20,
                "scale_ddof": 1,
                "shrink_weight": "t^2/(1+t^2)",
                "t_statistic": "abs(trimmed_mean)/(sample_sd/sqrt(n))",
                "failure": "nonfinite or zero scale gives zero shrink weight",
            },
            "bic_auto_arma_mean": {
                "history_window": 756,
                "minimum_observations_for_ARIMA": 80,
                "candidate_orders": [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],
                "selection": "minimum finite BIC with constant trend",
                "minimum_residual_observations": 30,
                "fallback": "factor_premium_near_zero_alpha_shrinkage when every ARIMA fit fails",
                "statsmodels_fit_failure": "candidate is skipped; all-candidate failure uses fallback",
            },
        },
        "volatility_models": {
            "constant_sample_volatility": {
                "scale_factor": 100.0,
                "estimator": "sample standard deviation of scaled residuals",
                "ddof": 1,
                "sigma_floor": 1.0e-6,
            },
            "garch_1_1_volatility": {
                "arch_model": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "power": 2.0},
                "mean": "Zero",
                "history_window": 1260,
                "minimum_observations": 80,
                "scaled_input": 100.0,
                "optimizer": {"maxiter": 80, "ftol": 1.0e-6, "disp": "off"},
                "conditional_sigma_floor": 1.0e-6,
                "minimum_returned_sigma_values": 20,
                "failure": "strict source fit failure; no implicit fallback",
            },
            "gjr_tarch_1_1_volatility": {
                "arch_model": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "power": 2.0},
                "mean": "Zero",
                "history_window": 1260,
                "minimum_observations": 80,
                "scaled_input": 100.0,
                "optimizer": {"maxiter": 80, "ftol": 1.0e-6, "disp": "off"},
                "conditional_sigma_floor": 1.0e-6,
                "minimum_returned_sigma_values": 20,
                "failure": "strict source fit failure; no implicit fallback",
            },
            "egarch_1_1_volatility": {
                "arch_model": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
                "mean": "Zero",
                "history_window": 1260,
                "minimum_observations": 80,
                "scaled_input": 100.0,
                "optimizer": {"maxiter": 80, "ftol": 1.0e-6, "disp": "off"},
                "conditional_sigma_floor": 1.0e-6,
                "minimum_returned_sigma_values": 20,
                "failure": "strict source fit failure; no implicit fallback",
            },
        },
        "innovations": {
            "standardization": {
                "center": "sample mean",
                "scale_ddof": 1,
                "fit_clip": [-12.0, 12.0],
                "generated_clip": [-20.0, 20.0],
                "zero_scale_failure": "return no valid innovation pool",
            },
            "gaussian_iid_standardized_innovations": {
                "distribution": "standard normal",
                "seed": "caller RNG",
            },
            "student_t_standardized_innovations": {
                "distribution": "arch.univariate.StudentsT",
                "nu_default": 8.0,
                "nu_bounds": [2.05, 80.0],
                "standardize_generated_draws": True,
            },
            "skew_t_standardized_innovations": {
                "distribution": "arch.univariate.SkewStudent",
                "eta_default": 8.0,
                "eta_bounds": [2.05, 80.0],
                "skew_lambda_default": 0.0,
                "skew_lambda_bounds": [-0.95, 0.95],
                "standardize_generated_draws": True,
            },
            "stationary_bootstrap": {
                "block_length": "max(Politis-White(z), Politis-White(z^2))",
                "restart_probability": "1/block_length",
                "initial_index": "uniform integer over innovation pool",
                "restart_index": "uniform integer over innovation pool",
                "row_assembly_threshold": 128,
            },
            "automated_evt_pot_gpd_tail": {
                "exceedance_share": "clip(sqrt(n)/n, 0.02, 0.10)",
                "central_pool": "between lower and upper empirical quantiles",
                "minimum_excesses_per_side": 10,
                "shape_clip": [-0.45, 0.45],
                "output_clip": [-20.0, 20.0],
                "failure": "fall back to empirical draws when GPD fit fails",
            },
        },
    }


def _frontier_definition() -> dict[str, Any]:
    return {
        "candidate": copy.deepcopy(FRONTIER_CANDIDATE),
        "marginal_parameterization": {
            "minimum_observations": 5,
            "sv_measurement": {
                "log_chi_square_mean": -1.2703628454614782,
                "log_chi_square_variance": "pi^2/2",
                "observed_variance_floor": 1.0e-16,
                "log_variance_clip": [-18.0, 18.0],
            },
            "dlm": {
                "phi_bounds": [0.0, 0.9995],
                "state_variance_ratio_bounds": [1.0e-10, (2.5**2) / 252.0],
                "objective": "bounded likelihood with three ratio starts",
                "annual_drift_sharpe_sd_cap": 2.5,
            },
            "posterior_sigma_points": {
                "center_and_local_points": 11,
                "phi_bounds": [0.0, 0.995],
                "eta_bounds": [0.02, 2.50],
                "rho_bounds": [-0.95, 0.95],
            },
            "multiscale": {
                "grid": [5.0, 21.0, 63.0, 252.0],
                "k_star": 4,
                "log_variance_clip": [-18.0, 18.0],
                "ridge_fraction": 0.05,
            },
            "failure": "nonfinite fit, insufficient history, or invalid innovation pool raises a source-specific ValueError",
        },
        "dependence_parameterization": {
            "minimum_observations": 80,
            "minimum_assets": 2,
            "pseudo_observation_clip": [1.0e-8, 1.0 - 1.0e-8],
            "covariance_psd_tolerance": -1.0e-7,
            "eigenvalue_floor": 1.0e-8,
            "factor_explained_variance_target": 0.80,
            "maximum_factors": 3,
            "residual_variance_floor": 1.0e-7,
            "phi_clip": [-0.995, 0.995],
            "innovation_variance_floor": 1.0e-7,
            "future_uniform_clip": [1.0e-8, 1.0 - 1.0e-8],
        },
        "portfolio_rejoin": {
            "turnover_cost_bps": 15.0,
            "weight_policy": "nonnegative target weights normalized to sum one",
            "rebalance_frequency_aliases": {
                "weekly": "W",
                "biweekly": "2W",
                "monthly": "ME",
                "quarterly": "QE",
                "annually": "YE",
                "annual": "YE",
                "yearly": "YE",
            },
            "future_calendar": "business days from training end plus one BDay",
            "failure": "require explicit training_dates, origin_date, future_dates and strict calendar alignment",
        },
    }


def _mcmc_contract() -> dict[str, Any]:
    return {
        "minimum_observations": 60,
        "fit_input": "finite daily portfolio log returns",
        "measurement_equation": {
            "default": "log_chi_square_mean_corrected_winsorized",
            "unclipped": "log_chi_square_mean_corrected_unwinsorized",
            "empirical_bias": "mean(log(max(z^2, tiny))) with source standardization",
            "measurement_mean": -1.2703628454614782,
            "measurement_variance": "pi^2/2",
            "observed_log_variance_clip": [-18.0, 18.0],
        },
        "latent_state": {
            "state": "level + AR(1) log variance",
            "phi_bounds": [0.0, 0.999],
            "eta_bounds": [1.0e-4, 5.0],
            "initial_parameter_bounds": {"level": [-18.0, 18.0], "rho": [-0.95, 0.95]},
            "no_ar_aliases": ["none", "iid", "no_ar"],
        },
        "sampler": {
            "fixed_defaults": {
                "iterations": 160,
                "burn": 60,
                "thin": 10,
                "minimum_iterations": 100,
                "check_interval": 40,
                "proposal": "bounded_random_walk_metropolis_hastings",
            },
            "transformed_parameterization": {
                "phi": "logit transform on (0, 0.999)",
                "eta": "log transform",
                "proposal": "stationary_transformed_metropolis_hastings",
            },
            "adaptive_defaults": {
                "stopping": "adaptive_ess_forecast_stability",
                "minimum_iterations": 120,
                "check_interval": 40,
                "forecast_stability_tolerance": 0.05,
                "min_ess": 100.0,
                "rhat_threshold": 1.01,
                "chains": 1,
                "adaptation_iterations": 60,
                "adaptation_target_acceptance": 0.234,
                "proposal": "adaptive_metropolis_warmup_covariance",
            },
            "seed": "deterministic_seed(full_mcmc_sv_overlay_fit, candidate signature, n, rounded sample mean)",
        },
        "innovations": {
            "pool_standardization": "mean/std or candidate median_mad; finite only",
            "pool_clip_when_raw_false": [-12.0, 12.0],
            "raw_pool": "unclipped finite standardized residuals",
            "resampling": {
                "iid": "uniform empirical pool draws",
                "stationary_bootstrap": "Politis-White block bootstrap",
                "circular_block_bootstrap": "uniform circular blocks",
                "paired_stationary_bootstrap": "paired return/state innovation blocks",
            },
            "parametric": {
                "student_t": {"df_bounds": [2.05, 80.0]},
                "skew_t": {"eta_bounds": [2.05, 80.0], "lambda_bounds": [-0.95, 0.95]},
            },
            "generated_clip": [-20.0, 20.0],
        },
        "simulation": {
            "output": "daily log-return increments, shape (paths, horizon)",
            "sigma": "sv_sigma_scale * exp(0.5*log_variance)/100",
            "leverage": "rho*lagged-or-same-period innovation + sqrt(1-rho^2)*state shock",
            "return_clip_default": [-1.0, 1.0],
            "nonfinite_output": "return empty output and caller fails closed",
        },
        "failure": {
            "short_history": "return no fit below 60 finite observations",
            "invalid_state_or_innovation_pool": "return no fit",
            "harx_factor_anchor": "fail closed until the verified factor panel is supplied",
        },
        "harx_ff6_factor_anchor": {
            "status": "blocked_factor_panel_closure",
            "model": "ridge_harx_ff6",
            "factor_model": "ff6",
            "required_columns": ["RF", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"],
            "source_column_reference": "source app/engine.py _factor_model_frame_config and _ml_forecast_feature_panel",
            "alignment": "normalize factor dates, reindex to return dates, numeric coercion, drop rows with missing factors",
            "features": "RF level and 21-day change; each factor one-day return, 21/63-day means, 21-day volatility, rolling beta",
            "minimum_history": 60,
            "leakage_rule": "factor rows must be available on or before each training date; no proxy or refresh",
            "required_evidence": "authorized factor panel digest and source-compatible Ridge preprocessing digest",
            "package_behavior": "_fit_harx_ff6_current_log_variance_anchor returns None until evidence is attached",
        },
    }


def _seed_identity() -> dict[str, Any]:
    """Return the source RNG call contexts used by each accepted family.

    The available-horizon tuple is part of the source seed.  The canonical
    all-daily plan therefore passes the dense tuple ``(1, ..., H)``; using
    ``(H,)`` would identify a different experiment.  Frontier retains its two
    historical aliases because its marginal and dependence draws are seeded
    independently by the source wrapper.
    """

    return {
        "checkpoint_seed_contract": "origin_task.seed_to_forecast_context.seed.v1",
        "base_and_full_mcmc": {
            "contract": "forecast_oos_candidate_seed(origin_date, tuple(range(1, horizon_days + 1)), model_id, simulations)",
            "parts": [
                "forecast_oos_candidate",
                "origin_date",
                "tuple(range(1, horizon_days + 1))",
                "model_id",
                "simulations",
            ],
            "horizon_values": "dense daily tuple 1..H, inclusive",
            "implementation": "simfolio_forecasting_methodology.seeds.forecast_oos_candidate_seed",
        },
        "frontier": {
            "contract": "two deterministic_seed calls from the historical Frontier wrapper",
            "marginal": {
                "alias": "asset_level_current_engine",
                "parts": [
                    "asset_level_current_engine",
                    "ticker",
                    "origin_date",
                    "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_fast_map_laplace_sigma_points",
                    "horizon_days",
                    "simulations",
                ],
            },
            "dependence": {
                "alias": "copula_alternatives",
                "parts": [
                    "copula_alternatives",
                    "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced",
                    "origin_date",
                    "horizon_days",
                    "simulations",
                ],
            },
            "implementation": "simfolio_forecasting_methodology.models.asset_level.frontier.HistoricalFrontierModel",
        },
    }


def _plain(value: Any) -> Any:
    """Convert module-level mapping proxies and tuples to JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _canonical_stack_factor_proxy_contract() -> dict[str, Any]:
    """Describe the complete frozen factor-prior input and fit semantics."""

    proxy_map = tuple(_canonical_stack_reference._CANONICAL_PROXY_MAP)
    series_digests = _canonical_stack_reference._CANONICAL_PROXY_SERIES_SHA256
    manifest = _canonical_stack_reference.SOURCE_ARTIFACTS[
        "canonical_proxy_snapshot_manifest"
    ]
    return {
        "status": "complete_frozen_manifest_whitelisted_source_snapshot",
        "dispatch": "_fit_factor_drift_prior",
        "manifest": {
            "path": str(manifest["path"]),
            "sha256": str(manifest["sha256"]),
            "validation": "content hash and every manifest series entry are checked before loading",
        },
        "series": [
            {
                "ticker": str(ticker),
                "label": str(label),
                "sha256": str(series_digests[ticker]),
            }
            for ticker, label in proxy_map
        ],
        "expected_factor_count": len(proxy_map),
        "minimum_history": 252,
        "minimum_joint_overlap": "max(252, min(756, len(y) // 3))",
        "alignment": {
            "index": "normalized portfolio dates",
            "factor_frame": "reindex frozen factor frame to the portfolio index",
            "joint_fit": "concatenate portfolio and usable factors then drop rows with any missing value",
        },
        "usable_factor_rule": "retain columns with at least minimum_joint_overlap nonmissing aligned rows",
        "overlap_reduction": "while joint rows are below the threshold, drop the factor with the lowest valid count; stop at one factor",
        "regression": {
            "center_portfolio": "y_arr - mean(y_arr)",
            "center_factors": "x_arr - mean(x_arr, axis=0)",
            "ridge_penalty": "max(trace(X.T @ X) / p * (p / n), 1e-12)",
            "solver": "solve(X.T @ X + ridge * I, X.T @ y); use pinv(X) @ y on arithmetic/runtime/value failure",
            "adjusted_r_squared": "clip(1 - (1 - r2) * ((n - 1) / max(n - p - 1, 1)), 0, 0.95)",
        },
        "alpha_shrinkage": {
            "raw": "mean(y_arr) - mean(x_arr, axis=0) @ beta",
            "residual_standard_deviation": "sample std of centered residuals, ddof=1 when n > p + 1",
            "weight": "t^2 / (1 + t^2), t = abs(alpha_raw) / (resid_sigma / sqrt(n))",
            "estimate": "alpha_raw * weight",
        },
        "factor_mean_shrinkage": "_sample_mean_near_zero_shrinkage for every aligned factor column",
        "daily_log_mean": "alpha_shrunk + factor_mean_arr @ beta; nonfinite result becomes 0.0",
        "no_overlap_behavior": {
            "insufficient_history": "status skipped, daily_log_mean 0.0, adjusted_r_squared 0.0, factor_count 0",
            "no_factor_proxy_overlap": "status skipped, daily_log_mean 0.0, adjusted_r_squared 0.0, factor_count 0",
            "insufficient_joint_overlap": "status skipped with retained factor count and combined observation count",
        },
        "downstream_forecast_weight": "clip(adjusted_r_squared * (observation_count / len(portfolio)), 0, 0.50)",
    }


def _portfolio_parameter_contract(module_name: str) -> dict[str, Any]:
    """Return the resolved numerical defaults for one extracted module."""

    seed_model_argument = (
        "candidate_id" if module_name == "canonical_stack" else "public_model_id"
    )
    common = {
        "input_unit": "portfolio_daily_log_return",
        "finite_observation_policy": "drop nonfinite observations before fitting",
        "simulated_return_clip": [-1.0, 1.0],
        "unknown_model_policy": "exact ID map; unknown IDs raise ValueError",
        "fit_failure_policy": "raise source-specific error; no generic fallback",
        "rng": "numpy.random.default_rng",
        "forecast_seed_arguments": [
            "forecast_oos_candidate",
            "origin_date",
            "dense_horizon_tuple",
            seed_model_argument,
            "simulations",
        ],
        "forecast_seed_model_argument": seed_model_argument,
        "dense_horizon_tuple": "tuple(range(1, horizon_days + 1))",
    }
    contracts: dict[str, dict[str, Any]] = {
        "reference_families": {
            **common,
            "output_semantics": "direct terminal log-return samples by dense horizon",
            "minimum_finite_training_observations": {
                "constant_mean_gaussian": 30,
                "constant_mean_student_t": 30,
                "naive_iid_historical_portfolio_bootstrap": 1,
                "zero_mean_gaussian_vol_only": 1,
            },
            "gaussian": {
                "mean": "arithmetic sample mean",
                "scale": "sample standard deviation, ddof=1",
                "terminal_scale": "sigma * sqrt(horizon)",
            },
            "student_t": {
                "df": "clip(4 + 6 / excess_kurtosis, 4, 30); default 30",
                "innovation_scale": "sigma / sqrt(df / (df - 2))",
            },
            "naive": {"index_sampling": "iid uniform integer over finite history"},
            "zero_mean": {"mean": 0.0, "scale": "sample standard deviation, ddof=1"},
        },
        "sv_reference": {
            **common,
            "output_semantics": "cumulative terminal samples from source daily SV paths",
            "minimum_finite_training_observations": 252,
            "measurement_equation": {
                "mean": -1.2703628454614782,
                "variance": "pi^2 / 2",
                "winsorize": "5th and 95th percentile only when observed count >= 80",
                "log_variance_clip": [-18.0, 18.0],
            },
            "state_space_fit": {
                "phi_bounds": [0.001, 0.995],
                "eta_bounds": [0.02, 2.5],
                "optimizer": "L-BFGS-B",
                "starts": 4,
            },
            "innovation": {
                "empirical_clip": [-12.0, 12.0],
                "student_t_df_bounds": [4.0, 30.0],
                "stationary_bootstrap": "max(_politis_white_block_length(z), 1)",
            },
        },
        "sv_extensions": {
            **common,
            "output_semantics": "cumulative terminal samples from source daily extension paths",
            "minimum_finite_training_observations": 504,
            "dp_mixture": {
                "max_components": 6,
                "max_fit_observations": 5000,
                "weight_concentration_prior": 0.5,
                "mixture_max_iter": 120,
                "transition_prior_alpha": 0.5,
                "rolling_windows": {"absolute_return_mean": 21, "trend_sum": 63},
            },
            "observable_markov": {
                "state_count": 12,
                "transition_prior_alpha": 0.5,
                "rolling_windows": {"volatility": [21, 10], "trend": [63, 21]},
            },
        },
        "sv_mcmc_reference": {
            **common,
            "output_semantics": "cumulative terminal samples from source daily MCMC-SV paths",
            "minimum_finite_training_observations": 504,
            "fit_seed_arguments": [
                "sv_mcmc_fit",
                "finite_observation_count",
                "round(finite_observation_mean, 10)",
            ],
            "measurement_equation": {
                "mean": -1.2703628454614782,
                "variance": "pi^2 / 2",
                "log_variance_clip": [-18.0, 18.0],
            },
            "sampler": {
                "iterations": 420,
                "burn": 120,
                "thin": 4,
                "proposal": "bounded random walk over level, phi, eta",
                "phi_bounds": [0.001, 0.994],
                "eta_bounds": [1.0e-4, 5.0],
            },
            "stationary_bootstrap": "_politis_white_block_length plus source index kernel",
        },
        "gas_reference": {
            **common,
            "output_semantics": "cumulative terminal samples from source daily GAS paths",
            "minimum_finite_training_observations": 504,
            "max_fit_observations": 5000,
            "grid": {"alpha": [0.03, 0.06, 0.10, 0.16], "beta": [0.85, 0.93, 0.97, 0.985]},
            "df_bounds": [4.0, 30.0],
            "innovation": "Jones-Faddy skew-t when fit succeeds; source empirical pool on fit failure",
            "log_variance_clip": [-18.0, 18.0],
            "innovation_clip": [-20.0, 20.0],
        },
        "gjr_reference": {
            **common,
            "output_semantics": "daily log-return increments",
            "minimum_finite_training_observations": 60,
            "mean_fit": {
                "model": "empirical_bayes_hac_sharpe",
                "prior_sr": 0.75,
                "sr_cap": 0.75,
                "posterior_mu_draws": False,
            },
            "arch_fit": {
                "vol": "GARCH",
                "p": 1,
                "o": 1,
                "q": 1,
                "power": 2.0,
                "max_fit_observations": 1260,
                "optimizer": {"maxiter": 80, "ftol": 1.0e-6},
            },
            "innovations": {
                "standardized_clip": [-20.0, 20.0],
                "block_length": "max(_politis_white_block_length(z), _politis_white_block_length(z*z))",
                "moving_block": "non-circular Kunsch blocks from recovery wrapper",
            },
        },
        "bayesian_vol": {
            **common,
            "output_semantics": "daily log-return increments",
            "minimum_finite_training_observations": 60,
            "shared_mean_fit": "_fit_bayesian_constrained_sbb and explicit source mean branches",
            "innovation": {
                "residual_scale": 100.0,
                "standardized_clip": [-20.0, 20.0],
                "block_length": "max(_politis_white_block_length(z), _politis_white_block_length(z*z))",
                "resampling": "stationary_bootstrap",
            },
            "overlay_multiplier_clip": [0.01, 100.0],
            "arch_fit": {"max_fit_observations": 1260, "optimizer": {"maxiter": 80, "ftol": 1.0e-6}},
            "ml_factor_policy": "raw descriptor controls branch; frozen dated FF6 frame only",
        },
        "factor_residual": {
            **common,
            "output_semantics": "daily log-return increments",
            "minimum_finite_training_observations": 180,
            "factor_model": "ff6 source default because raw descriptors omit factor_model",
            "factor_columns": ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"],
            "risk_free_column": "RF",
            "alignment": "normalize dates, align and drop missing factors, require final date equality",
            "regressor": {
                "pipeline": ["StandardScaler", "Ridge"],
                "alpha": 10.0,
                "sample_weight": "recent_exponential_weights",
                "half_life_days": 504.0,
            },
            "residual_overlay": "none source default because raw descriptors omit residual_overlay",
            "stationary_bootstrap": "max(_politis_white_block_length(y_excess), _politis_white_block_length(residuals*residuals))",
        },
        "canonical_stack": {
            **common,
            "output_semantics": "terminal log-return samples by dense horizon",
            "minimum_finite_training_observations": 30,
            "canonical_fit": {
                "history_window": _canonical_stack_reference.CANONICAL_REGIME_FIT_MAX_OBS,
                "mean": "sample_mean_positive_part_t_stat_shrinkage_to_zero",
                "volatility": "estimated_decay_ewma_volatility",
                "ewma_lambda_bounds": [0.80, 0.995],
                "ewma_lambda_default": 0.94,
                "ewma_objective": "sum(log(h_t) + x_t^2 / h_t)",
                "scaled_returns": 100.0,
                "residual_standardized_clip": [-12.0, 12.0],
                "minimum_standardized_residuals": 20,
            },
            "regime_filter": {
                "states": 3,
                "fit_max_iterations": _canonical_stack_reference.CANONICAL_REGIME_EM_MAX_ITER,
                "fit_min_iterations": _canonical_stack_reference.CANONICAL_REGIME_EM_MIN_ITER,
                "convergence_tolerance": _canonical_stack_reference.CANONICAL_REGIME_EM_TOL,
                "fallback_transition_matrix": [
                    [0.97, 0.03, 0.0],
                    [0.015, 0.97, 0.015],
                    [0.0, 0.03, 0.97],
                ],
                "sticky_diagonal_prior": 25.0,
                "log_sigma_quantiles": [0.20, 0.55, 0.85],
                "state_sigma_quantiles": [0.25, 0.55, 0.85],
                "log_variance_floor": 1.0e-4,
                "sigma_floor": 1.0e-6,
                "regime_blend_bounds": [0.005, 0.08],
            },
            "innovation": {
                "resampling": "stationary_bootstrap",
                "block_length": "max(_politis_white_block_length(z), _politis_white_block_length(residuals*residuals))",
                "row_assembly_threshold": _canonical_stack_reference.STATIONARY_BOOTSTRAP_ROW_ASSEMBLY_MIN_DAYS,
                "evt_threshold": 0.90,
                "evt_exceedance_share": "clip(sqrt(n)/n, 0.02, 0.10)",
                "evt_shape_clip": [-0.45, 0.45],
                "generated_clip": [-20.0, 20.0],
            },
            "factor_proxy": {
                **_canonical_stack_factor_proxy_contract(),
            },
            "stack_dispatch": {
                "mix_method": "source_terminal_prefix_concatenation",
                "canonical_weight_rounding": "round(weight * n_sims)",
                "weight_bounds": [0.0, 1.0],
                "empty_component_policy": "return the nonempty component",
            },
        },
    }
    return contracts[module_name]


def _portfolio_runtime_dependencies(module_name: str) -> dict[str, Any]:
    dependencies: dict[str, dict[str, Any]] = {
        "reference_families": {
            "items": ["numpy", "scipy"],
            "constraints": ["numpy>=2.0,<3", "scipy>=1.13,<2"],
        },
        "sv_reference": {
            "items": ["numpy", "scipy"],
            "constraints": ["numpy>=2.0,<3", "scipy>=1.13,<2"],
        },
        "sv_extensions": {
            "items": ["numpy", "pandas", "scikit-learn"],
            "constraints": ["numpy>=2.0,<3", "pandas>=2.2,<3", "scikit-learn>=1.9,<2"],
        },
        "sv_mcmc_reference": {
            "items": ["numpy", "scipy"],
            "constraints": ["numpy>=2.0,<3", "scipy>=1.13,<2"],
        },
        "gas_reference": {
            "items": ["numpy", "scipy"],
            "constraints": ["numpy>=2.0,<3", "scipy>=1.13,<2"],
        },
        "gjr_reference": {
            "items": ["numpy", "arch"],
            "constraints": ["numpy>=2.0,<3", "arch version pin unresolved in source"],
        },
        "bayesian_vol": {
            "items": ["numpy", "pandas", "scipy", "scikit-learn", "arch"],
            "constraints": [
                "numpy>=2.0,<3",
                "pandas>=2.2,<3",
                "scipy>=1.13,<2",
                "scikit-learn>=1.9,<2",
                "arch>=8,<9",
            ],
        },
        "factor_residual": {
            "items": ["numpy", "pandas", "scikit-learn", "scipy"],
            "constraints": [
                "numpy>=2.0,<3",
                "pandas>=2.2,<3",
                "scikit-learn>=1.9,<2",
                "scipy>=1.13,<2",
            ],
        },
        "canonical_stack": {
            "items": ["numpy", "pandas", "scipy"],
            "constraints": [
                "numpy>=2.0,<3",
                "pandas>=2.2,<3",
                "scipy>=1.13,<2",
            ],
        },
    }
    result = copy.deepcopy(dependencies[module_name])
    result.update(
        {
            "verified": True,
            "status": "imports and bounded source-kernel parity recorded in family audit",
        }
    )
    return result


def _portfolio_source_closure(module_name: str, module: Any) -> dict[str, Any]:
    artifacts = _plain(getattr(module, "SOURCE_ARTIFACTS", {}))
    source_revision = BASE_SOURCE_REVISION
    revision_entry = artifacts.get("source_revision")
    if isinstance(revision_entry, dict) and revision_entry.get("revision"):
        source_revision = str(revision_entry["revision"])
    primary_label = "research_gate" if "research_gate" in artifacts else "engine"
    primary = artifacts.get(primary_label, {})
    if not isinstance(primary, dict) or not primary.get("path") or not primary.get("sha256"):
        raise ValueError(f"portfolio source module has no hashed primary artifact: {module_name}")
    dependency_artifacts = []
    for label, artifact in artifacts.items():
        if label == primary_label or not isinstance(artifact, dict) or not artifact.get("path"):
            continue
        dependency_artifacts.append(
            {
                "label": label,
                "path": artifact["path"],
                "sha256": artifact.get("sha256"),
                **({"status": artifact["status"]} if artifact.get("status") else {}),
            }
        )
    line_ranges = getattr(module, "SOURCE_FUNCTION_LINE_RANGES", None)
    closure = {
        "module": f"simfolio_forecasting_methodology.models.portfolio.{module_name}",
        "source_revision": source_revision,
        "primary_artifact": {
            "label": primary_label,
            "path": primary["path"],
            "sha256": primary["sha256"],
        },
        "dependency_artifacts": dependency_artifacts,
        "source_function_names": list(getattr(module, "SOURCE_FUNCTION_NAMES", ())),
        "source_functions_sha256": str(getattr(module, "SOURCE_FUNCTIONS_SHA256", "")),
        "source_function_line_ranges": _plain(line_ranges) if line_ranges else None,
        "runtime_dependencies": _portfolio_runtime_dependencies(module_name),
        "resolved_parameter_contract": _portfolio_parameter_contract(module_name),
    }
    if len(closure["source_functions_sha256"]) != 64:
        raise ValueError(f"portfolio source function digest is not SHA-256: {module_name}")
    return closure


def _portfolio_shared_components() -> dict[str, Any]:
    source_closures = {
        module_name: _portfolio_source_closure(module_name, module)
        for module_name, module, _ in _PORTFOLIO_SOURCES
    }
    return {
        "contract": {
            "raw_descriptor_policy": "exact recovered source fields are immutable and seed-bearing",
            "resolved_descriptor_policy": "source defaults are added in resolved_candidate only",
            "daily_path_output": "shape (simulations, horizon_days) for increment adapters",
            "terminal_output": "shape (simulations, horizon_days) for terminal adapters",
            "failure_policy": "unknown IDs, missing dependencies, fit failures, bad dates, and nonfinite output raise",
            "historical_score_policy": "source-kernel parity does not claim retained score reproduction",
        },
        "seed_identity": {
            "forecast_contract": "forecast_oos_candidate_seed(origin_date, tuple(range(1, horizon_days + 1)), model_id, simulations)",
            "forecast_parts": [
                "forecast_oos_candidate",
                "origin_date",
                "dense_horizon_tuple",
                "public_model_id for existing adapters; candidate_id for canonical_stack",
                "simulations",
            ],
            "model_argument_policy": {
                "existing_adapters": "public_model_id",
                "canonical_stack": "candidate_id",
            },
            "fit_contract_exceptions": {
                "sv_mcmc_reference": "deterministic_seed('sv_mcmc_fit', finite_observation_count, round(finite_observation_mean, 10))",
                "factor_residual": "deterministic_seed('factor_residual_sbb', factor_model, len(combined), len(factor_cols))",
            },
            "historical_panel_seed": 20260528,
            "panel_seed_status": "source-declared seed retained separately; exact experiment schedule unresolved",
        },
        "source_closures": source_closures,
    }


def _portfolio_source_candidate(module_name: str, module: Any, model_id: str) -> dict[str, Any]:
    if module_name == "factor_residual":
        source_map = module.RAW_FACTOR_RESIDUAL_SPECS
    elif module_name == "bayesian_vol":
        source_map = module.RAW_CANDIDATE_SPECS
    else:
        source_map = module.SOURCE_CANDIDATE_SPECS
    candidate = source_map.get(model_id)
    if candidate is None:
        raise ValueError(f"portfolio source descriptor missing for {model_id!r}")
    return _plain(candidate)


def _portfolio_resolved_candidate(
    module_name: str, module: Any, model_id: str, source_candidate: dict[str, Any]
) -> dict[str, Any]:
    resolved = copy.deepcopy(source_candidate)
    resolved.update(_portfolio_parameter_contract(module_name))
    if module_name == "bayesian_vol":
        resolved.update(_plain(module.RESOLVED_STATISTICAL_SPECS[model_id]))
    elif module_name == "factor_residual":
        resolved.update(_plain(module.RESOLVED_FACTOR_RESIDUAL_SPECS[model_id]))
    elif module_name == "sv_reference" and model_id == "stochastic_volatility_ar1_student_t":
        # The retained row is labelled Student-t but its raw historical
        # descriptor omits innovation.  The source dispatcher therefore takes
        # its empirical default; keep that dispatch in resolved fields only.
        resolved["innovation_dispatch"] = {
            "source_default": "empirical",
            "parameter_source": "SVReferenceModel source default because raw descriptor omits innovation",
        }
    # The source candidate remains unchanged above.  The module contract and,
    # where available, the extracted family specification are resolved fields
    # only; neither object is passed to the source seed call.
    resolved["id"] = model_id
    return resolved


def _portfolio_seed_descriptor(module_name: str, module: Any, model_id: str) -> dict[str, Any]:
    seed_model_argument = (
        "candidate_id" if module_name == "canonical_stack" else "public_model_id"
    )
    descriptor = {
        "source_model_key": model_id,
        "forecast_seed_contract": str(module.SOURCE_SEED_CONTRACT),
        "forecast_seed_arguments": [
            "forecast_oos_candidate",
            "origin_date",
            "dense_horizon_tuple",
            seed_model_argument,
            "simulations",
        ],
        "status": "source seed arguments retained separately from resolved defaults",
    }
    if module_name == "bayesian_vol":
        descriptor.update(_plain(module.RAW_SEED_DESCRIPTORS[model_id]))
    elif module_name in {"gjr_reference", "canonical_stack"}:
        descriptor["panel_seed"] = 20260528
        descriptor["panel_seed_status"] = "source-declared; exact experiment schedule unresolved"
    if module_name == "sv_mcmc_reference":
        descriptor["fit_seed_contract"] = (
            "deterministic_seed('sv_mcmc_fit', finite_observation_count, round(finite_observation_mean, 10))"
        )
    if module_name == "factor_residual":
        descriptor["fit_seed_contract"] = (
            "deterministic_seed('factor_residual_sbb', factor_model, len(combined), len(factor_cols))"
        )
    return descriptor


_PORTFOLIO_FACTORY_MAPS = {
    "reference_families": ("REFERENCE_FACTORIES", "ReferencePortfolioModel"),
    "sv_reference": ("REFERENCE_FACTORIES", "SVReferenceModel"),
    "sv_extensions": ("REFERENCE_FACTORIES", "SuffixSVExtensionModel"),
    "sv_mcmc_reference": ("REFERENCE_FACTORIES", "BayesianMCMCSVModel"),
    "gas_reference": ("REFERENCE_FACTORIES", "GasScoreDrivenSkewTModel"),
    "gjr_reference": ("REFERENCE_FACTORIES", "PortfolioGJRGARCHModel"),
    "bayesian_vol": ("BAYESIAN_VOL_FACTORIES", "BayesianVolOverlayModel"),
    "factor_residual": ("FACTOR_RESIDUAL_FACTORIES", "FactorResidualSBBModel"),
    "canonical_stack": ("REFERENCE_FACTORIES", "CanonicalStackModel"),
}


def _resolve_portfolio_model(
    row: dict[str, Any], shared_components: dict[str, Any]
) -> dict[str, Any]:
    model_id = str(row["public_model_id"])
    module_name, module = _PORTFOLIO_SOURCE_BY_ID[model_id]
    source_candidate = _portfolio_source_candidate(module_name, module, model_id)
    resolved_candidate = _portfolio_resolved_candidate(
        module_name, module, model_id, source_candidate
    )
    closure = _shared_component(shared_components, f"portfolio.source_closures.{module_name}")
    source_reference = {
        "path": closure["primary_artifact"]["path"],
        "revision": closure["source_revision"],
        "sha256": closure["primary_artifact"]["sha256"],
        "entrypoints": closure["source_function_names"],
        "dependency_artifacts": closure["dependency_artifacts"],
    }
    factory_map, class_name = _PORTFOLIO_FACTORY_MAPS[module_name]
    definition = {
        "family": str(row["model_family"]),
        "source_id": model_id,
        "source_candidate": source_candidate,
        "resolved_candidate": resolved_candidate,
        "source_seed_descriptor": _portfolio_seed_descriptor(module_name, module, model_id),
        "source_seed_contract": str(module.SOURCE_SEED_CONTRACT),
        "source_reference": source_reference,
        "source_component_ref": f"portfolio.source_closures.{module_name}",
        "factory": {
            "module": f"simfolio_forecasting_methodology.models.portfolio.{module_name}",
            "map": factory_map,
            "class": class_name,
            "exact_id_dispatch": True,
        },
        "factory_seed_contract": str(module.SOURCE_SEED_CONTRACT),
        "source_seed_context_ref": PORTFOLIO_SOURCE_SEED_CONTEXT_REF,
        "failure_semantics": {
            "unknown_model": "raise ValueError",
            "fit_failure": "raise source-specific ValueError or RuntimeError",
            "nonfinite_output": "raise ValueError",
            "historical_score": "not verified by this resolved specification",
        },
    }
    return _bind_shared_component_digests(definition, shared_components)


def _load_inla_asset_fragment() -> dict[str, Any]:
    """Load the checked-in two-row source fragment without importing its generator."""

    fragment = json.loads(INLA_ASSET_FRAGMENT_PATH.read_text(encoding="utf-8"))
    accepted = fragment.get("accepted_model_ids")
    definitions = fragment.get("resolved_definitions")
    components = fragment.get("shared_components")
    if not isinstance(accepted, list) or not isinstance(definitions, dict):
        raise TypeError("INLA/asset fragment has no complete accepted definitions")
    if set(accepted) != set(definitions) or not isinstance(components, dict):
        raise ValueError("INLA/asset fragment membership or components are inconsistent")
    return fragment


def _merge_inla_asset_components(
    shared_components: dict[str, Any], fragment: dict[str, Any]
) -> None:
    """Merge fragment components while retaining their dotted references."""

    fragment_components = fragment["shared_components"]
    for name in ("asset_exact", "inla"):
        if name in shared_components:
            raise ValueError(f"INLA/asset component collides with central component: {name}")
        shared_components[name] = copy.deepcopy(fragment_components[name])
    fragment_seed_identity = fragment_components.get("seed_identity", {})
    if not isinstance(fragment_seed_identity, dict):
        raise TypeError("INLA/asset fragment seed identity is not an object")
    central_seed_identity = shared_components["seed_identity"]
    for name, value in fragment_seed_identity.items():
        if name in central_seed_identity:
            raise ValueError(f"INLA/asset seed component collides with central component: {name}")
        central_seed_identity[name] = copy.deepcopy(value)


def _fragment_factory_seed_contract(model_id: str) -> str:
    """Return the actual source seed calls for the two extension adapters."""

    if model_id == "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced":
        return (
            "deterministic_seed('asset_level_current_engine', ticker, origin_date, "
            "selected_model_id, horizon_days, simulations); "
            "deterministic_seed('copula_alternatives', asset_model_id, origin_date, "
            "horizon_days, simulations)"
        )
    if model_id == (
        "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_"
        "full_inla_laplace_quadrature_centered_multiscale"
    ):
        return "deterministic_seed('forecast_final_fixed', selected_model_id, horizon_days, simulations)"
    raise ValueError(f"unknown INLA/asset fragment model: {model_id}")


def _resolve_fragment_model(
    row: dict[str, Any], fragment: dict[str, Any], shared_components: dict[str, Any]
) -> dict[str, Any]:
    """Bind a fragment definition into the central resource without rewriting it."""

    model_id = str(row["public_model_id"])
    try:
        source_definition = fragment["resolved_definitions"][model_id]
    except KeyError as exc:
        raise ValueError(f"INLA/asset fragment missing ledger row: {model_id}") from exc
    definition = copy.deepcopy(source_definition)
    # The fragment was generated before the seed audit distinguished adapter
    # calls from the runner checkpoint contract.  Correct only that resolved
    # metadata field; the raw source_candidate and component graph stay intact.
    definition["source_seed_contract"] = _fragment_factory_seed_contract(model_id)
    definition["factory_seed_contract"] = definition["source_seed_contract"]
    return _bind_shared_component_digests(definition, shared_components)


def _resolve_mcmc_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply the source loop's documented defaults to one candidate row."""

    out = dict(candidate)
    candidate_id = str(out.get("id", ""))
    out.setdefault("type", "bayesian_sbb_full_mcmc_sv_overlay")
    leverage = bool(out.get("leverage", False))
    transformed = bool(out.get("transformed_parameter_mcmc", False))
    unclipped_measurement = bool(out.get("unclipped_sv_measurement", False))
    raw_innovations = bool(out.get("unclipped_empirical_innovations", False))
    out.setdefault("overlay_model", "mcmc_sv_ar1_leverage" if leverage else "mcmc_sv_ar1")
    out.setdefault("mean_model", "positive_sample_mean")
    out.setdefault("vol_model", "full_latent_mcmc_stochastic_log_volatility_overlay")
    out.setdefault("innovation_method", "empirical_standardized_residuals")
    out.setdefault("tail_method", "filtered_empirical_tail")
    out.setdefault(
        "path_generator", "stationary_bootstrap_standardized_residuals_scaled_by_latent_sv_paths"
    )
    out.setdefault("mcmc_iterations", 160)
    out.setdefault("mcmc_burn", 60)
    out.setdefault("mcmc_thin", 10)
    out.setdefault("mcmc_stopping", "fixed")
    out.setdefault("mcmc_min_iterations", max(40, int(out["mcmc_burn"]) + int(out["mcmc_thin"])))
    out.setdefault("mcmc_check_interval", max(int(out["mcmc_thin"]), 40))
    out.setdefault("mcmc_initial_iterations", int(out["mcmc_iterations"]))
    out.setdefault("mcmc_extend_iterations", int(out["mcmc_check_interval"]))
    out.setdefault("mcmc_max_iterations", int(out["mcmc_iterations"]))
    out.setdefault("mcmc_chains", 1)
    out.setdefault("mcmc_rhat_threshold", 1.01)
    out.setdefault("mcmc_min_ess", 100.0)
    out.setdefault("mcmc_forecast_stability_tolerance", 0.05)
    out.setdefault("mcmc_proposal_adaptation", "fixed")
    out.setdefault("mcmc_adaptation_iterations", int(out["mcmc_burn"]))
    out.setdefault("mcmc_adaptation_target_acceptance", 0.234)
    out.setdefault("leverage", leverage)
    out.setdefault("leverage_alignment", "lagged_return")
    out.setdefault("leverage_correlation_method", "pearson")
    out.setdefault("leverage_correlation_scope", "per_state_path")
    out.setdefault("state_innovation_distribution", "gaussian")
    out.setdefault("state_innovation_source", "posterior_ffbs_path")
    out.setdefault("state_innovation_standardization", "mean_std")
    out.setdefault("state_innovation_resampling", "iid")
    out.setdefault("state_innovation_coupling", "correlation_mixture")
    out.setdefault("sv_sigma_scale_method", "none")
    out.setdefault("sv_measurement_bias", "log_chi_square_theoretical")
    out.setdefault("latent_vol_persistence", "ar1")
    has_harx_anchor = "harx_ff6_vol_anchor" in candidate_id
    out.setdefault("vol_anchor_model", "ridge_harx_ff6" if has_harx_anchor else "none")
    out.setdefault(
        "vol_anchor_target",
        "current_latent_log_variance" if has_harx_anchor else "none",
    )
    out.setdefault("mean_state_scaling", "none")
    out.setdefault("vol_path_model", "none")
    out.setdefault("vol_path_blend", "none")
    out.setdefault("bdes_multiscale_k_star", 4)
    out.setdefault("bdes_multiscale_shock_coupling", "convex_state_independent_mix")
    out.setdefault(
        "measurement_equation",
        "empirical_log_squared_standardized_residual_bias_unwinsorized"
        if out["sv_measurement_bias"] == "empirical_standardized_residuals"
        else "log_chi_square_mean_corrected_unwinsorized"
        if unclipped_measurement
        else "log_chi_square_mean_corrected_winsorized",
    )
    out.setdefault(
        "parameter_sampler",
        "stationary_transformed_metropolis_hastings"
        if transformed
        else "bounded_random_walk_metropolis_hastings",
    )
    out.setdefault(
        "residual_tail_method",
        "raw_empirical_untruncated" if raw_innovations else "clipped_empirical",
    )
    out.setdefault("innovation_tail_splice", "none")
    out.setdefault("innovation_standardization", "mean_std")
    out.setdefault("innovation_resampling", "stationary_bootstrap")
    out.setdefault("innovation_conditioning", "none")
    out.setdefault("innovation_pool_source", "sample_standardized")
    out.setdefault("unclipped_sv_measurement", unclipped_measurement)
    out.setdefault("transformed_parameter_mcmc", transformed)
    out.setdefault("unclipped_empirical_innovations", raw_innovations)
    out.setdefault("clip_simulated_returns", True)
    out.setdefault("sv_research_baseline", False)
    out.setdefault("validation_status", "paired_80_portfolio_full_latent_mcmc_sv_overlay_candidate")
    return out


def _resolve_base_model(model_id: str) -> dict[str, Any]:
    spec = parse_compositional_spec(model_id)
    innovation = (
        {
            "model": "empirical_standardized_residuals",
            "resampling": spec.resampling,
            "tail": spec.tail_method,
        }
        if spec.innovation_model == "empirical"
        else {"model": spec.innovation_model, "resampling": None, "tail": "parametric"}
    )
    return {
        "family": "base",
        "source_id": model_id,
        "source_reference": {
            "path": "app/engine.py",
            "entrypoints": [
                "SimfolioEngine._fit_auto_forecast_mean",
                "SimfolioEngine._fit_auto_forecast_base",
                "SimfolioEngine._simulate_candidate_log_paths",
            ],
            "revision": BASE_SOURCE_REVISION,
            "sha256": BASE_SOURCE_SHA256,
        },
        "mean": {"ref": f"base.mean_models.{spec.mean_model}"},
        "volatility": {"ref": f"base.volatility_models.{spec.volatility_model}"},
        "innovation": innovation,
        "path_generator": "parametric_monte_carlo"
        if spec.is_parametric
        else (
            "stationary_bootstrap_standardized_residuals"
            if spec.resampling == "stationary_bootstrap"
            else "filtered_historical_simulation"
        ),
        "source_seed_contract": BASE_FACTORY_SEED_CONTRACT,
        "factory_seed_contract": BASE_FACTORY_SEED_CONTRACT,
        "source_seed_context_ref": "seed_identity.base_and_full_mcmc",
        "failure_semantics": {
            "short_history": "reject below 30 observations",
            "arch": "strict unless explicit fallback",
        },
    }


def _resolve_frontier_model(model_id: str) -> dict[str, Any]:
    return {
        "family": "frontier",
        "source_id": model_id,
        "source_reference": {
            "path": "source-research/app/engine.py",
            "entrypoints": [
                "SimfolioEngine._fit_bdes_non_mcmc_sv_forecast_base",
                "_simulate_full_mcmc_sv_bdes_log_paths_serial_numba",
            ],
            "revision": BASE_SOURCE_REVISION,
            "sha256": BASE_SOURCE_SHA256,
            "dependency_artifacts": [
                {
                    "path": "asset_level_full_exact_crps.py",
                    "sha256": "5beb318b918ea367f7048d71a4e53e1bcc9b343faccda2ef493ddcf81bf3d465",
                    "entrypoints": ["_asset_terminal_matrix"],
                },
                {
                    "path": "simfolio_oos_copula_alternatives.py",
                    "sha256": "413d2ca7f74cda13dd228c8974f822ce23e690cc56e098babf3fd2c121fbf95e",
                    "entrypoints": ["_exact_kalman_rejoined_portfolio_paths", "_map_uniforms_to_marginal_paths", "_rebalanced_portfolio_log_paths_numba"],
                },
                {
                    "path": "simfolio_adaptive_pgas.py",
                    "sha256": "f7988a6cfbdf674c1efeb1ca6836b34e5f34c1ee241e97ef44551ded7ba87c69",
                    "entrypoints": ["_fit_dynamic_factor_model", "_kalman_terminal_posterior_numba_information_small", "_simulate_future_gaussian_uniforms_flat_numba"],
                },
            ],
        },
        "marginal": {
            "ref": "frontier.marginal_parameterization",
            "candidate": copy.deepcopy(FRONTIER_CANDIDATE),
        },
        "dependence": {"ref": "frontier.dependence_parameterization"},
        "portfolio_rejoin": {"ref": "frontier.portfolio_rejoin"},
        "source_seed_contract": FRONTIER_FACTORY_SEED_CONTRACT,
        "factory_seed_contract": FRONTIER_FACTORY_SEED_CONTRACT,
        "source_seed_context_ref": "seed_identity.frontier",
        "failure_semantics": {
            "calendar": "strict source business-day rejoin",
            "paths": "reject nonfinite output",
        },
    }


def build_resource() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MCMC_MANIFEST_PATH.read_text(encoding="utf-8"))
    inla_asset_fragment = _load_inla_asset_fragment()
    shared_components = {
        "base": _base_components(),
        "frontier": _frontier_definition(),
        "full_mcmc_sv": {"contract": _mcmc_contract()},
        "seed_identity": _seed_identity(),
        "portfolio": _portfolio_shared_components(),
    }
    _merge_inla_asset_components(shared_components, inla_asset_fragment)
    rows = ledger["models"]
    base_rows = [row for row in rows if row["model_family"] == "base"]
    mcmc_rows = [row for row in rows if row["model_family"] == "bayesian_sbb_full_mcmc_sv_overlay"]
    frontier_rows = [
        row
        for row in rows
        if row["public_model_id"] == "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced"
    ]
    mcmc_by_id = {
        str(row["id"]): {
            "source_candidate": copy.deepcopy(row),
            "resolved_candidate": _resolve_mcmc_candidate(dict(row)),
        }
        for row in manifest["entries"]
    }
    bindings: dict[str, dict[str, Any]] = {}
    resolved: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        model_id = str(row["public_model_id"])
        definition = _bind_shared_component_digests(
            _resolve_base_model(model_id), shared_components
        )
        bindings[model_id] = {
            "family": "base",
            "component_refs": definition["shared_component_refs"],
            "seed_contract": definition["factory_seed_contract"],
            "source_reference": definition["source_reference"],
            "definition_fingerprint": _digest(definition),
        }
        resolved[model_id] = definition
    for row in frontier_rows:
        model_id = str(row["public_model_id"])
        definition = _bind_shared_component_digests(
            _resolve_frontier_model(model_id), shared_components
        )
        bindings[model_id] = {
            "family": "frontier",
            "component_refs": definition["shared_component_refs"],
            "seed_contract": definition["factory_seed_contract"],
            "source_reference": definition["source_reference"],
            "definition_fingerprint": _digest(definition),
        }
        resolved[model_id] = definition
    for row in mcmc_rows:
        model_id = str(row["public_model_id"])
        candidate_record = mcmc_by_id.get(model_id)
        if candidate_record is None:
            raise ValueError(f"MCMC row missing from source manifest: {model_id}")
        definition = {
            "family": "full_mcmc_sv",
            "source_id": model_id,
            "source_reference": {
                "path": "source-research/scripts/forecast_oos_research_gate.py",
                "entrypoints": [
                    "_fit_bayesian_sbb_full_mcmc_sv_overlay",
                    "_apply_full_mcmc_sv_simulation_options",
                    "_simulate_bayesian_sbb_full_mcmc_sv_overlay",
                ],
                "revision": MCMC_SOURCE_REVISION,
                "sha256": MCMC_SOURCE_SHA256,
            },
            "source_candidate": candidate_record["source_candidate"],
            "resolved_candidate": candidate_record["resolved_candidate"],
            "source_seed_contract": FULL_MCMC_FACTORY_SEED_CONTRACT,
            "factory_seed_contract": FULL_MCMC_FACTORY_SEED_CONTRACT,
            "source_seed_context_ref": "seed_identity.base_and_full_mcmc",
            "fit_contract": {"ref": "full_mcmc_sv.contract"},
            "failure_semantics": {"ref": "full_mcmc_sv.contract.failure"},
        }
        definition = _bind_shared_component_digests(definition, shared_components)
        bindings[model_id] = {
            "family": "full_mcmc_sv",
            "source_manifest_ref": "resources/catalogs/canonical_40_full_mcmc_sv_specs.json",
            "component_refs": definition["shared_component_refs"],
            "seed_contract": definition["factory_seed_contract"],
            "source_reference": definition["source_reference"],
            "definition_fingerprint": _digest(definition),
        }
        resolved[model_id] = definition
    portfolio_rows = {
        str(row["public_model_id"]): row
        for row in rows
        if str(row["public_model_id"]) in _PORTFOLIO_MODEL_IDS
    }
    if set(portfolio_rows) != set(_PORTFOLIO_MODEL_IDS):
        missing = sorted(set(_PORTFOLIO_MODEL_IDS) - set(portfolio_rows))
        raise ValueError(f"portfolio source rows missing from canonical ledger: {missing}")
    for model_id in _PORTFOLIO_MODEL_IDS:
        row = portfolio_rows[model_id]
        definition = _resolve_portfolio_model(row, shared_components)
        bindings[model_id] = {
            "family": definition["family"],
            "component_refs": definition["shared_component_refs"],
            "seed_contract": definition["factory_seed_contract"],
            "source_seed_contract": definition["source_seed_contract"],
            "source_reference": definition["source_reference"],
            "factory": definition["factory"],
            "definition_fingerprint": _digest(definition),
        }
        resolved[model_id] = definition

    fragment_model_ids = tuple(str(model_id) for model_id in inla_asset_fragment["accepted_model_ids"])
    fragment_rows = {
        str(row["public_model_id"]): row
        for row in rows
        if str(row["public_model_id"]) in fragment_model_ids
    }
    if set(fragment_rows) != set(fragment_model_ids):
        missing = sorted(set(fragment_model_ids) - set(fragment_rows))
        raise ValueError(f"INLA/asset source rows missing from canonical ledger: {missing}")
    for model_id in fragment_model_ids:
        definition = _resolve_fragment_model(
            fragment_rows[model_id], inla_asset_fragment, shared_components
        )
        bindings[model_id] = {
            "family": definition["family"],
            "component_refs": definition["shared_component_refs"],
            "seed_contract": definition["factory_seed_contract"],
            "source_reference": definition["source_reference"],
            "factory": definition["factory"],
            "definition_fingerprint": _digest(definition),
        }
        resolved[model_id] = definition

    ledger_model_ids = {str(row["public_model_id"]) for row in rows}
    if set(resolved) != ledger_model_ids:
        missing = sorted(ledger_model_ids - set(resolved))
        extra = sorted(set(resolved) - ledger_model_ids)
        raise ValueError(f"resolved canonical membership mismatch; missing={missing}, extra={extra}")
    extended_model_ids = sorted(set(_PORTFOLIO_MODEL_IDS) | set(fragment_model_ids))
    resource = {
        "schema_version": 1,
        "scope": "canonical_175_resolved_statistical_definitions",
        "portfolio_model_ids": extended_model_ids,
        "fingerprint": {
            "algorithm": "sha256(canonical JSON sorted keys, compact separators, UTF-8; includes shared_component_digests)",
            "shared_component_digest_algorithm": "sha256(canonical JSON sorted keys, compact separators, UTF-8) per dotted reference",
            "definition_excludes": [
                "historical_score",
                "rank",
                "protocol_fingerprint",
                "dataset_fingerprint",
                "panel_fingerprint",
            ],
        },
        "source_provenance": {
            "base_source_revision": BASE_SOURCE_REVISION,
            "base_source_code_sha256": BASE_SOURCE_SHA256,
            "mcmc_source_revision": MCMC_SOURCE_REVISION,
            "mcmc_source_script_sha256": MCMC_SOURCE_SHA256,
            "mcmc_candidate_manifest": "resources/catalogs/canonical_40_full_mcmc_sv_specs.json",
            "mcmc_candidate_manifest_sha256": _file_digest(MCMC_MANIFEST_PATH),
            "inla_asset_spec_fragment": {
                "path": "resources/specifications/canonical_inla_asset_spec_fragment.json",
                "sha256": _file_digest(INLA_ASSET_FRAGMENT_PATH),
                "accepted_model_ids": list(fragment_model_ids),
                "descriptor_policy": "fragment source_candidate fields remain raw; resolved defaults remain separate",
            },
            "historical_candidate_identity_reference": "retained academic publication_master_catalog_snapshot.json (source evidence; not package input)",
            "catalog_identity": {
                "historical_publication_catalog": {
                    "path": "docs/forecast_oos_publication_master_catalog.json",
                    "sha256": "b80e3e3c9616916f090b0449cdbe3a79f150340dd98846d1f40b0691dae537a8",
                    "retained_snapshot": "publication_master_catalog_snapshot.json",
                    "retained_snapshot_sha256": "b80e3e3c9616916f090b0449cdbe3a79f150340dd98846d1f40b0691dae537a8",
                    "descriptor_policy": "The exact sparse descriptors recovered from this historical catalog are the seed-bearing source_candidate fields; resolved defaults are separate.",
                },
                "canonical_run_binding_evidence": {
                    "manifest_path": "academic/whitepaper_oos_canonical_20260823_fastmap/canonical_manifest.json",
                    "report_path": "academic/whitepaper_oos_canonical_20260823_fastmap/full_panel_report.md",
                    "manifest_catalog_sha256": "b80e3e3c9616916f090b0449cdbe3a79f150340dd98846d1f40b0691dae537a8",
                    "manifest_catalog_snapshot": "publication_master_catalog_snapshot.json",
                    "report_catalog_path": "docs/forecast_oos_publication_master_catalog.json",
                    "result_archive": "academic/whitepaper_oos_canonical_20260823_fastmap/full_panel_results.json.gz",
                    "result_archive_sha256": "1244270ed1e637f55d776efab2d1c3ad8f498d63cac16fc808b22cf4343d061a",
                    "result_metadata_path": "meta.whitepaper_catalog.catalog_path",
                    "result_mapping_path": "outputs/019f1c90-82e5-72b0-88d6-1149e944528e/bdes_cagr_recovery_package/academic/replication_audit/appendix_b_model_id_mapping_210.csv",
                    "result_mapping_sha256": "cf521190d8656f1661f7b15e55540031209890c2384f358e5029855b9819587e",
                    "wrapper_source": "tmp/oos_all_daily_exact_crps.py::_configure_whitepaper_catalog",
                    "wrapper_source_sha256": "88aaa532d8a0a63e24964305a26ee6fa45d0d9099dd5da800ab691afad583ea7",
                    "wrapper_catalog_section": "full_current_catalog",
                    "owned_descriptor_count": 40,
                    "exact_row_match_verified": True,
                },
            "current_source_core_catalog": {
                    "path": "scripts/forecast_oos_research_gate.py::_core_catalog(include_slow=True)",
                    "source_revision": MCMC_SOURCE_REVISION,
                    "source_script_sha256": MCMC_SOURCE_SHA256,
                    "status": "separate current-core evidence; its expanded defaults and current INLA baseline must not replace the historical source_candidate descriptors",
                    "comparison": "The 39 shared MCMC IDs have different fit-seed signatures from the retained historical descriptors; the historical MCMC baseline is absent from the current core catalog.",
                },
                "runtime_binding_status": "The retained result metadata records the catalog path; that file matches the retained snapshot byte-for-byte, and all 40 owned descriptors exact-match its full_current_catalog rows. The current source _core_catalog output is separate evidence and is not used by the retained historical wrapper.",
                "portfolio_source_modules": {
                    module_name: {
                        "module": closure["module"],
                        "source_revision": closure["source_revision"],
                        "source_functions_sha256": closure["source_functions_sha256"],
                    }
                    for module_name, closure in shared_components["portfolio"]["source_closures"].items()
                },
            },
        },
        "shared_components": shared_components,
        "resolved_definitions": resolved,
        "bindings": bindings,
        "accepted_model_ids": sorted(resolved),
    }
    return resource, resolved


def patch_ledger(resolved: dict[str, dict[str, Any]]) -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    for row in ledger["models"]:
        model_id = str(row["public_model_id"])
        definition = resolved.get(model_id)
        if definition is None:
            continue
        row["structured_specification"] = {
            "schema_version": 1,
            "resource": "resources/specifications/canonical_statistical_specifications.json",
            "resolved_definition": definition,
        }
        row["specification_fingerprint"] = {
            "value": _digest(definition),
            "kind": "resolved_statistical_definition_sha256",
            "status": "full resolved statistical definition",
        }
        row["specification_recovered"] = True
        row["source_reference_verified"] = True
        artifact = row.get("source_artifact_digest")
        if not isinstance(artifact, dict):
            artifact = {}
            row["source_artifact_digest"] = artifact
        source_code = artifact.setdefault("source_code", {})
        if model_id in {
            r["public_model_id"] for r in ledger["models"] if r["model_family"] == "base"
        }:
            source_code["sha256"] = BASE_SOURCE_SHA256
        elif row["model_family"] == "bayesian_sbb_full_mcmc_sv_overlay":
            source_code["sha256"] = MCMC_SOURCE_SHA256
        else:
            source_code["sha256"] = BASE_SOURCE_SHA256
        factory = row.get("implementation_factory")
        if isinstance(factory, dict) and factory.get("callable"):
            factory["seed_contract"] = definition.get(
                "factory_seed_contract", factory.get("seed_contract")
            )
            mapped = ledger.get("implementation_factory_map", {}).get(model_id)
            if isinstance(mapped, dict):
                mapped["seed_contract"] = definition.get(
                    "factory_seed_contract", mapped.get("seed_contract")
                )
    ledger["identity_policy"]["full_statistical_specifications_confirmed"] = len(resolved)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def build_portfolio_ledger_patch(
    resource: dict[str, Any], ledger: dict[str, Any]
) -> dict[str, Any]:
    """Build the extension-row update without mutating the canonical ledger.

    The ledger owns membership, score, protocol, and dataset identity.  This
    patch contains only the resolved-specification and implementation metadata
    required to apply the already-audited source-backed portfolio rows.  A
    coordinator can apply it after registry wiring without risking a rewrite
    of unrelated canonical fields.
    """

    rows = {str(row["public_model_id"]): row for row in ledger["models"]}
    records: list[dict[str, Any]] = []
    for model_id in sorted(_PORTFOLIO_MODEL_IDS, key=lambda value: rows[value]["canonical_rank"]):
        row = rows[model_id]
        definition = resource["resolved_definitions"][model_id]
        binding = resource["bindings"][model_id]
        source = definition["source_reference"]
        closure = _shared_component(
            resource["shared_components"], definition["source_component_ref"]
        )
        existing_factory = row.get("implementation_factory")
        factory_name = (
            existing_factory.get("name")
            if isinstance(existing_factory, dict) and existing_factory.get("name")
            else f"{definition['factory']['module']}:{definition['factory']['class']}"
        )
        records.append(
            {
                "public_model_id": model_id,
                "canonical_rank": row["canonical_rank"],
                "historical_rank": row["historical_rank"],
                "model_family": row["model_family"],
                "structured_specification": {
                    "schema_version": 1,
                    "resource": "resources/specifications/canonical_statistical_specifications.json",
                    "resolved_definition": definition,
                },
                "specification_fingerprint": {
                    "value": binding["definition_fingerprint"],
                    "kind": "resolved_statistical_definition_sha256",
                    "status": "full resolved source-backed portfolio definition",
                },
                "specification_recovered": True,
                "source_reference_verified": True,
                "source_artifact_digest_update": {
                    "source_code": {
                        "path": source["path"],
                        "sha256": source["sha256"],
                        "extracted_functions_sha256": closure["source_functions_sha256"],
                        "status": "source_kernel_closure_verified; retained score identity unresolved",
                    },
                    "parameter_dictionary": {
                        "ledger_root_reference": "source_provenance.parameter_dictionary",
                        "sha256": "6364818645b005e6bbab981b69bcbd9d91023f69732ac8e862ffefbcd87d3574",
                        "status": "dictionary_only; full experiment closure unresolved",
                    },
                },
                "implementation_factory": {
                    "name": factory_name,
                    "callable": True,
                    "status": "source_kernel_parity_verified_pending_coordinator_registry_wiring",
                    "seed_contract": definition["factory_seed_contract"],
                },
                "required_dependencies": closure["runtime_dependencies"],
                "seed_identity": definition["source_seed_descriptor"],
                "verification_status": (
                    "source_daily_path_parity_verified_experiment_identity_unresolved"
                    if definition["resolved_candidate"].get("output_semantics")
                    == "daily log-return increments"
                    else "source_terminal_path_parity_verified_experiment_identity_unresolved"
                ),
                "notes": [
                    "Resolved defaults are complete for the extracted source closure and are kept separate from the exact raw descriptor.",
                    "Unknown IDs and numerical failures fail closed; no generic family fallback is permitted.",
                    "Historical score, protocol, dataset, and panel identity remain unresolved and are not changed by this patch.",
                ],
            }
        )
    return {
        "schema_version": "canonical-175-statistical-specifications-portfolio-patch-v2",
        "scope": f"exact {len(_PORTFOLIO_MODEL_IDS)} source-backed portfolio models, including canonical and stack extensions",
        "base_ledger_membership_digest": ledger["membership"]["membership_digest"],
        "resource": "resources/specifications/canonical_statistical_specifications.json",
        "resource_scope": resource["scope"],
        "source_artifact_policy": "all paths are repository-relative; no private or temporary paths are serialized",
        "raw_descriptor_policy": "source_candidate and source_seed_descriptor are immutable seed-bearing fields; resolved_candidate is separate",
        "preserved_ledger_fields": [
            "public_model_id",
            "canonical_rank",
            "historical_rank",
            "historical_score",
            "score_precision",
            "protocol_fingerprint",
            "dataset_fingerprint",
            "panel_fingerprint",
            "membership",
        ],
        "records": records,
    }


def render_readable(resource: dict[str, Any]) -> str:
    """Render one concise, source-linked section for every resolved model."""

    lines = [
        "# Canonical resolved statistical specifications",
        "",
        "Generated from `resources/specifications/canonical_statistical_specifications.json`.",
        "The machine-readable resource contains the complete definitions; this file keeps each accepted model's source identity, seed contract, and resolved component references visible to reviewers.",
        "The resource contains all 175 canonical definitions, with source-backed portfolio and asset extensions merged into the same fingerprinted component graph. `audit/statistical-specifications-portfolio-ledger-patch.json` is the generated extension metadata patch; canonical membership, scores, and protocol identity remain ledger-owned.",
        "Raw seed-bearing descriptors remain in `source_candidate` and `source_seed_descriptor`; resolved defaults are separate and never replace those fields.",
        "",
    ]
    bindings = resource["bindings"]
    resolved = resource["resolved_definitions"]
    for index, model_id in enumerate(resource["accepted_model_ids"], start=1):
        definition = resolved[model_id]
        binding = bindings[model_id]
        source = definition["source_reference"]
        lines.extend(
            [
                f"## {index:03d}. `{model_id}`",
                "",
                f"- Family: `{definition['family']}`",
                f"- Resolved-definition SHA-256: `{binding['definition_fingerprint']}`",
                f"- Source: `{source['path']}` at `{source['revision']}` (SHA-256 `{source['sha256']}`).",
                f"- Source entrypoints: {', '.join(f'`{item}`' for item in source['entrypoints'])}.",
                f"- Factory seed contract: `{definition['factory_seed_contract']}`.",
                "- Runner checkpoint contract: `origin_task.seed_to_forecast_context.seed.v1`.",
                f"- Source seed context: `{definition['source_seed_context_ref']}`.",
            ]
        )
        if definition["family"] == "base":
            lines.extend(
                [
                    f"- Mean component: `{definition['mean']['ref']}`.",
                    f"- Volatility component: `{definition['volatility']['ref']}`.",
                    f"- Innovation: `{json.dumps(definition['innovation'], sort_keys=True)}`.",
                    f"- Path generator: `{definition['path_generator']}`.",
                ]
            )
        elif definition["family"] == "frontier":
            lines.extend(
                [
                    f"- Marginal candidate: `{definition['marginal']['candidate']['id']}`.",
                    f"- Dependence component: `{definition['dependence']['ref']}`.",
                    f"- Portfolio rejoin: `{definition['portfolio_rejoin']['ref']}`.",
                ]
            )
        else:
            source_candidate = definition["source_candidate"]
            resolved_candidate = definition.get("resolved_candidate", definition.get("resolved_defaults"))
            lines.extend(
                [
                    "- Source descriptor policy: exact recovered historical candidate fields below; resolved defaults are kept in a separate object and are not passed to the seed-bearing source descriptor.",
                    f"- Exact source descriptor fields: `{', '.join(sorted(source_candidate))}`.",
                    "- Resolved defaults:",
                ]
            )
            if resolved_candidate is None:
                raise ValueError(f"resolved candidate missing for {model_id}")
            for key in sorted(resolved_candidate):
                lines.append(
                    f"  - `{key}` = `{json.dumps(resolved_candidate[key], sort_keys=True)}`"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    resource, _resolved = build_resource()
    RESOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESOURCE_PATH.write_text(
        json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    READABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    READABLE_PATH.write_text(render_readable(resource), encoding="utf-8")
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    portfolio_patch = build_portfolio_ledger_patch(resource, ledger)
    portfolio_patch["resource_sha256"] = _file_digest(RESOURCE_PATH)
    portfolio_patch["generator_sha256"] = _file_digest(Path(__file__))
    PORTFOLIO_LEDGER_PATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_LEDGER_PATCH_PATH.write_text(
        json.dumps(portfolio_patch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(resource['resolved_definitions'])} resolved definitions and "
        f"{len(portfolio_patch['records'])} extension ledger updates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
