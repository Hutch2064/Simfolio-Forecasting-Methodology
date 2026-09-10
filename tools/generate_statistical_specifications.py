"""Generate the resolved statistical-definition resource and ledger patch.

The canonical score catalogue remains the membership authority.  This utility
only resolves numerical defaults for the accepted base, Frontier, and full
MCMC-SV rows, computes a digest over each complete definition, and updates the
corresponding specification fields in ``canonical_175/ledger.json``.

The source manifest is intentionally kept separate from this resource.  It is
the row-level historical candidate identity; this file adds the defaults and
failure semantics needed to execute that candidate without changing its ID.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from simfolio_forecasting_methodology.models.numerical.bdes_fastmap import (
    FRONTIER_CANDIDATE,
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

BASE_SOURCE_REVISION = "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
BASE_SOURCE_SHA256 = "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
MCMC_SOURCE_REVISION = "511fb82c0be43564b79df3694ee570677f3137ed"
MCMC_SOURCE_SHA256 = "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997"


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
        "factory_seed_contract": "origin_task.seed_to_forecast_context.seed.v1",
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
        "factory_seed_contract": "origin_task.seed_to_forecast_context.seed.v1",
        "source_seed_context_ref": "seed_identity.frontier",
        "failure_semantics": {
            "calendar": "strict source business-day rejoin",
            "paths": "reject nonfinite output",
        },
    }


def build_resource() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MCMC_MANIFEST_PATH.read_text(encoding="utf-8"))
    shared_components = {
        "base": _base_components(),
        "frontier": _frontier_definition(),
        "full_mcmc_sv": {"contract": _mcmc_contract()},
        "seed_identity": _seed_identity(),
    }
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
            "factory_seed_contract": "origin_task.seed_to_forecast_context.seed.v1",
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
    resource = {
        "schema_version": 1,
        "scope": "canonical_125_resolved_statistical_definitions",
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
            factory["seed_contract"] = "origin_task.seed_to_forecast_context.seed.v1"
            mapped = ledger.get("implementation_factory_map", {}).get(model_id)
            if isinstance(mapped, dict):
                mapped["seed_contract"] = "origin_task.seed_to_forecast_context.seed.v1"
    ledger["identity_policy"]["full_statistical_specifications_confirmed"] = len(resolved)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def render_readable(resource: dict[str, Any]) -> str:
    """Render one concise, source-linked section for every resolved model."""

    lines = [
        "# Canonical resolved statistical specifications",
        "",
        "Generated from `resources/specifications/canonical_statistical_specifications.json`.",
        "The machine-readable resource contains the complete definitions; this file keeps each accepted model's source identity, seed contract, and resolved component references visible to reviewers.",
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
                f"- Factory/checkpoint seed contract: `{definition['factory_seed_contract']}`.",
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
            resolved_candidate = definition["resolved_candidate"]
            lines.extend(
                [
                    "- Source descriptor policy: exact recovered historical candidate fields below; resolved defaults are kept in a separate object and are not passed to the seed-bearing source descriptor.",
                    f"- Exact source descriptor fields: `{', '.join(sorted(source_candidate))}`.",
                    "- Resolved defaults:",
                ]
            )
            for key in sorted(resolved_candidate):
                lines.append(
                    f"  - `{key}` = `{json.dumps(resolved_candidate[key], sort_keys=True)}`"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    resource, resolved = build_resource()
    RESOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESOURCE_PATH.write_text(
        json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    READABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    READABLE_PATH.write_text(render_readable(resource), encoding="utf-8")
    patch_ledger(resolved)
    print(f"wrote {len(resolved)} resolved definitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
