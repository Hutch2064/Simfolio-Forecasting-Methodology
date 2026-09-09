"""Readable model labels without mutating historical research identity.

`model_id` is immutable provenance. `public_name` is presentation metadata only.
This distinction lets old artifacts remain auditable while tables and command
output use concise, interpretable names.
"""

from __future__ import annotations

import re

from .catalog import CanonicalRow

_SPECIAL_NAMES = {
    "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced": (
        "Asset FastMAP + Dynamic Gaussian Factor"
    ),
    "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced": (
        "Asset Exact-Kalman + Dynamic Gaussian Factor"
    ),
    "canonical": "Canonical Gaussian Baseline",
    "naive_iid_historical_portfolio_bootstrap": "IID Historical Bootstrap",
    "constant_mean_gaussian": "Constant-Mean Gaussian",
    "constant_mean_student_t": "Constant-Mean Student-t",
    "zero_mean_gaussian_vol_only": "Zero-Mean Gaussian Volatility",
    "gas_score_driven_skewt": "Score-Driven Skew-t",
    "dp_mixture_sv_sbb": "DP-Mixture SV + Stationary Bootstrap",
    "observable_markov_state_sbb": "Observable Markov State + Stationary Bootstrap",
    "stochastic_volatility_ar1_empirical": "AR(1) SV + Empirical Innovations",
    "stochastic_volatility_ar1_empirical_sbb": "AR(1) SV + Stationary Bootstrap",
    "stochastic_volatility_ar1_student_t": "AR(1) SV + Student-t",
    "bayesian_mcmc_stochastic_volatility_sbb": "Bayesian MCMC SV + Stationary Bootstrap",
}

_TOKEN_LABELS = {
    "bic_auto_arma_mean": "BIC ARMA Mean",
    "expanding_sample_mean": "Expanding Mean",
    "factor_premium_near_zero_alpha_shrinkage": "Factor-Premium Shrunk Mean",
    "constant_sample_volatility": "Constant Volatility",
    "garch_1_1_volatility": "GARCH(1,1)",
    "gjr_tarch_1_1_volatility": "GJR/TARCH(1,1)",
    "egarch_1_1_volatility": "EGARCH(1,1)",
    "gaussian_iid_standardized_innovations": "Gaussian Innovations",
    "student_t_standardized_innovations": "Student-t Innovations",
    "skew_t_standardized_innovations": "Skew-t Innovations",
    "empirical": "Empirical Innovations",
    "iid": "IID Resampling",
    "stationary_bootstrap": "Stationary Bootstrap",
    "filtered_empirical_tail": "Filtered Empirical Tail",
    "automated_evt_pot_gpd_tail": "EVT POT/GPD Tail",
    "parametric": "Parametric Tail",
}

_REPLACEMENTS = (
    ("bayesian_sbb_overlay_", "Bayesian SBB + "),
    ("sv_live_baseline_", "SV Baseline: "),
    ("mcmc_sv_ar1_leverage", "MCMC SV-AR1 Leverage"),
    ("sv_ar1_logvol_bias_corrected", "Bias-Corrected SV-AR1"),
    ("gjr_garch_1_1", "GJR-GARCH(1,1)"),
    ("garch_1_1", "GARCH(1,1)"),
    ("egarch_1_1", "EGARCH(1,1)"),
    ("harch_1_5_22", "HARCH(1,5,22)"),
    ("figarch_1_d_1", "FIGARCH(1,d,1)"),
    ("ewma_absolute", "Absolute-EWMA"),
    ("empirical_bayes", "Empirical-Bayes"),
    ("hierarchical_eb", "Hierarchical-EB"),
    ("horizon_credibility", "Horizon-Credibility"),
    ("filtered_sbb", "Filtered SBB"),
    ("optimal_block", "Optimal Block"),
    ("adaptive_metropolis_proposal_mcmc", "Adaptive-Metropolis MCMC"),
    ("adaptive_mcmc", "Adaptive MCMC"),
)


def descriptive_name(model_id: str) -> str:
    """Return a concise description for a historical source identifier."""

    if model_id in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[model_id]
    if model_id.startswith("stack_canonical_"):
        suffix = model_id.removeprefix("stack_canonical_")
        return f"Canonical Distribution Stack ({suffix.replace('_', ' ')})"
    if "|" in model_id:
        return " · ".join(_TOKEN_LABELS.get(part, part.replace("_", " ").title()) for part in model_id.split("|"))

    text = model_id
    for source, label in _REPLACEMENTS:
        text = text.replace(source, label)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" :-")
    if len(text) > 96:
        text = text[:93].rstrip() + "..."
    return text


def public_name(row: CanonicalRow) -> str:
    """Stable compact label suitable for tables and CLI output."""

    return f"M{row.canonical_rank:03d} — {descriptive_name(row.model_id)}"
