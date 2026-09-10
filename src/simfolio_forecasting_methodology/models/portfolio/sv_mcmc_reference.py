"""Source-backed Bayesian SV plus stationary-bootstrap reference family.

The retained ``sv_mcmc_sbb`` branch is a separate source candidate from the
larger full MCMC overlay family.  Its fit and path generator are kept as the
source gate defines them: a fixed random-walk posterior update over the AR(1)
log-variance state, followed by stationary-bootstrap standardized residuals.
The adapter returns cumulative terminal samples by horizon, matching the
source gate's terminal-ensemble semantics.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ...runner import TERMINAL_FORECAST_SEMANTICS, ForecastContext, TrainingData
from .sv_reference import (
    _fit_sv_ar1,
    _politis_white_block_length,
    _stationary_bootstrap_indices,
    _sv_initial_log_var,
    _sv_kalman_filter,
    _sv_observed_log_variance,
)

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
    "_deterministic_seed",
    "_sample_mean_near_zero_shrinkage",
    "_sv_observed_log_variance",
    "_sv_observed_log_variance_from_residuals",
    "_sv_kalman_filter",
    "_initial_sv_state_space_params",
    "_fit_sv_state_space_params",
    "_fit_sv_ar1_with_mean",
    "_fit_sv_ar1",
    "_sv_mcmc_log_posterior",
    "_fit_sv_mcmc_from_base",
    "_fit_sv_mcmc",
    "_sv_initial_log_var",
    "_politis_white_block_length",
    "_stationary_bootstrap_indices_from_draws_fast",
    "_stationary_bootstrap_indices_from_draws",
    "_stationary_bootstrap_indices",
    "_simulate_sv_mcmc_sbb",
)

# SHA-256 over the exact AST-extracted source segments above, in this order,
# joined by one blank line and followed by a newline.  The implementation
# imports the already extracted shared SV helpers; this digest records their
# source closure without broadening the public catalogue.
SOURCE_FUNCTIONS_SHA256 = "1ccb06cf25e440f7cdb815da6ee3bd0068ef67604f55d53af936916aaf3d406e"

SOURCE_SEED_CONTRACT = (
    "fit: blake2b-64-little-mod-2^32-1; args=('sv_mcmc_fit', finite_observation_count, "
    "round(finite_observation_mean, 10)); forecast: blake2b-64-little-mod-2^32-1; "
    "args=('forecast_oos_candidate', origin_date, dense_horizon_tuple, public_model_id, simulations)"
)

REFERENCE_MODEL_IDS: tuple[str, ...] = ("bayesian_mcmc_stochastic_volatility_sbb",)

# These are the explicit fields present for this row in the retained source
# gate's core catalogue.  Numerical defaults live in the extracted functions;
# current expanded catalogue descriptors are intentionally not substituted.
SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "bayesian_mcmc_stochastic_volatility_sbb": MappingProxyType(
            {
                "id": "bayesian_mcmc_stochastic_volatility_sbb",
                "type": "sv_mcmc_sbb",
                "innovation": "empirical",
            }
        )
    }
)

SV_LOG_CHI_SQUARE_VAR = (math.pi * math.pi) / 2.0


def _deterministic_seed(*parts: Any) -> int:
    """Pinned engine ``SimfolioEngine._deterministic_seed``."""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


def _sample_mean_near_zero_shrinkage(log_returns: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Pinned engine ``SimfolioEngine._sample_mean_near_zero_shrinkage``."""
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


def _sv_mcmc_log_posterior(y: np.ndarray, level: float, phi: float, eta: float) -> float:
    """Source gate ``_sv_mcmc_log_posterior``."""
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


def _fit_sv_mcmc_from_base(
    train_values: np.ndarray,
    base: dict[str, Any] | None,
    *,
    seed_label: str,
) -> dict[str, Any] | None:
    """Source gate fixed random-walk MCMC fit over an AR(1) SV base fit."""
    if base is None:
        return None
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 504:
        return None
    residual_center = float(base.get("residual_center", base["mu"]))
    y = np.asarray(base.get("observed_log_var"), dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 252:
        y, _ = _sv_observed_log_variance(x, residual_center)
    rng = np.random.default_rng(
        _deterministic_seed(seed_label, len(x), round(float(np.mean(x)), 10))
    )
    level = float(base["level"])
    phi = float(np.clip(base["phi"], 0.01, 0.985))
    eta = float(np.clip(base["eta_sd"] if base["eta_sd"] > 1e-4 else 0.35, 0.05, 2.0))
    current_lp = _sv_mcmc_log_posterior(y, level, phi, eta)
    samples: list[tuple[float, float, float]] = []
    iterations = 420
    burn = 120
    thin = 4
    accepted = 0
    for step in range(iterations):
        proposal_level = level + rng.normal(0.0, 0.03)
        proposal_phi = float(np.clip(phi + rng.normal(0.0, 0.012), 0.001, 0.994))
        proposal_eta = float(np.clip(eta * math.exp(rng.normal(0.0, 0.05)), 1e-4, 5.0))
        proposal_lp = _sv_mcmc_log_posterior(y, proposal_level, proposal_phi, proposal_eta)
        if math.log(max(float(rng.random()), 1e-300)) < proposal_lp - current_lp:
            level, phi, eta = proposal_level, proposal_phi, proposal_eta
            current_lp = proposal_lp
            accepted += 1
        if step >= burn and (step - burn) % thin == 0:
            samples.append((float(level), float(phi), float(eta)))
    if not samples:
        samples = [(float(base["level"]), float(base["phi"]), float(max(base["eta_sd"], 0.05)))]
    state_samples: list[tuple[float, float, float, float, float]] = []
    for sample_level, sample_phi, sample_eta in samples:
        _, filtered, filtered_var = _sv_kalman_filter(y, sample_level, sample_phi, sample_eta)
        last_mean = float(filtered[-1]) if filtered.size else float(base.get("last_state_mean", sample_level))
        last_var = float(filtered_var[-1]) if filtered_var.size else float(base.get("last_state_var", 0.0))
        state_samples.append((float(sample_level), float(sample_phi), float(sample_eta), last_mean, last_var))
    fitted = dict(base)
    fitted["posterior_samples"] = state_samples
    fitted["mcmc_iterations"] = iterations
    fitted["mcmc_acceptance_rate"] = float(accepted / max(iterations, 1))
    fitted["mcmc_loglikelihood_model"] = "kalman_log_chi_square_state_space"
    return fitted


def _fit_sv_mcmc(train_values: np.ndarray) -> dict[str, Any] | None:
    """Source gate ``sv_mcmc_sbb`` fit dispatch."""
    return _fit_sv_mcmc_from_base(train_values, _fit_sv_ar1(train_values), seed_label="sv_mcmc_fit")


def _simulate_sv_mcmc_sbb(
    fit: dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source gate MCMC-SBB daily path generator."""
    total_days = int(total_days)
    n_paths = int(n_paths)
    samples = list(fit.get("posterior_samples") or [(fit["level"], fit["phi"], max(fit["eta_sd"], 0.05))])
    sample_idx = rng.integers(0, len(samples), size=n_paths)
    levels = np.asarray([samples[int(idx)][0] for idx in sample_idx], dtype=np.float64)
    phis = np.asarray([samples[int(idx)][1] for idx in sample_idx], dtype=np.float64)
    etas = np.asarray([samples[int(idx)][2] for idx in sample_idx], dtype=np.float64)
    if len(samples[0]) >= 5:
        last_means = np.asarray([samples[int(idx)][3] for idx in sample_idx], dtype=np.float64)
        last_vars = np.asarray([max(samples[int(idx)][4], 0.0) for idx in sample_idx], dtype=np.float64)
        log_var = rng.normal(last_means, np.sqrt(last_vars)).astype(np.float64)
    else:
        log_var = _sv_initial_log_var(fit, n_paths, rng)
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
    out = np.empty((n_paths, total_days), dtype=np.float64)
    for day in range(total_days):
        log_var = levels + phis * (log_var - levels) + rng.normal(0.0, etas, size=n_paths)
        log_var = np.clip(log_var, -18.0, 18.0)
        sigma = np.exp(0.5 * log_var) / 100.0
        out[:, day] = mu + sigma * z_draws[:, day]
    return np.clip(out, -1.0, 1.0)


@dataclass(frozen=True)
class BayesianMCMCSVModel:
    """Exact-ID adapter for the retained Bayesian MCMC SV-SBB candidate."""

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
            raise ValueError(f"unknown Bayesian MCMC SV model: {self.model_id!r}")

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
        fit = _fit_sv_mcmc(values)
        if fit is None:
            raise ValueError(f"{self.model_id} requires at least 504 finite observations")
        paths = _simulate_sv_mcmc_sbb(
            fit,
            total_days=int(context.horizon_days),
            n_paths=int(context.simulations),
            rng=self._source_rng(context),
        )
        terminals = np.cumsum(paths, axis=1, dtype=np.float64)
        expected_shape = (int(context.simulations), int(context.horizon_days))
        if terminals.shape != expected_shape or not np.all(np.isfinite(terminals)):
            raise ValueError(f"{self.model_id} produced an invalid terminal ensemble")
        return terminals


REFERENCE_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {
        model_id: (lambda model_id=model_id: BayesianMCMCSVModel(model_id))
        for model_id in REFERENCE_MODEL_IDS
    }
)


def make_sv_mcmc_reference_model(model_id: str) -> BayesianMCMCSVModel:
    """Construct the exact-ID MCMC-SBB model; unknown IDs fail closed."""
    factory = REFERENCE_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"no source-backed Bayesian MCMC SV factory for canonical model {model_id!r}")
    return factory()


__all__ = [
    "REFERENCE_FACTORIES",
    "REFERENCE_MODEL_IDS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "BayesianMCMCSVModel",
    "make_sv_mcmc_reference_model",
]
