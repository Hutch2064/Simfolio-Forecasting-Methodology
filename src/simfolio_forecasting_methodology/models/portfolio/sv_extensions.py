"""Source-backed state-transition volatility extensions.

The two candidates here are the explicit observable-Markov and truncated
Dirichlet-process SV branches from the retained research gate.  They produce
daily simulated paths, which are accumulated to the terminal ensemble with
the same semantics as the source gate.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ...runner import TERMINAL_FORECAST_SEMANTICS, ForecastContext, TrainingData
from .sv_reference import _sample_mean_near_zero_shrinkage

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
    "_categorical_next_states",
    "_fit_observable_markov_state_sbb",
    "_simulate_observable_markov_state_sbb",
    "_fit_dp_mixture_sv_sbb",
    "_simulate_dp_mixture_sv_sbb",
    "_sample_mean_near_zero_shrinkage",
    "_deterministic_seed",
)
SOURCE_FUNCTIONS_SHA256 = "84a78bd1bc1f2bf9195be6682fd72429276547602061e31acbd2ac76546a2985"

SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, public_model_id, simulations)"
)

REFERENCE_MODEL_IDS: tuple[str, ...] = (
    "dp_mixture_sv_sbb",
    "observable_markov_state_sbb",
)

SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "dp_mixture_sv_sbb": MappingProxyType(
            {
                "id": "dp_mixture_sv_sbb",
                "type": "dp_mixture_sv_sbb",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "sample_mean_near_zero_shrinkage",
                "vol_model": "truncated_dirichlet_process_mixture_stochastic_volatility_proxy",
                "innovation_method": "state_conditioned_empirical_standardized_residuals",
                "tail_method": "state_conditioned_empirical_tail",
                "path_generator": "dirichlet_process_mixture_stochastic_volatility_stationary_bootstrap",
                "max_components": 6,
                "max_fit_obs": 5000,
                "candidate_role": "stochastic_volatility_extension",
            }
        ),
        "observable_markov_state_sbb": MappingProxyType(
            {
                "id": "observable_markov_state_sbb",
                "type": "observable_markov_state_sbb",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "state_conditioned_observable_markov_bootstrap",
                "vol_model": "observable_rolling_volatility_tertiles",
                "innovation_method": "state_conditioned_empirical_residuals",
                "tail_method": "state_conditioned_empirical_tail",
                "path_generator": "observable_markov_chain_state_specific_bootstrap",
                "state_count": 12,
                "state_alpha": 0.5,
                "candidate_role": "state_transition_volatility_extension",
            }
        ),
    }
)


def _categorical_next_states(
    transition_matrix: np.ndarray,
    current_states: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source gate `_categorical_next_states`."""
    transition = np.asarray(transition_matrix, dtype=np.float64)
    states = np.asarray(current_states, dtype=np.int64)
    cdf = np.cumsum(transition[states], axis=1)
    draws = rng.random(states.size)
    return np.sum(draws[:, None] > cdf, axis=1).astype(np.int64)


def _fit_observable_markov_state_sbb(
    train: pd.Series,
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Source gate `_fit_observable_markov_state_sbb`."""
    series = pd.Series(train).dropna().astype(float)
    if len(series) < 504:
        return None
    r = series.to_numpy(dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 504:
        return None
    vol_proxy = pd.Series(np.abs(r)).rolling(21, min_periods=10).mean().to_numpy(dtype=np.float64)
    trend_proxy = pd.Series(r).rolling(63, min_periods=21).sum().to_numpy(dtype=np.float64)
    wealth = np.exp(np.cumsum(r))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    valid = np.isfinite(vol_proxy) & np.isfinite(trend_proxy) & np.isfinite(drawdown)
    if int(np.sum(valid)) < 252:
        return None
    vol_edges = np.quantile(vol_proxy[valid], [1.0 / 3.0, 2.0 / 3.0])
    dd_edge = float(np.quantile(drawdown[valid], 0.25))
    vol_state = np.digitize(vol_proxy, vol_edges, right=False)
    trend_state = (trend_proxy > 0.0).astype(int)
    drawdown_state = (drawdown < dd_edge).astype(int)
    states = (vol_state + 3 * trend_state + 6 * drawdown_state).astype(int)
    states = states[valid]
    returns = r[valid]
    state_count = 12
    alpha = float(max(candidate.get("state_alpha", 0.5), 1e-6))
    counts = np.full((state_count, state_count), alpha, dtype=np.float64)
    for left, right in pairwise(states):
        if 0 <= int(left) < state_count and 0 <= int(right) < state_count:
            counts[int(left), int(right)] += 1.0
    transition = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    global_pool = returns[np.isfinite(returns)]
    if global_pool.size < 20:
        return None
    state_pools = []
    for state in range(state_count):
        pool = returns[states == state]
        pool = pool[np.isfinite(pool)]
        if pool.size < 12:
            pool = global_pool
        state_pools.append(np.clip(pool.astype(np.float64), -1.0, 1.0))
    return {
        "transition_matrix": transition.astype(np.float64),
        "initial_state": int(states[-1]),
        "state_pools": state_pools,
        "global_pool": np.clip(global_pool.astype(np.float64), -1.0, 1.0),
        "state_count": state_count,
        "meta": {
            "method": "observable_markov_chain_state_specific_bootstrap",
            "state_definition": "volatility_tertile_x_trend_sign_x_drawdown_quartile",
            "transition_estimator": "dirichlet_smoothed_transition_count_mle",
            "state_alpha": alpha,
        },
    }


def _simulate_observable_markov_state_sbb(
    fit: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source gate `_simulate_observable_markov_state_sbb`."""
    transition = np.asarray(fit.get("transition_matrix"), dtype=np.float64)
    state_count = int(transition.shape[0]) if transition.ndim == 2 else 0
    if state_count <= 0:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    states = np.full(int(n_paths), int(np.clip(fit.get("initial_state", 0), 0, state_count - 1)), dtype=np.int64)
    out = np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    pools = list(fit.get("state_pools") or [])
    global_pool = np.asarray(fit.get("global_pool", [0.0]), dtype=np.float64)
    global_pool = global_pool[np.isfinite(global_pool)]
    if global_pool.size == 0:
        global_pool = np.array([0.0], dtype=np.float64)
    for _day in range(int(total_days)):
        states = _categorical_next_states(transition, states, rng)
        values = np.empty(int(n_paths), dtype=np.float64)
        for state in range(state_count):
            mask = states == state
            if not np.any(mask):
                continue
            pool = np.asarray(pools[state] if state < len(pools) else global_pool, dtype=np.float64)
            pool = pool[np.isfinite(pool)]
            if pool.size == 0:
                pool = global_pool
            values[mask] = rng.choice(pool, size=int(np.sum(mask)), replace=True)
        out[:, _day] = values
    return np.clip(out, -1.0, 1.0)


def _fit_dp_mixture_sv_sbb(
    train_values: np.ndarray,
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Source gate `_fit_dp_mixture_sv_sbb`."""
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 504:
        return None
    max_fit_obs = int(max(candidate.get("max_fit_obs", 5000), 504))
    x = x[-max_fit_obs:]
    mu, meta = _sample_mean_near_zero_shrinkage(x)
    eps_x = (x - float(mu)) * 100.0
    if float(np.std(eps_x, ddof=1)) <= 1e-8:
        return None
    log_abs = np.log(np.maximum(np.abs(eps_x) * math.sqrt(math.pi / 2.0), 1e-8))
    log_abs = np.clip(log_abs, *np.quantile(log_abs, [0.01, 0.99]))
    try:
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import BayesianGaussianMixture

        max_components = int(np.clip(candidate.get("max_components", 6), 2, 10))
        model = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.5,
            max_iter=120,
            random_state=_deterministic_seed("dp_mixture_sv", len(x), max_components),
            reg_covar=1e-5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(log_abs.reshape(-1, 1))
        raw_labels = model.predict(log_abs.reshape(-1, 1)).astype(int)
    except Exception:
        logging.getLogger(__name__).warning("Using retained source failure rule", exc_info=True)
        edges = np.quantile(log_abs, [0.2, 0.4, 0.6, 0.8])
        raw_labels = np.digitize(log_abs, edges, right=False).astype(int)
    unique = sorted(int(value) for value in np.unique(raw_labels))
    mapping = {state: idx for idx, state in enumerate(unique)}
    labels = np.asarray([mapping[int(value)] for value in raw_labels], dtype=np.int64)
    k = len(unique)
    if k <= 0:
        return None
    alpha = 0.5
    counts = np.full((k, k), alpha, dtype=np.float64)
    for left, right in pairwise(labels):
        counts[int(left), int(right)] += 1.0
    transition = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    global_sigma_x = float(max(np.std(eps_x, ddof=1), 1e-6))
    global_z = eps_x / global_sigma_x
    state_sigmas = np.empty(k, dtype=np.float64)
    state_pools: list[np.ndarray] = []
    for state in range(k):
        mask = labels == state
        state_eps = eps_x[mask]
        sigma_x = float(np.sqrt(np.mean(state_eps * state_eps))) if state_eps.size >= 8 else global_sigma_x
        sigma_x = float(np.clip(sigma_x, global_sigma_x * 0.05, global_sigma_x * 20.0))
        state_sigmas[state] = sigma_x
        pool = state_eps / max(sigma_x, 1e-8) if state_eps.size >= 8 else global_z
        pool = pool[np.isfinite(pool)]
        if pool.size < 8:
            pool = global_z[np.isfinite(global_z)]
        state_pools.append(np.clip(pool.astype(np.float64), -20.0, 20.0))
    return {
        "mu": float(mu),
        "mean_meta": meta,
        "transition_matrix": transition.astype(np.float64),
        "initial_state": int(labels[-1]),
        "state_sigmas_x": state_sigmas.astype(np.float64),
        "state_pools": state_pools,
        "active_component_count": k,
        "meta": {
            "method": "truncated_dirichlet_process_mixture_stochastic_volatility_proxy",
            "transition_estimator": "dirichlet_smoothed_mixture_state_transitions",
            "max_fit_obs": int(max_fit_obs),
            "active_component_count": k,
        },
    }


def _simulate_dp_mixture_sv_sbb(
    fit: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source gate `_simulate_dp_mixture_sv_sbb`."""
    transition = np.asarray(fit.get("transition_matrix"), dtype=np.float64)
    sigmas_x = np.asarray(fit.get("state_sigmas_x"), dtype=np.float64)
    k = int(transition.shape[0]) if transition.ndim == 2 else 0
    if k <= 0 or sigmas_x.size < k:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    states = np.full(int(n_paths), int(np.clip(fit.get("initial_state", 0), 0, k - 1)), dtype=np.int64)
    pools = list(fit.get("state_pools") or [])
    global_pool = np.concatenate([np.asarray(pool, dtype=np.float64) for pool in pools]) if pools else np.array([0.0])
    global_pool = global_pool[np.isfinite(global_pool)]
    if global_pool.size == 0:
        global_pool = np.array([0.0], dtype=np.float64)
    out = np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    mu = float(fit.get("mu", 0.0))
    for _day in range(int(total_days)):
        states = _categorical_next_states(transition, states, rng)
        z = np.empty(int(n_paths), dtype=np.float64)
        for state in range(k):
            mask = states == state
            if not np.any(mask):
                continue
            pool = np.asarray(pools[state] if state < len(pools) else global_pool, dtype=np.float64)
            pool = pool[np.isfinite(pool)]
            if pool.size == 0:
                pool = global_pool
            z[mask] = rng.choice(pool, size=int(np.sum(mask)), replace=True)
        out[:, _day] = mu + (sigmas_x[states] / 100.0) * z
    return np.clip(out, -1.0, 1.0)


def _deterministic_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


@dataclass(frozen=True)
class SuffixSVExtensionModel:
    """Exact-ID adapter for one source state-transition volatility model."""

    model_id: str
    forecast_output_semantics: str = TERMINAL_FORECAST_SEMANTICS
    seed_contract: str = SOURCE_SEED_CONTRACT
    dependency_identity: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {
                "source_imports": ("hashlib", "math", "numpy", "pandas", "scikit-learn"),
                "runtime_constraints": ("numpy>=2.0,<3", "pandas>=2.0,<4", "scikit-learn>=1.0,<2"),
            }
        )
    )

    def __post_init__(self) -> None:
        if self.model_id not in REFERENCE_MODEL_IDS:
            raise ValueError(f"unknown SV extension model: {self.model_id!r}")

    @property
    def family(self) -> str:
        return str(SOURCE_CANDIDATE_SPECS[self.model_id]["type"])

    @property
    def source_specification(self) -> Mapping[str, Any]:
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
        return np.random.default_rng(
            _deterministic_seed(
                "forecast_oos_candidate",
                source_origin,
                dense_horizon_tuple,
                self.model_id,
                int(context.simulations),
            )
        )

    def simulate_terminal_log_returns(
        self,
        training: TrainingData,
        context: ForecastContext,
    ) -> np.ndarray:
        training.validate()
        values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
        candidate = SOURCE_CANDIDATE_SPECS[self.model_id]
        if self.model_id == "observable_markov_state_sbb":
            fit = _fit_observable_markov_state_sbb(pd.Series(values), candidate)
            if fit is None:
                raise ValueError(f"{self.model_id} requires at least 504 finite observations")
            paths = _simulate_observable_markov_state_sbb(fit, context.horizon_days, context.simulations, self._source_rng(context))
        else:
            fit = _fit_dp_mixture_sv_sbb(values, candidate)
            if fit is None:
                raise ValueError(f"{self.model_id} requires at least 504 finite observations")
            paths = _simulate_dp_mixture_sv_sbb(fit, context.horizon_days, context.simulations, self._source_rng(context))
        terminals = np.cumsum(paths, axis=1, dtype=np.float64)
        expected_shape = (int(context.simulations), int(context.horizon_days))
        if terminals.shape != expected_shape or not np.all(np.isfinite(terminals)):
            raise ValueError(f"{self.model_id} produced an invalid terminal ensemble")
        return terminals


REFERENCE_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {model_id: (lambda model_id=model_id: SuffixSVExtensionModel(model_id)) for model_id in REFERENCE_MODEL_IDS}
)


def make_sv_extension_model(model_id: str) -> SuffixSVExtensionModel:
    """Construct one exact-ID source-backed SV extension model."""
    factory = REFERENCE_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"no source-backed SV extension factory for canonical model {model_id!r}")
    return factory()


__all__ = [
    "REFERENCE_FACTORIES",
    "REFERENCE_MODEL_IDS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "SuffixSVExtensionModel",
    "make_sv_extension_model",
]
