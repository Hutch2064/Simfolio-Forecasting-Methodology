"""Source-extracted canonical portfolio reference families.

The four executable families in this module are the small, explicit terminal
ensemble branches used by the retained research gate.  Their numerical
functions are extracted from the source-main ``SimfolioEngine`` methods and
keep the source's float64 operations, minimum-history rules, and NumPy RNG
calls.  The adapter exposes terminal ensembles directly because the retained
research gate scored these families at each requested horizon; it does not
invent a daily increment path from a terminal distribution.

The full-INLA BDES candidate is recorded as a blocked source candidate below.
The accepted FastMAP Frontier closure is a separate asset-level model and is
intentionally not aliased to this univariate candidate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import stats

from ...runner import (
    TERMINAL_FORECAST_SEMANTICS,
    ForecastContext,
    TrainingData,
)

# The source artifact is referenced only by repository-relative paths.  These
# digests are provenance metadata; they are not a claim that the retained
# scalar scores have been reproduced.
SOURCE_ARTIFACTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "engine": MappingProxyType(
            {
                "path": "source-main/app/engine.py",
                "sha256": "c2fcb7ad07ed94317d3102d69bd5160ed2a385703e69786ad0e2cedbf733cd4e",
            }
        ),
        "research_gate": MappingProxyType(
            {
                "path": "source-research/scripts/forecast_oos_research_gate.py",
                "sha256": "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997",
            }
        ),
    }
)

SOURCE_FUNCTION_NAMES: tuple[str, ...] = (
    "_deterministic_seed",
    "_constant_mean_gaussian_log_terminal_samples",
    "_iid_historical_bootstrap_log_terminal_samples",
    "_fit_constant_mean_student_t_log_params",
    "_constant_mean_student_t_log_paths",
    "_constant_mean_student_t_log_terminal_samples",
)

# SHA-256 over the exact AST-extracted source function segments, in the order
# above, each separated by one blank line and followed by a newline.
SOURCE_FUNCTIONS_SHA256 = "a76c48963e9bc8fcc3375e15542f38bb079f76d9df4b01b287d71594bf0b86a6"

SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_label, dense_horizon_tuple, public_model_id, simulations)"
)

BDES_BLOCKED_MODEL_ID = (
    "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_"
    "conditional_sharpe_full_inla_laplace_quadrature_centered_multiscale"
)

REFERENCE_MODEL_IDS: tuple[str, ...] = (
    "constant_mean_gaussian",
    "constant_mean_student_t",
    "naive_iid_historical_portfolio_bootstrap",
    "zero_mean_gaussian_vol_only",
)


# These dictionaries preserve the explicit candidate metadata recovered from
# the source catalogue.  They are intentionally small: unspecified source
# fields remain unresolved rather than being filled with defaults.
SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "constant_mean_gaussian": MappingProxyType(
            {
                "id": "constant_mean_gaussian",
                "type": "gaussian",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "constant_mean_gaussian_log_returns",
                "vol_model": "constant_sample_volatility",
                "innovation_method": "gaussian_iid_standardized_innovations",
                "tail_method": "native_distribution_tail",
                "path_generator": "parametric_monte_carlo",
                "candidate_role": "gaussian_shrinkage_component",
            }
        ),
        "constant_mean_student_t": MappingProxyType(
            {
                "id": "constant_mean_student_t",
                "type": "student_t",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "constant_mean_student_t_log_returns",
                "vol_model": "constant_sample_volatility",
                "innovation_method": "student_t_iid_standardized_innovations",
                "tail_method": "native_student_t_tail",
                "path_generator": "parametric_student_t_monte_carlo",
                "candidate_role": "heavy_tailed_shrinkage_component",
            }
        ),
        "naive_iid_historical_portfolio_bootstrap": MappingProxyType(
            {
                "id": "naive_iid_historical_portfolio_bootstrap",
                "type": "naive_iid_historical_portfolio_bootstrap",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "none",
                "vol_model": "none",
                "innovation_method": "iid_resampled_historical_portfolio_returns",
                "tail_method": "empirical_historical_tail",
                "path_generator": "iid_historical_portfolio_return_bootstrap",
                "candidate_role": "naive_benchmark_candidate",
            }
        ),
        "zero_mean_gaussian_vol_only": MappingProxyType(
            {
                "id": "zero_mean_gaussian_vol_only",
                "type": "zero_gaussian",
                "return_target": "portfolio_daily_log_return",
                "forecast_level": "portfolio_return",
                "mean_model": "zero",
                "vol_model": "constant_sample_volatility",
                "innovation_method": "gaussian_iid_standardized_innovations",
                "tail_method": "native_distribution_tail",
                "path_generator": "parametric_monte_carlo",
                "candidate_role": "zero_mean_volatility_reference",
            }
        ),
    }
)


class _SourceKernel:
    """The exact source-main numerical function closure.

    The six methods below were AST-extracted from ``app/engine.py``.  Keep
    them as a small class so the classmethod dispatch in the Student-t source
    method remains unchanged.
    """

    @staticmethod
    def _deterministic_seed(*parts: Any) -> int:
        digest = hashlib.blake2b(digest_size=8)
        for part in parts:
            digest.update(str(part).encode("utf-8"))
            digest.update(b"\0")
        return int.from_bytes(digest.digest(), "little") % (2 ** 32 - 1)

    @staticmethod
    def _constant_mean_gaussian_log_terminal_samples(
        train_log_returns: np.ndarray,
        horizon: int,
        n_sims: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        train_values = np.asarray(train_log_returns, dtype=np.float64)
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size < 30:
            return np.array([], dtype=np.float64)
        mu = float(np.mean(train_values))
        sigma = float(np.std(train_values, ddof=1)) if train_values.size > 1 else 0.0
        mean_h = float(mu * int(horizon))
        std_h = float(max(sigma, 0.0) * math.sqrt(float(horizon)))
        if std_h <= 1e-12:
            return np.full(int(max(1, n_sims)), mean_h, dtype=np.float64)
        return rng.normal(mean_h, std_h, size=int(max(1, n_sims))).astype(np.float64)

    @staticmethod
    def _iid_historical_bootstrap_log_terminal_samples(
        train_log_returns: np.ndarray,
        horizon: int,
        n_sims: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        train_values = np.asarray(train_log_returns, dtype=np.float64)
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size == 0:
            return np.array([], dtype=np.float64)
        horizon = int(max(1, horizon))
        n_sims = int(max(1, n_sims))
        sample_indices = rng.integers(0, train_values.size, size=(n_sims, horizon))
        return np.sum(train_values[sample_indices], axis=1, dtype=np.float64)

    @staticmethod
    def _fit_constant_mean_student_t_log_params(
        train_log_returns: np.ndarray,
    ) -> dict[str, float] | None:
        train_values = np.asarray(train_log_returns, dtype=np.float64)
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size < 30:
            return None
        mu = float(np.mean(train_values))
        sigma = float(np.std(train_values, ddof=1)) if train_values.size > 1 else 0.0
        if sigma <= 1e-12:
            return {
                "mu": mu,
                "sigma": 0.0,
                "df": 30.0,
                "excess_kurtosis": 0.0,
                "n_obs": float(train_values.size),
            }
        standardized = (train_values - mu) / sigma
        try:
            excess_kurtosis = float(stats.kurtosis(standardized, fisher=True, bias=False))
        except Exception:  # noqa: BLE001 - preserve the source kernel fallback
            excess_kurtosis = 0.0
        if np.isfinite(excess_kurtosis) and excess_kurtosis > 0.1:
            df = float(np.clip(4.0 + 6.0 / excess_kurtosis, 4.0, 30.0))
        else:
            df = 30.0
        return {
            "mu": mu,
            "sigma": sigma,
            "df": df,
            "excess_kurtosis": excess_kurtosis if np.isfinite(excess_kurtosis) else 0.0,
            "n_obs": float(train_values.size),
        }

    @classmethod
    def _constant_mean_student_t_log_paths(
        cls,
        train_log_returns: np.ndarray,
        total_days: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        params = cls._fit_constant_mean_student_t_log_params(train_log_returns)
        days = int(max(1, total_days))
        paths = int(max(1, n_paths))
        if params is None:
            return np.empty((0, days), dtype=np.float64)
        mu = float(params["mu"])
        sigma = float(params["sigma"])
        df = float(max(params["df"], 2.01))
        if sigma <= 1e-12:
            return np.full((paths, days), mu, dtype=np.float64)
        scale = sigma / math.sqrt(df / (df - 2.0))
        return (mu + scale * rng.standard_t(df, size=(paths, days))).astype(np.float64)

    @classmethod
    def _constant_mean_student_t_log_terminal_samples(
        cls,
        train_log_returns: np.ndarray,
        horizon: int,
        n_sims: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        paths = cls._constant_mean_student_t_log_paths(
            train_log_returns,
            total_days=int(horizon),
            n_paths=int(n_sims),
            rng=rng,
        )
        if paths.size == 0:
            return np.array([], dtype=np.float64)
        return np.sum(paths, axis=1, dtype=np.float64)


@dataclass(frozen=True)
class ReferencePortfolioModel:
    """Explicit source-backed adapter for one canonical portfolio model."""

    model_id: str

    # Runner/checkpoint metadata.  The source code digest is computed from the
    # installed module by runner._implementation_digest; this provenance digest
    # identifies the source fragments from which the closure was extracted.
    forecast_output_semantics: str = TERMINAL_FORECAST_SEMANTICS
    seed_contract: str = SOURCE_SEED_CONTRACT
    dependency_identity: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {
                "source_imports": ("hashlib", "math", "numpy", "scipy.stats"),
                "runtime_constraints": ("numpy>=2.0,<3", "scipy>=1.13,<2"),
            }
        )
    )

    def __post_init__(self) -> None:
        if self.model_id not in REFERENCE_MODEL_IDS:
            raise ValueError(f"unknown reference portfolio model: {self.model_id!r}")

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

    def _source_rng(self, context: ForecastContext, horizon_grid: tuple[int, ...]) -> np.random.Generator:
        # These are the exact seed call arguments at the source gate.  The
        # canonical runner's task seed remains part of the task identity, but
        # the retained source candidate RNG did not pass it to this call.
        seed = _SourceKernel._deterministic_seed(
            "forecast_oos_candidate",
            context.origin_label,
            horizon_grid,
            self.model_id,
            int(context.simulations),
        )
        return np.random.default_rng(seed)

    def simulate_terminal_log_returns(
        self,
        training: TrainingData,
        context: ForecastContext,
    ) -> np.ndarray:
        """Return direct terminal samples for every dense horizon 1..H."""

        training.validate()
        horizon = int(context.horizon_days)
        simulations = int(context.simulations)
        if horizon < 1 or simulations < 1:
            raise ValueError("reference forecast requires positive horizon and simulation count")
        horizon_grid = tuple(range(1, horizon + 1))
        rng = self._source_rng(context, horizon_grid)
        values = np.asarray(training.portfolio_log_returns, dtype=np.float64)

        if self.model_id == "constant_mean_gaussian":
            if values.size < 30:
                raise ValueError("constant_mean_gaussian requires at least 30 finite observations")
            columns = [
                _SourceKernel._constant_mean_gaussian_log_terminal_samples(
                    values, day, simulations, rng
                )
                for day in horizon_grid
            ]
            result = np.column_stack(columns)
        elif self.model_id == "constant_mean_student_t":
            paths = _SourceKernel._constant_mean_student_t_log_paths(
                values, total_days=horizon, n_paths=simulations, rng=rng
            )
            if paths.size == 0:
                raise ValueError("constant_mean_student_t requires at least 30 finite observations")
            result = np.cumsum(paths, axis=1, dtype=np.float64)
        elif self.model_id == "naive_iid_historical_portfolio_bootstrap":
            columns = [
                _SourceKernel._iid_historical_bootstrap_log_terminal_samples(
                    values, horizon=day, n_sims=simulations, rng=rng
                )
                for day in horizon_grid
            ]
            result = np.column_stack(columns)
        elif self.model_id == "zero_mean_gaussian_vol_only":
            train_values = values[np.isfinite(values)]
            sigma = float(np.std(train_values, ddof=1)) if train_values.size > 1 else 0.0
            result = np.column_stack(
                [
                    rng.normal(
                        0.0,
                        max(sigma, 0.0) * math.sqrt(float(day)),
                        size=simulations,
                    ).astype(np.float64)
                    for day in horizon_grid
                ]
            )
        else:  # pragma: no cover - guarded by __post_init__ and exact branches
            raise AssertionError(f"unhandled reference model: {self.model_id}")

        expected_shape = (simulations, horizon)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise ValueError(
                f"source terminal closure returned invalid shape or values: {result.shape}"
            )
        return np.asarray(result, dtype=np.float64)


def _factory(model_id: str):
    return lambda: ReferencePortfolioModel(model_id)


# Exact-ID map; no aliases or substring dispatch are permitted here.
REFERENCE_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {model_id: _factory(model_id) for model_id in REFERENCE_MODEL_IDS}
)


def make_reference_model(model_id: str) -> ReferencePortfolioModel:
    """Construct one explicitly mapped model or reject the requested ID."""

    if model_id == BDES_BLOCKED_MODEL_ID:
        raise ValueError(
            f"canonical model '{model_id}' is blocked: full-INLA numerical closure is not recovered"
        )
    factory = REFERENCE_FACTORIES.get(model_id)
    if factory is None:
        raise ValueError(f"unknown canonical reference model: {model_id!r}")
    return factory()


__all__ = [
    "BDES_BLOCKED_MODEL_ID",
    "REFERENCE_FACTORIES",
    "REFERENCE_MODEL_IDS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_NAMES",
    "ReferencePortfolioModel",
    "make_reference_model",
]
