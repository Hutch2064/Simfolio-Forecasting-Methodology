"""Source-backed score-driven skew-t portfolio reference family.

The fit and simulation functions in this module are extracted from the pinned
research gate.  The adapter keeps the gate's daily path construction and
returns its cumulative terminal ensemble by horizon; it does not recreate a
daily path from terminal samples.
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
    "_gas_t_score",
    "_fit_gas_score_driven_skewt",
    "_simulate_gas_score_driven_skewt",
)

# SHA-256 over the exact AST-extracted source function segments above, in the
# order listed, joined by one blank line and followed by a newline.
SOURCE_FUNCTIONS_SHA256 = "51849e868f01149e6011e68cad3f7745622b44a0cd6763030c763053ae2ba01b"

SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, public_model_id, simulations)"
)

REFERENCE_MODEL_IDS: tuple[str, ...] = ("gas_score_driven_skewt",)

# This is the exact retained publication snapshot descriptor.  The numerical
# defaults used by the source closure remain in the extracted functions; no
# current expanded catalogue fields are substituted here.
SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "gas_score_driven_skewt": MappingProxyType(
            {
                "id": "gas_score_driven_skewt",
                "type": "gas_score_driven_skewt",
                "feature_family": "gas_distributional_baseline",
                "mean_component": "near_zero_shrinkage_existing_candidate",
                "state_inference": "score_driven_observed_volatility",
            }
        )
    }
)


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


def _gas_t_score(z: np.ndarray, df: float) -> np.ndarray:
    """Source gate ``_gas_t_score``."""
    nu = float(max(df, 2.1))
    z2 = np.asarray(z, dtype=np.float64) ** 2
    return 0.5 * (((nu + 1.0) * z2) / np.maximum((nu - 2.0) + z2, 1e-8) - 1.0)


def _fit_gas_score_driven_skewt(train_values: np.ndarray, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """Source gate ``_fit_gas_score_driven_skewt``."""
    x = np.asarray(train_values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 504:
        return None
    max_fit_obs = int(max(candidate.get("max_fit_obs", 5000), 504))
    x = x[-max_fit_obs:]
    mu, meta = _sample_mean_near_zero_shrinkage(x)
    eps_x = (x - float(mu)) * 100.0
    long_var = float(np.var(eps_x, ddof=1))
    if not np.isfinite(long_var) or long_var <= 1e-8:
        return None
    standardized = eps_x / math.sqrt(long_var)
    try:
        from scipy import stats

        kurt = float(stats.kurtosis(standardized, fisher=True, bias=False))
    except Exception:  # noqa: BLE001 - preserve the source's explicit fallback
        kurt = 0.0
    df = float(np.clip(4.0 + 6.0 / max(kurt, 0.1), 4.0, 30.0)) if np.isfinite(kurt) and kurt > 0.1 else 12.0
    t_scale = math.sqrt((df - 2.0) / df)
    long_h = math.log(max(long_var, 1e-8))
    best: tuple[float, float, float, np.ndarray] | None = None
    for alpha in (0.03, 0.06, 0.10, 0.16):
        for beta in (0.85, 0.93, 0.97, 0.985):
            h = long_h
            ll = 0.0
            h_path = np.empty(eps_x.size, dtype=np.float64)
            for idx, eps in enumerate(eps_x):
                sigma = math.exp(0.5 * float(np.clip(h, -18.0, 18.0)))
                z = float(eps) / max(sigma, 1e-8)
                try:
                    from scipy import stats

                    ll += float(stats.t.logpdf(z / t_scale, df) - math.log(max(t_scale, 1e-8)) - math.log(max(sigma, 1e-8)))
                except Exception:  # noqa: BLE001 - preserve the source's explicit fallback
                    ll += -0.5 * (z * z + math.log(max(sigma * sigma, 1e-8)))
                score = float(_gas_t_score(np.asarray([z], dtype=np.float64), df)[0])
                h_path[idx] = h
                h = long_h + float(beta) * (h - long_h) + float(alpha) * score
            if best is None or ll > best[0]:
                best = (float(ll), float(alpha), float(beta), h_path)
    if best is None:
        return None
    _, alpha, beta, h_path = best
    sigma_path = np.exp(0.5 * np.clip(h_path, -18.0, 18.0))
    z_fit = eps_x / np.maximum(sigma_path, 1e-8)
    z_fit = z_fit[np.isfinite(z_fit)]
    z_fit = np.clip(z_fit, -20.0, 20.0)
    jf_params: tuple[float, float, float, float] | None = None
    try:
        from scipy import stats

        params = stats.jf_skew_t.fit(z_fit)
        jf_params = tuple(float(value) for value in params)
    except Exception:  # noqa: BLE001 - preserve the source's explicit fallback
        jf_params = None
    return {
        "mu": float(mu),
        "mean_meta": meta,
        "long_log_variance_x": float(long_h),
        "last_log_variance_x": float(h_path[-1]),
        "alpha": float(alpha),
        "beta": float(beta),
        "df": float(df),
        "t_scale": float(t_scale),
        "jf_skew_t_params": jf_params,
        "standardized_residuals": z_fit.astype(np.float64),
        "meta": {
            "method": "gas_score_driven_volatility_with_jones_faddy_skew_t_innovations",
            "distribution": "jf_skew_t",
            "max_fit_obs": int(max_fit_obs),
            "df_proxy": float(df),
            "alpha": float(alpha),
            "beta": float(beta),
        },
    }


def _simulate_gas_score_driven_skewt(
    fit: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source gate ``_simulate_gas_score_driven_skewt``."""
    n_paths = int(n_paths)
    total_days = int(total_days)
    out = np.empty((n_paths, total_days), dtype=np.float64)
    h = np.full(n_paths, float(fit.get("last_log_variance_x", fit.get("long_log_variance_x", 0.0))), dtype=np.float64)
    long_h = float(fit.get("long_log_variance_x", 0.0))
    alpha = float(np.clip(fit.get("alpha", 0.06), 0.0, 0.5))
    beta = float(np.clip(fit.get("beta", 0.93), 0.0, 0.999))
    df = float(max(fit.get("df", 8.0), 2.1))
    mu = float(fit.get("mu", 0.0))
    z_pool = np.asarray(fit.get("standardized_residuals"), dtype=np.float64)
    z_pool = z_pool[np.isfinite(z_pool)]
    if z_pool.size == 0:
        z_pool = np.array([0.0], dtype=np.float64)
    jf_params = fit.get("jf_skew_t_params")
    for day in range(total_days):
        sigma = np.exp(0.5 * np.clip(h, -18.0, 18.0)) / 100.0
        if jf_params is not None:
            try:
                from scipy import stats

                z = np.asarray(stats.jf_skew_t.rvs(*jf_params, size=n_paths, random_state=rng), dtype=np.float64)
            except Exception:  # noqa: BLE001 - preserve the source's explicit fallback
                z = rng.choice(z_pool, size=n_paths, replace=True)
        else:
            z = rng.choice(z_pool, size=n_paths, replace=True)
        z = np.clip(np.where(np.isfinite(z), z, 0.0), -20.0, 20.0)
        out[:, day] = mu + sigma * z
        score = _gas_t_score(z, df)
        h = long_h + beta * (h - long_h) + alpha * score
    return np.clip(out, -1.0, 1.0)


@dataclass(frozen=True)
class GasScoreDrivenSkewTModel:
    """Exact-ID adapter for the retained score-driven skew-t candidate."""

    model_id: str
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
            raise ValueError(f"unknown gas reference model: {self.model_id!r}")

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
        fit = _fit_gas_score_driven_skewt(values, SOURCE_CANDIDATE_SPECS[self.model_id])
        if fit is None:
            raise ValueError(f"{self.model_id} requires at least 504 finite observations")
        paths = _simulate_gas_score_driven_skewt(
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
    {model_id: (lambda model_id=model_id: GasScoreDrivenSkewTModel(model_id)) for model_id in REFERENCE_MODEL_IDS}
)


def make_gas_reference_model(model_id: str) -> GasScoreDrivenSkewTModel:
    """Construct one exact-ID source-backed gas model."""
    factory = REFERENCE_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"no source-backed gas factory for canonical model {model_id!r}")
    return factory()


__all__ = [
    "REFERENCE_FACTORIES",
    "REFERENCE_MODEL_IDS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "GasScoreDrivenSkewTModel",
    "_fit_gas_score_driven_skewt",
    "_gas_t_score",
    "_simulate_gas_score_driven_skewt",
    "make_gas_reference_model",
]
