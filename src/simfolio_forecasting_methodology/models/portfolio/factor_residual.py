"""Source-backed FF6 factor residual stationary-bootstrap models.

The two function bodies below are extracted from the pinned research gate
without rewriting their regression, date alignment, factor/RF convention,
block-length rule, or daily path arithmetic.  The package boundary binds the
source engine calls to the shared offline factor snapshot and extracted
stationary-bootstrap kernel.

The research code is user-authorized and had no destination license.  This
attribution records provenance without inventing a license.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ...runner import ForecastContext, TrainingData
from ...seeds import deterministic_seed, forecast_oos_candidate_seed
from ..numerical.base_models import _politis_white_block_length, _stationary_bootstrap_indices
from ..numerical.factors import load_packaged_factor_frame
from .bayesian_vol import (
    _factor_model_frame_config,
    _fit_bayesian_sbb_vol_overlay,
    _fit_sklearn_regressor,
    _overlay_vol_multiplier_curve,
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
        "source_revision": MappingProxyType(
            {"revision": "773bc1c325559e6bf57a567f1d8bf473a3427fbc"}
        ),
    }
)

SOURCE_FUNCTION_NAMES: tuple[str, ...] = (
    "_fit_factor_residual_sbb",
    "_simulate_factor_residual_sbb",
)
SOURCE_FUNCTION_LINE_RANGES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"_fit_factor_residual_sbb": (7363, 7417), "_simulate_factor_residual_sbb": (7420, 7456)}
)
SOURCE_FUNCTIONS_SHA256 = "a2e1abb838a6f8eaf557c2383f0dcd5573c171e94d0022d493ef4f73fc9420ca"
SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; fit args=('factor_residual_sbb', factor_model, "
    "len(combined), len(factor_cols)); forecast args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, public_model_id, simulations)"
)

RAW_FACTOR_RESIDUAL_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "factor_ff6_ridge_residual_sbb_none": MappingProxyType(
            {
                "id": "factor_ff6_ridge_residual_sbb_none",
                "type": "factor_residual_sbb",
                "factor_model": "ff6",
                "residual_overlay": "none",
            }
        ),
        "factor_ff6_ridge_residual_sbb_absolute_ewma": MappingProxyType(
            {
                "id": "factor_ff6_ridge_residual_sbb_absolute_ewma",
                "type": "factor_residual_sbb",
                "factor_model": "ff6",
                "residual_overlay": "absolute_ewma",
            }
        ),
    }
)

RESOLVED_FACTOR_RESIDUAL_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        model_id: MappingProxyType(
            {
                "family": "factor_residual_sbb",
                "fit_function": "_fit_factor_residual_sbb",
                "minimum_finite_training_observations": 180,
                "factor_model": "ff6",
                "factor_columns": ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"],
                "risk_free_column": "RF",
                "alignment": "normalize dates, concat portfolio and factor frame, dropna, require final aligned date equals final training date",
                "regressor": {
                    "kind": "ridge",
                    "pipeline": ["StandardScaler", "Ridge"],
                    "alpha": 10.0,
                    "sample_weight": "recent_exponential_weights",
                    "half_life_days": 504.0,
                },
                "residual_definition": "portfolio_return - risk_free - ridge_factor_prediction",
                "block_length": "max(_politis_white_block_length(y_excess), _politis_white_block_length(residuals*residuals))",
                "block_length_floor": 1,
                "resampling": "stationary_bootstrap",
                "stationary_kernel": "_stationary_bootstrap_indices",
                "path_definition": "risk_free_draw + factor_prediction_draw + residual_draw, clipped to [-1, 1]",
                "residual_overlay": RAW_FACTOR_RESIDUAL_SPECS[model_id]["residual_overlay"],
                "seed": {
                    "fit": [
                        "factor_residual_sbb",
                        "factor_model",
                        "len(combined)",
                        "len(factor_cols)",
                    ],
                    "forecast": [
                        "forecast_oos_candidate",
                        "origin_date",
                        "dense_horizon_tuple",
                        "public_model_id",
                        "simulations",
                    ],
                },
            }
        )
        for model_id in RAW_FACTOR_RESIDUAL_SPECS
    }
)


class _FactorEngine:
    _deterministic_seed = staticmethod(deterministic_seed)

    @staticmethod
    def _load_french_factor_frame() -> pd.DataFrame:
        return load_packaged_factor_frame("ff6")


SimfolioEngine = _FactorEngine


def _fit_factor_residual_sbb(
    train: pd.Series,
    candidate: dict[str, Any],
    engine: Any,
    *,
    factor_frame: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    clean = pd.Series(train).dropna().astype(float)
    if len(clean) < 180:
        return None
    clean.index = pd.to_datetime(clean.index).normalize()
    factor_model = str(candidate.get("factor_model", "ff6")).lower()
    cfg = _factor_model_frame_config(engine, factor_model, factor_frame=factor_frame)
    if cfg is None or cfg.get("frame") is None:
        return None
    factors = cfg["frame"].copy()
    factors.index = pd.to_datetime(factors.index).normalize()
    factor_cols = list(cfg["factor_cols"])
    rf_col = str(cfg["rf_col"])
    combined = pd.concat([clean.rename("Rp"), factors[[rf_col, *factor_cols]]], axis=1).dropna()
    if len(combined) < 180 or pd.Timestamp(combined.index[-1]) != pd.Timestamp(clean.index[-1]):
        return None
    X = combined[factor_cols].to_numpy(dtype=np.float64)
    rf = combined[rf_col].to_numpy(dtype=np.float64)
    y_excess = combined["Rp"].to_numpy(dtype=np.float64) - rf
    seed = SimfolioEngine._deterministic_seed(
        "factor_residual_sbb", factor_model, len(combined), len(factor_cols)
    )
    model = _fit_sklearn_regressor("ridge", X, y_excess, seed)
    if model is None:
        return None
    predicted = np.asarray(model.predict(X), dtype=np.float64)
    residuals = y_excess - predicted
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size != len(combined) or float(np.std(residuals, ddof=1)) <= 1e-12:
        return None
    residual_overlay = str(candidate.get("residual_overlay", "none"))
    overlay_fit = None
    if residual_overlay == "absolute_ewma":
        overlay_fit = _fit_bayesian_sbb_vol_overlay(residuals, {"overlay_model": "ewma_absolute"})
    block_length = max(
        _politis_white_block_length(y_excess), _politis_white_block_length(residuals * residuals)
    )
    return {
        "model": model,
        "factor_values": X,
        "rf_values": rf,
        "residuals": residuals.astype(np.float64),
        "block_length": int(max(1, block_length)),
        "residual_overlay_fit": overlay_fit,
        "meta": {
            "method": "ridge_factor_model_with_joint_stationary_bootstrap_factor_residual_paths",
            "factor_model": factor_model,
            "factor_cols": factor_cols,
            "rf_col": rf_col,
            "residual_overlay": residual_overlay,
            "training_rows": len(combined),
        },
    }


def _simulate_factor_residual_sbb(
    fit: dict[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    factor_values = np.asarray(fit.get("factor_values"), dtype=np.float64)
    residuals = np.asarray(fit.get("residuals"), dtype=np.float64)
    rf_values = np.asarray(fit.get("rf_values"), dtype=np.float64)
    if factor_values.ndim != 2 or residuals.ndim != 1 or rf_values.ndim != 1:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    n = min(factor_values.shape[0], residuals.size, rf_values.size)
    if n <= 0:
        return np.empty((int(n_paths), int(total_days)), dtype=np.float64)
    idx = _stationary_bootstrap_indices(
        n=n,
        block_length=int(max(1, fit.get("block_length", 1))),
        total_days=int(total_days),
        n_paths=int(n_paths),
        rng=rng,
    )
    factor_draws = factor_values[idx]
    flat = factor_draws.reshape(-1, factor_values.shape[1])
    factor_pred = np.asarray(fit["model"].predict(flat), dtype=np.float64).reshape(
        int(n_paths), int(total_days)
    )
    resid_draws = residuals[idx]
    overlay_fit = fit.get("residual_overlay_fit")
    if overlay_fit is not None:
        curve = _overlay_vol_multiplier_curve(overlay_fit, int(total_days))
        if curve.size == int(total_days):
            resid_draws = resid_draws * curve[None, :]
    paths = rf_values[idx] + factor_pred + resid_draws
    return np.clip(paths, -1.0, 1.0)


def _training_series(training: TrainingData) -> pd.Series:
    training.validate()
    if training.training_dates is None:
        raise ValueError("this factor residual model requires training_dates")
    dates = pd.DatetimeIndex(
        np.asarray(training.training_dates, dtype="datetime64[ns]")
    ).normalize()
    values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
    if dates.size != values.size or dates.duplicated().any():
        raise ValueError("training_dates must be unique and aligned with returns")
    return pd.Series(values, index=dates, dtype=np.float64)


@dataclass(frozen=True)
class FactorResidualSBBModel:
    """Concrete adapter for one exact source factor-residual model ID."""

    model_id: str

    def __post_init__(self) -> None:
        if self.model_id not in RAW_FACTOR_RESIDUAL_SPECS:
            raise ValueError(f"unknown factor residual model: {self.model_id!r}")

    @property
    def source_specification(self) -> Mapping[str, Any]:
        return RAW_FACTOR_RESIDUAL_SPECS[self.model_id]

    @property
    def resolved_statistical_specification(self) -> Mapping[str, Any]:
        return RESOLVED_FACTOR_RESIDUAL_SPECS[self.model_id]

    @property
    def source_function_names(self) -> tuple[str, ...]:
        return SOURCE_FUNCTION_NAMES

    @property
    def source_fragment_digest(self) -> str:
        return SOURCE_FUNCTIONS_SHA256

    def _source_rng(self, context: ForecastContext) -> np.random.Generator:
        origin = (
            str(context.origin_date)
            if context.origin_date is not None
            else str(context.origin_label)
        )
        horizons = tuple(range(1, int(context.horizon_days) + 1))
        seed = forecast_oos_candidate_seed(
            origin, horizons, self.model_id, int(context.simulations)
        )
        return np.random.default_rng(seed)

    def fit(self, training: TrainingData) -> dict[str, Any]:
        series = _training_series(training)
        factor_frame = load_packaged_factor_frame("ff6")
        fit = _fit_factor_residual_sbb(
            series,
            dict(RAW_FACTOR_RESIDUAL_SPECS[self.model_id]),
            SimfolioEngine(),
            factor_frame=factor_frame,
        )
        if fit is None:
            raise ValueError(f"{self.model_id} source fit failed closed")
        return fit

    def simulate_daily_log_returns(
        self, training: TrainingData, context: ForecastContext
    ) -> np.ndarray:
        fit = self.fit(training)
        paths = _simulate_factor_residual_sbb(
            fit, int(context.horizon_days), int(context.simulations), self._source_rng(context)
        )
        paths = np.asarray(paths, dtype=np.float64)
        expected = (int(context.simulations), int(context.horizon_days))
        if paths.shape != expected or not np.all(np.isfinite(paths)):
            raise ValueError(
                f"{self.model_id} returned invalid daily paths {paths.shape}; expected {expected}"
            )
        return paths


FACTOR_RESIDUAL_MODEL_IDS: tuple[str, ...] = tuple(RAW_FACTOR_RESIDUAL_SPECS)
FACTOR_RESIDUAL_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {
        model_id: (lambda model_id=model_id: FactorResidualSBBModel(model_id))
        for model_id in FACTOR_RESIDUAL_MODEL_IDS
    }
)


def make_factor_residual_model(model_id: str) -> FactorResidualSBBModel:
    factory = FACTOR_RESIDUAL_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"no source-backed factor residual factory for {model_id!r}")
    return factory()


__all__ = [
    "FACTOR_RESIDUAL_FACTORIES",
    "FACTOR_RESIDUAL_MODEL_IDS",
    "RAW_FACTOR_RESIDUAL_SPECS",
    "RESOLVED_FACTOR_RESIDUAL_SPECS",
    "SOURCE_ARTIFACTS",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_LINE_RANGES",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "FactorResidualSBBModel",
    "make_factor_residual_model",
]
