"""Portfolio adapter for the retained full-INLA BDES candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...runner import ForecastContext, TrainingData
from ..numerical.bdes_fastmap import deterministic_seed
from ..numerical.inla_bdes import INLA_MODEL_ID, fit_bdes_full_inla, simulate_bdes_full_inla


def _source_engine_log_returns(values: np.ndarray) -> np.ndarray:
    """Match the retained control's simple-return round trip before fitting."""
    simple = np.expm1(np.asarray(values, dtype=np.float64))
    simple = np.clip(simple.astype(np.float64), -0.999999, None)
    return np.log1p(simple)


def _forecast_candidate_seed(context: ForecastContext) -> int:
    if context.origin_date is None or not str(context.origin_date).strip():
        raise ValueError("inla_requires_explicit_origin_date")
    horizon = int(context.horizon_days)
    simulations = int(context.simulations)
    if horizon < 1 or simulations < 2:
        raise ValueError("inla_requires_positive_horizon_and_two_simulations")
    # The retained current-engine control calls BASE._engine_forecast_log_paths
    # with seed_parts=("forecast_final_fixed",).  That helper appends the
    # selected engine model id, horizon, and simulation count; origin is used
    # for task slicing but is deliberately absent from this seed.
    return deterministic_seed("forecast_final_fixed", INLA_MODEL_ID, horizon, simulations)


@dataclass(frozen=True)
class FullINLABDESModel:
    """Exact source-derived portfolio model for the retained INLA ID."""

    model_id: str = INLA_MODEL_ID

    def __post_init__(self) -> None:
        if self.model_id != INLA_MODEL_ID:
            raise ValueError("model ID is not the retained full-INLA BDES candidate")

    def simulate_daily_log_returns(
        self, training: TrainingData, context: ForecastContext
    ) -> np.ndarray:
        training.validate()
        if context.model_id != self.model_id:
            raise ValueError("inla_context_model_id_mismatch")
        source_log_returns = _source_engine_log_returns(training.portfolio_log_returns)
        fit = fit_bdes_full_inla(source_log_returns)
        paths = simulate_bdes_full_inla(
            fit,
            int(context.simulations),
            int(context.horizon_days),
            _forecast_candidate_seed(context),
        )
        expected = (int(context.simulations), int(context.horizon_days))
        if paths.shape != expected or not np.all(np.isfinite(paths)):
            raise ValueError(f"full-INLA returned invalid paths {paths.shape}; expected {expected}")
        return np.asarray(paths, dtype=np.float64)


INLABDESModel = FullINLABDESModel

__all__ = ["FullINLABDESModel", "INLABDESModel"]
