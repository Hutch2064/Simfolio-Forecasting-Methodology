"""Bounded portfolio-adapter coverage for the retained full-INLA candidate."""

from __future__ import annotations

import hashlib

import numpy as np

from simfolio_forecasting_methodology.models.numerical.adaptive_pgas import (
    adaptive_pgas_uniform_paths,
)
from simfolio_forecasting_methodology.models.numerical.inla_bdes import INLA_MODEL_ID
from simfolio_forecasting_methodology.models.portfolio.inla import (
    FullINLABDESModel,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData


def _values() -> np.ndarray:
    return np.sin(np.arange(180, dtype=float) / 8.0) * 0.01 + np.random.default_rng(2).normal(
        0.0, 0.01, 180
    )


def test_full_inla_portfolio_adapter_uses_current_engine_control_seed_context() -> None:
    values = _values()
    training = TrainingData(portfolio_log_returns=values)
    context = ForecastContext(
        model_id=INLA_MODEL_ID,
        portfolio_id="fixture",
        origin_label="fixture-origin",
        horizon_days=10,
        simulations=4,
        seed=999,
        origin_date="2026-05-13",
    )

    output = FullINLABDESModel().simulate_daily_log_returns(training, context)

    assert output.shape == (4, 10)
    assert np.isfinite(output).all()
    digest = hashlib.sha256(np.ascontiguousarray(output).tobytes(order="C")).hexdigest()
    assert digest == "3d8672a98d39d03b225424cb976734d52be85db1eddf91949f0ecbda708c1ff2"


def test_full_inla_portfolio_adapter_rejects_missing_origin_date() -> None:
    context = ForecastContext(
        model_id=INLA_MODEL_ID,
        portfolio_id="fixture",
        origin_label="fixture-origin",
        horizon_days=10,
        simulations=4,
        seed=999,
        origin_date=None,
    )
    with np.testing.assert_raises(ValueError):
        FullINLABDESModel().simulate_daily_log_returns(
            TrainingData(portfolio_log_returns=_values()), context
        )


def test_exact_dispatch_rejects_unassigned_student_scale_mixture() -> None:
    with np.testing.assert_raises_regex(ValueError, "student_scale_mixture_is_not_authorized"):
        adaptive_pgas_uniform_paths(
            np.zeros((80, 2), dtype=np.float64),
            4,
            2,
            np.random.default_rng(7),
            pseudo_observations=lambda values: np.full_like(values, 0.5),
            student_copula_fit=lambda values: (np.eye(values.shape[1]), 8.0),
            stable_correlation=lambda values: np.eye(values.shape[1]),
            config={"student_scale_mixture": True},
        )
