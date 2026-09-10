"""Bounded portfolio-adapter coverage for the retained full-INLA candidate."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

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
    path = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "test_fixtures", "inla", "source_input.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload["input_values"]
    values = np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
    assert list(values.shape) == record["shape"]
    assert str(values.dtype) == record["dtype"]
    assert hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() == record["sha256"]
    return values


def _control_fixture() -> tuple[dict[str, object], np.ndarray]:
    path = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "test_fixtures", "inla", "source_parity_py312.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload["comparison"]["adapter_control_paths"]
    values = np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
    assert list(values.shape) == record["shape"]
    assert str(values.dtype) == record["dtype"]
    assert hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() == record["sha256"]
    return payload, values


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
    source_fixture, expected = _control_fixture()
    assert source_fixture["source_engine_sha256"] == (
        "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
    )
    assert source_fixture["source_revision"] == "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
    control = source_fixture["adapter_control"]
    assert control["source_wrapper_sha256"] == (
        "88aaa532d8a0a63e24964305a26ee6fa45d0d9099dd5da800ab691afad583ea7"
    )
    assert source_fixture["generator"]["input_values_sha256"] == (
        "fd5053f93e6bd654b88f08f760dd57d0ef9fea7e14d9896426d7114a01210e6d"
    )
    assert source_fixture["comparison"]["adapter_control_paths"]["sha256"] == (
        "3d8672a98d39d03b225424cb976734d52be85db1eddf91949f0ecbda708c1ff2"
    )
    np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-12)


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
