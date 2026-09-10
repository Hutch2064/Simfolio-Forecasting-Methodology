"""Bounded adapter coverage for the exact asset-level INLA composition."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import numpy as np

from simfolio_forecasting_methodology.models.asset_level.exact_kalman import (
    EXACT_KALMAN_MODEL_ID,
    SOURCE_ASSET_WRAPPER_SHA256,
    SOURCE_DEPENDENCE_WRAPPER_SHA256,
    SOURCE_PGAS_WRAPPER_SHA256,
    ExactKalmanDynamicGaussianFactorModel,
)
from simfolio_forecasting_methodology.runner import (
    ForecastContext,
    PortfolioPolicy,
    TrainingData,
)


def _fixture() -> dict[str, np.ndarray]:
    fixture = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "test_fixtures", "frontier", "source_reference.npz"
    )
    with fixture.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _source_fixture() -> dict[str, object]:
    path = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "test_fixtures", "inla", "exact_asset_source_parity.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _retained_array(record: dict[str, object]) -> np.ndarray:
    values = np.asarray(record["values"], dtype=np.dtype(str(record["dtype"])))
    assert list(values.shape) == record["shape"]
    assert str(values.dtype) == record["dtype"]
    assert hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() == record["sha256"]
    return values


def test_exact_kalman_asset_adapter_has_deterministic_bounded_forecast() -> None:
    fixture = _fixture()
    tickers = ("ALPHA", "BETA", "GAMMA", "DELTA")
    weights = tuple(float(value) for value in fixture["weights"])
    training = TrainingData(
        portfolio_log_returns=np.asarray(fixture["asset_log_returns"] @ fixture["weights"]),
        asset_log_returns=fixture["asset_log_returns"],
        policy=PortfolioPolicy(tickers, weights, "monthly"),
        training_dates=fixture["training_dates"],
    )
    context = ForecastContext(
        model_id=EXACT_KALMAN_MODEL_ID,
        portfolio_id="fixture",
        origin_label="fixture-origin",
        horizon_days=8,
        simulations=16,
        seed=123,
        future_dates=fixture["future_dates"],
        origin_date="2026-05-13",
    )

    output = ExactKalmanDynamicGaussianFactorModel().simulate_daily_log_returns(training, context)

    assert output.shape == (16, 8)
    assert np.isfinite(output).all()
    source_fixture = _source_fixture()
    source_dispatch = source_fixture["source_dispatch"]
    assert source_dispatch["candidate_id"] == EXACT_KALMAN_MODEL_ID
    assert source_dispatch["source_dependence_wrapper_sha256"] == SOURCE_DEPENDENCE_WRAPPER_SHA256
    assert source_dispatch["source_pgas_wrapper_sha256"] == SOURCE_PGAS_WRAPPER_SHA256
    assert source_dispatch["source_marginal_wrapper_sha256"] == SOURCE_ASSET_WRAPPER_SHA256
    assert source_dispatch["source_engine_sha256"] == (
        "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
    )
    assert source_dispatch["source_revision"] == "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
    expected = _retained_array(source_fixture["comparison"]["portfolio_daily_log_paths"])
    assert output.shape == expected.shape
    assert output.dtype == expected.dtype
    assert source_fixture["comparison"]["portfolio_daily_log_paths"]["sha256"] == (
        "e99d8227c5a7e634db29b782387cf8ba701bec076cbe2a706bc54a754a99c6ea"
    )
    np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-12)
    assert source_fixture["comparison"]["source_local_exact"] is True


def test_exact_kalman_asset_adapter_rejects_non_source_model_id() -> None:
    fixture = _fixture()
    training = TrainingData(
        portfolio_log_returns=np.asarray(fixture["asset_log_returns"] @ fixture["weights"]),
        asset_log_returns=fixture["asset_log_returns"],
        policy=PortfolioPolicy(
            ("ALPHA", "BETA", "GAMMA", "DELTA"),
            tuple(float(value) for value in fixture["weights"]),
            "monthly",
        ),
        training_dates=fixture["training_dates"],
    )
    context = ForecastContext(
        model_id="wrong",
        portfolio_id="fixture",
        origin_label="fixture-origin",
        horizon_days=8,
        simulations=16,
        seed=123,
        future_dates=fixture["future_dates"],
        origin_date="2026-05-13",
    )
    with np.testing.assert_raises(ValueError):
        ExactKalmanDynamicGaussianFactorModel(model_id="wrong").simulate_daily_log_returns(
            training, context
        )
