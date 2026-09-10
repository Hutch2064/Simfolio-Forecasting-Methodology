import json
from importlib.resources import files

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.portfolio.gjr_reference import (
    REFERENCE_MODEL_IDS,
    SOURCE_CANDIDATE_SPECS,
    SOURCE_FUNCTIONS_SHA256,
    PortfolioGJRGARCHModel,
    make_gjr_reference_model,
)
from simfolio_forecasting_methodology.runner import (
    ForecastContext,
    OriginTask,
    TrainingData,
    evaluate_origin_task,
)

FIXTURE = files("simfolio_forecasting_methodology").joinpath("resources/test_fixtures/portfolio/gjr_reference_source_parity.json")


def _fixture_context(model_id: str, horizon_days: int = 5, simulations: int = 7) -> ForecastContext:
    return ForecastContext(
        model_id=model_id,
        portfolio_id="fixture",
        origin_label="2026-01-31",
        origin_date="2026-01-31",
        horizon_days=horizon_days,
        simulations=simulations,
        seed=19,
    )


def test_gjr_source_fixture_parity_and_repeatability():
    fixture = json.loads(FIXTURE.read_text())
    training = TrainingData(np.asarray(fixture["training_values"], dtype=np.float64))
    for model_id in REFERENCE_MODEL_IDS:
        model = make_gjr_reference_model(model_id)
        context = _fixture_context(model_id)
        actual = model.simulate_daily_log_returns(training, context)
        repeated = model.simulate_daily_log_returns(training, context)
        expected = np.asarray(fixture["expected_daily_log_returns"][model_id], dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, repeated)
        assert actual.shape == (7, 5)
        assert np.all(np.isfinite(actual))
        assert not hasattr(model, "simulate_terminal_log_returns")


def test_gjr_daily_paths_are_cumulative_scored_by_runner():
    fixture = json.loads(FIXTURE.read_text())
    training = TrainingData(np.asarray(fixture["training_values"], dtype=np.float64))
    model_id = REFERENCE_MODEL_IDS[0]
    model = make_gjr_reference_model(model_id)
    realized = np.zeros(5, dtype=np.float64)
    task = OriginTask(
        portfolio_id="fixture",
        origin_label="2026-01-31",
        training=training,
        realized_future_daily_log_returns=realized,
        seed=19,
        origin_date="2026-01-31",
    )
    losses = evaluate_origin_task(model, task, simulations=7)
    assert losses.shape == (5,)
    assert np.all(np.isfinite(losses))


def test_gjr_source_descriptor_and_closure_are_explicit():
    assert SOURCE_FUNCTIONS_SHA256 == json.loads(FIXTURE.read_text())["source_fragment_digest"]
    assert set(REFERENCE_MODEL_IDS) == {
        "portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal",
        "portfolio_gjr_garch_eb_sampler_moving_block_optimal",
    }
    assert SOURCE_CANDIDATE_SPECS[REFERENCE_MODEL_IDS[0]]["residual_resampling"] == "stationary_bootstrap"
    assert SOURCE_CANDIDATE_SPECS[REFERENCE_MODEL_IDS[1]]["residual_resampling"] == "moving_block_bootstrap"
    assert SOURCE_CANDIDATE_SPECS[REFERENCE_MODEL_IDS[0]]["overlay_model"] == "gjr_garch_1_1"


def test_gjr_short_history_and_exact_id_fail_closed():
    model = make_gjr_reference_model(REFERENCE_MODEL_IDS[0])
    short = TrainingData(np.linspace(-0.01, 0.01, 59, dtype=np.float64))
    with pytest.raises(RuntimeError, match="at least 60"):
        model.simulate_daily_log_returns(short, _fixture_context(REFERENCE_MODEL_IDS[0]))
    with pytest.raises(ValueError, match="no source-backed GJR factory"):
        make_gjr_reference_model("portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal_suffix")
    with pytest.raises(ValueError, match="unknown GJR reference model"):
        PortfolioGJRGARCHModel("portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal_suffix")
