import json
from importlib.resources import files

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.portfolio.sv_reference import (
    REFERENCE_MODEL_IDS,
    SOURCE_CANDIDATE_SPECS,
    SOURCE_FUNCTION_NAMES,
    SOURCE_FUNCTIONS_SHA256,
    SVReferenceModel,
    make_sv_reference_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

FIXTURE = files("simfolio_forecasting_methodology").joinpath("resources/test_fixtures/portfolio/sv_reference_source_parity.json")


def _fixture_context(model_id: str) -> ForecastContext:
    return ForecastContext(
        model_id=model_id,
        portfolio_id="fixture",
        origin_label="2026-01-31",
        origin_date="2026-01-31",
        horizon_days=5,
        simulations=7,
        seed=19,
    )


def test_sv_source_fixture_parity_and_repeatability():
    fixture = json.loads(FIXTURE.read_text())
    training = TrainingData(np.asarray(fixture["training_values"], dtype=np.float64))
    for model_id in REFERENCE_MODEL_IDS:
        model = make_sv_reference_model(model_id)
        actual = model.simulate_terminal_log_returns(training, _fixture_context(model_id))
        expected = np.asarray(fixture["expected_terminal_log_returns"][model_id], dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual,
            model.simulate_terminal_log_returns(training, _fixture_context(model_id)),
        )
        assert actual.shape == (7, 5)
        assert np.all(np.isfinite(actual))


def test_sv_source_closure_and_short_history_fail_closed():
    assert SOURCE_FUNCTION_NAMES[0] == "_sv_observed_log_variance"
    assert len(SOURCE_FUNCTIONS_SHA256) == 64
    model = make_sv_reference_model("stochastic_volatility_ar1_empirical")
    short = TrainingData(np.linspace(-0.01, 0.01, 251, dtype=np.float64))
    with pytest.raises(ValueError, match="at least 252"):
        model.simulate_terminal_log_returns(short, _fixture_context(model.model_id))


def test_sv_factory_is_exact_id_only():
    assert isinstance(make_sv_reference_model(REFERENCE_MODEL_IDS[0]), SVReferenceModel)
    with pytest.raises(ValueError, match="no source-backed SV factory"):
        make_sv_reference_model("stochastic_volatility_ar1_empirical_suffix")
    with pytest.raises(ValueError, match="unknown SV reference model"):
        SVReferenceModel("stochastic_volatility_ar1_empirical_suffix")


def test_student_t_public_row_preserves_raw_historical_descriptor():
    spec = SOURCE_CANDIDATE_SPECS["stochastic_volatility_ar1_student_t"]
    assert dict(spec) == {
        "id": "stochastic_volatility_ar1_student_t",
        "type": "sv",
    }
    # The source dispatcher resolves an omitted innovation with its explicit
    # empirical fallback; the public Student-t label is not a runtime switch.
    assert "innovation" not in spec
