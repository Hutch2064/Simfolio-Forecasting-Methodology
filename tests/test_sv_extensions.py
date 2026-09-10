import json
from importlib.resources import files

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.portfolio.sv_extensions import (
    REFERENCE_MODEL_IDS,
    SOURCE_FUNCTION_NAMES,
    SOURCE_FUNCTIONS_SHA256,
    SuffixSVExtensionModel,
    make_sv_extension_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

FIXTURE = files("simfolio_forecasting_methodology").joinpath("resources/test_fixtures/portfolio/sv_extension_source_parity.json")


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


def test_sv_extension_source_fixture_parity_and_repeatability():
    fixture = json.loads(FIXTURE.read_text())
    training = TrainingData(np.asarray(fixture["training_values"], dtype=np.float64))
    for model_id in REFERENCE_MODEL_IDS:
        model = make_sv_extension_model(model_id)
        actual = model.simulate_terminal_log_returns(training, _fixture_context(model_id))
        expected = np.asarray(fixture["expected_terminal_log_returns"][model_id], dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual,
            model.simulate_terminal_log_returns(training, _fixture_context(model_id)),
        )
        assert actual.shape == (7, 5)
        assert np.all(np.isfinite(actual))


def test_sv_extension_closure_and_short_history_fail_closed():
    assert SOURCE_FUNCTION_NAMES[0] == "_categorical_next_states"
    assert len(SOURCE_FUNCTIONS_SHA256) == 64
    model = make_sv_extension_model("dp_mixture_sv_sbb")
    short = TrainingData(np.linspace(-0.01, 0.01, 503, dtype=np.float64))
    with pytest.raises(ValueError, match="at least 504"):
        model.simulate_terminal_log_returns(short, _fixture_context(model.model_id))


def test_sv_extension_factory_is_exact_id_only():
    assert isinstance(make_sv_extension_model(REFERENCE_MODEL_IDS[0]), SuffixSVExtensionModel)
    with pytest.raises(ValueError, match="no source-backed SV extension factory"):
        make_sv_extension_model("dp_mixture_sv_sbb_suffix")
    with pytest.raises(ValueError, match="unknown SV extension model"):
        SuffixSVExtensionModel("dp_mixture_sv_sbb_suffix")
