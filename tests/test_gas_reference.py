import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.portfolio.gas_reference import (
    REFERENCE_MODEL_IDS,
    SOURCE_FUNCTION_NAMES,
    SOURCE_FUNCTIONS_SHA256,
    GasScoreDrivenSkewTModel,
    make_gas_reference_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

FIXTURE = Path(__file__).parent / "fixtures" / "gas_reference_source_parity.json"
TRAINING_FIXTURE = Path(__file__).parent / "fixtures" / "sv_extension_source_parity.json"


def _fixture_context() -> ForecastContext:
    return ForecastContext(
        model_id="gas_score_driven_skewt",
        portfolio_id="fixture",
        origin_label="2026-01-31",
        origin_date="2026-01-31",
        horizon_days=5,
        simulations=7,
        seed=19,
    )


def test_gas_source_fixture_parity_and_repeatability():
    fixture = json.loads(FIXTURE.read_text())
    training_fixture = json.loads(TRAINING_FIXTURE.read_text())
    training = TrainingData(np.asarray(training_fixture["training_values"], dtype=np.float64))
    model = make_gas_reference_model("gas_score_driven_skewt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = model.simulate_terminal_log_returns(training, _fixture_context())
        repeated = model.simulate_terminal_log_returns(training, _fixture_context())
    expected = np.asarray(
        fixture["expected_terminal_log_returns"]["gas_score_driven_skewt"],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, repeated)
    assert actual.shape == (7, 5)
    assert np.all(np.isfinite(actual))


def test_gas_source_closure_and_short_history_fail_closed():
    assert SOURCE_FUNCTION_NAMES == (
        "_deterministic_seed",
        "_sample_mean_near_zero_shrinkage",
        "_gas_t_score",
        "_fit_gas_score_driven_skewt",
        "_simulate_gas_score_driven_skewt",
    )
    assert len(SOURCE_FUNCTIONS_SHA256) == 64
    model = make_gas_reference_model("gas_score_driven_skewt")
    short = TrainingData(np.linspace(-0.01, 0.01, 503, dtype=np.float64))
    with pytest.raises(ValueError, match="at least 504"):
        model.simulate_terminal_log_returns(short, _fixture_context())


def test_gas_factory_is_exact_id_only():
    assert isinstance(make_gas_reference_model(REFERENCE_MODEL_IDS[0]), GasScoreDrivenSkewTModel)
    with pytest.raises(ValueError, match="no source-backed gas factory"):
        make_gas_reference_model("gas_score_driven_skewt_suffix")
    with pytest.raises(ValueError, match="unknown gas reference model"):
        GasScoreDrivenSkewTModel("gas_score_driven_skewt_suffix")
