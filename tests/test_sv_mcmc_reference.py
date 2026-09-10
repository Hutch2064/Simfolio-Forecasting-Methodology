import json
from importlib.resources import files

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.portfolio.sv_mcmc_reference import (
    REFERENCE_MODEL_IDS,
    SOURCE_FUNCTION_NAMES,
    SOURCE_FUNCTIONS_SHA256,
    BayesianMCMCSVModel,
    _fit_sv_mcmc,
    make_sv_mcmc_reference_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

FIXTURE = files("simfolio_forecasting_methodology").joinpath("resources/test_fixtures/portfolio/sv_mcmc_reference_source_parity.json")
TRAINING_FIXTURE = files("simfolio_forecasting_methodology").joinpath("resources/test_fixtures/portfolio/sv_extension_source_parity.json")


def _fixture_context() -> ForecastContext:
    return ForecastContext(
        model_id="bayesian_mcmc_stochastic_volatility_sbb",
        portfolio_id="fixture",
        origin_label="2026-01-31",
        origin_date="2026-01-31",
        horizon_days=5,
        simulations=7,
        seed=19,
    )


def test_mcmc_source_fixture_parity_and_repeatability():
    fixture = json.loads(FIXTURE.read_text())
    training_fixture = json.loads(TRAINING_FIXTURE.read_text())
    training = TrainingData(np.asarray(training_fixture["training_values"], dtype=np.float64))
    model = make_sv_mcmc_reference_model(REFERENCE_MODEL_IDS[0])
    actual = model.simulate_terminal_log_returns(training, _fixture_context())
    expected = np.asarray(
        fixture["expected_terminal_log_returns"][REFERENCE_MODEL_IDS[0]], dtype=np.float64
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual,
        model.simulate_terminal_log_returns(training, _fixture_context()),
    )
    assert actual.shape == (7, 5)
    assert np.all(np.isfinite(actual))


def test_mcmc_source_closure_and_fixed_sampler_settings():
    fixture = json.loads(FIXTURE.read_text())
    assert SOURCE_FUNCTION_NAMES[0] == "_deterministic_seed"
    assert SOURCE_FUNCTION_NAMES[-1] == "_simulate_sv_mcmc_sbb"
    assert SOURCE_FUNCTIONS_SHA256 == fixture["source_fragment_digest"]
    fit = _fit_sv_mcmc(
        np.asarray(json.loads(TRAINING_FIXTURE.read_text())["training_values"], dtype=np.float64)
    )
    assert fit is not None
    assert fit["mcmc_iterations"] == fixture["mcmc_settings"]["iterations"] == 420
    assert len(fit["posterior_samples"]) == fixture["mcmc_settings"]["posterior_sample_count"] == 75
    assert fit["mcmc_loglikelihood_model"] == "kalman_log_chi_square_state_space"


def test_mcmc_short_history_and_exact_id_fail_closed():
    model = make_sv_mcmc_reference_model(REFERENCE_MODEL_IDS[0])
    short = TrainingData(np.linspace(-0.01, 0.01, 503, dtype=np.float64))
    with pytest.raises(ValueError, match="at least 504"):
        model.simulate_terminal_log_returns(short, _fixture_context())
    with pytest.raises(ValueError, match="no source-backed Bayesian MCMC SV factory"):
        make_sv_mcmc_reference_model("bayesian_mcmc_stochastic_volatility_sbb_suffix")
    with pytest.raises(ValueError, match="unknown Bayesian MCMC SV model"):
        BayesianMCMCSVModel("bayesian_mcmc_stochastic_volatility_sbb_suffix")
