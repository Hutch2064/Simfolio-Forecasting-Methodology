import json
from importlib.resources import files

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.portfolio.reference_families import (
    BDES_BLOCKED_MODEL_ID,
    REFERENCE_MODEL_IDS,
    SOURCE_ARTIFACTS,
    SOURCE_CANDIDATE_SPECS,
    SOURCE_FUNCTION_NAMES,
    SOURCE_FUNCTIONS_SHA256,
    make_reference_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

FIXTURE = files("simfolio_forecasting_methodology").joinpath("resources/test_fixtures/portfolio/reference_family_source_parity.json")


def _fixture_payload() -> dict:
    return json.loads(FIXTURE.read_text())


def test_exact_factory_map_contains_only_the_four_extracted_ids() -> None:
    assert REFERENCE_MODEL_IDS == (
        "constant_mean_gaussian",
        "constant_mean_student_t",
        "naive_iid_historical_portfolio_bootstrap",
        "zero_mean_gaussian_vol_only",
    )
    assert set(SOURCE_CANDIDATE_SPECS) == set(REFERENCE_MODEL_IDS)
    for model_id in REFERENCE_MODEL_IDS:
        model = make_reference_model(model_id)
        assert model.model_id == model_id
        assert model.source_function_names == SOURCE_FUNCTION_NAMES
        assert model.source_fragment_digest == SOURCE_FUNCTIONS_SHA256


def test_unknown_and_unresolved_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown canonical reference model"):
        make_reference_model("constant_mean_gaussian_extra")
    with pytest.raises(ValueError, match="full-INLA numerical closure is not recovered"):
        make_reference_model(BDES_BLOCKED_MODEL_ID)


def test_source_fixture_matches_each_terminal_closure() -> None:
    payload = _fixture_payload()
    training = TrainingData(np.asarray(payload["training_log_returns"], dtype=np.float64))
    for model_id, expected in payload["expected_terminal_log_returns"].items():
        model = make_reference_model(model_id)
        context = ForecastContext(
            model_id=model_id,
            portfolio_id="fixture-portfolio",
            origin_label=payload["origin_label"],
            horizon_days=int(payload["horizon_days"]),
            simulations=int(payload["simulations"]),
            seed=20260528,
            origin_date=payload["origin_label"],
        )
        actual = model.simulate_terminal_log_returns(training, context)
        assert actual.shape == (payload["simulations"], payload["horizon_days"])
        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float64))


def test_terminal_models_do_not_make_a_daily_increment_claim() -> None:
    payload = _fixture_payload()
    training = TrainingData(np.asarray(payload["training_log_returns"], dtype=np.float64))
    model = make_reference_model("constant_mean_gaussian")
    context = ForecastContext(
        model_id=model.model_id,
        portfolio_id="fixture-portfolio",
        origin_label=payload["origin_label"],
        horizon_days=5,
        simulations=7,
        seed=20260528,
        origin_date=payload["origin_label"],
    )
    result = model.simulate_terminal_log_returns(training, context)
    # Every column is generated as a source terminal ensemble.  The adapter
    # must not force column j to be a cumulative sum of column j-1.
    assert not np.allclose(result[:, 1], result[:, 0] + result[:, 0])


def test_gaussian_and_student_t_source_minimum_history_is_explicit() -> None:
    short_training = TrainingData(np.arange(29, dtype=np.float64))
    for model_id in ("constant_mean_gaussian", "constant_mean_student_t"):
        model = make_reference_model(model_id)
        context = ForecastContext(model_id, "p", "o", 1, 3, 1)
        with pytest.raises(ValueError, match="at least 30"):
            model.simulate_terminal_log_returns(short_training, context)


def test_source_provenance_paths_are_relative_and_digested() -> None:
    assert set(SOURCE_ARTIFACTS) == {"engine", "research_gate"}
    for record in SOURCE_ARTIFACTS.values():
        assert not record["path"].startswith(("/", "~"))
        assert len(record["sha256"]) == 64
