import json
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from simfolio_forecasting_methodology.models.portfolio.canonical_stack_reference import (
    CANONICAL_MODEL_ID,
    REFERENCE_MODEL_IDS,
    SOURCE_ARTIFACTS,
    SOURCE_CANDIDATE_SPECS,
    SOURCE_FUNCTION_NAMES,
    SOURCE_FUNCTIONS_SHA256,
    STACK_MODEL_IDS,
    _fit_factor_drift_prior,
    make_canonical_stack_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

FIXTURE = files("simfolio_forecasting_methodology").joinpath("resources/test_fixtures/portfolio/canonical_stack_source_parity.json")


def _payload() -> dict:
    return json.loads(FIXTURE.read_text())


def _context(model_id: str, payload: dict) -> ForecastContext:
    return ForecastContext(
        model_id=model_id,
        portfolio_id="canonical-stack-fixture",
        origin_label=payload["origin_label"],
        horizon_days=int(payload["horizon_days"]),
        simulations=int(payload["simulations"]),
        seed=20260910,
        origin_date=payload["origin_label"],
    )


def test_factory_contains_exact_canonical_plus_twelve_stack_ids() -> None:
    assert REFERENCE_MODEL_IDS == (CANONICAL_MODEL_ID, *STACK_MODEL_IDS)
    assert len(REFERENCE_MODEL_IDS) == 13
    assert len(set(REFERENCE_MODEL_IDS)) == 13
    for model_id in REFERENCE_MODEL_IDS:
        assert make_canonical_stack_model(model_id).model_id == model_id


def test_source_specs_preserve_sparse_raw_stack_descriptors() -> None:
    assert dict(SOURCE_CANDIDATE_SPECS[CANONICAL_MODEL_ID]) == {
        "id": CANONICAL_MODEL_ID,
        "type": "canonical",
    }
    for model_id in STACK_MODEL_IDS:
        assert dict(SOURCE_CANDIDATE_SPECS[model_id]) == {"id": model_id, "type": "stack"}


def test_source_fixture_matches_all_thirteen_terminal_closures() -> None:
    payload = _payload()
    training = TrainingData(np.asarray(payload["training_log_returns"], dtype=np.float64))
    for model_id, expected in payload["expected_terminal_log_returns"].items():
        model = make_canonical_stack_model(model_id)
        actual = model.simulate_terminal_log_returns(training, _context(model_id, payload))
        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float64))


def test_terminal_output_is_not_recast_as_daily_increments() -> None:
    payload = _payload()
    training = TrainingData(np.asarray(payload["training_log_returns"], dtype=np.float64))
    model = make_canonical_stack_model(CANONICAL_MODEL_ID)
    result = model.simulate_terminal_log_returns(training, _context(model.model_id, payload))
    assert result.shape == (int(payload["simulations"]), int(payload["horizon_days"]))
    assert not np.allclose(result[:, 1], result[:, 0] + result[:, 0])


def test_unknown_ids_fail_closed_and_short_history_is_visible() -> None:
    with pytest.raises(ValueError, match="unknown canonical stack model"):
        make_canonical_stack_model("stack_canonical_gaussian_w0.60_extra")
    model = make_canonical_stack_model(CANONICAL_MODEL_ID)
    with pytest.raises(RuntimeError, match="at least 30 usable observations"):
        model.simulate_terminal_log_returns(
            TrainingData(np.arange(29, dtype=np.float64)),
            ForecastContext(CANONICAL_MODEL_ID, "p", "o", 5, 3, 1),
        )


def test_source_provenance_is_relative_and_factor_gap_is_explicit() -> None:
    payload = _payload()
    assert payload["source_function_digest"] == SOURCE_FUNCTIONS_SHA256
    assert "complete manifest-whitelisted nine-series factor closure" in payload["factor_branch"]
    assert set(SOURCE_ARTIFACTS) == {
        "engine",
        "research_gate",
        "pinned_source_revision",
        "canonical_proxy_snapshot_manifest",
    }
    assert len(SOURCE_FUNCTION_NAMES) >= 15
    for artifact in SOURCE_ARTIFACTS.values():
        for key, value in artifact.items():
            if key == "path":
                assert not str(value).startswith(("/", "~"))


def test_factor_prior_uses_all_aligned_columns_and_is_repeatable() -> None:
    index = pd.date_range("2020-01-01", periods=300, freq="D")
    base = np.linspace(-0.002, 0.002, len(index), dtype=np.float64)
    factors = pd.DataFrame(
        {
            f"factor-{idx}": base + (idx + 1) * 1e-5 * np.sin(np.arange(len(index)) / (idx + 2.0))
            for idx in range(9)
        },
        index=index,
    )
    portfolio = pd.Series(
        0.0002 + 0.4 * factors["factor-0"] - 0.2 * factors["factor-1"],
        index=index,
        name="portfolio",
    )
    first = _fit_factor_drift_prior(portfolio, factor_frame=factors)
    second = _fit_factor_drift_prior(portfolio, factor_frame=factors)
    assert first["status"] == "complete"
    assert first["factor_count"] == 9
    assert first["factor_names"] == list(factors.columns)
    assert first == second
