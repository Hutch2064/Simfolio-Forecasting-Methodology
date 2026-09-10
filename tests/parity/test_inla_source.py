"""Source parity checks for the recovered full-INLA BDES candidate."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import numpy as np

from simfolio_forecasting_methodology.models.numerical.inla_bdes import (
    INLA_CANDIDATE,
    fit_bdes_full_inla,
    simulate_bdes_full_inla,
    source_candidate_digest,
)


def _resource(name: str) -> bytes:
    return (
        resources.files("simfolio_forecasting_methodology")
        .joinpath("resources", "test_fixtures", "inla", name)
        .read_bytes()
    )


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _retained_array(record: dict[str, object]) -> np.ndarray:
    values = np.asarray(record["values"], dtype=np.dtype(str(record["dtype"])))
    assert list(values.shape) == record["shape"]
    assert str(values.dtype) == record["dtype"]
    assert _digest(values) == record["sha256"]
    return values


def _frozen_input() -> np.ndarray:
    fixture = json.loads(_resource("source_input.json").decode("utf-8"))
    return _retained_array(fixture["input_values"])


def test_raw_source_descriptor_matches_ast_recovery_fixture() -> None:
    descriptor = json.loads(_resource("source_candidate.json").decode("utf-8"))
    assert descriptor == INLA_CANDIDATE
    assert len(descriptor) == 82
    assert (
        source_candidate_digest()
        == "472d8bac243f3b1b123071a21a52d96394e775e8c4accb8d5fa18fd676451034"
    )


def test_full_inla_fit_and_path_recurrence_match_source_fixture() -> None:
    fixture = json.loads(_resource("source_parity_py312.json").decode("utf-8"))
    values = _frozen_input()
    assert fixture["source_engine_sha256"] == (
        "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
    )
    assert fixture["source_revision"] == "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
    generator = fixture["generator"]
    assert generator["input_fixture"] == "source_input.json"
    assert generator["input_values_sha256"] == _digest(values)
    assert generator["input_values_shape"] == list(values.shape)
    assert generator["input_values_dtype"] == str(values.dtype)
    fit = fit_bdes_full_inla(values, INLA_CANDIDATE)
    expected = fixture["comparison"]
    for key in (
        "posterior_samples",
        "posterior_sample_weights",
        "innovation_pool",
        "standardized_residuals",
    ):
        actual = np.asarray(fit[key])
        retained = _retained_array(expected[key])
        assert list(actual.shape) == list(retained.shape)
        assert str(actual.dtype) == str(retained.dtype)
        np.testing.assert_allclose(actual, retained, rtol=0.0, atol=2e-12)
    bdes = fit["bdes_multiscale_vol"]
    for key in ("q_last", "phis", "b", "q_var"):
        actual = np.asarray(bdes[key])
        record = expected[f"bdes_{key}"]
        retained = _retained_array(record)
        assert list(actual.shape) == list(retained.shape)
        assert str(actual.dtype) == str(retained.dtype)
        np.testing.assert_allclose(actual, retained, rtol=0.0, atol=2e-12)
    for key in (
        "residual_last",
        "residual_phi",
        "residual_innovation_sd",
        "residual_common_loading",
    ):
        np.testing.assert_allclose(
            fit["bdes_multiscale_vol"][key], expected[f"bdes_{key}"], rtol=0.0, atol=2e-12
        )
    np.testing.assert_allclose(
        np.asarray(fit["posterior_center"], dtype=np.float64),
        np.asarray(expected["posterior_center"], dtype=np.float64),
        rtol=0.0,
        atol=2e-12,
    )
    paths = simulate_bdes_full_inla(fit, 4, 10, 123)
    retained_paths = _retained_array(expected["paths"])
    assert list(paths.shape) == list(retained_paths.shape)
    assert str(paths.dtype) == str(retained_paths.dtype)
    np.testing.assert_allclose(paths, retained_paths, rtol=0.0, atol=2e-12)


def test_inla_candidate_is_seed_identity_strict() -> None:
    altered = dict(INLA_CANDIDATE)
    altered["map_maxiter"] = 99
    with np.testing.assert_raises(ValueError):
        fit_bdes_full_inla(np.linspace(-0.01, 0.01, 80), altered)
