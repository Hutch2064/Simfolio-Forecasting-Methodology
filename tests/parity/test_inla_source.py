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
    values = np.sin(np.arange(180, dtype=float) / 8.0) * 0.01 + np.random.default_rng(2).normal(
        0.0, 0.01, 180
    )
    fit = fit_bdes_full_inla(values, INLA_CANDIDATE)
    expected = fixture["comparison"]
    for key in (
        "posterior_samples",
        "posterior_sample_weights",
        "innovation_pool",
        "standardized_residuals",
    ):
        actual = np.asarray(fit[key])
        assert list(actual.shape) == expected[key]["shape"]
        assert str(actual.dtype) == expected[key]["dtype"]
        assert _digest(actual) == expected[key]["sha256"]
    bdes = fit["bdes_multiscale_vol"]
    for key in ("q_last", "phis", "b", "q_var"):
        actual = np.asarray(bdes[key])
        record = expected[f"bdes_{key}"]
        assert list(actual.shape) == record["shape"]
        assert str(actual.dtype) == record["dtype"]
        assert _digest(actual) == record["sha256"]
    for key in (
        "residual_last",
        "residual_phi",
        "residual_innovation_sd",
        "residual_common_loading",
    ):
        assert fit["bdes_multiscale_vol"][key] == expected[f"bdes_{key}"]
    np.testing.assert_array_equal(np.asarray(fit["posterior_center"]), expected["posterior_center"])
    paths = simulate_bdes_full_inla(fit, 4, 10, 123)
    assert list(paths.shape) == expected["paths"]["shape"]
    assert _digest(paths) == expected["paths"]["sha256"]


def test_inla_candidate_is_seed_identity_strict() -> None:
    altered = dict(INLA_CANDIDATE)
    altered["map_maxiter"] = 99
    with np.testing.assert_raises(ValueError):
        fit_bdes_full_inla(np.linspace(-0.01, 0.01, 80), altered)
