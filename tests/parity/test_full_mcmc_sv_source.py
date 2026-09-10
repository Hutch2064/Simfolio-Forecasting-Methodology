"""Bounded numerical parity against source-derived canonical SV fixtures."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from importlib.resources import files

import numpy as np

from simfolio_forecasting_methodology.models.numerical.mcmc_sv import (
    canonical_full_mcmc_sv_ids,
    fit_full_mcmc_sv,
    load_canonical_full_mcmc_sv_specs,
    simulate_full_mcmc_sv,
)

_FIXTURE = files("simfolio_forecasting_methodology").joinpath(
    "resources/test_fixtures/mcmc/full_mcmc_sv_source_parity.json"
)
_PY311_FIXTURE = files("simfolio_forecasting_methodology").joinpath(
    "resources/test_fixtures/mcmc/full_mcmc_sv_source_parity_py311_numpy246_scipy1171.json"
)
_MANIFEST = files("simfolio_forecasting_methodology").joinpath(
    "resources/catalogs/canonical_40_full_mcmc_sv_specs.json"
)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _runtime_environment() -> dict[str, str]:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return {
        "label": f"python{python_version}-numpy{np.__version__}-scipy{importlib.metadata.version('scipy')}",
        "python": python_version,
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
    }


def _environment_fixture():
    environment = _runtime_environment()
    if environment["label"] == "python3.12-numpy2.5.3-scipy1.18.1":
        return _FIXTURE
    if environment["label"] == "python3.11-numpy2.4.6-scipy1.17.1":
        return _PY311_FIXTURE
    raise AssertionError(f"no byte-exact fixture for {environment['label']}; install requirements-lock.txt")


def test_owned_full_mcmc_sv_catalogue_is_exactly_40_source_entries():
    entries = load_canonical_full_mcmc_sv_specs()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    runtime = manifest["scored_runtime_catalog"]
    assert len(entries) == 40
    assert tuple(entry["id"] for entry in entries) == canonical_full_mcmc_sv_ids()
    assert {entry["type"] for entry in entries} == {"bayesian_sbb_full_mcmc_sv_overlay"}
    assert runtime["catalog_section_consumed_by_wrapper"] == "full_current_catalog"
    assert runtime["catalog_sha256"] == manifest["historical_descriptor_source_sha256"]
    assert runtime["exact_row_match_verified"] is True
    assert runtime["status"] == (
        "scored-runtime-catalog-input-resolved-to-historical-publication-file"
    )


def test_full_mcmc_sv_fit_and_paths_match_source_fixture():
    fixture = json.loads(_environment_fixture().read_text(encoding="utf-8"))
    assert fixture["execution_environment"] == _runtime_environment()
    manifest_digest = hashlib.sha256(_MANIFEST.read_bytes()).hexdigest()
    assert fixture["source_candidate_manifest"] == (
        "src/simfolio_forecasting_methodology/resources/catalogs/"
        "canonical_40_full_mcmc_sv_specs.json"
    )
    assert fixture["source_candidate_manifest_sha256"] == manifest_digest
    values = np.random.default_rng(173).normal(0.0002, 0.01, 420).astype(np.float64)
    assert fixture["fixture_input"]["length"] == values.size
    simulation = fixture["simulation"]

    for expected in fixture["entries"]:
        fitted = fit_full_mcmc_sv(values, expected["id"])
        assert fitted is not None, expected["id"]
        posterior = np.asarray(fitted["posterior_samples"], dtype=np.float64)
        assert list(posterior.shape) == expected["posterior_samples_shape"]
        assert _digest(posterior) == expected["posterior_samples_sha256"]

        paths = simulate_full_mcmc_sv(
            fitted,
            int(simulation["horizon_days"]),
            int(simulation["n_paths"]),
            np.random.default_rng(int(simulation["seed"])),
        )
        assert list(paths.shape) == expected["paths_shape"]
        assert np.all(np.isfinite(paths))
        assert _digest(paths) == expected["paths_sha256"]


def test_source_fixtures_are_explicitly_environment_labelled():
    current = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    py311 = json.loads(_PY311_FIXTURE.read_text(encoding="utf-8"))
    assert current["execution_environment"] == {
        "label": "python3.12-numpy2.5.3-scipy1.18.1",
        "python": "3.12",
        "numpy": "2.5.3",
        "scipy": "1.18.1",
    }
    assert py311["execution_environment"] == {
        "label": "python3.11-numpy2.4.6-scipy1.17.1",
        "python": "3.11",
        "numpy": "2.4.6",
        "scipy": "1.17.1",
    }
    current_nig = {
        row["id"]: row["paths_sha256"] for row in current["entries"] if "_nig_" in row["id"]
    }
    py311_nig = {row["id"]: row["paths_sha256"] for row in py311["entries"] if "_nig_" in row["id"]}
    assert current_nig != py311_nig
