"""Bounded numerical parity against source-derived canonical SV fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from simfolio_forecasting_methodology.models.numerical.mcmc_sv import (
    canonical_full_mcmc_sv_ids,
    fit_full_mcmc_sv,
    load_canonical_full_mcmc_sv_specs,
    simulate_full_mcmc_sv,
)

_FIXTURE = Path(__file__).parents[1] / "data" / "full_mcmc_sv_source_parity.json"


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def test_owned_full_mcmc_sv_catalogue_is_exactly_40_source_entries():
    entries = load_canonical_full_mcmc_sv_specs()
    assert len(entries) == 40
    assert tuple(entry["id"] for entry in entries) == canonical_full_mcmc_sv_ids()
    assert {entry["type"] for entry in entries} == {"bayesian_sbb_full_mcmc_sv_overlay"}


def test_full_mcmc_sv_fit_and_paths_match_source_fixture():
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
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
