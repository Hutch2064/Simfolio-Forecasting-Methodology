#!/usr/bin/env python3
"""Regenerate the bounded source comparison for the full-INLA BDES closure.

The command imports the pinned source app and the local numerical adapter in a
fresh process-local Numba cache.  It resolves source identity before fitting,
registers the imported module through the normal import machinery, and writes
retained numerical arrays with their shapes, hashes, and scalar metadata.  It
never copies source code or private financial data into the repository.

Example::

    PYTHONDONTWRITEBYTECODE=1 python tools/generate_inla_source_fixture.py \
      --source-root /path/to/source-research \
      --source-revision 773bc1c325559e6bf57a567f1d8bf473a3427fbc \
      --input-fixture src/simfolio_forecasting_methodology/resources/test_fixtures/inla/source_input.json \
      --output /tmp/inla-source-parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

EXPECTED_SOURCE_ENGINE_SHA256 = "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        "values": array.tolist(),
    }


def _load_frozen_input(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"frozen input must be an object: {path}")
    record = payload.get("input_values")
    if not isinstance(record, dict):
        raise TypeError(f"frozen input record missing: {path}")
    values = np.asarray(record.get("values"), dtype=np.float64)
    if list(values.shape) != record.get("shape"):
        raise ValueError(f"frozen input shape mismatch: {path}")
    if str(values.dtype) != record.get("dtype"):
        raise ValueError(f"frozen input dtype mismatch: {path}")
    digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    if digest != record.get("sha256"):
        raise ValueError(f"frozen input digest mismatch: {path}")
    if values.ndim != 1 or values.size < 5 or not np.all(np.isfinite(values)):
        raise ValueError(f"frozen input values are invalid: {path}")
    return values


def _source_revision(source_root: Path, requested: str | None) -> str:
    if requested:
        return requested
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "--source-revision is required when source-root is not a git checkout"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--input-fixture",
        type=Path,
        help="JSON fixture containing the frozen source input vector",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/simfolio_forecasting_methodology/resources/test_fixtures/inla/source_parity.json"
        ),
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    input_fixture = args.input_fixture or (
        repository_root
        / "src/simfolio_forecasting_methodology/resources/test_fixtures/inla/source_input.json"
    )
    input_fixture = input_fixture.resolve()
    values = _load_frozen_input(input_fixture)

    source_root = args.source_root.resolve()
    source_engine_path = source_root / "app" / "engine.py"
    if not source_engine_path.is_file():
        raise FileNotFoundError(source_engine_path)
    source_digest = _sha256(source_engine_path)
    if source_digest != EXPECTED_SOURCE_ENGINE_SHA256:
        raise ValueError(
            f"source engine digest mismatch: expected {EXPECTED_SOURCE_ENGINE_SHA256}, got {source_digest}"
        )
    source_revision = _source_revision(source_root, args.source_revision)

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    with TemporaryDirectory(prefix="simfolio-inla-source-numba-") as cache_dir:
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
        sys.path.insert(0, str(source_root))
        source = importlib.import_module("app.engine")
        sys.path.insert(0, str(repository_root / "src"))
        from simfolio_forecasting_methodology.models.numerical.inla_bdes import (
            INLA_CANDIDATE,
            fit_bdes_full_inla,
            simulate_bdes_full_inla,
            source_candidate_digest,
        )

        source_engine = source.SimfolioEngine()
        source_candidate = source_engine._bdes_cagr_candidate()
        if source_candidate != INLA_CANDIDATE:
            raise AssertionError("source candidate differs from the packaged raw descriptor")
        source_fit = source_engine._fit_bdes_non_mcmc_sv_forecast_base(values, source_candidate)
        local_fit = fit_bdes_full_inla(values, INLA_CANDIDATE)
        if source_fit is None:
            raise RuntimeError("source full-INLA fit returned no fit")

        comparison: dict[str, Any] = {}
        for key in (
            "posterior_samples",
            "posterior_sample_weights",
            "innovation_pool",
            "standardized_residuals",
        ):
            source_value = np.asarray(source_fit[key], dtype=np.float64)
            local_value = np.asarray(local_fit[key], dtype=np.float64)
            if not np.array_equal(source_value, local_value):
                raise AssertionError(f"source/local fit parity failed for {key}")
            comparison[key] = _array_record(local_value)
        for key in ("q_last", "phis", "b", "q_var"):
            source_value = np.asarray(source_fit["bdes_multiscale_vol"][key], dtype=np.float64)
            local_value = np.asarray(local_fit["bdes_multiscale_vol"][key], dtype=np.float64)
            if not np.array_equal(source_value, local_value):
                raise AssertionError(f"source/local BDES parity failed for {key}")
            comparison[f"bdes_{key}"] = _array_record(local_value)
        for key in (
            "residual_last",
            "residual_phi",
            "residual_innovation_sd",
            "residual_common_loading",
        ):
            source_value = source_fit["bdes_multiscale_vol"][key]
            local_value = local_fit["bdes_multiscale_vol"][key]
            if source_value != local_value:
                raise AssertionError(f"source/local BDES scalar parity failed for {key}")
            comparison[f"bdes_{key}"] = float(local_value)
        source_center = np.asarray(source_fit["posterior_center"], dtype=np.float64)
        local_center = np.asarray(local_fit["posterior_center"], dtype=np.float64)
        if not np.array_equal(source_center, local_center):
            raise AssertionError("source/local posterior center parity failed")
        comparison["posterior_center"] = [float(value) for value in local_center]
        source_paths = source_engine._simulate_full_mcmc_sv_forecast_log_paths(
            source_fit, 10, 4, np.random.default_rng(123)
        )
        local_paths = simulate_bdes_full_inla(local_fit, 4, 10, 123)
        if not np.array_equal(source_paths, local_paths):
            raise AssertionError("source/local full-INLA path parity failed")
        comparison["paths"] = _array_record(local_paths)

        control_values = np.log1p(np.clip(np.expm1(values), -0.999999, None))
        control_fit = source_engine._fit_bdes_non_mcmc_sv_forecast_base(
            control_values, source_candidate
        )
        if control_fit is None:
            raise RuntimeError("source control fit returned no fit")
        control_seed = source_engine._deterministic_seed(
            "forecast_final_fixed", str(source_candidate["id"]), 10, 4
        )
        control_paths = source_engine._simulate_full_mcmc_sv_forecast_log_paths(
            control_fit,
            10,
            4,
            np.random.default_rng(control_seed),
        )
        comparison["adapter_control_paths"] = _array_record(control_paths)
        comparison["source_candidate_digest"] = source_candidate_digest()
        comparison["source_candidate_key_count"] = len(INLA_CANDIDATE)
        comparison["source_fit_path_exact"] = True

    payload = {
        "schema_version": 1,
        "source_engine_path": "source-research/app/engine.py",
        "source_engine_sha256": source_digest,
        "source_revision": source_revision,
        "source_candidate_method": "SimfolioEngine._bdes_cagr_candidate",
        "generator": {
            "expression": "np.sin(np.arange(180,dtype=float)/8)*0.01 + np.random.default_rng(2).normal(0,0.01,180)",
            "input_fixture": "source_input.json",
            "input_values_sha256": hashlib.sha256(
                np.ascontiguousarray(values).tobytes()
            ).hexdigest(),
            "input_values_shape": list(values.shape),
            "input_values_dtype": str(values.dtype),
            "length": int(values.size),
            "fit_min_obs": 5,
            "simulations": 4,
            "horizon_days": 10,
            "simulation_seed": 123,
        },
        "adapter_control": {
            "source_wrapper_path": "tmp/oos_all_daily_exact_crps.py",
            "source_wrapper_sha256": "88aaa532d8a0a63e24964305a26ee6fa45d0d9099dd5da800ab691afad583ea7",
            "fit_method": "SimfolioEngine._fit_bdes_non_mcmc_sv_forecast_base",
            "simulation_method": "SimfolioEngine._simulate_full_mcmc_sv_forecast_log_paths",
            "input_transform": "np.log1p(np.clip(np.expm1(values), -0.999999, None))",
            "seed_label": "forecast_final_fixed",
            "simulations": 4,
            "horizon_days": 10,
            "context_seed": 999,
            "simulation_seed": int(
                source_engine._deterministic_seed(
                    "forecast_final_fixed", str(INLA_CANDIDATE["id"]), 10, 4
                )
            ),
        },
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote source/local INLA parity fixture to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
