"""Regenerate the bounded full-MCMC-SV source-comparison fixture.

Run this utility from a Python environment containing the pinned source
checkout's dependencies, for example:

    PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/simfolio-numba-cache \
    python tests/parity/generate_full_mcmc_sv_source_fixture.py \
      --source-root /path/to/source-checkout \
      --source-script scripts/forecast_oos_research_gate.py \
      --source-revision 511fb82c0be43564b79df3694ee570677f3137ed \
      --output src/simfolio_forecasting_methodology/resources/test_fixtures/mcmc/full_mcmc_sv_source_parity.json

The script imports the source module and the local numerical adapter, evaluates
one deterministic synthetic series for every owned entry, and refuses to write
unless posterior samples and forecast paths are byte-identical.  It records
only generator parameters, hashes, shapes, and source identity; it does not
copy financial arrays or source data into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

EXPECTED_SOURCE_SCRIPT_SHA256 = "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997"
EXPECTED_MCMC_MANIFEST_SHA256 = "30d868665f277ecf2494ad6256c60aa1a8fd9a8fce006ab384d667e02e32c078"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _load_source_module(source_root: Path, source_script: Path) -> Any:
    sys.path.insert(0, str(source_root))
    module_name = "simfolio_pinned_source_gate"
    spec = importlib.util.spec_from_file_location(module_name, source_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source script: {source_script}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution.  Source-side decorators and Numba's dynamic
    # cache loader resolve the defining module through sys.modules.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _source_revision(source_root: Path, requested: str | None) -> str:
    if requested:
        return requested
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "--source-revision is required when source-root is not a git checkout"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-script", default="scripts/forecast_oos_research_gate.py", type=Path
    )
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/simfolio_forecasting_methodology/resources/test_fixtures/mcmc/"
            "full_mcmc_sv_source_parity.json"
        ),
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    source_script = (source_root / args.source_script).resolve()
    if not source_script.is_file():
        raise FileNotFoundError(source_script)
    source_digest = _sha256(source_script)
    if source_digest != EXPECTED_SOURCE_SCRIPT_SHA256:
        raise ValueError(
            f"source script digest mismatch: expected {EXPECTED_SOURCE_SCRIPT_SHA256}, got {source_digest}"
        )

    # Resolve identity before importing or running numerical code.  A source
    # checkout may be a detached snapshot, so callers can provide its exact
    # revision explicitly.
    source_revision = _source_revision(source_root, args.source_revision)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    with TemporaryDirectory(prefix="simfolio-source-numba-") as cache_dir:
        # A fresh directory prevents stale Numba cache modules from changing
        # the source import or leaking into the user's checkout.
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
        source = _load_source_module(source_root, source_script)
        repository_root = Path(__file__).resolve().parents[2]
        candidate_manifest = (
            repository_root / "src/simfolio_forecasting_methodology/resources/catalogs/"
            "canonical_40_full_mcmc_sv_specs.json"
        )
        candidate_manifest_digest = _sha256(candidate_manifest)
        if candidate_manifest_digest != EXPECTED_MCMC_MANIFEST_SHA256:
            raise ValueError(
                "candidate manifest digest mismatch: "
                f"expected {EXPECTED_MCMC_MANIFEST_SHA256}, got {candidate_manifest_digest}"
            )
        sys.path.insert(0, str(repository_root / "src"))
        from simfolio_forecasting_methodology.models.numerical import mcmc_sv as local

        values = np.random.default_rng(173).normal(0.0002, 0.01, 420).astype(np.float64)
        horizon_days = 17
        n_paths = 9
        simulation_seed = 9841
        rows: list[dict[str, object]] = []
        for entry in local.load_canonical_full_mcmc_sv_specs():
            candidate_id = str(entry["id"])
            source_fit = source._fit_bayesian_sbb_full_mcmc_sv_overlay(values, entry)
            local_fit = local._fit_bayesian_sbb_full_mcmc_sv_overlay(values, entry)
            if source_fit is None or local_fit is None:
                raise RuntimeError(f"source/local fit failed for {candidate_id}")
            source_posterior = np.asarray(source_fit["posterior_samples"], dtype=np.float64)
            local_posterior = np.asarray(local_fit["posterior_samples"], dtype=np.float64)
            if not np.array_equal(source_posterior, local_posterior):
                raise AssertionError(f"posterior parity failed for {candidate_id}")
            source_fit = source._apply_full_mcmc_sv_simulation_options(source_fit, entry)
            local_fit = local._apply_full_mcmc_sv_simulation_options(local_fit, entry)
            source_paths = source._simulate_bayesian_sbb_full_mcmc_sv_overlay(
                source_fit, horizon_days, n_paths, np.random.default_rng(simulation_seed)
            )
            local_paths = local._simulate_bayesian_sbb_full_mcmc_sv_overlay(
                local_fit, horizon_days, n_paths, np.random.default_rng(simulation_seed)
            )
            if not np.array_equal(source_paths, local_paths):
                raise AssertionError(f"simulation parity failed for {candidate_id}")
            rows.append(
                {
                    "id": candidate_id,
                    "mcmc_acceptance_rate": float(local_fit["mcmc_acceptance_rate"]),
                    "mcmc_retained_samples": int(local_fit["mcmc_retained_samples"]),
                    "paths_sha256": _array_digest(local_paths),
                    "paths_shape": list(local_paths.shape),
                    "posterior_samples_sha256": _array_digest(local_posterior),
                    "posterior_samples_shape": list(local_posterior.shape),
                }
            )

    payload = {
        "entries": rows,
        "fixture_input": {
            "generator": "np.random.default_rng(173).normal(0.0002,0.01,420).astype(np.float64)",
            "length": int(values.size),
        },
        "schema_version": 1,
        "simulation": {
            "horizon_days": horizon_days,
            "n_paths": n_paths,
            "seed": simulation_seed,
        },
        "source_revision": source_revision,
        "source_script": args.source_script.as_posix(),
        "source_script_sha256": source_digest,
        "source_candidate_manifest": (
            "src/simfolio_forecasting_methodology/resources/catalogs/"
            "canonical_40_full_mcmc_sv_specs.json"
        ),
        "source_candidate_manifest_sha256": candidate_manifest_digest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} source-parity entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
