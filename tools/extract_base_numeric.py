#!/usr/bin/env python3
"""Generate a source-reference fixture for the canonical base closure.

The source checkout is a development-only input.  The output contains only
source-generated arrays and metadata, so package tests do not import that
checkout.  Numba's cache directory is redirected into a temporary directory
inside the destination clone before the source engine is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from simfolio_forecasting_methodology.models.numerical.base_models import (
    historical_base_seed,
)

SOURCE_REVISION = "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
SOURCE_ENGINE_SHA256 = "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
DEFAULT_MEAN_MODELS = ("expanding_sample_mean",)
VOL_MODELS = (
    "constant_sample_volatility",
    "garch_1_1_volatility",
    "gjr_tarch_1_1_volatility",
    "egarch_1_1_volatility",
)
PARAMETRIC_INNOVATIONS = (
    "gaussian_iid_standardized_innovations",
    "student_t_standardized_innovations",
    "skew_t_standardized_innovations",
)
EMPIRICAL_VARIANTS = (
    ("stationary_bootstrap", "filtered_empirical_tail"),
    ("stationary_bootstrap", "automated_evt_pot_gpd_tail"),
    ("iid", "filtered_empirical_tail"),
    ("iid", "automated_evt_pot_gpd_tail"),
)
PARAM_KEYS = ("sigma_x", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu", "eta", "lambda")


@contextmanager
def _isolated_runtime(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix=".base-numba-cache-", dir=destination))
    prior_cache = os.environ.get("NUMBA_CACHE_DIR")
    prior_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    prior_flag = sys.dont_write_bytecode
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = prior_flag
        if prior_cache is None:
            os.environ.pop("NUMBA_CACHE_DIR", None)
        else:
            os.environ["NUMBA_CACHE_DIR"] = prior_cache
        if prior_bytecode is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = prior_bytecode
        shutil.rmtree(cache_dir, ignore_errors=True)


def _source_path_and_tail(resampling: str | None, tail: str) -> tuple[str, str]:
    if resampling is None:
        return "parametric_monte_carlo", "native_distribution_tail"
    path = (
        "stationary_bootstrap_standardized_residuals"
        if resampling == "stationary_bootstrap"
        else "filtered_historical_simulation"
    )
    return path, tail


def _candidate_rows(mean_models: tuple[str, ...]) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for mean_model in mean_models:
        for vol in VOL_MODELS:
            for innovation in PARAMETRIC_INNOVATIONS:
                rows.append(
                    {
                        "model_id": f"{mean_model}|{vol}|{innovation}|parametric",
                        "mean_model": mean_model,
                        "vol_model": vol,
                        "innovation_method": innovation,
                        "resampling": None,
                        "tail_method": "parametric",
                    }
                )
            for resampling, tail in EMPIRICAL_VARIANTS:
                rows.append(
                    {
                        "model_id": f"{mean_model}|{vol}|empirical|{resampling}|{tail}",
                        "mean_model": mean_model,
                        "vol_model": vol,
                        "innovation_method": "empirical_standardized_residuals",
                        "resampling": resampling,
                        "tail_method": tail,
                    }
                )
    return rows


def _training() -> np.ndarray:
    rng = np.random.default_rng(20260823)
    days = np.arange(600, dtype=np.float64)
    return (
        0.00015
        + 0.006 * rng.standard_normal(days.size)
        + 0.0015 * np.sin(days / 17.0)
        + 0.0008 * np.cos(days / 43.0)
    ).astype(np.float64)


def _source_arrays(source_root: Path, values: np.ndarray, rows: list[dict[str, str | None]], origin: str, horizon: int, simulations: int) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(source_root))
    try:
        source_engine = importlib.import_module("app.engine")
        engine = source_engine.SimfolioEngine()
        source_fit = []
        source_paths = []
        source_params = []
        source_status: list[str] = []
        seed_values: list[int] = []
        for row in rows:
            source_mean = engine._fit_auto_forecast_mean(values, str(row["mean_model"]))
            path, source_tail = _source_path_and_tail(row["resampling"], str(row["tail_method"]))
            candidate = {
                "id": row["model_id"],
                "type": "base",
                "mean_model": row["mean_model"],
                "vol_model": row["vol_model"],
                "innovation_method": row["innovation_method"],
                "tail_method": source_tail,
                "path_generator": path,
            }
            fit = engine._fit_auto_forecast_base(
                pd.Series(values),
                str(row["mean_model"]),
                str(row["vol_model"]),
                str(row["innovation_method"]),
                mean_fit=source_mean,
                candidate=candidate,
            )
            if fit is None:
                raise RuntimeError(f"source fit returned None for {row['model_id']}")
            seed = historical_base_seed(str(row["model_id"]), origin, horizon, simulations)
            path_values = engine._simulate_candidate_log_paths(
                fit, source_tail, path, horizon, simulations, np.random.default_rng(seed)
            )
            sigma = np.asarray(fit["vol_fit"]["sigma_x"], dtype=np.float64)
            residuals = np.asarray(fit["residuals"], dtype=np.float64)
            standardized = np.asarray(fit["standardized_residuals"], dtype=np.float64)
            if residuals.size != values.size or sigma.size != values.size or standardized.size != values.size:
                raise RuntimeError(f"source fixture arrays are not fixed-width for {row['model_id']}")
            params = fit["vol_fit"].get("params", {}) or {}
            source_fit.append((float(fit["mu"]), residuals, standardized, sigma))
            source_paths.append(np.asarray(path_values, dtype=np.float64))
            source_params.append([float(params.get(key, np.nan)) for key in PARAM_KEYS])
            source_status.append(str(fit["vol_fit"].get("fit_status", "")))
            seed_values.append(seed)
        mu = np.asarray([item[0] for item in source_fit], dtype=np.float64)
        residuals = np.stack([item[1] for item in source_fit])
        standardized = np.stack([item[2] for item in source_fit])
        sigma = np.stack([item[3] for item in source_fit])
        return {
            "training_log_returns": values,
            "model_ids": np.asarray([str(row["model_id"]) for row in rows]),
            "fit_mu": mu,
            "fit_residuals": residuals,
            "fit_standardized_residuals": standardized,
            "fit_sigma_x": sigma,
            "fit_params": np.asarray(source_params, dtype=np.float64),
            "paths": np.stack(source_paths),
            "seed": np.asarray(seed_values, dtype=np.uint64),
            "fit_status": np.asarray(source_status),
        }
    finally:
        sys.path.remove(str(source_root))


def generate(source_root: Path, output: Path, mean_models: tuple[str, ...]) -> None:
    values = _training()
    rows = _candidate_rows(mean_models)
    origin = "2026-05-13"
    horizon = 16
    simulations = 16
    with _isolated_runtime(output.parent / ".base-source-runtime"):
        source_arrays = _source_arrays(source_root, values, rows, origin, horizon, simulations)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **source_arrays)
    metadata = {
        "schema": "canonical-base-source-fixture-v1",
        "source_revision": SOURCE_REVISION,
        "source_engine_sha256": SOURCE_ENGINE_SHA256,
        "source_root": "development-only source-research checkout (path intentionally omitted)",
        "source_execution": "app.engine.SimfolioEngine._fit_auto_forecast_mean/_fit_auto_forecast_base/_simulate_candidate_log_paths",
        "origin": origin,
        "horizon_days": horizon,
        "simulations": simulations,
        "training_generator": "default_rng(20260823), 600 synthetic finite daily log returns",
        "model_count": len(rows),
        "mean_models": list(mean_models),
        "array_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "cache_policy": "NUMBA_CACHE_DIR temporary under destination clone; PYTHONDONTWRITEBYTECODE=1",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": metadata["array_sha256"], "models": len(rows)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mean-models",
        default=",".join(DEFAULT_MEAN_MODELS),
        help="comma-delimited canonical mean IDs; default generates the first 28-row batch",
    )
    args = parser.parse_args()
    mean_models = tuple(value.strip() for value in str(args.mean_models).split(",") if value.strip())
    if not mean_models:
        raise ValueError("at least one mean model is required")
    generate(args.source_root.resolve(), args.output.resolve(), mean_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
