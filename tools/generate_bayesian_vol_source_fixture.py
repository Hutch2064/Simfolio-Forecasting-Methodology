"""Generate the bounded Bayesian-volatility source parity fixture.

Usage (from the destination checkout, with dependencies installed):

    NUMBA_CACHE_DIR="$PWD/.numba-cache" PYTHONDONTWRITEBYTECODE=1 \\
      python tools/generate_bayesian_vol_source_fixture.py \\
      --source-root /path/to/source-research \\
      --output src/simfolio_forecasting_methodology/resources/test_fixtures/portfolio/bayesian_vol_source_parity.json

The source checkout is read-only.  The redirected Numba cache must point into
this checkout or another disposable task-owned directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from importlib import resources
from pathlib import Path
from typing import Any

# These must be set before loading the source engine, which imports Numba.  The
# tool owns the redirect even when the caller supplied a different cache path.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = str(Path(__file__).resolve().parents[1] / ".numba-cache")

import numpy as np
import pandas as pd

ENGINE_SHA256 = "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665"
RESEARCH_GATE_SHA256 = "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997"
SOURCE_REVISION = "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
CACHE_POLICY_LABEL = "task_owned_redirected_cache"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_record(values: Any) -> dict[str, Any] | None:
    if values is None:
        return None
    array = np.ascontiguousarray(np.asarray(values))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _fit_summary(fit: dict[str, Any]) -> dict[str, Any]:
    base = dict(fit.get("base_fit", {}) or {})
    curve = dict(fit.get("curve_fit", {}) or {})
    meta = dict(fit.get("meta", {}) or {})
    base_keys = (
        "sample_mu",
        "sigma",
        "posterior_mean",
        "posterior_sd",
        "mu_cap",
        "sample_mu_days",
        "nonnegative_drift",
        "posterior_mu_draws",
    )
    curve_keys = (
        "curve_type",
        "persistence",
        "last_variance_x",
        "long_variance_x",
        "last_log_variance_x",
        "long_log_variance_x",
        "last_log_variance_error_variance",
        "fit_status",
        "objective",
    )
    meta_keys = (
        "method",
        "overlay_model",
        "overlay_persistence",
        "overlay_half_life_days",
        "overlay_fit_status",
        "ml_model",
        "feature_set",
        "feature_count",
        "training_rows",
    )
    summary: dict[str, Any] = {
        "base": {key: base[key] for key in base_keys if key in base},
        "curve": {key: curve[key] for key in curve_keys if key in curve},
        "meta": {key: meta[key] for key in meta_keys if key in meta},
        "standardized_residuals": _array_record(base.get("standardized_residuals")),
    }
    if "sigma_x" in curve:
        summary["curve_sigma_x"] = _array_record(curve.get("sigma_x"))
    if "arch_params" in curve:
        summary["arch_params"] = {
            str(key): _finite_float(value) for key, value in dict(curve["arch_params"]).items()
        }
    if "sigma_clip_bounds_x" in curve:
        summary["sigma_clip_bounds_x"] = [
            _finite_float(value) for value in curve["sigma_clip_bounds_x"]
        ]
    if "vol_anchor" in fit:
        anchor = dict(fit["vol_anchor"])
        summary["vol_anchor"] = {
            key: anchor[key]
            for key in (
                "model",
                "target",
                "anchor_log_variance_x",
                "anchor_log_variance_error_variance",
                "predicted_sigma_x",
                "training_rows",
                "feature_count",
                "factor_model",
                "factor_cols",
                "anchor_weight",
                "pre_anchor_log_variance_x",
                "pre_anchor_log_variance_error_variance",
            )
            if key in anchor
        }
    return summary


def _source_module(source_root: Path):
    sys.path.insert(0, str(source_root / "scripts"))
    return importlib.import_module("forecast_oos_research_gate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    engine_path = source_root / "app" / "engine.py"
    gate_path = source_root / "scripts" / "forecast_oos_research_gate.py"
    if _sha256_file(engine_path) != ENGINE_SHA256:
        raise SystemExit("source engine SHA-256 does not match pinned artifact")
    if _sha256_file(gate_path) != RESEARCH_GATE_SHA256:
        raise SystemExit("source research-gate SHA-256 does not match pinned artifact")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from simfolio_forecasting_methodology.models.portfolio import bayesian_vol as local

    source = _source_module(source_root)
    factor_frame = local._load_packaged_factor_frame("ff6")
    factor_resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources",
        "data",
        "canonical_snapshot",
        "app",
        "factor_data",
        "french_daily.csv.gz",
    )

    # Exercise the original source loader against the exact packaged bytes,
    # while disabling its refresh callback in memory.  This proves the local
    # frame before it is passed as an explicit fit input and avoids source
    # checkout writes or remote refreshes.
    # Resolve the engine already loaded by the digest-verified source gate.
    source_engine_module = sys.modules[source.SimfolioEngine.__module__]

    previous_refresh = source_engine_module.refresh_factor_frame
    source_engine_module.refresh_factor_frame = lambda key, frame: (
        frame.copy(deep=True) if frame is not None else None
    )
    try:
        with resources.as_file(factor_resource) as factor_path:

            class FrozenFactorSourceEngine(source.SimfolioEngine):
                def _bundled_factor_path(self, filename):
                    if filename == self.FRENCH_FACTOR_BUNDLE:
                        return str(factor_path)
                    return super()._bundled_factor_path(filename)

            source_engine = FrozenFactorSourceEngine()
            source_factor_frame = source_engine._load_french_factor_frame()
    finally:
        source_engine_module.refresh_factor_frame = previous_refresh

    if source_factor_frame is None or not factor_frame.equals(source_factor_frame):
        raise SystemExit("packaged factor frame differs from original source loader frame")
    rng = np.random.default_rng(123)
    n_obs = 720
    values = (
        0.00015
        + 0.010 * np.sin(np.arange(n_obs, dtype=np.float64) / 17.0)
        + rng.normal(0.0, 0.007, n_obs)
    ).astype(np.float64)
    dates = pd.bdate_range("2000-01-03", periods=n_obs)
    horizon = 5
    simulations = 4
    origin = "2026-05-13"
    cases: list[dict[str, Any]] = []
    for model_id in local.BAYESIAN_VOL_MODEL_IDS:
        candidate = dict(local.SOURCE_CANDIDATE_SPECS[model_id])
        if candidate["type"] == "bayesian_sbb_ml_vol_overlay":
            source_fit = source._fit_bayesian_sbb_ml_vol_overlay(
                pd.Series(values, index=dates),
                candidate,
                source_engine,
                factor_frame=source_factor_frame,
            )
        else:
            source_fit = source._fit_bayesian_sbb_vol_overlay(values, candidate)
            if candidate.get("vol_anchor_model"):
                source_fit = source._apply_bayesian_sbb_vol_overlay_vol_anchor(
                    source_fit,
                    candidate,
                    train=pd.Series(values, index=dates),
                    engine=source_engine,
                )
        if source_fit is None:
            raise SystemExit(f"source fit failed for {model_id}")
        source_seed = source.SimfolioEngine._deterministic_seed(
            "forecast_oos_candidate",
            origin,
            tuple(range(1, horizon + 1)),
            model_id,
            simulations,
        )
        source_rng = np.random.default_rng(source_seed)
        if candidate["type"] == "bayesian_sbb_ml_vol_overlay":
            paths = source._simulate_bayesian_sbb_ml_vol_overlay(
                source_fit, horizon, simulations, source_rng
            )
        else:
            paths = source._simulate_bayesian_sbb_vol_overlay(
                source_fit, horizon, simulations, source_rng
            )
        cases.append(
            {
                "model_id": model_id,
                "source_specification": candidate,
                "source_forecast_seed": int(source_seed),
                "fit": _fit_summary(source_fit),
                "paths": np.asarray(paths, dtype=np.float64).tolist(),
                "paths_array": _array_record(paths),
            }
        )
    payload = {
        "schema": "bayesian-vol-source-parity-v1",
        "source_artifacts": {
            "engine": {"sha256": ENGINE_SHA256, "path": "app/engine.py"},
            "research_gate": {
                "sha256": RESEARCH_GATE_SHA256,
                "path": "scripts/forecast_oos_research_gate.py",
            },
            "revision": SOURCE_REVISION,
        },
        "source_function_digest": local.SOURCE_FUNCTIONS_SHA256,
        "source_function_names": list(local.SOURCE_FUNCTION_NAMES),
        "source_function_line_ranges": {
            name: list(ranges) for name, ranges in local.SOURCE_FUNCTION_LINE_RANGES.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "numba_cache_dir": CACHE_POLICY_LABEL,
        },
        "training": {
            "n_obs": n_obs,
            "values": values.tolist(),
            "dates": [value.strftime("%Y-%m-%d") for value in dates],
            "values_array": _array_record(values),
            "factor_snapshot": "resources/data/canonical_snapshot/app/factor_data/french_daily.csv.gz",
            "factor_snapshot_sha256": _sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/simfolio_forecasting_methodology/resources/data/canonical_snapshot/app/factor_data/french_daily.csv.gz"
            ),
            "source_loader_frame": {
                "rows": len(source_factor_frame),
                "columns": list(source_factor_frame.columns),
                "values_array": _array_record(source_factor_frame.to_numpy(dtype=np.float64)),
                "frame_equals_packaged": True,
            },
        },
        "context": {
            "origin_date": origin,
            "horizon_days": horizon,
            "simulations": simulations,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(cases),
                "source_function_digest": local.SOURCE_FUNCTIONS_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
