"""Generate a bounded source parity fixture for the two FF6 residual models.

The source checkout is read-only.  The source engine is pointed at the
packaged factor bytes and its refresh callback is replaced in memory so this
generator cannot refresh or write source-side factor data.

Usage:

    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
      python tools/generate_factor_residual_source_fixture.py \
      --source-root /path/to/source-research \
      --output src/simfolio_forecasting_methodology/resources/test_fixtures/portfolio/factor_residual_source_parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

# Numba is imported by the source engine.  Always redirect its cache before
# that import, regardless of a caller's inherited environment.
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


def _array_record(values: Any, *, include_values: bool = False) -> dict[str, Any] | None:
    if values is None:
        return None
    array = np.ascontiguousarray(np.asarray(values))
    record: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    if include_values:
        record["values"] = array.tolist()
    return record


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _nested_fit_summary(
    fit: Mapping[str, Any] | None, *, include_values: bool = False
) -> dict[str, Any] | None:
    if fit is None:
        return None
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
        "base": {key: _plain(base[key]) for key in base_keys if key in base},
        "curve": {key: _plain(curve[key]) for key in curve_keys if key in curve},
        "meta": {key: _plain(meta[key]) for key in meta_keys if key in meta},
        "standardized_residuals": _array_record(
            base.get("standardized_residuals"), include_values=include_values
        ),
    }
    if "sigma_x" in curve:
        summary["curve_sigma_x"] = _array_record(
            curve.get("sigma_x"), include_values=include_values
        )
    if "arch_params" in curve:
        summary["arch_params"] = {
            str(key): float(value) for key, value in dict(curve["arch_params"]).items()
        }
    if "sigma_clip_bounds_x" in curve:
        summary["sigma_clip_bounds_x"] = [float(value) for value in curve["sigma_clip_bounds_x"]]
    return summary


def _fit_summary(fit: Mapping[str, Any], *, include_values: bool = False) -> dict[str, Any]:
    model = fit.get("model")
    steps = getattr(model, "named_steps", {})
    scaler = steps.get("standardscaler")
    ridge = steps.get("ridge")
    return {
        "factor_values": _array_record(fit.get("factor_values"), include_values=include_values),
        "rf_values": _array_record(fit.get("rf_values"), include_values=include_values),
        "residuals": _array_record(fit.get("residuals"), include_values=include_values),
        "block_length": int(fit.get("block_length", 0)),
        "meta": _plain(fit.get("meta", {})),
        "model": {
            "steps": list(steps),
            "scaler_mean": _array_record(
                getattr(scaler, "mean_", None), include_values=include_values
            ),
            "scaler_scale": _array_record(
                getattr(scaler, "scale_", None), include_values=include_values
            ),
            "ridge_coef": _array_record(
                getattr(ridge, "coef_", None), include_values=include_values
            ),
            "ridge_intercept": _plain(getattr(ridge, "intercept_", None)),
        },
        "residual_overlay_fit": _nested_fit_summary(
            fit.get("residual_overlay_fit"), include_values=include_values
        ),
    }


def _source_module(source_root: Path) -> Any:
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
    from simfolio_forecasting_methodology.models.portfolio import factor_residual as local

    source = _source_module(source_root)
    factor_frame = local.load_packaged_factor_frame("ff6")
    factor_resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources",
        "data",
        "canonical_snapshot",
        "app",
        "factor_data",
        "french_daily.csv.gz",
    )

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
    for model_id in local.FACTOR_RESIDUAL_MODEL_IDS:
        candidate = dict(local.RAW_FACTOR_RESIDUAL_SPECS[model_id])
        source_fit = source._fit_factor_residual_sbb(
            pd.Series(values, index=dates),
            candidate,
            source_engine,
            factor_frame=source_factor_frame,
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
        paths = source._simulate_factor_residual_sbb(
            source_fit,
            horizon,
            simulations,
            np.random.default_rng(source_seed),
        )
        dispatcher_caches = {"fits": {}, "terminals": {}}
        dispatcher_horizons = tuple(range(1, horizon + 1))
        dispatcher_terminals = source._terminal_logs_for_candidate(
            source_engine,
            candidate,
            pd.Series(values, index=dates),
            values,
            dispatcher_horizons,
            simulations,
            (origin, dispatcher_horizons),
            dispatcher_caches,
        )
        if dispatcher_terminals is None:
            raise SystemExit(f"source dispatcher failed for {model_id}")
        dispatcher_fit_key = ("factor_residual_sbb", model_id, n_obs)
        dispatcher_fit = dispatcher_caches["fits"].get(dispatcher_fit_key)
        if dispatcher_fit is None:
            raise SystemExit(f"source dispatcher did not retain fit for {model_id}")
        if _fit_summary(dispatcher_fit) != _fit_summary(source_fit):
            raise SystemExit(f"source dispatcher fit differs from direct fit for {model_id}")
        dispatcher_terminal_records = {
            str(horizon_key): {
                "values": np.asarray(values_for_horizon, dtype=np.float64).tolist(),
                "array": _array_record(values_for_horizon),
            }
            for horizon_key, values_for_horizon in dispatcher_terminals.items()
        }
        cases.append(
            {
                "dispatcher": {
                    "fit": _fit_summary(dispatcher_fit, include_values=True),
                    "horizons": list(dispatcher_horizons),
                    "terminals": dispatcher_terminal_records,
                },
                "model_id": model_id,
                "source_specification": candidate,
                "source_forecast_seed": int(source_seed),
                "fit": _fit_summary(source_fit, include_values=True),
                "paths": np.asarray(paths, dtype=np.float64).tolist(),
                "paths_array": _array_record(paths),
            }
        )

    snapshot_path = (
        Path(__file__).resolve().parents[1]
        / "src/simfolio_forecasting_methodology/resources/data/canonical_snapshot/app/factor_data/french_daily.csv.gz"
    )
    payload = {
        "schema": "factor-residual-source-parity-v1",
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
            "factor_snapshot_sha256": _sha256_file(snapshot_path),
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
