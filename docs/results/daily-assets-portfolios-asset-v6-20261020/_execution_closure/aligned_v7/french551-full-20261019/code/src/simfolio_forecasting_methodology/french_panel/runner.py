"""Bounded, resumable research execution for the frozen French panel.

No model is selected or registered here. A score exists only after every
series/origin and every series/horizon origin count satisfies the frozen plan.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..evaluation import empirical_crps_by_horizon
from ..results.checkpoint import (
    CheckpointCorruptError,
    CheckpointIdentityError,
    stable_digest,
    stable_json_bytes,
)
from ..runner import (
    TERMINAL_FORECAST_SEMANTICS,
    ForecastContext,
    ForecastModel,
    PortfolioPolicy,
    TrainingData,
    _dependency_identity,
)
from ..seeds import coherent_daily_seed
from .data import PROTOCOL_ID, SIMULATIONS, load_dates, load_panel, load_returns, schedule


@lru_cache(maxsize=1)
def _native_dependency_identity() -> dict:
    """Bind retained compiler headers and transitive native binaries as installed."""
    packages = {}
    for name in ("pybind11", "pyvinecopulib"):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
            continue
        artifacts = {}
        for entry in distribution.files or ():
            if Path(entry).suffix in {".h", ".hpp", ".so", ".dylib", ".pyd"}:
                artifacts[str(entry)] = hashlib.sha256(distribution.locate_file(entry).read_bytes()).hexdigest()
        packages[name] = {"version": distribution.version, "native_artifacts": artifacts}
    return packages


def dependency_identity(model) -> dict:
    return {**_dependency_identity(model, {}), "french_native_dependencies": _native_dependency_identity()}


def _source_closure() -> dict[str, str]:
    """Bind execution/scoring/seeds, adapters, and retained protocol resources."""
    root = Path(__file__).resolve().parents[1]
    paths = set(root.rglob("*.py"))
    for directory in ("protocols", "french1926"):
        paths.update(path for path in (root / "resources" / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix not in {".pyc", ".pyo"})
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(stable_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest(root: Path, identity: dict) -> str:
    fingerprint = stable_digest(identity)
    path = root / "execution.json"
    if path.exists():
        try:
            stored = json.loads(path.read_text())
            if stable_digest(stored["identity"]) != stored["fingerprint"]:
                raise CheckpointCorruptError("execution manifest checksum mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckpointCorruptError("invalid execution manifest") from exc
        if stored["fingerprint"] != fingerprint:
            raise CheckpointIdentityError("panel/model/implementation/runtime/protocol identity drift")
    else:
        if any(root.rglob("*.npz")) or any((root / "errors").rglob("*.json")):
            raise CheckpointCorruptError("checkpoint records exist without an execution manifest")
        _atomic_json(path, {"identity": identity, "fingerprint": fingerprint})
    return fingerprint


def _write_checkpoint(path: Path, identity: dict, losses: np.ndarray, diagnostics: dict) -> None:
    values = np.asarray(losses, dtype="<f8")
    metadata = {
        "identity": identity,
        "diagnostics": diagnostics,
        "loss_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    envelope = {"payload": metadata, "checksum": stable_digest(metadata)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, losses=values, metadata=stable_json_bytes(envelope).decode())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_checkpoint(path: Path, identity: dict) -> tuple[np.ndarray, dict]:
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            if set(checkpoint.files) != {"losses", "metadata"}:
                raise ValueError("unexpected checkpoint arrays")
            losses = checkpoint["losses"]
            envelope = json.loads(str(checkpoint["metadata"].item()))
        metadata = envelope["payload"]
        if stable_digest(metadata) != envelope["checksum"]:
            raise ValueError("metadata checksum mismatch")
        if metadata["identity"] != identity:
            raise ValueError("task identity mismatch")
        if losses.dtype != np.dtype("<f8") or losses.shape != (identity["origin"]["max_horizon"],):
            raise ValueError("loss vector shape/dtype mismatch")
        if not np.all(np.isfinite(losses)) or np.any(losses < 0):
            raise ValueError("invalid loss vector")
        if hashlib.sha256(losses.tobytes()).hexdigest() != metadata["loss_sha256"]:
            raise ValueError("loss checksum mismatch")
        diagnostics = metadata["diagnostics"]
        elapsed = diagnostics["forecast_and_score_seconds"]
        if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid forecast and scoring timing")
        return losses, diagnostics
    except Exception as exc:
        # Corrupt files are never silently replaced by a fresh simulation.
        raise CheckpointCorruptError(f"invalid task checkpoint: {path}") from exc


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values)
    # A bytes owner prevents adapters from re-enabling writeability or reaching
    # held-out values through ndarray.base (a read-only view alone does neither).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _forecast(model: ForecastModel, series_id: str, values: np.ndarray,
              dates: np.ndarray, origin, seed: int) -> tuple[np.ndarray, dict]:
    stop = int(origin.position) + 1
    horizon = int(origin.max_horizon)
    history = _readonly_copy(np.asarray(values[:stop], dtype=np.float64))
    training = TrainingData(
        portfolio_log_returns=history,
        asset_log_returns=history.reshape(-1, 1),
        policy=PortfolioPolicy((series_id,), (1.0,), "none"),
        training_dates=_readonly_copy(dates[:stop]),
    )
    training.validate()
    # Targets and training use observed French dates. Preserve the established
    # experiment.py policy calendar separately: synthetic weekdays after origin.
    future_dates = pd.bdate_range(
        start=pd.Timestamp(origin.origin_date) + pd.offsets.BDay(1), periods=horizon
    ).to_numpy(dtype="datetime64[ns]")
    context = ForecastContext(
        model_id=str(model.model_id), portfolio_id=series_id,
        origin_label=origin.origin_label, horizon_days=horizon, simulations=SIMULATIONS,
        seed=seed, future_dates=_readonly_copy(future_dates), origin_date=origin.origin_date,
    )
    started = perf_counter()
    terminal_method = getattr(model, "simulate_terminal_log_returns", None)
    if callable(terminal_method):
        if getattr(model, "forecast_output_semantics", None) != TERMINAL_FORECAST_SEMANTICS:
            raise ValueError("terminal ensemble must declare the existing forecast semantics")
        predictions = np.asarray(terminal_method(training, context), dtype=np.float64)
    else:
        predictions = np.asarray(model.simulate_daily_log_returns(training, context),
                                 dtype=np.float64)
    if predictions.shape != (SIMULATIONS, horizon) or not np.all(np.isfinite(predictions)):
        raise ValueError(f"forecast must be finite with shape {(SIMULATIONS, horizon)}")
    terminal = predictions if callable(terminal_method) else np.cumsum(predictions, axis=1)
    realized = np.cumsum(np.asarray(values[stop:stop + horizon], dtype=np.float64))
    losses = empirical_crps_by_horizon(terminal, realized)
    elapsed = perf_counter() - started
    with np.errstate(over="ignore", invalid="ignore"):
        forecast_cagr = np.expm1(terminal[:, -1] * (252.0 / horizon))
        diagnostics = {
            "forecast_and_score_seconds": elapsed,
            "timing_scope": "forecast adapter and exact scoring worker compute; excludes checkpoint IO",
            "horizon_days": horizon,
            "annualization_days": 252,
            "forecast_mean_cagr": _finite_float(np.mean(forecast_cagr)),
            "forecast_median_cagr": _finite_float(np.median(forecast_cagr)),
            "training_historical_cagr": _finite_float(np.expm1(np.mean(history) * 252.0)),
            "nonfinite_cagr_paths": int(np.count_nonzero(~np.isfinite(forecast_cagr))),
            "diagnostic_only_no_exclusions": True,
        }
    return losses, diagnostics


def run_panel(model: ForecastModel, panel_root: Path, output_root: Path, *,
              implementation_digest: str, max_tasks: int | None = None,
              should_continue: Callable[[], bool] | None = None,
              progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Resume valid checkpoints and attempt at most ``max_tasks`` new origins.

    Ordinary model failures are recorded separately and retried on the next
    call. Identity drift and checkpoint corruption fail closed; repair requires
    an explicit separate action, never an implicit resimulation. A false
    ``should_continue`` pauses new work at an origin boundary; saved scores are
    still loaded and aggregated. Progress reports include only saved compute.
    """
    if not isinstance(implementation_digest, str) or not implementation_digest.strip():
        raise ValueError("implementation_digest is required")
    if not isinstance(getattr(model, "model_id", None), str) or not model.model_id.strip():
        raise ValueError("a nonempty model_id is required")
    if max_tasks is not None and (isinstance(max_tasks, bool) or int(max_tasks) != max_tasks
                                  or max_tasks < 0):
        raise ValueError("max_tasks must be a nonnegative integer or None")
    panel_root, output_root = Path(panel_root), Path(output_root)
    panel = load_panel(panel_root)
    if panel["simulations_per_origin"] != SIMULATIONS:
        raise ValueError("French panel execution requires exactly 240 simulations")
    dates, returns = load_dates(panel_root), load_returns(panel_root)
    series = panel["series"]
    series_ids = [item["series_id"] for item in series]
    origins = tuple(schedule(panel_root))
    if (not series_ids or len(set(series_ids)) != len(series_ids) or not origins
            or returns.shape != (len(series), len(dates))):
        raise ValueError("invalid panel dimensions or duplicate series")
    labels = [origin.origin_label for origin in origins]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate origin labels")
    for origin in origins:
        if (origin.position < 0 or origin.max_horizon < 1
                or origin.position + 1 + origin.max_horizon > len(dates)
                or str(dates[origin.position].astype("datetime64[D]")) != origin.origin_date):
            raise ValueError("origin does not align with actual panel dates")
    identity = {
        "protocol_id": PROTOCOL_ID,
        "panel_fingerprint": panel["panel_fingerprint"],
        "series_ids": series_ids,
        "model_id": model.model_id,
        "implementation_digest": implementation_digest,
        "runtime": dependency_identity(model),
        "simulations_per_origin": SIMULATIONS,
        "source_closure": _source_closure(),
        "origins": [asdict(origin) for origin in origins],
        "seed_contract": "coherent_daily_seed(origin_date,max_horizon,240)",
        "aggregation": "origin_mean_per_series_horizon_then_equal_cell_mean",
        "future_dates": "synthetic_weekdays_after_origin_actual_dates_for_train_and_targets",
    }
    fingerprint = _manifest(output_root, identity)
    capacity = max(origin.max_horizon for origin in origins)
    expected_counts = np.zeros(capacity, dtype=np.int64)
    for origin in origins:
        expected_counts[:origin.max_horizon] += 1
    completed = attempted = resumed = failed = 0
    paused = False
    expected_tasks = len(series) * len(origins)

    def report_progress() -> None:
        if progress_callback is not None:
            progress_callback({
                "model_id": model.model_id, "expected_tasks": expected_tasks,
                "completed_tasks": completed, "failed_tasks": failed,
                "completed_forecast_and_score_seconds": completed_compute_seconds,
                "paused": paused,
            })

    complete_series: list[str] = []
    complete_cell_sum = 0.0
    completed_compute_seconds = 0.0
    complete_cells = 0
    for series_index, series_id in enumerate(series_ids):
        counts = np.zeros(capacity, dtype=np.int64)
        sums = np.zeros(capacity, dtype=np.float64)
        for origin_index, origin in enumerate(origins):
            seed = coherent_daily_seed(origin.origin_date, origin.max_horizon, SIMULATIONS)
            task_identity = {
                "execution_fingerprint": fingerprint, "series_id": series_id,
                "origin": asdict(origin), "seed": seed,
            }
            stem = f"{series_index:06d}/{origin_index:03d}"
            checkpoint_path = output_root / "tasks" / f"{stem}.npz"
            error_path = output_root / "errors" / f"{stem}.json"
            if should_continue is not None and not paused and not should_continue():
                paused = True
            new_checkpoint = False
            if checkpoint_path.exists():
                losses, diagnostics = _read_checkpoint(checkpoint_path, task_identity)
                resumed += 1
            elif not paused and (max_tasks is None or attempted < max_tasks):
                attempted += 1
                try:
                    losses, diagnostics = _forecast(
                        model, series_id, returns[series_index], dates, origin, seed
                    )
                except Exception as exc:  # noqa: BLE001 - record each model failure and continue.
                    _atomic_json(error_path, {
                        "identity": task_identity, "exception_type": type(exc).__name__,
                        "message": str(exc),
                    })
                    failed += 1
                    report_progress()
                    continue
                _write_checkpoint(checkpoint_path, task_identity, losses, diagnostics)
                new_checkpoint = True
                error_path.unlink(missing_ok=True)
            else:
                if error_path.exists():
                    failed += 1
                continue
            completed += 1
            completed_compute_seconds += diagnostics["forecast_and_score_seconds"]
            counts[:origin.max_horizon] += 1
            sums[:origin.max_horizon] += losses
            if new_checkpoint:
                report_progress()
        if np.array_equal(counts, expected_counts):
            complete_series.append(series_id)
            complete_cell_sum += float(np.sum(sums / counts))
            complete_cells += capacity
    report_progress()
    is_complete = (completed == expected_tasks and failed == 0
                   and len(complete_series) == len(series))
    score = complete_cell_sum / complete_cells if complete_cells else None
    result = {
        "protocol_id": PROTOCOL_ID, "execution_fingerprint": fingerprint,
        "panel_fingerprint": panel["panel_fingerprint"], "model_id": model.model_id,
        "status": "complete" if is_complete else "incomplete", "paused": paused,
        "simulations_per_origin": SIMULATIONS,
        "expected_tasks": expected_tasks, "completed_tasks": completed,
        "attempted_tasks_this_call": attempted, "resumed_tasks": resumed,
        "completed_forecast_and_score_seconds": completed_compute_seconds,
        "timing_scope": "sum of saved completed-origin forecast and scoring worker compute",
        "failed_tasks": failed, "pending_tasks": expected_tasks - completed - failed,
        "expected_cells": len(series) * capacity,
        "complete_series": complete_series, "complete_series_cells": complete_cells,
        "final_score": score if is_complete else None,
        "partial_complete_series_score": None if is_complete else score,
        "partial_score_scope": "matching complete series subset only; compare identical series IDs",
        "diagnostics": "per-origin final-horizon CAGR stored in task checkpoints; no exclusions",
    }
    _atomic_json(output_root / "summary.json", result)
    return result
