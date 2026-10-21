"""Versioned, incremental scoring for the expanded daily cross-asset panel.

This reporter only reads the panel and completed task checkpoints. It never
creates forecasts or changes the legacy French-panel score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

SCORE_PROTOCOL_ID = "daily_cross_asset_panel_crps_v3_logcalendar_hierarchical"
SENSITIVITY_PROTOCOL_ID = "daily_cross_asset_panel_crps_v3_logcalendar_hierarchical_sqrt_h"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_scores (
    panel_fingerprint TEXT NOT NULL,
    scorer_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    execution_fingerprint TEXT NOT NULL,
    task_key TEXT NOT NULL,
    series_id TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    family TEXT NOT NULL,
    origin_date TEXT NOT NULL,
    quarter TEXT NOT NULL,
    primary_score REAL NOT NULL,
    sensitivity_score REAL NOT NULL,
    checkpoint_path TEXT NOT NULL,
    checkpoint_size INTEGER NOT NULL,
    checkpoint_mtime_ns INTEGER NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    loss_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    scan_id TEXT NOT NULL,
    PRIMARY KEY (panel_fingerprint, scorer_hash, model_id,
                 execution_fingerprint, task_key)
);
CREATE INDEX IF NOT EXISTS task_scores_scan
ON task_scores(panel_fingerprint, scorer_hash, model_id, scan_id);
"""
_INPUT_HASH_CACHE: dict[str, tuple[int, int, int, int, str]] = {}
_PANEL_CACHE: dict[str, tuple[str, tuple, dict]] = {}


class ControlChangedDuringScan(RuntimeError):
    """The sweep cohort changed while this read-only scan was running."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def scorer_hash() -> str:
    """Bind cache entries to this exact scoring implementation."""
    return _sha256_bytes(Path(__file__).read_bytes())


def quarter_key(value: object) -> str:
    """Return the actual calendar quarter for a canonical origin date."""
    if isinstance(value, np.datetime64):
        value = str(value.astype("datetime64[D]"))
    if isinstance(value, (datetime, date)):
        year, month = value.year, value.month
    else:
        text = str(value)
        try:
            parsed = date.fromisoformat(text[:10])
        except ValueError as exc:
            raise ValueError(f"invalid origin date {value!r}") from exc
        year, month = parsed.year, parsed.month
    return f"{year:04d}-Q{(month - 1) // 3 + 1}"


def calendar_log_weights(origin_date: object, future_dates: np.ndarray) -> np.ndarray:
    """Positive normalized log-calendar interval weights for native horizons.

    If d_h is elapsed calendar days at the h-th observed target and d_0=0,
    interval h receives log((d_h+1)/(d_(h-1)+1)). This integrates the
    logarithmic calendar-lead measure over each observed interval, including
    weekend/holiday gaps without inventing target observations.
    """
    origin = np.datetime64(str(origin_date)[:10], "D").astype(np.int64)
    dates = np.asarray(future_dates).astype("datetime64[D]")
    if dates.ndim != 1 or dates.size == 0 or np.any(np.isnat(dates)):
        raise ValueError("future_dates must be a nonempty one-dimensional date array")
    target_days = dates.astype(np.int64)
    elapsed = target_days - origin
    if np.any(elapsed <= 0) or np.any(elapsed[1:] <= elapsed[:-1]):
        raise ValueError("native target dates must follow the origin and increase strictly")
    edges = np.concatenate((np.array([0.0]), np.log1p(elapsed.astype(np.float64))))
    mass = np.diff(edges)
    total = float(edges[-1])
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0) or not math.isfinite(total) or total <= 0.0:
        raise ValueError("calendar log-lead weights are invalid")
    return mass / total


def normalized_origin_scores(
    horizon_crps: np.ndarray,
    training_log_returns: np.ndarray,
    origin_date: object,
    future_dates: np.ndarray,
) -> tuple[float, float]:
    """Return primary and sqrt(h) sensitivity scores from the same CRPS vector.

    For observation horizon h and n training observations, the scale is
    s_train * sqrt(h + h^2/n), with sample standard deviation s_train (ddof=1).
    A zero or invalid training scale fails closed; no held-out statistic or
    arbitrary epsilon is substituted.
    """
    losses = np.asarray(horizon_crps, dtype=np.float64)
    history = np.asarray(training_log_returns, dtype=np.float64)
    dates = np.asarray(future_dates).astype("datetime64[D]")
    if losses.ndim != 1 or losses.size == 0 or dates.shape != losses.shape:
        raise ValueError("CRPS vector and native future dates must have matching nonempty horizons")
    if history.ndim != 1 or history.size < 2 or not np.all(np.isfinite(history)):
        raise ValueError("training history must contain at least two finite observations")
    if not np.all(np.isfinite(losses)) or np.any(losses < 0.0):
        raise ValueError("exact CRPS losses must be finite and nonnegative")
    s_train = float(np.std(history, ddof=1))
    n_train = float(history.size)
    if not math.isfinite(s_train) or s_train <= 0.0:
        raise ValueError("training-only sample standard deviation must be positive and finite")
    h = np.arange(1, losses.size + 1, dtype=np.float64)
    primary_scale = s_train * np.sqrt(h + h * h / n_train)
    sensitivity_scale = s_train * np.sqrt(h)
    if (not np.all(np.isfinite(primary_scale)) or np.any(primary_scale <= 0.0)
            or not np.all(np.isfinite(sensitivity_scale)) or np.any(sensitivity_scale <= 0.0)):
        raise ValueError("training-only horizon scales are invalid")
    weights = calendar_log_weights(origin_date, dates)
    primary = float(np.dot(losses / primary_scale, weights))
    sensitivity = float(np.dot(losses / sensitivity_scale, weights))
    if not math.isfinite(primary) or not math.isfinite(sensitivity):
        raise ValueError("normalized origin score is nonfinite")
    return primary, sensitivity


def normalized_origin_score(
    horizon_crps: np.ndarray,
    training_log_returns: np.ndarray,
    origin_date: object,
    future_dates: np.ndarray,
) -> float:
    """Backward-compatible alias for the primary, estimation-uncertainty scale."""
    return normalized_origin_scores(
        horizon_crps, training_log_returns, origin_date, future_dates,
    )[0]


def aggregate_records(
    records: Iterable[Mapping[str, object]],
    series_metadata: Mapping[str, Mapping[str, object]],
    *, score_key: str = "score",
) -> dict:
    """Aggregate origins -> series/quarter -> family -> class -> equal quarters.

    Within each quarter, origins are averaged for each series first. Available
    series are then averaged within family, available families within class,
    and available classes equally. Finally, available calendar quarters are
    equally weighted. The support and denominators depend only on the frozen
    schedule/checkpoint intersection, never on loss magnitude.
    """
    origin_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in records:
        series_id = str(item["series_id"])
        if series_id not in series_metadata:
            raise ValueError(f"no panel metadata for series {series_id}")
        q = quarter_key(item["origin_date"])
        score = float(item[score_key])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError("origin scores must be finite and nonnegative")
        origin_values[(q, series_id)].append(score)
    if not origin_values:
        return {"score": None, "quarter_count": 0, "quarter_scores": {},
                "quarter_class_scores": {}, "quarter_family_scores": {}}

    quarter_family: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for (q, sid), values in origin_values.items():
        meta = series_metadata[sid]
        cls, family = str(meta.get("asset_class", "")).strip(), str(meta.get("family", "")).strip()
        if not cls or not family:
            raise ValueError(f"series {sid} needs nonempty asset_class and family metadata")
        quarter_family[q][(cls, family)].append(float(np.mean(values)))

    quarter_class_families: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    quarter_family_scores: dict[str, dict[str, dict[str, float]]] = {}
    for q, families in quarter_family.items():
        rendered: dict[str, dict[str, float]] = defaultdict(dict)
        for (cls, family), values in families.items():
            family_score = float(np.mean(values))
            rendered[cls][family] = family_score
            quarter_class_families[q][cls].append(family_score)
        quarter_family_scores[q] = dict(rendered)

    quarter_class_scores: dict[str, dict[str, float]] = {}
    quarter_scores: dict[str, float] = {}
    for q, classes in quarter_class_families.items():
        class_scores = {cls: float(np.mean(values)) for cls, values in classes.items()}
        quarter_class_scores[q] = class_scores
        quarter_scores[q] = float(np.mean(list(class_scores.values())))
    return {
        "score": float(np.mean(list(quarter_scores.values()))),
        "quarter_count": len(quarter_scores),
        "quarter_scores": quarter_scores,
        "quarter_class_scores": quarter_class_scores,
        "quarter_family_scores": quarter_family_scores,
    }


def _get_api(api=None):
    if api is not None:
        return api
    panel_dir = Path(__file__).resolve().parent
    if str(panel_dir) not in sys.path:
        sys.path.insert(0, str(panel_dir))
    return importlib.import_module("sweep")


def _read_model_manifest(model_root: Path, panel_fingerprint: str, model_id: str, api) -> tuple[dict, str]:
    path = model_root / "execution.json"
    try:
        manifest = json.loads(path.read_text())
        identity, fingerprint = manifest["identity"], manifest["fingerprint"]
        expected_fingerprint = api.original.stable_digest(identity)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid execution manifest at {path}") from exc
    if fingerprint != expected_fingerprint:
        raise ValueError("execution manifest fingerprint mismatch")
    if identity.get("panel_fingerprint") != panel_fingerprint:
        raise ValueError("model execution is bound to a different panel fingerprint")
    if identity.get("model_id") != model_id:
        raise ValueError("model execution manifest identity differs from requested model")
    return identity, str(fingerprint)


def _open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(task_scores)")}
    if not {"primary_score", "sensitivity_score", "checkpoint_sha256", "status"} <= columns:
        connection.close()
        raise ValueError("score cache has an obsolete schema; choose a new cache path")
    return connection


def _task_key(series_id: str, origin: Mapping[str, object]) -> str:
    return _canonical([series_id, origin])


def _series_meta(rows: list[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output = {}
    for row in rows:
        sid = str(row["series_id"])
        if sid in output:
            raise ValueError(f"duplicate panel series_id {sid}")
        if not str(row.get("asset_class", "")).strip() or not str(row.get("family", "")).strip():
            raise ValueError(f"panel series {sid} has no class/family classification")
        output[sid] = row
    return output


def _verify_panel_inputs(panel_root: Path, panel: Mapping[str, object]) -> None:
    """Verify frozen input hashes once, then use file stats for cheap watch polls."""
    root = panel_root.resolve()
    declarations = dict(panel.get("files", {}))
    for row in panel["series"]:
        declarations.update(row["files"])
    for relative, expected in declarations.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("panel input checksum entry is malformed")
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError("panel input path must be relative")
        path = (root / candidate).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"panel input path is missing or escapes the frozen root: {relative}")
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)
        cached = _INPUT_HASH_CACHE.get(str(path))
        if cached is not None and cached[:4] == signature:
            digest = cached[4]
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            _INPUT_HASH_CACHE[str(path)] = (*signature, digest)
        if digest != expected:
            raise ValueError(f"panel input checksum mismatch: {relative}")


def _panel_file_signature(panel_root: Path, panel_manifest: Mapping[str, object]) -> tuple:
    root = panel_root.resolve()
    entries = []
    declarations = dict(panel_manifest.get("files", {}))
    for row in panel_manifest["series"]:
        declarations.update(row.get("files", {}))
    for relative in sorted(declarations):
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError("panel input path must be relative")
        path = (root / candidate).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"panel input path is missing or escapes the frozen root: {relative}")
        stat = path.stat()
        entries.append((relative, stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns))
    return tuple(entries)


def _load_panel(api, panel_root: Path) -> dict:
    """Validate once per unchanged frozen input snapshot, cheap-check on watch polls."""
    manifest_path = panel_root / "panel.json"
    if not manifest_path.is_file():
        return api.load_panel(panel_root)
    manifest = json.loads(manifest_path.read_text())
    identity = {key: value for key, value in manifest.items() if key != "panel_fingerprint"}
    if api.original.stable_digest(identity) != manifest.get("panel_fingerprint"):
        raise ValueError("panel manifest fingerprint mismatch")
    signature = (_file_stat_signature(manifest_path),
                 _panel_file_signature(panel_root, manifest))
    key = str(panel_root.resolve())
    cached = _PANEL_CACHE.get(key)
    if cached is not None and cached[0] == manifest["panel_fingerprint"] and cached[1] == signature:
        return cached[2]
    try:
        supports_skip_hashes = "verify_files" in inspect.signature(api.load_panel).parameters
    except (TypeError, ValueError):
        supports_skip_hashes = False
    if supports_skip_hashes:
        panel = api.load_panel(panel_root, verify_files=False)
        _verify_panel_inputs(panel_root, panel)
    else:
        panel = api.load_panel(panel_root)
    _PANEL_CACHE[key] = (str(panel["panel_fingerprint"]), signature, panel)
    return panel


def _file_stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_series(api, panel_root: Path, row: Mapping[str, object]):
    try:
        supports_skip_hashes = "verify_files" in inspect.signature(api.load_series).parameters
    except (TypeError, ValueError):
        supports_skip_hashes = False
    if supports_skip_hashes:
        return api.load_series(panel_root, row, verify_files=False)
    return api.load_series(panel_root, row)


def _collect_model(
    panel_root: Path,
    panel: Mapping[str, object],
    model_root: Path,
    model_id: str,
    cache_path: Path,
    api,
    control_spec: Mapping[str, object],
) -> tuple[dict, dict[str, dict]]:
    panel_fingerprint = str(panel["panel_fingerprint"])
    identity, execution_fingerprint = _read_model_manifest(model_root, panel_fingerprint, model_id, api)
    if (identity.get("factory") != control_spec.get("factory")
            or identity.get("implementation_digest") != control_spec.get("implementation_digest")):
        raise ValueError("execution manifest factory/digest differs from current enabled control")
    protocol_hash = scorer_hash()
    scan_id = hashlib.sha256(os.urandom(32)).hexdigest()
    expected_tasks = 0
    completed = failed = missing = processed = 0
    errors: list[dict] = []
    latest_mtime_ns: int | None = None
    expected_keys: set[str] = set()
    metadata = _series_meta(list(panel["series"]))
    connection = _open_cache(cache_path)
    try:
        with connection:
            # Keep only the active implementation identity for this model;
            # older adapter checkpoints cannot contribute to current scores.
            connection.execute(
                "DELETE FROM task_scores WHERE panel_fingerprint=? AND scorer_hash=? "
                "AND model_id=? AND execution_fingerprint<>?",
                (panel_fingerprint, protocol_hash, model_id, execution_fingerprint),
            )
            for series_index, row in enumerate(panel["series"]):
                sid = str(row["series_id"])
                dates, returns, origins = _load_series(api, panel_root, row)
                dates = np.asarray(dates).astype("datetime64[D]")
                returns = np.asarray(returns, dtype=np.float64)
                origins = list(origins)
                scheduled = list(api.schedule(panel_root, row))
                if _canonical(origins) != _canonical(scheduled):
                    raise ValueError(f"loaded origins differ from the frozen schedule for {sid}")
                if dates.ndim != 1 or returns.shape != dates.shape or not dates.size:
                    raise ValueError(f"invalid actual dates/returns for {sid}")
                for origin_index, origin in enumerate(origins):
                    expected_tasks += 1
                    position = int(origin["position"])
                    horizon = int(origin["max_horizon"])
                    origin_date = str(origin["origin_date"])
                    if (position < 1 or horizon < 1 or position + horizon >= dates.size
                            or str(dates[position]) != origin_date):
                        raise ValueError(f"invalid frozen origin bounds for {sid}/{origin_date}")
                    future_dates = dates[position + 1:position + 1 + horizon]
                    if future_dates.size != horizon:
                        raise ValueError(f"truncated actual target dates for {sid}/{origin_date}")
                    key = _task_key(sid, origin)
                    if key in expected_keys:
                        raise ValueError(f"duplicate scheduled task for {sid}/{origin_date}")
                    expected_keys.add(key)
                    checkpoint, error_path = api.task_paths(model_root, (series_index, origin_index))
                    checkpoint = Path(checkpoint)
                    error_path = Path(error_path)
                    cached = connection.execute(
                        "SELECT primary_score, sensitivity_score, checkpoint_size, checkpoint_mtime_ns, "
                        "checkpoint_path, checkpoint_sha256, loss_sha256, status FROM task_scores "
                        "WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? "
                        "AND execution_fingerprint=? AND task_key=?",
                        (panel_fingerprint, protocol_hash, model_id, execution_fingerprint, key),
                    ).fetchone()
                    if not checkpoint.is_file():
                        if cached is not None:
                            connection.execute(
                                "UPDATE task_scores SET status='missing', scan_id=? WHERE panel_fingerprint=? "
                                "AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?",
                                (scan_id, panel_fingerprint, protocol_hash, model_id, execution_fingerprint, key),
                            )
                        if error_path.is_file():
                            failed += 1
                            if len(errors) < 20:
                                errors.append({"series_id": sid, "origin_date": origin_date,
                                               "kind": "task_error_recorded"})
                        else:
                            missing += 1
                        continue
                    stat = checkpoint.stat()
                    latest_mtime_ns = max(latest_mtime_ns or stat.st_mtime_ns, stat.st_mtime_ns)
                    checkpoint_digest = _file_sha256(checkpoint)
                    if cached is not None and cached["checkpoint_path"] != str(checkpoint):
                        connection.execute(
                            "UPDATE task_scores SET status='drift', scan_id=? WHERE panel_fingerprint=? "
                            "AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?",
                            (scan_id, panel_fingerprint, protocol_hash, model_id, execution_fingerprint, key),
                        )
                        failed += 1
                        if len(errors) < 20:
                            errors.append({"series_id": sid, "origin_date": origin_date,
                                           "kind": "checkpoint_path_drift"})
                        continue
                    if cached is not None and cached["checkpoint_sha256"] == checkpoint_digest:
                        connection.execute(
                            "UPDATE task_scores SET status='ok', scan_id=? WHERE panel_fingerprint=? "
                            "AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?",
                            (scan_id, panel_fingerprint, protocol_hash, model_id, execution_fingerprint, key),
                        )
                        primary_score = float(cached["primary_score"])
                        sensitivity_score = float(cached["sensitivity_score"])
                        completed += 1
                    elif cached is not None:
                        # A checkpoint is immutable under its execution identity.
                        # Catch in-place mutations even if size and mtime are kept.
                        connection.execute(
                            "UPDATE task_scores SET status='drift', scan_id=? WHERE panel_fingerprint=? "
                            "AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?",
                            (scan_id, panel_fingerprint, protocol_hash, model_id, execution_fingerprint, key),
                        )
                        failed += 1
                        if len(errors) < 20:
                            errors.append({"series_id": sid, "origin_date": origin_date,
                                           "kind": "checkpoint_byte_drift"})
                        continue
                    else:
                        identity = api.task_identity(execution_fingerprint, model_id, sid, origin)
                        try:
                            losses, _ = api.original._read_checkpoint(checkpoint, identity)
                            losses = np.asarray(losses, dtype=np.float64)
                            if losses.shape != (horizon,):
                                raise ValueError("checkpoint horizon does not match frozen origin")
                            primary_score, sensitivity_score = normalized_origin_scores(
                                losses, returns[:position + 1], origin_date, future_dates,
                            )
                            with np.load(checkpoint, allow_pickle=False) as data:
                                envelope = json.loads(str(data["metadata"].item()))
                            loss_digest = str(envelope["payload"]["loss_sha256"])
                            if _file_sha256(checkpoint) != checkpoint_digest:
                                raise ValueError("checkpoint changed while it was being ingested")
                            connection.execute(
                                "INSERT OR REPLACE INTO task_scores "
                                "(panel_fingerprint, scorer_hash, model_id, execution_fingerprint, task_key, "
                                "series_id, asset_class, family, origin_date, quarter, primary_score, "
                                "sensitivity_score, checkpoint_path, checkpoint_size, checkpoint_mtime_ns, "
                                "checkpoint_sha256, loss_sha256, status, scan_id) "
                                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (panel_fingerprint, protocol_hash, model_id, execution_fingerprint,
                                 key, sid, str(row["asset_class"]), str(row["family"]), origin_date,
                                 quarter_key(origin_date), primary_score, sensitivity_score, str(checkpoint),
                                 stat.st_size, stat.st_mtime_ns, checkpoint_digest, loss_digest, "ok", scan_id),
                            )
                            processed += 1
                            completed += 1
                        except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                                AssertionError, IndexError) as exc:
                            connection.execute(
                                "DELETE FROM task_scores WHERE panel_fingerprint=? AND scorer_hash=? "
                                "AND model_id=? AND execution_fingerprint=? AND task_key=?",
                                (panel_fingerprint, protocol_hash, model_id, execution_fingerprint, key),
                            )
                            failed += 1
                            if len(errors) < 20:
                                errors.append({"series_id": sid, "origin_date": origin_date,
                                               "kind": type(exc).__name__, "message": str(exc)[:240]})
                            continue
            # Remove cached tasks absent from this schedule or missing/invalid
            # this scan; only tasks explicitly validated or stat-cache matched
            # have the current scan_id.
            connection.execute(
                "DELETE FROM task_scores WHERE panel_fingerprint=? AND scorer_hash=? "
                "AND model_id=? AND execution_fingerprint=? AND scan_id<>?",
                (panel_fingerprint, protocol_hash, model_id, execution_fingerprint, scan_id),
            )
            task_rows = connection.execute(
                "SELECT task_key, series_id, origin_date, quarter, primary_score, sensitivity_score "
                "FROM task_scores "
                "WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? "
                "AND execution_fingerprint=? AND scan_id=? AND status='ok'",
                (panel_fingerprint, protocol_hash, model_id, execution_fingerprint, scan_id),
            ).fetchall()
    finally:
        connection.close()

    records = {str(record["task_key"]): {
        "series_id": str(record["series_id"]), "origin_date": str(record["origin_date"]),
        "quarter": str(record["quarter"]),
        "score": float(record["primary_score"]),
        "primary_score": float(record["primary_score"]),
        "sensitivity_score": float(record["sensitivity_score"]),
    } for record in task_rows}
    summary = aggregate_records(records.values(), metadata, score_key="primary_score")
    sensitivity_summary = aggregate_records(records.values(), metadata, score_key="sensitivity_score")
    coverage = {
        "completed_tasks": completed,
        "expected_tasks": expected_tasks,
        "failed_tasks": failed,
        "missing_tasks": missing,
        "fraction": completed / expected_tasks if expected_tasks else 0.0,
        "complete": completed == expected_tasks and failed == 0 and missing == 0,
    }
    return ({"id": model_id, "completed_tasks": completed, "expected_tasks": expected_tasks,
             "failed_tasks": failed, "missing_tasks": missing,
             "score_on_own_coverage": summary["score"],
             "sensitivity_score_on_own_coverage": sensitivity_summary["score"],
             "coverage": coverage,
             "quarter_count": summary["quarter_count"], "errors": errors,
             "latest_checkpoint_mtime_ns": latest_mtime_ns, "processed": processed}, records)


def _pairwise_score(
    model_id: str,
    model_records: Mapping[str, Mapping[str, object]],
    public_records: Mapping[str, Mapping[str, object]],
    series_metadata: Mapping[str, Mapping[str, object]],
    expected_keys: set[str],
    model_complete: bool,
    public_complete: bool,
    *, score_key: str = "primary_score",
) -> dict:
    common = sorted(set(model_records) & set(public_records))
    model_summary = aggregate_records((model_records[key] for key in common), series_metadata,
                                      score_key=score_key)
    public_summary = aggregate_records((public_records[key] for key in common), series_metadata,
                                       score_key=score_key)
    model_score, public_score = model_summary["score"], public_summary["score"]
    relative = None
    if model_score is not None and public_score is not None and public_score > 0:
        relative = 100.0 * (public_score - model_score) / public_score
    return {
        "matched_tasks": len(common),
        "expected_tasks": len(expected_keys),
        "matched_series": len({str(model_records[key]["series_id"]) for key in common}),
        "quarter_count": model_summary["quarter_count"],
        "model_score": model_score,
        "public_score": public_score,
        "pct_better": relative,
        "preliminary": not (model_complete and public_complete and set(common) == expected_keys),
    }


def build_scoreboard(
    panel_root: Path,
    output_root: Path,
    *,
    control_path: Path,
    cache_path: Path | None = None,
    model_ids: Iterable[str] | None = None,
    public_model_id: str | None = None,
    api=None,
    write: bool = True,
) -> dict:
    """Read completed checkpoints for currently enabled control entries only."""
    api = _get_api(api)
    panel_root, output_root = Path(panel_root), Path(output_root)
    control_path = Path(control_path)
    if not hasattr(api, "read_control"):
        raise ValueError("scheduler API does not expose the validated control reader")
    control_digest = _file_sha256(control_path)
    control = api.read_control(control_path)
    if _file_sha256(control_path) != control_digest:
        raise ControlChangedDuringScan("control file changed while its scoring snapshot was read")
    panel = _load_panel(api, panel_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_path) if cache_path else output_root / "score-cache-v3.sqlite3"
    enabled_specs = {str(mid): spec for mid, spec in control["models"].items() if spec["enabled"]}
    requested_ids = None if model_ids is None else list(dict.fromkeys(str(mid) for mid in model_ids))
    included_ids = [mid for mid in enabled_specs if requested_ids is None or mid in requested_ids]
    model_metadata = _series_meta(list(panel["series"]))
    expected_keys = {
        _task_key(str(row["series_id"]), origin)
        for row in panel["series"]
        for origin in api.schedule(panel_root, row)
    }
    models: dict[str, dict] = {}
    model_records: dict[str, dict[str, dict]] = {}
    processed = error_count = 0
    latest_mtime_ns: int | None = None
    for mid in included_ids:
        model_root = Path(api.model_directory(output_root, mid))
        try:
            summary, records = _collect_model(
                panel_root, panel, model_root, mid, cache_path, api, enabled_specs[mid],
            )
            processed += int(summary.pop("processed"))
            error_count += int(summary["failed_tasks"])
            latest = summary.pop("latest_checkpoint_mtime_ns")
            if latest is not None:
                latest_mtime_ns = max(latest_mtime_ns or latest, latest)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            count = len(expected_keys)
            summary = {
                "id": mid, "completed_tasks": 0, "expected_tasks": count,
                "failed_tasks": count, "missing_tasks": 0,
                "score_on_own_coverage": None,
                "sensitivity_score_on_own_coverage": None,
                "coverage": {"completed_tasks": 0, "expected_tasks": count,
                             "failed_tasks": count, "missing_tasks": 0,
                             "fraction": 0.0, "complete": False},
                "quarter_count": 0,
                "errors": [{"kind": type(exc).__name__, "message": str(exc)[:240]}],
            }
            records = {}
            error_count += count
        models[mid] = summary
        model_records[mid] = records
    pairwise = {}
    if public_model_id and public_model_id in model_records:
        public = model_records[public_model_id]
        for mid, records in model_records.items():
            if mid == public_model_id:
                continue
            pairwise[mid] = _pairwise_score(
                mid, records, public, model_metadata, expected_keys,
                bool(models[mid]["coverage"]["complete"]),
                bool(models[public_model_id]["coverage"]["complete"]),
            )
    def common_board(score_field: str):
        all_common = bool(expected_keys) and bool(models) and all(
            models[mid]["coverage"]["complete"] and set(model_records[mid]) == expected_keys
            for mid in included_ids
        )
        if not all_common:
            return None
        return [{"id": mid, "score": models[mid][score_field]}
                for mid in sorted(included_ids, key=lambda name: models[name][score_field])]

    common_leaderboard = common_board("score_on_own_coverage")
    sensitivity_leaderboard = common_board("sensitivity_score_on_own_coverage")
    sensitivity_pairwise = {}
    if public_model_id and public_model_id in model_records:
        public = model_records[public_model_id]
        for mid, records in model_records.items():
            if mid == public_model_id:
                continue
            sensitivity_pairwise[mid] = _pairwise_score(
                mid, records, public, model_metadata, expected_keys,
                bool(models[mid]["coverage"]["complete"]),
                bool(models[public_model_id]["coverage"]["complete"]),
                score_key="sensitivity_score",
            )
    now_ns = time.time_ns()
    latest_lag = ((now_ns - latest_mtime_ns) / 1e9) if latest_mtime_ns is not None else None
    report = {
        "score_protocol_id": SCORE_PROTOCOL_ID,
        "control": {"sha256": control_digest,
                    "enabled_model_ids": list(enabled_specs),
                    "included_model_ids": included_ids},
        "panel_fingerprint": panel["panel_fingerprint"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "ingestion": {"processed": processed, "errors": error_count,
                      "latest_checkpoint_lag_seconds": latest_lag},
        "models": models,
        "pairwise_public": pairwise,
        "commonleaderboard": common_leaderboard,
        "sensitivity": {
            "score_protocol_id": SENSITIVITY_PROTOCOL_ID,
            "models": {mid: {"score_on_own_coverage": data["sensitivity_score_on_own_coverage"],
                              "coverage": data["coverage"]}
                       for mid, data in models.items()},
            "pairwise_public": sensitivity_pairwise,
            "commonleaderboard": sensitivity_leaderboard,
        },
    }
    current_control = api.read_control(control_path)
    if (_file_sha256(control_path) != control_digest
            or _canonical(current_control) != _canonical(control)):
        raise ControlChangedDuringScan("control file changed during scoring; report was not published")
    if write:
        output = output_root / "scoreboard.json"
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--public-model")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    while True:
        try:
            report = build_scoreboard(
                args.panel_root, args.output_root, control_path=args.control, cache_path=args.cache,
                model_ids=args.models, public_model_id=args.public_model,
            )
        except ControlChangedDuringScan as exc:
            print(json.dumps({"status": "retry_next_scan", "reason": str(exc)}), flush=True)
            if not args.watch:
                return 2
            time.sleep(args.poll_seconds)
            continue
        print(json.dumps({"processed": report["ingestion"]["processed"],
                          "errors": report["ingestion"]["errors"],
                          "models": {mid: item["coverage"] for mid, item in report["models"].items()}},
                         sort_keys=True), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
