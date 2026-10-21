"""Read-only v7 rescore of the frozen, audited 1,000-path WR/WOR pair.

Default mode checks the small preflight snapshot only. Full checkpoint and
input reads require both --execute and --profile-clearance. No forecasts run.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

V = Path(__file__).resolve().parent
O = V.parent
PANEL_ROOT = V / "inputs"
RESULTS = V / "results"
INDEX = RESULTS / "native-loss-index.sqlite3"
AUDIT = V / "full_pair_independent_audit.json"
LAUNCH = V / "launch_plan.json"
SCOREBOARD = RESULTS / "scoreboard.json"
STATUS = RESULTS / "status.json"
PROTOCOL = O / "final_scoring_protocol_v7.json"
PREFLIGHT = V / "rescore_full_pair_v7_preflight.json"
OUTPUT = V / "full_pair_rescore_v7_diagnostic.json"
SCORE_INDEX_OUTPUT = V / "full_pair_rescore_v7_scores.sqlite3"
EXPECTED_PROTOCOL = "daily_portfolio_panel_crps_v7_train_sd_sqrt_native_lead_equal_days_two_branch"
MODEL_IDS = (
    "model_testing_asset_joint_blocks_replacement_v6",
    "model_testing_asset_joint_blocks_history_partition_v6",
)
EXECUTION_FINGERPRINTS = {
    "model_testing_asset_joint_blocks_replacement_v6": "c78b20f5ab72dc1e4e33ad2e9ed3bad4316807a84e32cba3962c9e05dc859f20",
    "model_testing_asset_joint_blocks_history_partition_v6": "f9263d42137db6d7ea2f1fe986c996227c9ea2b11e4fa8a524d46b9fcd1040de",
}

sys.path.insert(0, str(V))
import common  # noqa: E402,F401
import native_index  # noqa: E402
import score_aligned_v7  # noqa: E402
import score_main  # noqa: E402
import sweep_v6 as api  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_preflight() -> dict:
    """Validate bounded metadata; deliberately does not enumerate checkpoints."""
    assert Path(common.OWN).resolve() == V.resolve()
    launch, audit = read(LAUNCH), read(AUDIT)
    scoreboard, status = read(SCOREBOARD), read(STATUS)
    protocol, panel = read(PROTOCOL), read(PANEL_ROOT / "panel.json")
    panel_identity = {k: v for k, v in panel.items() if k != "panel_fingerprint"}
    panel_digest = sha256_bytes(json.dumps(
        panel_identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())
    assert panel_digest == panel["panel_fingerprint"] == launch["panel_fingerprint"]
    assert protocol["protocol_id"] == score_aligned_v7.SCORE_PROTOCOL_ID == EXPECTED_PROTOCOL
    assert audit["complete_pair_audit"] and audit["same_successful_coverage"]
    assert audit["total_checkpoints_verified"] == audit["tasks_per_model"] * 2 == 128098
    assert audit["scoring_integrity_errors"] == audit["native_failures"] == 0
    assert audit["tasks_per_model"] == panel["expected_tasks"] == 64049
    assert audit["series"] == len(panel["series"]) == 1264
    assert set(audit["models"]) == set(MODEL_IDS) == set(launch["models"])
    assert status["completed_active_tasks"] == status["expected_active_tasks"] == 128098
    assert status["failed_active_tasks"] == 0
    assert status["active_workers"] == status["running_origins"] == 0
    assert set(status["models"]) == set(MODEL_IDS)
    assert all(
        model["status"] == "complete" and model["completed_tasks"] == 64049
        and model["failed_tasks"] == 0 and model["running_origins"] == 0
        for model in status["models"].values()
    )
    assert scoreboard["panel_fingerprint"] == panel["panel_fingerprint"]
    assert scoreboard["expected_tasks_per_model"] == 64049
    assert scoreboard["native_failures"] == scoreboard["scoring_integrity_errors"] == 0
    assert set(scoreboard["models"]) == set(MODEL_IDS)
    assert all(
        model["complete"] and model["completed_tasks"] == 64049
        and model["native_failures"] == model["scoring_integrity_errors"] == model["missing_tasks"] == 0
        for model in scoreboard["models"].values()
    )
    assert scoreboard["score_protocol_id"] == "daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch"
    assert scoreboard["score_source_sha256"] == sha256_file(V / "score_main.py")
    for source in ("score_main.py", "native_index.py", "sweep_v6.py"):
        assert sha256_file(V / source) == launch["source_hashes"][source], source
    assert not OUTPUT.exists(), f"refusing to overwrite {OUTPUT}"
    assert not SCORE_INDEX_OUTPUT.exists(), f"refusing to overwrite {SCORE_INDEX_OUTPUT}"
    roots = {}
    for mid in MODEL_IDS:
        root = Path(api.model_directory(RESULTS, mid))
        manifest = root / "execution.json"
        roots[mid] = {
            "path": str(root),
            "execution_fingerprint": EXECUTION_FINGERPRINTS[mid],
            "audited_checkpoint_commitment": audit["models"][mid]["checkpoint_commitment"],
            "audited_tasks": audit["models"][mid]["verified"],
            "execution_manifest_path": str(manifest),
            "execution_manifest_sha256": sha256_file(manifest),
        }
        assert root.is_dir()
        assert roots[mid]["audited_tasks"] == 64049
    source_hashes = {
        "panel_metadata": sha256_file(PANEL_ROOT / "panel.json"),
        "protocol": sha256_file(PROTOCOL),
        "v7_scorer": sha256_file(V / "score_aligned_v7.py"),
        "legacy_aggregator": sha256_file(V / "score_main.py"),
        "legacy_indexer": sha256_file(V / "native_index.py"),
        "legacy_checkpoint_reader": sha256_file(Path(native_index.verified.__file__)),
        "sweep_api": sha256_file(V / "sweep_v6.py"),
        "full_pair_audit": sha256_file(AUDIT),
        "launch_plan": sha256_file(LAUNCH),
        "legacy_scoreboard": sha256_file(SCOREBOARD),
        "runner": sha256_file(Path(__file__)),
        "execution_manifests": {mid: roots[mid]["execution_manifest_sha256"] for mid in MODEL_IDS},
    }
    return {
        "protocol_id": EXPECTED_PROTOCOL,
        "panel_fingerprint": panel["panel_fingerprint"],
        "models": roots,
        "tasks_per_model": 64049,
        "total_checkpoints": 128098,
        "paths_per_forecast": 1000,
        "checkpoint_index": str(INDEX),
        "checkpoint_index_access": "read-only SQLite query; never update the v6 index",
        "output_path": str(OUTPUT),
        "score_index_output_path": str(SCORE_INDEX_OUTPUT),
        "source_hashes": source_hashes,
        "no_forecast_calls": True,
    }


def _check_preflight_snapshot() -> tuple[dict, dict]:
    snapshot = read(PREFLIGHT)
    current = _metadata_preflight()
    assert current["panel_fingerprint"] == snapshot["panel_fingerprint"]
    assert current["models"] == snapshot["models"]
    assert current["tasks_per_model"] == snapshot["tasks_per_model"]
    assert current["total_checkpoints"] == snapshot["total_checkpoints"]
    assert current["source_hashes"] == snapshot["source_hashes"]
    return snapshot, current


def _index_rows(panel_fingerprint: str, roots: dict) -> tuple[dict, str]:
    """Read the frozen v6 index without opening it for writes."""
    scorer = native_index.source_digest()
    uri = "file:" + quote(str(INDEX.resolve()), safe="/") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        result = {}
        for mid, info in roots.items():
            rows = connection.execute(
                """SELECT task_key, series_id, origin_date, main_score, loss_sha256,
                          checkpoint_path, checkpoint_sha256, status, scan_id
                   FROM task_scores_main
                   WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=?
                     AND execution_fingerprint=? AND status='ok'""",
                (panel_fingerprint, scorer, mid, info["fingerprint"]),
            ).fetchall()
            by_key = {row["task_key"]: dict(row) for row in rows}
            if len(by_key) != 64049 or len(rows) != len(by_key):
                raise ValueError(f"index coverage is not 64,049 unique rows for {mid}: {len(rows)}")
            result[mid] = by_key
        return result, scorer
    finally:
        connection.close()


def _write_v7_score_index(
    rows_by_model: dict, metadata: dict, *, expected_per_model: int = 64049,
) -> tuple[str, dict]:
    """Publish a new compact score index atomically; never touch the v6 index."""
    if SCORE_INDEX_OUTPUT.exists():
        raise FileExistsError(SCORE_INDEX_OUTPUT)
    temporary = SCORE_INDEX_OUTPUT.with_name(f".{SCORE_INDEX_OUTPUT.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    connection = None
    created = False
    try:
        connection = sqlite3.connect(temporary)
        created = True
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript("""
            CREATE TABLE task_scores_v7 (
                model_id TEXT NOT NULL,
                task_key TEXT NOT NULL,
                panel_fingerprint TEXT NOT NULL,
                protocol_id TEXT NOT NULL,
                execution_fingerprint TEXT NOT NULL,
                case_index TEXT NOT NULL,
                series_index INTEGER NOT NULL,
                origin_index INTEGER NOT NULL,
                series_id TEXT NOT NULL,
                origin_date TEXT NOT NULL,
                native_horizon INTEGER NOT NULL,
                training_observations INTEGER NOT NULL,
                task_identity_json TEXT NOT NULL,
                source_checkpoint_sha256 TEXT NOT NULL,
                loss_sha256 TEXT NOT NULL,
                score_v7 REAL NOT NULL,
                legacy_score_v6 REAL NOT NULL,
                PRIMARY KEY (model_id, task_key)
            );
            CREATE INDEX task_scores_v7_case ON task_scores_v7(case_index, model_id);
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            PRAGMA user_version=1;
        """)
        with connection:
            connection.executemany(
                "INSERT INTO task_scores_v7 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row for model_rows in rows_by_model.values() for row in model_rows),
            )
            connection.executemany(
                "INSERT INTO metadata(key,value) VALUES(?,?)",
                ((key, json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
                 for key, value in metadata.items()),
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"new v7 score index failed integrity_check: {integrity}")
        counts = dict(connection.execute(
            "SELECT model_id,COUNT(*) FROM task_scores_v7 GROUP BY model_id"
        ).fetchall())
        if counts != {mid: expected_per_model for mid in MODEL_IDS}:
            raise ValueError(f"new v7 score index coverage mismatch: {counts}")
        total = connection.execute("SELECT COUNT(*) FROM task_scores_v7").fetchone()[0]
        expected_total = expected_per_model * len(MODEL_IDS)
        if total != expected_total:
            raise ValueError(f"new v7 score index row count mismatch: {total}")
        connection.close()
        connection = None
        digest = sha256_file(temporary)
        os.replace(temporary, SCORE_INDEX_OUTPUT)
        return digest, counts
    except BaseException:
        if connection is not None:
            connection.close()
        if created and temporary.exists():
            temporary.unlink()
        raise


def run_rescore(snapshot: dict, current: dict) -> dict:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    panel_start = time.perf_counter()
    panel = api.load_panel(PANEL_ROOT, verify_files=False)
    panel_metadata_wall = time.perf_counter() - panel_start
    assert panel["panel_fingerprint"] == snapshot["panel_fingerprint"]
    audit, launch = read(AUDIT), read(LAUNCH)
    roots = {}
    for mid in MODEL_IDS:
        root = Path(api.model_directory(RESULTS, mid))
        identity, fingerprint = native_index.verified._read_model_manifest(
            root, panel["panel_fingerprint"], mid, api,
        )
        spec = launch["models"][mid]
        if fingerprint != EXECUTION_FINGERPRINTS[mid]:
            raise ValueError(f"execution fingerprint differs from frozen full-pair identity: {mid}")
        if identity.get("factory") != spec["factory"] or identity.get("implementation_digest") != spec["implementation_digest"]:
            raise ValueError(f"model source identity differs from frozen launch plan: {mid}")
        roots[mid] = {"path": root, "fingerprint": fingerprint}
    index_start = time.perf_counter()
    indexed, index_scorer_digest = _index_rows(panel["panel_fingerprint"], roots)
    index_read_wall = time.perf_counter() - index_start

    score_records = {mid: [] for mid in MODEL_IDS}
    legacy_records = {mid: [] for mid in MODEL_IDS}
    score_index_rows = {mid: [] for mid in MODEL_IDS}
    commitments = {mid: hashlib.sha256() for mid in MODEL_IDS}
    verified_counts = {mid: 0 for mid in MODEL_IDS}
    series_validation_wall = checkpoint_wall = score_cpu_v7 = score_cpu_v6 = 0.0
    expected_task_keys = set()
    for i, row in enumerate(panel["series"]):
        input_start = time.perf_counter()
        dates, values, origins = api.load_series(PANEL_ROOT, row, verify_files=True)
        series_validation_wall += time.perf_counter() - input_start
        for j, origin in enumerate(origins):
            task_key = native_index.verified._task_key(row["series_id"], origin)
            expected_task_keys.add(task_key)
            history_end = origin["position"] + 1
            horizon = origin["max_horizon"]
            future_dates = dates[history_end:history_end + horizon]
            history = values[:history_end]
            for mid in MODEL_IDS:
                record = indexed[mid].get(task_key)
                if record is None:
                    raise ValueError(f"index task absent: {mid} {row['series_id']} {origin['origin_date']}")
                checkpoint, _ = api.task_paths(roots[mid]["path"], (i, j))
                if record["series_id"] != row["series_id"] or record["origin_date"] != origin["origin_date"]:
                    raise ValueError(f"index task identity differs: {mid} {task_key}")
                if record["checkpoint_path"] != str(checkpoint):
                    raise ValueError(f"indexed checkpoint path differs: {mid} {task_key}")
                read_start = time.perf_counter()
                blob = checkpoint.read_bytes()
                byte_digest = sha256_bytes(blob)
                if byte_digest != record["checkpoint_sha256"]:
                    raise ValueError(f"checkpoint byte digest differs from frozen index: {mid} {task_key}")
                identity = api.task_identity(roots[mid]["fingerprint"], mid, row["series_id"], origin)
                loss, _ = api.original._read_checkpoint(io.BytesIO(blob), identity)
                checkpoint_wall += time.perf_counter() - read_start
                loss_digest = hashlib.sha256(np.asarray(loss, dtype="<f8").tobytes()).hexdigest()
                if loss_digest != record["loss_sha256"]:
                    raise ValueError(f"loss vector differs from frozen index: {mid} {task_key}")
                commitments[mid].update(f"{i}/{j}:{byte_digest}\n".encode())
                verified_counts[mid] += 1
                cpu = time.process_time()
                v7_value = score_aligned_v7.origin_score(loss, history, origin["origin_date"], future_dates)
                score_cpu_v7 += time.process_time() - cpu
                cpu = time.process_time()
                v6_value = score_main.origin_score(loss, history, origin["origin_date"], future_dates)
                score_cpu_v6 += time.process_time() - cpu
                if abs(v6_value - record["main_score"]) > 2e-12:
                    raise ValueError(f"legacy indexed score does not replay: {mid} {task_key}")
                score_records[mid].append({"series_id": row["series_id"], "origin_date": origin["origin_date"], "score": v7_value})
                legacy_records[mid].append({"series_id": row["series_id"], "origin_date": origin["origin_date"], "score": v6_value})
                index_identity = api.task_identity(roots[mid]["fingerprint"], mid, row["series_id"], origin)
                score_index_rows[mid].append((
                    mid,
                    task_key,
                    panel["panel_fingerprint"],
                    EXPECTED_PROTOCOL,
                    roots[mid]["fingerprint"],
                    f"{i:06d}/{j:03d}",
                    i,
                    j,
                    row["series_id"],
                    origin["origin_date"],
                    horizon,
                    history_end,
                    json.dumps(index_identity, sort_keys=True, separators=(",", ":"), allow_nan=False),
                    byte_digest,
                    loss_digest,
                    v7_value,
                    v6_value,
                ))
    assert len(expected_task_keys) == 64049
    for mid in MODEL_IDS:
        assert verified_counts[mid] == 64049
        assert commitments[mid].hexdigest() == audit["models"][mid]["checkpoint_commitment"]
        assert len(indexed[mid]) == 64049
        if set(indexed[mid]) != expected_task_keys:
            raise ValueError(f"the frozen index does not match the exact panel task-key coverage: {mid}")

    metadata = {row["series_id"]: row for row in panel["series"]}
    reduction_start = time.process_time()
    reduced = {mid: score_main.aggregate_records(score_records[mid], metadata) for mid in MODEL_IDS}
    legacy_reduced = {mid: score_main.aggregate_records(legacy_records[mid], metadata) for mid in MODEL_IDS}
    reduction_cpu = time.process_time() - reduction_start
    expected_old = {mid: audit["models"][mid]["main_score"] for mid in MODEL_IDS}
    for mid in MODEL_IDS:
        if reduced[mid]["quarter_count"] != 343:
            raise ValueError(f"unexpected quarter support for {mid}")
        if abs(float(legacy_reduced[mid]["score"]) - expected_old[mid]) > 2e-12:
            raise ValueError(f"legacy hierarchy score does not replay the independent audit: {mid}")
    v7_scores = {mid: float(reduced[mid]["score"]) for mid in MODEL_IDS}
    # The conservative reference is chosen anew under v7: the higher Q wins.
    reference = max(MODEL_IDS, key=lambda mid: (v7_scores[mid], mid == MODEL_IDS[0]))
    reference_score = v7_scores[reference]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": EXPECTED_PROTOCOL,
        "status": "diagnostic_only_not_canonical_promotion",
        "coverage": {"series": 1264, "tasks_per_model": 64049, "total_checkpoints_verified": 128098,
                     "same_task_keys_each_model": True, "paths_per_forecast": 1000,
                     "new_forecasts": 0, "score_integrity_errors": 0, "native_failures": 0},
        "panel_fingerprint": panel["panel_fingerprint"],
        "index_scorer_digest": index_scorer_digest,
        "reference_model_id_under_v7": reference,
        "reference_rule": "higher v7 score of frozen WR/WOR pair, on identical 64,049-task coverage",
        "models": {
            mid: {
                "v7_Q": v7_scores[mid],
                "v6_Q_replayed": float(legacy_reduced[mid]["score"]),
                "old_v6_Q_from_independent_audit": expected_old[mid],
                "percent_better_vs_v7_reference": 100.0 * (reference_score - v7_scores[mid]) / reference_score,
                "quarter_count": reduced[mid]["quarter_count"],
                "portfolio_supported_quarters": sum(
                    weight > 0 for weight in reduced[mid]["portfolio_weight_by_quarter"].values()
                ),
                "maximum_portfolio_weight": reduced[mid]["maximum_portfolio_weight"],
                "checkpoint_commitment_verified": commitments[mid].hexdigest(),
                "task_count_verified": verified_counts[mid],
            } for mid in MODEL_IDS
        },
        "runtime": {
            "rescore_wall_seconds": time.perf_counter() - wall_start,
            "rescore_process_cpu_seconds": time.process_time() - cpu_start,
            "panel_metadata_validation_wall_seconds": panel_metadata_wall,
            "series_input_validation_wall_seconds": series_validation_wall,
            "index_read_wall_seconds": index_read_wall,
            "checkpoint_read_and_identity_verification_wall_seconds": checkpoint_wall,
            "v7_origin_score_cpu_seconds": score_cpu_v7,
            "old_v6_row_score_replay_cpu_seconds": score_cpu_v6,
            "hierarchy_reduction_cpu_seconds": reduction_cpu,
            "forecast_worker_compute_seconds_from_old_run_not_rescore_runtime": {
                mid: audit["models"][mid]["compute_seconds"] for mid in MODEL_IDS
            },
        },
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "v7_scorer": sha256_file(V / "score_aligned_v7.py"),
            "protocol": sha256_file(PROTOCOL),
            "legacy_aggregator": sha256_file(V / "score_main.py"),
            "full_pair_audit": sha256_file(AUDIT),
            "launch_plan": sha256_file(LAUNCH),
            "legacy_scoreboard_unchanged": sha256_file(SCOREBOARD),
        },
        "interpretation_limits": [
            "This reuses and verifies the frozen saved 1,000-path checkpoints; it does not refit or forecast.",
            "This is a deterministic metric re-score on the existing historical tasks, not independent validation or a significance test.",
            "It does not alter or promote the canonical v6 score or modify the legacy scoreboard/index/checkpoints.",
            "v7 equal-day, train-SD/sqrt(native-lead) scores are a distinct protocol and are not directly compared numerically with v6 scores.",
        ],
    }
    if _metadata_preflight()["source_hashes"] != snapshot["source_hashes"]:
        raise ValueError("a hash-bound source, panel, or execution manifest changed during diagnostic rescore")
    if OUTPUT.exists() or SCORE_INDEX_OUTPUT.exists():
        raise FileExistsError("refusing to overwrite a prior v7 rescore artifact")
    index_metadata = {
        "schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL,
        "panel_fingerprint": panel["panel_fingerprint"],
        "source_hashes": result["source_hashes"],
        "baseline_model_ids": list(MODEL_IDS),
        "reference_model_id_under_v7": reference,
        "reference_rule": result["reference_rule"],
        "coverage": result["coverage"],
        "models": {mid: result["models"][mid] for mid in MODEL_IDS},
        "task_index_row_count": 128098,
    }
    index_sha256, index_counts = _write_v7_score_index(score_index_rows, index_metadata)
    result["score_index"] = {
        "path": str(SCORE_INDEX_OUTPUT),
        "sha256": index_sha256,
        "row_count": sum(index_counts.values()),
        "rows_per_model": index_counts,
        "table": "task_scores_v7",
        "metadata_table": "metadata",
    }
    result["score_index_path"] = str(SCORE_INDEX_OUTPUT)
    result["score_index_sha256"] = index_sha256
    api.original._atomic_json(OUTPUT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="read and rescore the full frozen checkpoint pair")
    parser.add_argument("--profile-clearance", action="store_true", help="assert the parent has cleared the live profiler gate")
    args = parser.parse_args()
    snapshot, current = _check_preflight_snapshot()
    if not args.execute:
        print(json.dumps({"preflight_verified": True, **current}, indent=2, sort_keys=True))
        return
    if not args.profile_clearance:
        parser.error("full pair checkpoint I/O requires --profile-clearance from the parent")
    result = run_rescore(snapshot, current)
    print(json.dumps({k: v for k, v in result.items() if k in ("protocol_id", "coverage", "reference_model_id_under_v7", "models", "runtime")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
