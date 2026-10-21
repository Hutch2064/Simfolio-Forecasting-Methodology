"""Independent read-only coverage and hierarchy reduction of the v7 score index.

This verifier reads the compact SQLite scores and frozen origin schedules. It
does not reopen checkpoint NPZ files or rerun either scoring implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

V = Path(__file__).resolve().parent
O = V.parent
CAMPAIGN = V.parents[2]
PANEL_ROOT = V / "inputs"
PANEL_FILE = PANEL_ROOT / "panel.json"
AUDIT = V / "full_pair_independent_audit.json"
PREFLIGHT = V / "rescore_full_pair_v7_preflight.json"
RESULT = V / "full_pair_rescore_v7_diagnostic.json"
INDEX = V / "full_pair_rescore_v7_scores.sqlite3"
OUTPUT = V / "full_pair_rescore_v7_independent_verification.json"
PROTOCOL_ID = "daily_portfolio_panel_crps_v7_train_sd_sqrt_native_lead_equal_days_two_branch"
MODEL_IDS = (
    "model_testing_asset_joint_blocks_replacement_v6",
    "model_testing_asset_joint_blocks_history_partition_v6",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty hierarchy cell")
    return math.fsum(values) / len(values)


def quarter(date: str) -> str:
    year, month = int(date[:4]), int(date[5:7])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid origin month: {date}")
    return f"{year}-Q{(month - 1) // 3 + 1}"


def reduce_hierarchy(records: dict[tuple[str, str], list[float]], series_meta: dict[str, dict]) -> dict:
    """Origins -> series/quarter -> family -> class -> branch -> quarter."""
    by_quarter: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for (q, sid), origins in records.items():
        meta = series_meta[sid]
        branch = meta["panel_branch"]
        cls, family = meta["asset_class"], meta["family"]
        if branch not in ("assets", "portfolios") or not cls or not family:
            raise ValueError(f"invalid frozen hierarchy labels for {sid}")
        by_quarter[q][branch][cls][family].append(mean(origins))

    quarter_scores: dict[str, float] = {}
    quarter_branch_scores: dict[str, dict[str, float]] = {}
    for q, branches in by_quarter.items():
        if "assets" not in branches:
            raise ValueError(f"portfolio support without asset support in {q}")
        branch_scores: dict[str, float] = {}
        for branch, classes in branches.items():
            class_scores = []
            for families in classes.values():
                family_scores = [mean(series_values) for series_values in families.values()]
                class_scores.append(mean(family_scores))
            branch_scores[branch] = mean(class_scores)
        quarter_branch_scores[q] = branch_scores
        if "portfolios" in branch_scores:
            quarter_scores[q] = mean([branch_scores["assets"], branch_scores["portfolios"]])
        else:
            quarter_scores[q] = branch_scores["assets"]
    return {
        "score": mean(list(quarter_scores.values())),
        "quarter_count": len(quarter_scores),
        "portfolio_supported_quarters": sum("portfolios" in value for value in quarter_branch_scores.values()),
        "quarter_scores": quarter_scores,
        "quarter_branch_scores": quarter_branch_scores,
    }


def source_hashes_now(preflight: dict) -> dict:
    paths = {
        "panel_metadata": PANEL_FILE,
        "protocol": O / "final_scoring_protocol_v7.json",
        "v7_scorer": V / "score_aligned_v7.py",
        "legacy_aggregator": V / "score_main.py",
        "legacy_indexer": V / "native_index.py",
        "legacy_checkpoint_reader": CAMPAIGN / "scoring.py",
        "sweep_api": V / "sweep_v6.py",
        "full_pair_audit": AUDIT,
        "launch_plan": V / "launch_plan.json",
        "legacy_scoreboard": V / "results" / "scoreboard.json",
        "runner": V / "rescore_full_pair_v7.py",
    }
    sources = {name: sha256_file(path) for name, path in paths.items()}
    sources["execution_manifests"] = {
        mid: sha256_file(Path(preflight["models"][mid]["execution_manifest_path"]))
        for mid in MODEL_IDS
    }
    return sources


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    panel = json.loads(PANEL_FILE.read_text())
    result = json.loads(RESULT.read_text())
    audit = json.loads(AUDIT.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    identity = {key: value for key, value in panel.items() if key != "panel_fingerprint"}
    panel_fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if not (panel_fingerprint == panel["panel_fingerprint"] == result["panel_fingerprint"]):
        raise ValueError("panel fingerprint does not match the v7 receipt")
    if result["protocol_id"] != PROTOCOL_ID:
        raise ValueError("v7 protocol ID differs")
    if result["coverage"]["new_forecasts"] != 0:
        raise ValueError("receipt reports new forecasts")
    if sha256_file(INDEX) != result["score_index_sha256"]:
        raise ValueError("new score index SHA differs from its receipt")
    if result["score_index_path"] != str(INDEX):
        raise ValueError("receipt is bound to a different score index")

    current_sources = source_hashes_now(preflight)
    if preflight["source_hashes"] != current_sources:
        raise ValueError("source hashes differ from the frozen preflight")
    receipt_source_aliases = {"legacy_scoreboard_unchanged": "legacy_scoreboard"}
    for receipt_name, expected_sha in result["source_hashes"].items():
        source_name = receipt_source_aliases.get(receipt_name, receipt_name)
        if current_sources.get(source_name) != expected_sha:
            raise ValueError(f"receipt source hash differs from the frozen preflight: {receipt_name}")

    expected: dict[str, tuple[int, int, dict, dict]] = {}
    series_meta = {}
    schedule_hashes = {}
    for i, row in enumerate(panel["series"]):
        sid = row["series_id"]
        if sid in series_meta:
            raise ValueError(f"duplicate series metadata: {sid}")
        series_meta[sid] = row
        rel = Path(row["origins_file"])
        schedule_path = (PANEL_ROOT / rel).resolve()
        if rel.is_absolute() or PANEL_ROOT.resolve() not in schedule_path.parents:
            raise ValueError("origin schedule escapes frozen input root")
        expected_sha = row["files"].get(row["origins_file"])
        actual_sha = sha256_file(schedule_path)
        if actual_sha != expected_sha:
            raise ValueError(f"frozen origin schedule hash mismatch: {sid}")
        schedule_hashes[str(rel)] = actual_sha
        origins = json.loads(schedule_path.read_text())
        if len(origins) != row["origins_count"]:
            raise ValueError(f"origin count differs for {sid}")
        for j, origin in enumerate(origins):
            key = canonical([sid, origin])
            if key in expected:
                raise ValueError(f"duplicate scheduled task key: {key}")
            expected[key] = (i, j, row, origin)
    if len(expected) != panel["expected_tasks"] or len(expected) != 64049:
        raise ValueError(f"frozen schedule task count differs: {len(expected)}")

    uri = "file:" + quote(str(INDEX.resolve()), safe="/") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        metadata = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key,value FROM metadata")
        }
        if metadata["schema_version"] != 1 or metadata["protocol_id"] != PROTOCOL_ID:
            raise ValueError("score index metadata schema/protocol differs")
        if metadata["panel_fingerprint"] != panel_fingerprint:
            raise ValueError("score index metadata is bound to a different panel")
        if metadata["baseline_model_ids"] != list(MODEL_IDS):
            raise ValueError("score index model pair differs")
        if metadata["task_index_row_count"] != 128098:
            raise ValueError("score index metadata row count differs")
        if metadata["source_hashes"] != result["source_hashes"]:
            raise ValueError("SQLite source-hash metadata differs from JSON receipt")
        if (metadata["models"] != result["models"]
                or metadata["coverage"] != result["coverage"]
                or metadata["reference_rule"] != result["reference_rule"]):
            raise ValueError("SQLite score/coverage/reference metadata differs from JSON receipt")
        model_keys: dict[str, set[str]] = {}
        direct_scores = {}
        for mid in MODEL_IDS:
            task_keys = set()
            score_records: dict[tuple[str, str], list[float]] = defaultdict(list)
            legacy_records: dict[tuple[str, str], list[float]] = defaultdict(list)
            cursor = connection.execute(
                """SELECT model_id,task_key,panel_fingerprint,protocol_id,
                          execution_fingerprint,case_index,series_index,origin_index,
                          series_id,origin_date,native_horizon,training_observations,
                          task_identity_json,source_checkpoint_sha256,loss_sha256,
                          score_v7,legacy_score_v6
                   FROM task_scores_v7 WHERE model_id=? ORDER BY task_key""",
                (mid,),
            )
            count = 0
            for row in cursor:
                (model_id, task_key, row_panel, protocol_id, exec_fp, case_index,
                 series_index, origin_index, sid, origin_date, horizon, train_obs,
                 identity_json, checkpoint_sha, loss_sha, score_v7, score_v6) = row
                count += 1
                if task_key in task_keys:
                    raise ValueError(f"duplicate task key for {mid}: {task_key}")
                task_keys.add(task_key)
                expected_task = expected.get(task_key)
                if expected_task is None:
                    raise ValueError(f"unexpected task key for {mid}: {task_key}")
                i, j, series, origin = expected_task
                if (model_id != mid or row_panel != panel_fingerprint or protocol_id != PROTOCOL_ID
                        or exec_fp != preflight["models"][mid]["execution_fingerprint"]
                        or case_index != f"{i:06d}/{j:03d}" or series_index != i or origin_index != j
                        or sid != series["series_id"] or origin_date != origin["origin_date"]
                        or horizon != origin["max_horizon"] or train_obs != origin["position"] + 1):
                    raise ValueError(f"task identity/case metadata mismatch for {mid}: {task_key}")
                task_identity = json.loads(identity_json)
                if (task_identity.get("model_id") != mid
                        or task_identity.get("execution_fingerprint") != exec_fp
                        or task_identity.get("series_id") != sid
                        or task_identity.get("origin") != origin
                        or task_identity.get("training_observations") != train_obs):
                    raise ValueError(f"saved task identity JSON mismatch for {mid}: {task_key}")
                if (len(checkpoint_sha) != 64 or len(loss_sha) != 64
                        or any(ch not in "0123456789abcdef" for ch in checkpoint_sha + loss_sha)):
                    raise ValueError(f"invalid saved checkpoint/loss SHA format for {mid}")
                if not (math.isfinite(score_v7) and score_v7 >= 0.0
                        and math.isfinite(score_v6) and score_v6 >= 0.0):
                    raise ValueError(f"invalid saved origin score for {mid}")
                q = quarter(origin_date)
                score_records[(q, sid)].append(float(score_v7))
                legacy_records[(q, sid)].append(float(score_v6))
            if count != panel["expected_tasks"]:
                raise ValueError(f"expected 64,049 rows for {mid}, saw {count}")
            if task_keys != set(expected):
                raise ValueError(f"scheduled task key coverage differs for {mid}")
            model_keys[mid] = task_keys
            score_result = reduce_hierarchy(score_records, series_meta)
            legacy_result = reduce_hierarchy(legacy_records, series_meta)
            direct_scores[mid] = {"v7": score_result, "v6": legacy_result}
        if model_keys[MODEL_IDS[0]] != model_keys[MODEL_IDS[1]]:
            raise ValueError("the two baselines do not have identical task-key coverage")
    finally:
        connection.close()

    reference = max(MODEL_IDS, key=lambda mid: direct_scores[mid]["v7"]["score"])
    if reference != result["reference_model_id_under_v7"] or reference != MODEL_IDS[0]:
        raise ValueError("v7 worse-reference selection does not replay")
    models = {}
    for mid in MODEL_IDS:
        reported = result["models"][mid]
        independent_v7 = direct_scores[mid]["v7"]
        independent_v6 = direct_scores[mid]["v6"]
        if abs(independent_v7["score"] - reported["v7_Q"]) > 2e-12:
            raise ValueError(f"independent v7 Q differs for {mid}")
        if abs(independent_v6["score"] - audit["models"][mid]["main_score"]) > 2e-12:
            raise ValueError(f"independent legacy Q does not reconcile with frozen audit for {mid}")
        if independent_v7["quarter_count"] != reported["quarter_count"] or independent_v7["quarter_count"] != 343:
            raise ValueError(f"quarter support differs for {mid}")
        if independent_v7["portfolio_supported_quarters"] != reported["portfolio_supported_quarters"]:
            raise ValueError(f"portfolio branch support differs for {mid}")
        models[mid] = {
            "independently_reduced_v7_Q": independent_v7["score"],
            "reported_v7_Q": reported["v7_Q"],
            "absolute_v7_Q_difference": abs(independent_v7["score"] - reported["v7_Q"]),
            "independently_reduced_v6_Q": independent_v6["score"],
            "frozen_audit_v6_Q": audit["models"][mid]["main_score"],
            "quarter_count": independent_v7["quarter_count"],
            "portfolio_supported_quarters": independent_v7["portfolio_supported_quarters"],
            "task_count": panel["expected_tasks"],
        }
    reference_q = direct_scores[reference]["v7"]["score"]
    for mid in MODEL_IDS:
        expected_gain = 100.0 * (reference_q - direct_scores[mid]["v7"]["score"]) / reference_q
        actual_gain = result["models"][mid]["percent_better_vs_v7_reference"]
        if abs(expected_gain - actual_gain) > 2e-12:
            raise ValueError(f"percent-improvement display does not replay for {mid}")

    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "independent_index_coverage_source_hash_and_hierarchy_check_passed",
        "protocol_id": PROTOCOL_ID,
        "panel_fingerprint": panel_fingerprint,
        "score_index_path": str(INDEX),
        "score_index_sha256": sha256_file(INDEX),
        "sqlite_integrity": integrity,
        "index_rows_verified": 128098,
        "tasks_per_model": panel["expected_tasks"],
        "models": models,
        "reference_model_id_under_v7": reference,
        "reference_rule": "higher direct v7 Q of the frozen WR/WOR pair on exact scheduled task-key coverage",
        "scheduled_origin_files_sha256_verified": len(schedule_hashes),
        "checkpoint_npz_files_reopened": 0,
        "forecast_calls": 0,
        "native_failures": result["coverage"]["native_failures"],
        "score_integrity_errors": result["coverage"]["score_integrity_errors"],
        "infrastructure_failures_before_successful_run": 2,
        "infrastructure_failure_notes": [
            "First one-shot attempt exited before checkpoint iteration on a stale variable name; corrected and preserved its log.",
            "Second one-shot attempt was deliberately SIGTERM-stopped before the requested bounded smoke; no output artifact was published.",
        ],
        "source_hashes_reverified": current_sources,
        "verifier_sha256": sha256_file(Path(__file__)),
        "claim_boundary": "Independent reduction of stored origin scores; not a second CRPS-from-loss recomputation or significance test.",
    }
    if verification["native_failures"] != 0 or verification["score_integrity_errors"] != 0:
        raise ValueError("full pair reported a native failure or scoring integrity error")
    if len(schedule_hashes) != len(panel["series"]):
        raise ValueError("did not verify every frozen per-series origin schedule")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    tmp = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    if tmp.exists():
        raise FileExistsError(tmp)
    tmp.write_text(json.dumps(verification, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(OUTPUT)
    print(json.dumps({
        "status": verification["status"],
        "reference": reference,
        "models": {mid: models[mid]["independently_reduced_v7_Q"] for mid in MODEL_IDS},
        "index_rows_verified": verification["index_rows_verified"],
        "score_index_sha256": verification["score_index_sha256"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
