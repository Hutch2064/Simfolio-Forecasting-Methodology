"""Validate one frozen task per baseline through the v6 index and both scorers."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

V = Path(__file__).resolve().parent
O = V.parent
sys.path.insert(0, str(V))
import rescore_full_pair_v7 as runner  # noqa: E402

PANEL_ROOT = V / "inputs"
RESULTS = V / "results"
INDEX = RESULTS / "native-loss-index.sqlite3"
OUTPUT = V / "rescore_full_pair_v7_first_task_smoke.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    preflight = runner._metadata_preflight()
    panel = runner.api.load_panel(PANEL_ROOT, verify_files=False)
    assert panel["panel_fingerprint"] == preflight["panel_fingerprint"]
    series_index, origin_index = 0, 0
    series = panel["series"][series_index]
    dates, values, origins = runner.api.load_series(PANEL_ROOT, series, verify_files=True)
    origin = origins[origin_index]
    task_key = runner.native_index.verified._task_key(series["series_id"], origin)
    history_end = origin["position"] + 1
    horizon = origin["max_horizon"]
    future_dates = dates[history_end:history_end + horizon]
    history = values[:history_end]

    connection = sqlite3.connect(
        "file:" + quote(str(INDEX.resolve()), safe="/") + "?mode=ro", uri=True,
    )
    connection.row_factory = sqlite3.Row
    checked = {}
    manifests = {}
    checkpoint_hashes = {}
    try:
        scorer_digest = runner.native_index.source_digest()
        for mid in runner.MODEL_IDS:
            root = Path(runner.api.model_directory(RESULTS, mid))
            identity, fingerprint = runner.native_index.verified._read_model_manifest(
                root, panel["panel_fingerprint"], mid, runner.api,
            )
            assert fingerprint == runner.EXECUTION_FINGERPRINTS[mid]
            spec = runner.read(runner.LAUNCH)["models"][mid]
            assert identity["factory"] == spec["factory"]
            assert identity["implementation_digest"] == spec["implementation_digest"]
            manifest_path = root / "execution.json"
            manifests[mid] = runner.sha256_file(manifest_path)
            row = connection.execute(
                """SELECT task_key,series_id,origin_date,main_score,loss_sha256,
                          checkpoint_path,checkpoint_sha256,status
                   FROM task_scores_main
                   WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=?
                     AND execution_fingerprint=? AND task_key=? AND status='ok'""",
                (panel["panel_fingerprint"], scorer_digest, mid, fingerprint, task_key),
            ).fetchone()
            if row is None:
                raise ValueError(f"the exact first task is missing from the v6 index: {mid}")
            checkpoint, _ = runner.api.task_paths(root, (series_index, origin_index))
            if row["checkpoint_path"] != str(checkpoint):
                raise ValueError(f"the v6 index points to a different checkpoint: {mid}")
            blob = checkpoint.read_bytes()
            checkpoint_sha = sha256_bytes(blob)
            if checkpoint_sha != row["checkpoint_sha256"]:
                raise ValueError(f"the saved checkpoint SHA differs from the v6 index: {mid}")
            task_identity = runner.api.task_identity(fingerprint, mid, series["series_id"], origin)
            losses, _ = runner.api.original._read_checkpoint(io.BytesIO(blob), task_identity)
            loss_sha = hashlib.sha256(runner.np.asarray(losses, dtype="<f8").tobytes()).hexdigest()
            if loss_sha != row["loss_sha256"]:
                raise ValueError(f"the saved loss SHA differs from the v6 index: {mid}")
            v6_score = runner.score_main.origin_score(
                losses, history, origin["origin_date"], future_dates,
            )
            if abs(v6_score - row["main_score"]) > 2e-12:
                raise ValueError(f"the v6 origin score does not replay from checkpoint: {mid}")
            v7_score = runner.score_aligned_v7.origin_score(
                losses, history, origin["origin_date"], future_dates,
            )
            checked[mid] = {
                "task_key": task_key,
                "case_index": f"{series_index:06d}/{origin_index:03d}",
                "series_id": series["series_id"],
                "origin_date": origin["origin_date"],
                "native_horizon": horizon,
                "training_observations": history_end,
                "task_identity": task_identity,
                "checkpoint_sha256": checkpoint_sha,
                "loss_sha256": loss_sha,
                "legacy_v6_score_replayed": v6_score,
                "v7_score": v7_score,
            }
            checkpoint_hashes[mid] = checkpoint_sha
    finally:
        connection.close()

    if len(checked) != 2:
        raise ValueError("first-task smoke did not validate both frozen baselines")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "one_task_per_baseline_identity_and_score_smoke_passed",
        "protocol_id": runner.EXPECTED_PROTOCOL,
        "panel_fingerprint": panel["panel_fingerprint"],
        "task_count_per_baseline": 1,
        "forecast_calls": 0,
        "full_panel_rescore_performed": False,
        "task": checked,
        "source_hashes": {
            **preflight["source_hashes"],
            "legacy_native_index_digest": scorer_digest,
            "execution_manifests": manifests,
            "first_task_checkpoints": checkpoint_hashes,
            "smoke_runner": runner.sha256_file(Path(__file__)),
        },
    }
    runner.api.original._atomic_json(OUTPUT, result)
    print(json.dumps({"output": str(OUTPUT), "status": result["status"], "task_key": task_key}, indent=2))


if __name__ == "__main__":
    main()
