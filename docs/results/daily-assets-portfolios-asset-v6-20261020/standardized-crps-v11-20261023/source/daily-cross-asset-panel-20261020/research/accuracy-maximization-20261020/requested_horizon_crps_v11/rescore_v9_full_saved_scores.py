#!/usr/bin/env python3
"""Rescore the closed V9 full panel from its saved raw-request-mean cache."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D = ROOT / "short_horizon_final240_v3"
SCORE_DIR = ROOT / "series_buckets_v9_live"
SCORE_CACHE = SCORE_DIR / "raw_cache.sqlite3"
SCOREBOARD = SCORE_DIR / "scoreboard.json"
HANDOFF_ADMISSION = D / "root_launch_admission.json"
PLAN = D / "plan.json"
PANEL = ROOT / "asset_level_v6/inputs/panel.json"
SCALE_DB = HERE / "scale_index.sqlite3"
SCALE_RECEIPT = HERE / "scale_index_receipt.json"
PROTOCOL = HERE / "protocol.json"
OUT = HERE / "full_primary_rescore.json"
EXPECTED_REQUESTS = 320017
EXPECTED_CACHE_ROWS = 128041
EXPECTED_PLAN_FP = "c9bd17eeb30bc274aa48cf2974dd2246af14e3399696ad13527c3ca4db140291"
EXPECTED_WEIGHT_SHA = "207ee6316df2a0d622062e56c0ba5e9697b38f387458015db031250bc20b1132"
V9_ID = "daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters"
V11_ID = "daily_portfolio_panel_crps_v11_train_sd_sqrt_requested_h_equal_days_fixed_calendar_buckets"
WR = "model_testing_asset_joint_blocks_replacement_v6"
WOR = "model_testing_asset_joint_blocks_history_partition_v6"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from score_requested_horizon_crps_v11 import (  # noqa: E402
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    aggregate_records,
    compare,
    sha256_file,
)
from score_series_buckets_v9 import SCORE_PROTOCOL_ID as V9_SCORER_ID  # noqa: E402
from horizon_buckets_v8_reporter import MODELS, HORIZONS  # noqa: E402


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _read_scale_index() -> tuple[dict, dict, dict, dict]:
    receipt = json.loads(SCALE_RECEIPT.read_text())
    if receipt.get("status") != "complete" or receipt.get("protocol_id") != V11_ID:
        raise RuntimeError("V11 scale-index receipt is not complete or has wrong protocol")
    if sha256_file(SCALE_DB) != receipt.get("index_sha256"):
        raise RuntimeError("V11 scale-index SQLite SHA differs from its receipt")
    if receipt.get("protocol_sha256") != PROTOCOL_SHA256 or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("V11 definition protocol differs from frozen source")
    db = sqlite3.connect(SCALE_DB.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("V11 scale-index SQLite integrity check failed")
        meta = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata") if not key.startswith("source:")}
        if meta.get("protocol_id") != V11_ID or meta.get("request_count") != EXPECTED_REQUESTS:
            raise RuntimeError("V11 scale index is for a different protocol/support")
        if meta.get("weight_plan_sha256") != EXPECTED_WEIGHT_SHA:
            raise RuntimeError("V9 weight-plan fingerprint differs from frozen full panel")
        scales = {}
        weights_map = {}
        for (request_id, kind, case_index, sid, origin_index, origin_date, end_date, H, bucket, sd, scale, within, global_weight) in db.execute(
            "SELECT request_id,request_kind,case_index,series_id,origin_index,origin_date,requested_end_date,native_h,horizon_bucket,training_sd,requested_scale,within_bucket_weight,global_weight FROM request_scales ORDER BY request_id"
        ):
            scales[request_id] = {
                "request_kind": kind,
                "case_index": int(case_index),
                "series_id": sid,
                "origin_index": int(origin_index),
                "origin_date": origin_date,
                "requested_end_date": end_date,
                "native_h": int(H),
                "training_sd": float(sd),
            }
            weights_map[request_id] = {"bucket": bucket, "series_id": sid, "weight": float(within)}
        prefix_count = db.execute("SELECT COUNT(*) FROM prefix_scales").fetchone()[0]
    finally:
        db.close()
    if len(scales) != EXPECTED_REQUESTS:
        raise RuntimeError("V11 scale-index request count differs from full panel")
    weights = {
        "score_protocol_id": V9_ID,
        "weight_plan_sha256": EXPECTED_WEIGHT_SHA,
        "planned_request_count": EXPECTED_REQUESTS,
        "requests": weights_map,
    }
    return receipt, scales, weights, {"prefix_count": prefix_count, **meta}


def _read_checkpoint_path_map(plan: dict) -> tuple[dict, dict, str, str]:
    admission = json.loads(HANDOFF_ADMISSION.read_text())
    binding = admission.get("handoff")
    if not binding or sha256_file(Path(binding["path"])) != binding.get("sha256"):
        raise RuntimeError("Frozen task handoff binding differs")
    handoff = json.loads(Path(binding["path"]).read_text())
    path_map = {}
    counts = Counter()
    old_hashes = {}
    for text, bound in handoff["saved_old_slots"].items():
        i = int(text)
        cp = str(Path(bound["checkpoint_path"]))
        entry = (i, "original", int(bound["paths"]))
        if cp in path_map:
            raise RuntimeError("Duplicate checkpoint in frozen handoff")
        path_map[cp] = entry
        old_hashes[cp] = bound["checkpoint_sha256"]
        counts["saved_1000_original"] += 1
    for text, bound in handoff["inherited_short_groups"].items():
        i = int(text)
        cp = str(Path(bound["checkpoint_path"]))
        entry = (i, "short", int(bound["paths"]))
        if cp in path_map:
            raise RuntimeError("Duplicate checkpoint in frozen handoff")
        path_map[cp] = entry
        old_hashes[cp] = bound["checkpoint_sha256"]
        counts["saved_1000_short"] += 1
    if counts["saved_1000_original"] != 24574 or counts["saved_1000_short"] != 8833:
        raise RuntimeError("Saved 1000-path inherited task coverage differs")
    if plan.get("fingerprint") != EXPECTED_PLAN_FP:
        raise RuntimeError("V9 full scientific plan differs")
    return path_map, old_hashes, binding["path"], binding["sha256"]


def _resolve_task(path: str, path_map: dict, plan: dict, short_case_set: set[int]) -> tuple[int, str, int, str]:
    if path in path_map:
        i, kind, paths = path_map[path]
        return i, kind, paths, "handoff"
    checkpoint = Path(path)
    parent = checkpoint.parent
    if parent == D / "groups":
        i = int(checkpoint.stem)
        if i not in short_case_set:
            raise RuntimeError(f"Short task path has an unplanned case index: {path}")
        return i, "short", 240, "new_short"
    if parent == D / "new_long_groups":
        i = int(checkpoint.stem)
        if i < 0 or i >= len(plan["cases"]):
            raise RuntimeError(f"Original task path has an invalid case index: {path}")
        return i, "original", 240, "new_original"
    raise RuntimeError(f"Raw cache checkpoint path has no frozen request mapping: {path}")


def _request_id(kind: str, index: int, H: int) -> str:
    return _canonical(["short" if kind == "short" else "original", index, H])


def _load_raw_records(scales: dict, plan: dict, path_map: dict, old_hashes: dict) -> tuple[dict[str, list[dict]], dict]:
    signature_before = _signature(SCORE_CACHE)
    database_sha_before = sha256_file(SCORE_CACHE)
    source_rows = {model: [] for model in MODELS}
    model_digests = {model: hashlib.sha256() for model in MODELS}
    raw_commitment = hashlib.sha256()
    counts = Counter()
    seen = set()
    path_signatures = Counter()
    short_case_set = set(map(int, plan["group_indices"]))
    db = sqlite3.connect(SCORE_CACHE.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("V9 saved raw-mean cache integrity check failed")
        count = db.execute("SELECT COUNT(*) FROM raw_pairs").fetchone()[0]
        if count != EXPECTED_CACHE_ROWS:
            raise RuntimeError(f"Unexpected V9 raw-mean cache row count: {count}")
        for path, signature, checkpoint_sha, items_text in db.execute(
            "SELECT path,signature,checkpoint_sha256,items FROM raw_pairs ORDER BY path"
        ):
            path = str(path)
            if path in old_hashes and checkpoint_sha != old_hashes[path]:
                raise RuntimeError(f"Retained 1000-path checkpoint hash differs from handoff: {path}")
            index, kind, final_paths, generation = _resolve_task(path, path_map, plan, short_case_set)
            case = plan["cases"][index]
            items = json.loads(items_text)
            expected_h = [1, 5, 10, 21] if kind == "short" else [int(case["origin"]["max_horizon"])]
            if not isinstance(items, list) or [int(item["H"]) for item in items] != expected_h:
                raise RuntimeError(f"Raw-mean cache lead list differs for {path}")
            raw_commitment.update((path + "\n" + str(checkpoint_sha) + "\n" + str(signature) + "\n" + items_text + "\n").encode())
            path_signatures[generation] += 1
            counts[f"physical_{final_paths}"] += 1
            counts[f"generation_{generation}"] += 1
            for item in items:
                H = int(item["H"])
                request_id = _request_id(kind, index, H)
                if request_id in seen:
                    raise RuntimeError(f"Duplicate raw request identity {request_id}")
                seen.add(request_id)
                scale = scales.get(request_id)
                if scale is None:
                    raise RuntimeError(f"Raw cache request missing from frozen scale index: {request_id}")
                if scale["request_kind"] != kind or scale["case_index"] != index or scale["series_id"] != case["series_id"] or scale["origin_index"] != int(case["origin_index"]):
                    raise RuntimeError(f"Raw cache request identity differs from frozen plan: {request_id}")
                if scale["native_h"] != H or scale["origin_date"] != case["origin"]["origin_date"]:
                    raise RuntimeError(f"Raw cache native H/origin differs from frozen plan: {request_id}")
                values = item.get("values")
                if not isinstance(values, list) or len(values) != len(MODELS):
                    raise RuntimeError(f"Raw cache model triple missing for {request_id}")
                for model, value in zip(MODELS, values):
                    score = float(value)
                    if not math.isfinite(score) or score < 0:
                        raise RuntimeError(f"Invalid saved raw CRPS for {model} {request_id}")
                    model_digests[model].update((request_id + "\t" + score.hex() + "\n").encode())
                    source_rows[model].append({
                        "request_id": request_id,
                        "request_kind": kind,
                        "case_index": int(index),
                        "series_id": case["series_id"],
                        "origin_index": int(case["origin_index"]),
                        "origin_date": case["origin"]["origin_date"],
                        "requested_end_date": scale["requested_end_date"],
                        "requested_native_h": H,
                        "raw_mean_crps": score,
                    })
                counts[f"{kind}_requests"] += 1
    finally:
        db.close()
    if len(seen) != EXPECTED_REQUESTS or any(len(rows) != EXPECTED_REQUESTS for rows in source_rows.values()):
        raise RuntimeError(f"Saved raw-mean request count differs: {len(seen)}")
    expected_ids = set(scales)
    if seen != expected_ids:
        missing = sorted(expected_ids - seen)[:5]
        extra = sorted(seen - expected_ids)[:5]
        raise RuntimeError(f"Raw cache coverage differs from scale index; missing={missing} extra={extra}")
    if _signature(SCORE_CACHE) != signature_before or sha256_file(SCORE_CACHE) != database_sha_before:
        raise RuntimeError("V9 raw-mean cache changed during read")
    return source_rows, {
        "raw_cache_path": str(SCORE_CACHE),
        "raw_cache_sha256": database_sha_before,
        "raw_cache_bytes": SCORE_CACHE.stat().st_size,
        "raw_cache_row_count": EXPECTED_CACHE_ROWS,
        "raw_cache_checkpoint_item_digest_sha256": raw_commitment.hexdigest(),
        "raw_mean_request_digest_sha256": {m: model_digests[m].hexdigest() for m in MODELS},
        "request_count_by_kind": {"original": counts["original_requests"], "short": counts["short_requests"]},
        "checkpoint_group_count_by_final_paths": {"1000": counts["physical_1000"], "240": counts["physical_240"]},
        "checkpoint_groups_by_generation": {k.removeprefix("generation_"): v for k, v in counts.items() if k.startswith("generation_")},
        "raw_cache_paths_by_generation": dict(path_signatures),
    }


def main() -> None:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256 or PROTOCOL_ID != V11_ID:
        raise RuntimeError("Frozen V11 scoring definition differs")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("protocol_id") != V11_ID or protocol.get("comparative_scores_read_before_freeze") is not False:
        raise RuntimeError("V11 protocol is not the expected frozen score definition")
    if V9_SCORER_ID != V9_ID or tuple(MODELS) != ("endpoint", WR, WOR) or tuple(HORIZONS) != (1, 5, 10, 21):
        raise RuntimeError("Inherited V9 model/request protocol differs")
    v9_scoreboard = json.loads(SCOREBOARD.read_text())
    if (v9_scoreboard.get("complete") is not True or v9_scoreboard.get("paired_requests_per_model") != EXPECTED_REQUESTS or
            v9_scoreboard.get("scientific_plan_fingerprint") != EXPECTED_PLAN_FP or
            v9_scoreboard.get("score_protocol_id") != V9_ID or
            v9_scoreboard.get("weight_plan_sha256") != EXPECTED_WEIGHT_SHA):
        raise RuntimeError("V9 frozen full-score completeness/source metadata changed")
    plan = json.loads(PLAN.read_text())
    panel = json.loads(PANEL.read_text())
    scale_receipt, scales, weights, scale_meta = _read_scale_index()
    path_map, old_hashes, handoff_path, handoff_sha = _read_checkpoint_path_map(plan)
    source_rows, cache_receipt = _load_raw_records(scales, plan, path_map, old_hashes)
    metadata = {row["series_id"]: row for row in panel["series"]}
    results = {}
    for model in MODELS:
        results[model] = aggregate_records(
            source_rows[model], metadata, weights=weights, scales_by_request=scales,
            score_key="raw_mean_crps",
        )
        result = results[model]
        if result["complete"] is not True or result["observed_request_count"] != EXPECTED_REQUESTS:
            raise RuntimeError(f"V11 full coverage is incomplete for {model}")
        if result["weight_plan_sha256"] != EXPECTED_WEIGHT_SHA:
            raise RuntimeError(f"V11 hierarchy weights changed for {model}")
    endpoint_vs_wr = compare(results["endpoint"], results[WR])
    wor_vs_wr = compare(results[WOR], results[WR])

    terminal_path = D / "terminal.json"
    source_paths = [
        Path(__file__), PROTOCOL, HERE / "score_requested_horizon_crps_v11.py",
        HERE / "build_scale_index_v11.py", SCORE_CACHE, SCOREBOARD, terminal_path,
        PLAN, PANEL, HANDOFF_ADMISSION, Path(handoff_path), ROOT / "score_series_buckets_v9.py",
        ROOT / "score_horizon_buckets_v8.py", ROOT / "series_buckets_v9_reporter.py",
        ROOT / "horizon_buckets_v8_reporter.py", SCALE_DB, SCALE_RECEIPT,
    ]
    source_hashes = {str(p): sha256_file(p) for p in source_paths}
    if source_hashes[str(SCALE_DB)] != scale_receipt["index_sha256"]:
        raise RuntimeError("V11 scale-index SQLite changed during full score rescore")
    output = {
        "schema": "simfolio_v11_full_saved_score_rescore_v1",
        "status": "complete_pending_independent_score_replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "score_protocol_id": V11_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "scorer_sha256": sha256_file(HERE / "score_requested_horizon_crps_v11.py"),
        "hierarchy_protocol_id": V9_ID,
        "scope": "Complete 320017-request full planned cohort; raw request means reused from the closed V9 saved-score cache; no forecasts/fits rerun; V9 and V10 results are preserved.",
        "planned_requests_per_model": EXPECTED_REQUESTS,
        "observed_requests_per_model": {m: results[m]["observed_request_count"] for m in MODELS},
        "matched_request_coverage_complete": True,
        "same_fixed_request_weights": all(results[m]["weight_plan_sha256"] == EXPECTED_WEIGHT_SHA for m in MODELS),
        "weight_plan_sha256": EXPECTED_WEIGHT_SHA,
        "scale_index": {"path": str(SCALE_DB), "sha256": scale_receipt["index_sha256"], "receipt_path": str(SCALE_RECEIPT), "receipt_sha256": sha256_file(SCALE_RECEIPT), "prefixes": scale_meta["prefix_count"], "requests": scale_meta["request_count"]},
        "v9_saved_raw_means": cache_receipt,
        "source_hashes": source_hashes,
        "models": {
            model: {
                "Q": results[model]["score"],
                "bucket_Q": {bucket: results[model]["buckets"][bucket]["partial_score"] for bucket in ("short", "medium", "long")},
                "request_counts_by_bucket": {bucket: results[model]["buckets"][bucket]["observed_requests"] for bucket in ("short", "medium", "long")},
                "scale_data_commitment_sha256": results[model]["scale_data_commitment_sha256"],
                "complete": results[model]["complete"],
            } for model in MODELS
        },
        "comparisons": {"endpoint_vs_fixed_WR": endpoint_vs_wr, "WOR_vs_fixed_WR": wor_vs_wr},
        "fixed_reference": WR,
        "fullcoverage_not_holdout_warning": "Development-panel rescoring from saved outputs; adaptive/posthoc scoring rule, not untouched independent validation or production model promotion.",
        "canonical_pointer_changed": False,
        "forecasts_or_fits_run": False,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(OUT)
    print(json.dumps({
        "status": output["status"],
        "requests": output["observed_requests_per_model"],
        "Q": {m: output["models"][m]["Q"] for m in MODELS},
        "endpoint_percent_better_than_WR": endpoint_vs_wr["percent_better_than_baseline"],
        "WOR_percent_better_than_WR": wor_vs_wr["percent_better_than_baseline"],
        "output": str(OUT),
        "output_sha256": sha256_file(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
