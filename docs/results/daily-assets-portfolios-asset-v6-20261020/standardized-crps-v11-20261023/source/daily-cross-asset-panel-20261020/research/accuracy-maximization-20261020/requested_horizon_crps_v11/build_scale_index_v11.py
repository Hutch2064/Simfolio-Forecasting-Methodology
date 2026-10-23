#!/usr/bin/env python3
"""Build the read-only V11 prefix/request scale sidecar from frozen inputs."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_PATH = HERE / "protocol.json"
SCAN_PATH = ROOT / "training_sd_domain_scan_v1/scan_training_sd_domain.py"
DOMAIN_PATH = ROOT / "training_sd_domain_scan_v1/domain_scan.json"
PLAN_PATH = ROOT / "short_horizon_final240_v3/plan.json"
MAP_PATH = ROOT / "horizon_buckets_v8/requested_calendar_mapping.npz"
INPUTS = ROOT / "asset_level_v6/inputs"
PANEL_PATH = INPUTS / "panel.json"
CATALOGUE_PATH = INPUTS / "portfolio_catalogue.json"
OUT_DB = HERE / "scale_index.sqlite3"
OUT_RECEIPT = HERE / "scale_index_receipt.json"
EXPECTED_WEIGHT_PLAN_SHA256 = "207ee6316df2a0d622062e56c0ba5e9697b38f387458015db031250bc20b1132"
EXPECTED_FULL_REQUESTS = 320017


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_prefix_sd_helper():
    spec = importlib.util.spec_from_file_location("frozen_training_sd_domain_scan", SCAN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen score-blind Welford helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prefix_sd_by_stop


def _expected_sources(protocol: dict) -> dict[str, str]:
    result = {}
    for name, binding in protocol["source_bindings"].items():
        path = ROOT / binding["path"]
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise RuntimeError(f"Frozen protocol source changed: {name}")
        result[str(path)] = actual
    return result


def _check_full_input_manifest(panel: dict, expected_sha: str) -> tuple[str, int, int]:
    manifest = []
    total_bytes = 0
    for row in panel["series"]:
        for rel in (row["dates_file"], row["returns_file"], row["origins_file"]):
            path = INPUTS / rel
            actual = sha256_file(path)
            if actual != row["files"][rel]:
                raise RuntimeError(f"Input file differs from panel manifest: {rel}")
            manifest.append((rel, actual))
            total_bytes += path.stat().st_size
    blob = "\n".join(f"{rel}\t{digest}" for rel, digest in sorted(manifest)).encode()
    actual_manifest_sha = hashlib.sha256(blob).hexdigest()
    if actual_manifest_sha != expected_sha:
        raise RuntimeError("Verified input manifest differs from score-blind domain scan")
    return actual_manifest_sha, len(manifest), total_bytes


def _planned_rows(plan: dict, mapping) -> list[dict]:
    cases = plan["cases"]
    original = mapping["original"]
    if original.shape != (len(cases), 8):
        raise RuntimeError("Original request mapping shape changed")
    if not np.array_equal(original[:, 0], np.arange(len(cases))):
        raise RuntimeError("Original request case ordering changed")
    if not np.array_equal(original[:, 1], np.asarray([c["series_index"] for c in cases])):
        raise RuntimeError("Original series indices differ from frozen plan")
    if not np.array_equal(original[:, 2], np.asarray([c["origin_index"] for c in cases])):
        raise RuntimeError("Original origin indices differ from frozen plan")
    if not np.array_equal(original[:, 3], np.asarray([c["origin"]["position"] for c in cases])):
        raise RuntimeError("Original training positions differ from frozen plan")
    if not np.array_equal(original[:, 4], np.asarray([c["origin"]["max_horizon"] for c in cases])):
        raise RuntimeError("Original native horizons differ from frozen plan")
    rows = []
    for i, case in enumerate(cases):
        end_day = int(original[i, 6])
        H = int(case["origin"]["max_horizon"])
        rows.append({
            "request_id": json.dumps(["original", i, H], separators=(",", ":")),
            "request_kind": "original", "case_index": i,
            "series_id": str(case["series_id"]), "origin_index": int(case["origin_index"]),
            "origin_date": str(case["origin"]["origin_date"]),
            "requested_end_date": str(np.datetime64(end_day, "D")),
            "native_h": H,
        })

    short_cases = np.asarray(mapping["short_group_case_indices"], dtype=np.int64)
    short_horizons = np.asarray(mapping["short_horizons"], dtype=np.int64)
    short_ends = mapping["short_requested_end_days"]
    if not np.array_equal(short_horizons, np.array([1, 5, 10, 21], dtype=np.int64)):
        raise RuntimeError("Short requested horizons differ from frozen mapping")
    if short_ends.shape != (len(short_cases), len(short_horizons)):
        raise RuntimeError("Short requested-end mapping shape changed")
    if not np.array_equal(short_cases, np.asarray(plan["group_indices"], dtype=np.int64)):
        raise RuntimeError("Short-group cases differ from frozen plan")
    for gi, raw_case_index in enumerate(short_cases):
        i = int(raw_case_index)
        case = cases[i]
        for hj, raw_h in enumerate(short_horizons):
            H = int(raw_h)
            if H > int(case["origin"]["max_horizon"]):
                raise RuntimeError(f"Short requested H exceeds native horizon for case {i}")
            rows.append({
                "request_id": json.dumps(["short", i, H], separators=(",", ":")),
                "request_kind": "short", "case_index": i,
                "series_id": str(case["series_id"]), "origin_index": int(case["origin_index"]),
                "origin_date": str(case["origin"]["origin_date"]),
                "requested_end_date": str(np.datetime64(int(short_ends[gi, hj]), "D")),
                "native_h": H,
            })
    if len(rows) != EXPECTED_FULL_REQUESTS or len({r["request_id"] for r in rows}) != len(rows):
        raise RuntimeError("Frozen planned request inventory or identity uniqueness changed")
    return rows


def _create_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing scale index: {path}")
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.executescript("""
      CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE prefix_scales (
        case_index INTEGER PRIMARY KEY,
        series_id TEXT NOT NULL,
        series_index INTEGER NOT NULL,
        origin_index INTEGER NOT NULL,
        origin_date TEXT NOT NULL,
        origin_position INTEGER NOT NULL,
        training_n INTEGER NOT NULL,
        prefix_mean REAL NOT NULL,
        training_sd REAL NOT NULL,
        exact_constant INTEGER NOT NULL CHECK(exact_constant IN (0,1))
      );
      CREATE INDEX prefix_scales_series_origin ON prefix_scales(series_id, origin_index);
      CREATE TABLE request_scales (
        request_id TEXT PRIMARY KEY,
        request_kind TEXT NOT NULL CHECK(request_kind IN ('original','short')),
        case_index INTEGER NOT NULL,
        series_id TEXT NOT NULL,
        origin_index INTEGER NOT NULL,
        origin_date TEXT NOT NULL,
        requested_end_date TEXT NOT NULL,
        native_h INTEGER NOT NULL,
        horizon_bucket TEXT NOT NULL CHECK(horizon_bucket IN ('short','medium','long')),
        training_sd REAL NOT NULL,
        requested_scale REAL NOT NULL,
        within_bucket_weight REAL NOT NULL,
        global_weight REAL NOT NULL
      );
      CREATE INDEX request_scales_case ON request_scales(case_index);
      CREATE INDEX request_scales_bucket ON request_scales(horizon_bucket);
      CREATE INDEX request_scales_series ON request_scales(series_id);
    """)
    return con


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get("protocol_id") != "daily_portfolio_panel_crps_v11_train_sd_sqrt_requested_h_equal_days_fixed_calendar_buckets":
        raise RuntimeError("Unexpected V11 definition protocol")
    bindings = _expected_sources(protocol)
    plan = json.loads(PLAN_PATH.read_text())
    panel = json.loads(PANEL_PATH.read_text())
    domain = json.loads(DOMAIN_PATH.read_text())
    if plan["fingerprint"] != protocol["source_bindings"]["scientific_plan.json"]["fingerprint"]:
        raise RuntimeError("Scientific plan fingerprint differs from frozen V11 protocol")
    if panel["panel_fingerprint"] != protocol["source_bindings"]["panel.json"]["fingerprint"]:
        raise RuntimeError("Panel fingerprint differs from frozen V11 protocol")
    if domain.get("status") != "complete" or domain.get("score_blind") is not True:
        raise RuntimeError("Score-blind training scale domain receipt is not complete")
    if domain["prefix_domain"]["zero_sd_prefix_count"] != 0 or not domain["request_scale_domain"]["all_requested_scales_positive_finite"]:
        raise RuntimeError("The frozen score-blind domain proof no longer supports this definition")
    manifest_sha, input_count, input_bytes = _check_full_input_manifest(panel, domain["verified_inputs"]["sorted_path_sha256_manifest"])

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "asset_level_v6"))
    from inputs import AssetInputs
    import score_series_buckets_v9 as v9
    import score_horizon_buckets_v8 as v8
    if v9.SCORE_PROTOCOL_ID != "daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters":
        raise RuntimeError("Inherited V9 hierarchy protocol changed")
    if sha256_file(ROOT / "score_series_buckets_v9.py") != "c598301b977fb1d8405de640476289d071107633e8ba72fbedf3ef8712d4796c":
        raise RuntimeError("Inherited V9 hierarchy source changed")
    if sha256_file(ROOT / "score_horizon_buckets_v8.py") != "9a1222c60dc7adc7261fd467bfab835ab33681793e3baa8995597d690be1b70b":
        raise RuntimeError("Inherited V8 date-bucket source changed")

    mapping = np.load(MAP_PATH, allow_pickle=False)
    cases = plan["cases"]
    series_rows = panel["series"]
    by_id = {str(r["series_id"]): r for r in series_rows}
    metadata = {str(r["series_id"]): r for r in series_rows}
    planned = _planned_rows(plan, mapping)
    for row in planned:
        actual_bucket = v8.horizon_bucket(row["origin_date"], row["requested_end_date"])
        row["horizon_bucket"] = actual_bucket
    weights = v9.build_weights(planned, metadata)
    if weights["planned_request_count"] != EXPECTED_FULL_REQUESTS:
        raise RuntimeError("V9 frozen hierarchy produced an unexpected planned count")
    if weights["weight_plan_sha256"] != EXPECTED_WEIGHT_PLAN_SHA256:
        raise RuntimeError("Rebuilt V9 planned weights differ from the frozen V9 plan")

    case_lists: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, case in enumerate(cases):
        sid = str(case["series_id"])
        row = by_id.get(sid)
        if row is None or int(case["series_index"]) >= len(series_rows) or series_rows[int(case["series_index"])]["series_id"] != sid:
            raise RuntimeError(f"Frozen case/series index mismatch at {i}")
        case_lists[sid].append((i, case))

    prefix_sd_by_stop = _load_prefix_sd_helper()
    input_obj = AssetInputs(INPUTS)
    prefix_rows = {}
    portfolio_max_error = 0.0
    portfolio_rows_reconstructed = 0
    for row in series_rows:
        sid = str(row["series_id"])
        these = case_lists[sid]
        if not these:
            raise RuntimeError(f"Planned panel series has no training prefixes: {sid}")
        stops_by_case = {i: int(c["origin"]["position"]) + 1 for i, c in these}
        max_stop = max(stops_by_case.values())
        matrix, exact_weights, constituent_ids = input_obj.training(row, max_stop)
        if matrix.shape[0] != max_stop or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"Invalid training matrix for {sid}")
        if row["panel_branch"] == "assets":
            if matrix.shape[1] != 1 or tuple(constituent_ids) != (sid,):
                raise RuntimeError(f"Asset input identity mismatch for {sid}")
            returns = np.asarray(matrix[:, 0], dtype=np.float64)
        else:
            rational = [Fraction(n, d) for _, n, d in row["rational_weights"]]
            expected_weights = np.asarray([float(x) for x in rational], dtype=np.float64)
            expected_ids = tuple(x[0] for x in row["rational_weights"])
            if tuple(constituent_ids) != expected_ids or not np.array_equal(exact_weights, expected_weights):
                raise RuntimeError(f"Portfolio constituent order or weights mismatch for {sid}")
            returns = np.log1p(np.sum(np.expm1(matrix) * exact_weights[None, :], axis=1))
            target = np.load(INPUTS / row["returns_file"], mmap_mode="r", allow_pickle=False)[:max_stop]
            errors = np.abs(returns - target)
            maximum = float(np.max(errors))
            if maximum > 2e-14:
                raise RuntimeError(f"Portfolio reconstruction differs from panel target for {sid}")
            portfolio_max_error = max(portfolio_max_error, maximum)
            portfolio_rows_reconstructed += max_stop
        if not np.all(np.isfinite(returns)):
            raise RuntimeError(f"Nonfinite portfolio training stream for {sid}")
        stats = prefix_sd_by_stop(returns, list(stops_by_case.values()))
        for case_index, stop in stops_by_case.items():
            mean, sd = stats[stop]
            exact_constant = bool(np.all(returns[:stop] == returns[0]))
            if not math.isfinite(sd) or sd <= 0 or exact_constant:
                raise RuntimeError(f"V11 scale undefined or constant at planned prefix case {case_index}")
            case = cases[case_index]
            prefix_rows[case_index] = {
                "case_index": case_index,
                "series_id": sid,
                "series_index": int(case["series_index"]),
                "origin_index": int(case["origin_index"]),
                "origin_date": str(case["origin"]["origin_date"]),
                "origin_position": int(case["origin"]["position"]),
                "training_n": int(stop),
                "prefix_mean": float(mean),
                "training_sd": float(sd),
                "exact_constant": False,
            }
    if len(prefix_rows) != len(cases) or len(prefix_rows) != 64049:
        raise RuntimeError("Not every planned training prefix received a scale")

    request_rows = []
    for row in planned:
        case_index = int(row["case_index"])
        prefix = prefix_rows[case_index]
        if prefix["series_id"] != row["series_id"] or prefix["origin_date"] != row["origin_date"]:
            raise RuntimeError(f"Request/prefix identity mismatch: {row['request_id']}")
        H = int(row["native_h"])
        scale = prefix["training_sd"] * math.sqrt(H)
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"Invalid requested scale for {row['request_id']}")
        weight_record = weights["requests"][row["request_id"]]
        if weight_record["series_id"] != row["series_id"] or weight_record["bucket"] != row["horizon_bucket"]:
            raise RuntimeError(f"Frozen V9 weight identity mismatch: {row['request_id']}")
        request_rows.append({
            **row,
            "training_sd": prefix["training_sd"],
            "requested_scale": scale,
            "within_bucket_weight": float(weight_record["weight"]),
            "global_weight": float(weight_record["weight"]) / 3.0,
        })

    if OUT_DB.exists() or OUT_RECEIPT.exists():
        raise FileExistsError("V11 scale-index destination already exists; refusing overwrite")
    con = _create_db(OUT_DB)
    try:
        with con:
            con.executemany(
                "INSERT INTO prefix_scales VALUES(?,?,?,?,?,?,?,?,?,?)",
                [(r["case_index"], r["series_id"], r["series_index"], r["origin_index"], r["origin_date"], r["origin_position"], r["training_n"], r["prefix_mean"], r["training_sd"], int(r["exact_constant"])) for r in prefix_rows.values()],
            )
            con.executemany(
                "INSERT INTO request_scales VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(r["request_id"], r["request_kind"], r["case_index"], r["series_id"], r["origin_index"], r["origin_date"], r["requested_end_date"], r["native_h"], r["horizon_bucket"], r["training_sd"], r["requested_scale"], r["within_bucket_weight"], r["global_weight"]) for r in request_rows],
            )
            metadata = {
                "schema": "simfolio_v11_scale_index_v1",
                "protocol_id": protocol["protocol_id"],
                "plan_fingerprint": plan["fingerprint"],
                "plan_sha256": sha256_file(PLAN_PATH),
                "mapping_sha256": sha256_file(MAP_PATH),
                "panel_sha256": sha256_file(PANEL_PATH),
                "input_manifest_sha256": manifest_sha,
                "prefix_count": len(prefix_rows),
                "request_count": len(request_rows),
                "weight_plan_sha256": weights["weight_plan_sha256"],
                "scale_definition": "sample SD ddof=1 of exact causal training portfolio rows times sqrt(requested native H)",
            }
            for key, value in metadata.items():
                con.execute("INSERT INTO metadata VALUES(?,?)", (key, json.dumps(value, sort_keys=True, separators=(",", ":"))))
            for path, digest in bindings.items():
                con.execute("INSERT INTO metadata VALUES(?,?)", ("source:" + path, json.dumps(digest)))
            con.execute("INSERT INTO metadata VALUES(?,?)", ("prefix_sd_min", json.dumps(min(r["training_sd"] for r in prefix_rows.values()))))
            con.execute("INSERT INTO metadata VALUES(?,?)", ("prefix_sd_max", json.dumps(max(r["training_sd"] for r in prefix_rows.values()))))
            con.execute("INSERT INTO metadata VALUES(?,?)", ("portfolio_reconstruction_max_abs_error", json.dumps(portfolio_max_error)))
        check = con.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {check}")
        prefix_count = con.execute("SELECT COUNT(*) FROM prefix_scales").fetchone()[0]
        request_count = con.execute("SELECT COUNT(*) FROM request_scales").fetchone()[0]
        buckets = dict(con.execute("SELECT horizon_bucket,COUNT(*) FROM request_scales GROUP BY horizon_bucket"))
        if prefix_count != 64049 or request_count != EXPECTED_FULL_REQUESTS:
            raise RuntimeError("SQLite table row counts differ from frozen full schedule")
    finally:
        con.close()

    db_sha = sha256_file(OUT_DB)
    db_bytes = OUT_DB.stat().st_size
    if db_bytes > 100 * 1024 * 1024:
        raise RuntimeError(f"Scale sidecar exceeds 0.1 GiB bound: {db_bytes} bytes")
    receipt = {
        "schema": "simfolio_v11_scale_index_receipt_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "index_path": str(OUT_DB),
        "index_sha256": db_sha,
        "index_bytes": db_bytes,
        "sqlite_integrity_check": "ok",
        "prefix_count": prefix_count,
        "request_count": request_count,
        "request_count_by_bucket": buckets,
        "request_count_by_kind": {"original": len(cases), "short": len(request_rows) - len(cases)},
        "weight_plan_sha256": weights["weight_plan_sha256"],
        "verified_input_manifest_sha256": manifest_sha,
        "verified_input_file_count": input_count,
        "verified_input_bytes": input_bytes,
        "portfolio_reconstruction_rows": portfolio_rows_reconstructed,
        "portfolio_reconstruction_max_abs_error": portfolio_max_error,
        "prefix_sd_min": min(r["training_sd"] for r in prefix_rows.values()),
        "prefix_sd_max": max(r["training_sd"] for r in prefix_rows.values()),
        "all_scales_positive_finite": True,
        "source_hashes": bindings,
        "comparative_scores_read": False,
        "forecasts_or_fits_run": False,
    }
    OUT_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "index_path": str(OUT_DB), "index_sha256": db_sha, "index_bytes": db_bytes, "prefixes": prefix_count, "requests": request_count, "buckets": buckets}, sort_keys=True))


if __name__ == "__main__":
    main()
