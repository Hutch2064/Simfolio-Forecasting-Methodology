#!/usr/bin/env python3
"""Score-blind feasibility scan for training-portfolio SD * sqrt(native H)."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


O = Path(__file__).resolve().parents[1]
PLAN = O / "short_horizon_final240_v3/plan.json"
MAP = O / "horizon_buckets_v8/requested_calendar_mapping.npz"
INPUTS = O / "asset_level_v6/inputs"
PANEL = INPUTS / "panel.json"
CATALOGUE = INPUTS / "portfolio_catalogue.json"
DESIGN = O / "training_standardized_requested_horizon_crps_design_review_v1.md"
OUT = Path(__file__).resolve().parent / "domain_scan.json"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"count": 0}
    q = np.quantile(a, [0, .01, .10, .50, .90, .99, 1])
    return {
        "count": int(a.size),
        "nonfinite": int(np.count_nonzero(~np.isfinite(a))),
        "zero": int(np.count_nonzero(a == 0)),
        "positive": int(np.count_nonzero(a > 0)),
        "min": float(q[0]), "p01": float(q[1]), "p10": float(q[2]),
        "p50": float(q[3]), "p90": float(q[4]), "p99": float(q[5]),
        "max": float(q[6]),
    }


def prefix_sd_by_stop(values: np.ndarray, stops: list[int]) -> dict[int, tuple[float, float]]:
    """Welford sample SD in one pass; retain mean and SD at planned stops."""
    wanted = set(stops)
    mean = 0.0
    m2 = 0.0
    found = {}
    for n, raw in enumerate(values, start=1):
        x = float(raw)
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
        if n in wanted:
            var = m2 / (n - 1)
            if not math.isfinite(var) or var < 0:
                raise ArithmeticError(f"invalid Welford variance at prefix {n}: {var}")
            found[n] = (mean, math.sqrt(var))
    if found.keys() != wanted:
        raise AssertionError(f"missing requested training prefix stops: {sorted(wanted-found.keys())[:5]}")
    return found


def main() -> None:
    sys.path.insert(0, str(O / "asset_level_v6"))
    from inputs import AssetInputs

    plan = json.loads(PLAN.read_text())
    panel = json.loads(PANEL.read_text())
    catalogue = json.loads(CATALOGUE.read_text())
    mapping = np.load(MAP, allow_pickle=False)
    cases = plan["cases"]
    rows = panel["series"]
    by_id = {r["series_id"]: r for r in rows}
    case_lists: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    input_obj = AssetInputs(INPUTS)

    if plan["fingerprint"] != "c9bd17eeb30bc274aa48cf2974dd2246af14e3399696ad13527c3ca4db140291":
        raise AssertionError("unexpected frozen plan fingerprint")
    if panel["panel_fingerprint"] != plan["panel_fingerprint"]:
        raise AssertionError("plan/panel fingerprint mismatch")
    if mapping["original"].shape != (len(cases), 8):
        raise AssertionError("original request mapping shape changed")
    if not np.array_equal(mapping["original"][:, 0], np.arange(len(cases))):
        raise AssertionError("original request mapping case order changed")
    if not np.array_equal(mapping["original"][:, 1], np.array([c["series_index"] for c in cases])):
        raise AssertionError("original series indices differ from plan")
    if not np.array_equal(mapping["original"][:, 2], np.array([c["origin_index"] for c in cases])):
        raise AssertionError("original origin indices differ from plan")
    if not np.array_equal(mapping["original"][:, 3], np.array([c["origin"]["position"] for c in cases])):
        raise AssertionError("original training positions differ from plan")
    if not np.array_equal(mapping["original"][:, 4], np.array([c["origin"]["max_horizon"] for c in cases])):
        raise AssertionError("original native horizons differ from plan")

    # The parent training reader is the authoritative accounting implementation.
    input_path = O / "asset_level_v6/inputs.py"
    if file_sha(input_path) != plan["source_hashes"][str(input_path)]:
        raise AssertionError("training input reader source hash differs from frozen plan")

    accepted = {r["series_id"]: r for r in catalogue["accepted"]}
    portfolio_rows = [r for r in rows if r["panel_branch"] == "portfolios"]
    if len(portfolio_rows) != len(accepted):
        raise AssertionError("portfolio panel/catalogue cardinality mismatch")
    for r in portfolio_rows:
        c = accepted.get(r["series_id"])
        if not c or c["definition_sha256"] != r["definition_sha256"] or c["rational_weights"] != r["rational_weights"]:
            raise AssertionError(f"portfolio catalogue mismatch: {r['series_id']}")

    origin_lists = {}
    date_arrays = {}
    input_manifest = []
    input_bytes = 0
    for i, case in enumerate(cases):
        sid = case["series_id"]
        si = case["series_index"]
        if si >= len(rows) or rows[si]["series_id"] != sid:
            raise AssertionError(f"series index mismatch at case {i}")
        row = rows[si]
        oi = case["origin_index"]
        if sid not in date_arrays:
            date_arrays[sid] = np.load(INPUTS / row["dates_file"], mmap_mode="r", allow_pickle=False)
        if sid not in origin_lists:
            origin_lists[sid] = json.loads((INPUTS / row["origins_file"]).read_text())
        origins = origin_lists[sid]
        origin = case["origin"]
        if oi >= len(origins) or origins[oi] != origin:
            raise AssertionError(f"origin metadata mismatch at case {i}")
        pos = int(origin["position"])
        h = int(origin["max_horizon"])
        dates = date_arrays[sid]
        if pos < 1 or pos + h >= len(dates):
            raise AssertionError(f"invalid training/original-H bounds at case {i}")
        if np.datetime_as_string(dates[pos], unit="D") != origin["origin_date"]:
            raise AssertionError(f"origin date mismatch at case {i}")
        end_day = int(dates[pos + h].astype("datetime64[D]").astype(np.int64))
        origin_day = int(dates[pos].astype("datetime64[D]").astype(np.int64))
        mapped = mapping["original"][i]
        if origin_day != int(mapped[5]) or end_day != int(mapped[6]):
            raise AssertionError(f"original mapping dates mismatch at case {i}")
        case_lists[sid].append((i, case))

    short_cases = mapping["short_group_case_indices"].astype(np.int64)
    short_h = mapping["short_horizons"].astype(np.int64)
    if not np.array_equal(short_h, np.array([1, 5, 10, 21])):
        raise AssertionError("short requested horizons differ from frozen mapping")
    if len(np.unique(short_cases)) != len(short_cases):
        raise AssertionError("duplicate short-group case index")
    if not np.array_equal(short_cases, np.asarray(plan["group_indices"], dtype=np.int64)):
        raise AssertionError("short-group indices differ from frozen plan")
    if mapping["short_requested_end_days"].shape != (len(short_cases), len(short_h)):
        raise AssertionError("short requested date mapping shape changed")

    # Verify the actual date and return files referenced by every series row.
    for row in rows:
        for rel in (row["dates_file"], row["returns_file"], row["origins_file"]):
            p = INPUTS / rel
            expected = row["files"][rel]
            actual = file_sha(p)
            if actual != expected:
                raise AssertionError(f"input file hash mismatch: {rel}")
            input_manifest.append((rel, actual))
            input_bytes += p.stat().st_size

    prefix_rows = {}
    portfolio_reconstruction = []
    welford_total_maxstop = 0
    for row in rows:
        sid = row["series_id"]
        these = case_lists[sid]
        if not these:
            raise AssertionError(f"planned panel series has no origin cases: {sid}")
        stops_by_case = {i: int(c["origin"]["position"]) + 1 for i, c in these}
        max_stop = max(stops_by_case.values())
        matrix, weights, ids = input_obj.training(row, max_stop)
        if matrix.shape[0] != max_stop or not np.all(np.isfinite(matrix)):
            raise AssertionError(f"invalid training matrix: {sid}")
        if row["panel_branch"] == "assets":
            if matrix.shape[1] != 1 or ids != (sid,):
                raise AssertionError(f"asset input identity mismatch: {sid}")
            r = np.asarray(matrix[:, 0], dtype=np.float64)
        else:
            fractions = [Fraction(n, d) for _, n, d in row["rational_weights"]]
            exact_weights = np.asarray([float(x) for x in fractions], dtype=np.float64)
            if tuple(ids) != tuple(s for s, _, _ in row["rational_weights"]):
                raise AssertionError(f"portfolio constituent ordering mismatch: {sid}")
            if not np.array_equal(weights, exact_weights) or matrix.shape[1] != weights.size:
                raise AssertionError(f"portfolio rational weights mismatch: {sid}")
            if sum(fractions, Fraction(0, 1)) != Fraction(1, 1):
                raise AssertionError(f"portfolio weights do not sum exactly to one: {sid}")
            r = np.log1p(np.sum(np.expm1(matrix) * weights[None, :], axis=1))
            target = np.load(INPUTS / row["returns_file"], mmap_mode="r", allow_pickle=False)[:max_stop]
            err = np.abs(r - target)
            maxerr = float(np.max(err))
            if maxerr > 2e-14:
                raise AssertionError(f"portfolio return reconstruction outside frozen tolerance: {sid}")
            portfolio_reconstruction.append({"series_id": sid, "rows": max_stop, "max_abs_error": maxerr})
        if not np.all(np.isfinite(r)):
            raise AssertionError(f"nonfinite portfolio/asset training returns: {sid}")
        stop_stats = prefix_sd_by_stop(r, list(stops_by_case.values()))
        welford_total_maxstop += max_stop
        for case_i, stop in stops_by_case.items():
            mean, sd = stop_stats[stop]
            prefix_rows[case_i] = {
                "series_id": sid,
                "origin_index": cases[case_i]["origin_index"],
                "origin_date": cases[case_i]["origin"]["origin_date"],
                "training_n": stop,
                "sd": sd,
                "prefix_mean": mean,
                "exact_constant": bool(np.all(r[:stop] == r[0])),
            }

    if len(prefix_rows) != len(cases):
        raise AssertionError("not all planned training prefixes were measured")

    short_dates = mapping["short_requested_end_days"]
    for gi, case_i in enumerate(short_cases):
        case = cases[int(case_i)]
        dates = date_arrays[case["series_id"]]
        pos = int(case["origin"]["position"])
        for hj, h in enumerate(short_h):
            if pos + int(h) >= len(dates) or int(dates[pos + int(h)].astype("datetime64[D]").astype(np.int64)) != int(short_dates[gi, hj]):
                raise AssertionError(f"short request date/H binding mismatch: case {case_i}, H {h}")
            if int(h) > int(case["origin"]["max_horizon"]):
                raise AssertionError(f"short H exceeds planned native horizon: case {case_i}, H {h}")

    original_scales, short_scales = [], []
    per_h = {int(h): [] for h in short_h}
    for i, case in enumerate(cases):
        row = prefix_rows[i]
        h = int(case["origin"]["max_horizon"])
        scale = row["sd"] * math.sqrt(h)
        original_scales.append(scale)
    for case_i in short_cases:
        sd = prefix_rows[int(case_i)]["sd"]
        for h in short_h:
            scale = sd * math.sqrt(int(h))
            short_scales.append(scale)
            per_h[int(h)].append(scale)

    zeros = [
        {"case_index": i, **r}
        for i, r in prefix_rows.items() if r["sd"] == 0
    ]
    constants = [i for i, r in prefix_rows.items() if r["exact_constant"]]
    if len(zeros) != len(constants):
        raise AssertionError("zero Welford SD and exact-constant prefix sets differ")
    smallest = sorted(
        ({"case_index": i, **r} for i, r in prefix_rows.items() if r["sd"] > 0),
        key=lambda x: (x["sd"], x["series_id"], x["origin_index"]),
    )[:10]
    manifest_blob = "\n".join(f"{path}\t{digest}" for path, digest in sorted(input_manifest)).encode()
    input_manifest_sha = hashlib.sha256(manifest_blob).hexdigest()
    origin_rows = sum(len(v) for v in case_lists.values())
    if origin_rows != 64049 or len(short_cases) != 63992:
        raise AssertionError("unexpected frozen request inventory")
    request_count = origin_rows + len(short_cases) * len(short_h)
    if request_count != 320017:
        raise AssertionError("unexpected full planned request count")

    report = {
        "schema": "training-standardized-requested-horizon-domain-scan-v1",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "score_blind": True,
        "comparative_score_or_loss_files_read": False,
        "fits_or_forecasts_run": False,
        "primary_formula": "portfolio_r_t=log1p(sum_a w_a*expm1(x_ta)); sigma=sqrt(sum_t(r_t-mean(r))^2/(n-1)); scale=sigma*sqrt(requested_native_H); training rows include the origin observation",
        "variance_algorithm": "one-pass Welford sample variance; no epsilon/floor; exact-constant prefix independently checked",
        "plan": {"path": str(PLAN), "sha256": file_sha(PLAN), "fingerprint": plan["fingerprint"], "case_digest": plan["case_digest"], "cases": len(cases), "source_protocol": plan["score_protocol"]},
        "design_review": {"path": str(DESIGN), "sha256": file_sha(DESIGN)},
        "panel": {"path": str(PANEL), "sha256": file_sha(PANEL), "fingerprint": panel["panel_fingerprint"], "series": panel["series_count"], "asset_series": panel["asset_series_preserved"], "portfolio_series": panel["portfolio_series_added"]},
        "portfolio_catalogue": {"path": str(CATALOGUE), "sha256": file_sha(CATALOGUE), "accepted": len(accepted), "all_portfolio_definitions_and_rational_weights_match_panel": True},
        "requested_mapping": {"path": str(MAP), "sha256": file_sha(MAP), "original_cases": int(mapping["original"].shape[0]), "short_groups": len(short_cases), "short_horizons": [int(x) for x in short_h], "short_requests": len(short_cases) * len(short_h), "planned_requests_total": request_count, "all_origin_and_end_dates_match_panel": True},
        "training_input_reader": {"path": str(input_path), "sha256": file_sha(input_path), "matches_frozen_plan_source_hash": True},
        "verified_inputs": {"file_count": len(input_manifest), "bytes": input_bytes, "sorted_path_sha256_manifest": input_manifest_sha, "all_dates_returns_and_origin_files_match_panel_manifest": True},
        "prefix_domain": {"unique_prefixes": len(prefix_rows), "training_rows_min": min(r["training_n"] for r in prefix_rows.values()), "training_rows_max": max(r["training_n"] for r in prefix_rows.values()), "series_with_planned_prefixes": len(case_lists), "prefix_sd": summary([r["sd"] for r in prefix_rows.values()]), "zero_sd_prefix_count": len(zeros), "exact_constant_prefix_count": len(constants), "zero_sd_prefixes": zeros, "ten_smallest_positive_sd_prefixes": smallest, "welford_total_observations_scanned": welford_total_maxstop},
        "request_scale_domain": {"original_requests": summary(original_scales), "short_requests": summary(short_scales), "short_by_native_H": {str(h): summary(v) for h, v in per_h.items()}, "all_requested_scales_positive_finite": all(math.isfinite(x) and x > 0 for x in original_scales + short_scales)},
        "portfolio_accounting_check": {"method": "AssetInputs.training from exact frozen source; rational constituent interval returns and log1p(weighted expm1) aggregate; compared against precomputed panel portfolio target with the reader's absolute tolerance 2e-14", "series_checked": len(portfolio_reconstruction), "training_rows_reconstructed": sum(x["rows"] for x in portfolio_reconstruction), "max_absolute_difference": max((x["max_abs_error"] for x in portfolio_reconstruction), default=0.0), "all_within_2e-14": True, "per_series_error_summary": summary([x["max_abs_error"] for x in portfolio_reconstruction])},
        "implementation_scope": {
            "reuse": "Compute one sigma per immutable (series_id, origin position) from only that origin's panel training prefix; use the same sigma for every model and requested H at that prefix.",
            "request_loss": "Use already-saved per-request raw mean daily CRPS C_mq and divide by sigma_q*sqrt(H); do not use V7 per-lead divisors, candidate predictive SD, future outcomes, or baseline-loss division.",
            "aggregation": "Feed standardized request losses to unchanged V9 fixed within-bucket hierarchy; apply equal one-third across the existing three calendar buckets exactly once; retain the same matched support and WR baseline.",
            "unresolved_domain_rule": "If any sigma is exactly zero, the proposed score is undefined for those planned rows. Do not score, exclude, floor, or epsilon-adjust until a unit/reference convention is preregistered for zero-dispersion prefixes.",
            "scope_limit": "This scan establishes training-scale feasibility only. It computes no comparative model score, CRPS, or candidate ordering and does not activate V10 or any new canonical pointer."
        }
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "status": report["status"], "prefixes": len(prefix_rows), "requests": request_count, "zero_sd": len(zeros), "min_sd": report["prefix_domain"]["prefix_sd"]["min"], "max_sd": report["prefix_domain"]["prefix_sd"]["max"], "all_scales_positive_finite": report["request_scale_domain"]["all_requested_scales_positive_finite"]}, indent=2))


if __name__ == "__main__":
    main()
