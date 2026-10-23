"""V11 request-level training-SD/sqrt(requested-H) CRPS rescorer.

This module transforms already-computed raw mean daily CRPS values and delegates
the fixed hierarchy to the unchanged V9 scorer. It does not create forecasts.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import sys
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_PATH = HERE / "protocol.json"
PROTOCOL_ID = "daily_portfolio_panel_crps_v11_train_sd_sqrt_requested_h_equal_days_fixed_calendar_buckets"
PROTOCOL_SHA256 = "0472072ef517368d73bd581f11038de9f834e49efb39a5d0937bfe936ac27760"
V9_PATH = ROOT / "score_series_buckets_v9.py"
V9_SHA256 = "c598301b977fb1d8405de640476289d071107633e8ba72fbedf3ef8712d4796c"
V9_PROTOCOL_ID = "daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import score_series_buckets_v9 as _v9  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_frozen_sources() -> None:
    """Refuse to run if the frozen definition or inherited hierarchy changed."""
    if _v9.SCORE_PROTOCOL_ID != V9_PROTOCOL_ID:
        raise RuntimeError("The inherited V9 hierarchy protocol changed")
    if sha256_file(V9_PATH) != V9_SHA256:
        raise RuntimeError("The inherited V9 hierarchy scorer changed")
    if sha256_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("The frozen V11 definition protocol changed")
    with PROTOCOL_PATH.open() as stream:
        if json.load(stream).get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("The V11 protocol identifier changed")


def standardized_request_crps(
    raw_mean_crps: float,
    training_sd: float,
    requested_native_h: int,
) -> float:
    """Return C/(sample-SD(training portfolio)*sqrt(native H)), fail closed."""
    if isinstance(requested_native_h, bool) or not isinstance(requested_native_h, Integral):
        raise TypeError("requested_native_h must be a positive integer")
    h = int(requested_native_h)
    if h <= 0:
        raise ValueError("requested_native_h must be positive")
    raw = float(raw_mean_crps)
    sd = float(training_sd)
    if not math.isfinite(raw) or raw < 0:
        raise ValueError("raw mean CRPS must be finite and nonnegative")
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError("training sample SD must be positive and finite")
    scale = sd * math.sqrt(h)
    if not math.isfinite(scale) or scale <= 0:
        raise ArithmeticError("training SD times sqrt(H) is not positive and finite")
    result = raw / scale
    if not math.isfinite(result) or result < 0:
        raise ArithmeticError("standardized request CRPS is not finite and nonnegative")
    return result


def _scale_entry(scales_by_request: Mapping[str, Mapping], row: Mapping) -> tuple[float, int]:
    request_id = str(row.get("request_id", row.get("task_key", "")))
    if not request_id:
        raise ValueError("Each request requires its frozen V9 request_id")
    try:
        entry = scales_by_request[request_id]
    except KeyError as exc:
        raise KeyError(f"No frozen training scale for request {request_id}") from exc
    h = entry.get("native_h")
    sd = entry.get("training_sd")
    request_h = row.get("requested_native_h")
    if isinstance(request_h, bool) or not isinstance(request_h, Integral):
        raise TypeError(f"Observed request requires an explicit integer requested_native_h: {request_id}")
    if isinstance(h, bool) or not isinstance(h, Integral):
        raise TypeError(f"Invalid native_h in scale index for {request_id}")
    if int(h) != int(request_h):
        raise ValueError(f"Request H differs from frozen scale index for {request_id}")
    if "H" in row:
        row_h = row["H"]
        if isinstance(row_h, bool) or not isinstance(row_h, Integral) or int(row_h) != int(h):
            raise ValueError(f"Observed H field differs from frozen scale index for {request_id}")
    sid = row.get("series_id")
    if sid is not None and str(sid) != str(entry.get("series_id")):
        raise ValueError(f"Series identity differs from frozen scale index for {request_id}")
    if sid is None or entry.get("series_id") is None:
        raise ValueError(f"Request and scale index must bind series identity for {request_id}")
    if row.get("origin_date") is None or entry.get("origin_date") is None or str(row["origin_date"])[:10] != str(entry["origin_date"])[:10]:
        raise ValueError(f"Origin date differs from frozen scale index for {request_id}")
    if row.get("requested_end_date") is None or entry.get("requested_end_date") is None or str(row["requested_end_date"])[:10] != str(entry["requested_end_date"])[:10]:
        raise ValueError(f"Requested end date differs from frozen scale index for {request_id}")
    for identity_field in ("case_index", "origin_index"):
        if identity_field not in row or identity_field not in entry:
            raise ValueError(f"Request and scale index must bind {identity_field} for {request_id}")
        row_identity = row[identity_field]
        index_identity = entry[identity_field]
        if (isinstance(row_identity, bool) or not isinstance(row_identity, Integral) or
                isinstance(index_identity, bool) or not isinstance(index_identity, Integral)):
            raise TypeError(f"{identity_field} must be an integer for {request_id}")
        if int(row_identity) != int(index_identity):
            raise ValueError(f"{identity_field} differs from frozen scale index for {request_id}")
    return float(sd), int(h)


def aggregate_records(
    records: Sequence[Mapping],
    metadata: Mapping[str, Mapping],
    *,
    weights: Mapping,
    scales_by_request: Mapping[str, Mapping],
    score_key: str = "raw_mean_crps",
) -> dict:
    """Transform raw request means, then use unchanged V9 fixed planned weights.

    Partial aggregations retain V9's fixed coefficients and support accounting;
    a final score is present only when the supplied frozen support is complete.
    """
    assert_frozen_sources()
    if weights.get("score_protocol_id") != V9_PROTOCOL_ID:
        raise ValueError("A frozen V9 planned weight map is required")
    transformed = []
    seen = set()
    for source in records:
        row = dict(source)
        request_id = str(row.get("request_id", row.get("task_key", "")))
        if not request_id:
            raise ValueError("Each request requires its frozen V9 request_id")
        if request_id in seen:
            raise ValueError("Duplicate observed request identity: " + request_id)
        seen.add(request_id)
        sd, h = _scale_entry(scales_by_request, row)
        raw = row.pop(score_key)
        row["score"] = standardized_request_crps(raw, sd, h)
        row["requested_native_h"] = h
        transformed.append(row)
    out = _v9.aggregate_records(transformed, metadata, weights=weights, score_key="score")
    out["hierarchy_protocol_id"] = V9_PROTOCOL_ID
    out["score_protocol_id"] = PROTOCOL_ID
    out["normalization"] = "raw mean daily CRPS / (training portfolio sample SD ddof=1 * sqrt(requested native H))"
    out["observed_request_count"] = len(transformed)
    out["planned_request_count"] = int(weights["planned_request_count"])
    commitment_hash = hashlib.sha256()
    commitment_hash.update(b"[")
    for index, request_id in enumerate(sorted(seen)):
        entry = scales_by_request[request_id]
        commitment = {
            "request_id": request_id,
            "series_id": str(entry["series_id"]),
            "origin_date": str(entry["origin_date"])[:10],
            "requested_end_date": str(entry["requested_end_date"])[:10],
            "native_h": int(entry["native_h"]),
            "training_sd": float(entry["training_sd"]),
        }
        for field in ("case_index", "origin_index"):
            commitment[field] = int(entry[field])
        if index:
            commitment_hash.update(b",")
        commitment_hash.update(json.dumps(commitment, sort_keys=True, separators=(",", ":")).encode())
    commitment_hash.update(b"]")
    out["scale_data_commitment_sha256"] = commitment_hash.hexdigest()
    out["complete"] = bool(out["complete"] and len(seen) == int(weights["planned_request_count"]))
    if not out["complete"]:
        out["score"] = None
    return out


def compare(algorithm: Mapping, baseline: Mapping) -> dict:
    """Compare complete, exactly matched V11 Q values; return ratio only if defined."""
    if algorithm.get("score_protocol_id") != PROTOCOL_ID or baseline.get("score_protocol_id") != PROTOCOL_ID:
        raise ValueError("Both aggregates must use the frozen V11 protocol")
    if algorithm.get("hierarchy_protocol_id") != V9_PROTOCOL_ID or baseline.get("hierarchy_protocol_id") != V9_PROTOCOL_ID:
        raise ValueError("Both aggregates must inherit the V9 hierarchy")
    if algorithm.get("weight_plan_sha256") != baseline.get("weight_plan_sha256"):
        raise ValueError("Identical frozen planned weights are required")
    if algorithm.get("scale_data_commitment_sha256") != baseline.get("scale_data_commitment_sha256"):
        raise ValueError("Identical request training scales and source identities are required")
    if algorithm.get("observed_request_ids") != baseline.get("observed_request_ids"):
        raise ValueError("Identical matched request identities are required")
    if not algorithm.get("complete") or not baseline.get("complete"):
        raise ValueError("Final comparison requires complete matched planned coverage")
    q_model = float(algorithm["score"])
    q_baseline = float(baseline["score"])
    if not math.isfinite(q_model) or q_model < 0 or not math.isfinite(q_baseline) or q_baseline < 0:
        raise ValueError("Aggregated Q values must be finite and nonnegative")
    gain = 100.0 * (1.0 - q_model / q_baseline) if q_baseline > 0 else None
    return {
        "score_protocol_id": PROTOCOL_ID,
        "hierarchy_protocol_id": V9_PROTOCOL_ID,
        "algorithm_Q": q_model,
        "baseline_Q": q_baseline,
        "percent_better_than_baseline": gain,
        "relative_gain_defined": gain is not None,
        "undefined_reason": None if gain is not None else "Aggregated baseline Q is exactly zero; no epsilon or substitute denominator",
        "weight_plan_sha256": algorithm["weight_plan_sha256"],
        "matched_request_count": algorithm["observed_request_count"],
        "complete": True,
    }
