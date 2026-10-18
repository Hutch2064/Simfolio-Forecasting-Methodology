"""Separate completed single52 evidence, keyed by immutable catalogue identity."""

from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from pathlib import Path

from .catalogue import _assert_public_paths, load_canonical_models
from .safety_scan import scan_text

INDEX_RESOURCE = "resources/results/single52.json"
PROTOCOL = "catalogue52-terminal-log-empirical-crps-240-v1"
DENOMINATORS = {"portfolios": 52, "tasks": 2652, "cells": 455832, "simulations": 240}
CONTROL_ID = "asset_m256_data_timescales_existing_sv_rate"


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _score(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def validate_single52_index(payload, *, catalogue=None):
    """Reject incomplete scores, evidence gaps, and changed catalogue identities."""
    if (payload.get("schema_version") != 1 or payload.get("protocol") != PROTOCOL
            or payload.get("ranking_metric") != "single52_crps_only_lower_is_better"
            or payload.get("blended_score_used") is not False
            or payload.get("partial_results_ranked") is not False):
        raise ValueError("invalid separate single52 index protocol")
    if not _hash(payload.get("source_report_sha256")) or not _hash(payload.get("inventory_sha256")):
        raise ValueError("single52 index requires report and inventory hashes")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise TypeError("single52 index must be keyed by model ID")
    known = {row["public_model_id"]: row for row in
             (load_canonical_models() if catalogue is None else catalogue)}
    for model_id, row in models.items():
        record = known.get(model_id)
        if (record is None or row.get("model_id") != model_id
                or row.get("canonical_rank") != record["canonical_rank"]
                or row.get("specification_fingerprint") != record["specification_fingerprint"]):
            raise ValueError("single52 result does not match immutable catalogue identity")
        if any(type(row.get(key)) is not int or row[key] != value
               for key, value in DENOMINATORS.items()):
            raise ValueError("incomplete single52 result denominators")
        if not _score(row.get("single52_crps")) or not _hash(row.get("cells_sha256")):
            raise ValueError("single52 result requires a finite score and cells hash")
        history = record["historical_score"]["exact_empirical_crps"]
        if row.get("historical80_crps") != (None if history is None else float(history)):
            raise ValueError("single52 result changed the separate historical80 score")
        if type(row.get("historical80_evidence_verified")) is not bool:
            raise ValueError("single52 result requires separate historical80 evidence status")
        if row.get("role") == "validated_saved_production_control":
            if model_id != CONTROL_ID or not _hash(row.get("saved_control_verification_sha256")):
                raise ValueError("unverified saved single52 control")
        else:
            if any(not _hash(row.get(key)) for key in
                   ("manifest_sha256", "source_snapshot_sha256", "result_sha256")):
                raise ValueError("single52 result requires complete execution hashes")
            scores = row.get("portfolio_scores", {})
            if (len(scores) != 52 or any(not key.startswith("single_") for key in scores)
                    or any(not _score(value) for value in scores.values())
                    or not math.isclose(sum(scores.values()) / 52, row["single52_crps"],
                                        rel_tol=1e-12, abs_tol=1e-14)):
                raise ValueError("single52 asset score denominator or mean mismatch")
            raw = row.get("raw_evidence", {})
            if type(raw.get("archived_and_download_verified")) is not bool:
                raise ValueError("single52 result requires explicit raw archive status")
            if (raw["archived_and_download_verified"]
                    and any(not _hash(raw.get(key)) for key in
                            ("archive_sha256", "remote_receipt_sha256"))):
                raise ValueError("verified raw archive requires evidence hashes")
    _assert_public_paths(payload, "single52")
    if scan_text(json.dumps(payload, sort_keys=True)):
        raise ValueError("single52 index contains nonpublic material")


def load_single52_results(path: Path | None = None):
    """Load the independent single52 index without changing retained80 records."""
    source = files(__package__).joinpath(INDEX_RESOURCE) if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_single52_index(payload)
    return payload["models"]
