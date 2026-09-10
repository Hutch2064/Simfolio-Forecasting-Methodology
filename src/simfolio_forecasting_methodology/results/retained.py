"""Inspection of retained historical scores without creating run results."""

from __future__ import annotations

from typing import Any

from ..catalogue import load_canonical_ledger


def retained_score_report(model_id: str | None = None) -> dict[str, Any]:
    """Return lexical retained score evidence with its unresolved digest state."""

    ledger = load_canonical_ledger()
    if model_id is not None:
        rows = [
            row for row in ledger["models"] if row["public_model_id"] == model_id
        ]
        if not rows:
            raise ValueError(f"unknown canonical model ID: {model_id}")
    else:
        rows = list(ledger["models"])
    artifact = ledger["score_evidence"]["retained_score_artifact"]
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        output_rows.append(
            {
                "public_model_id": row["public_model_id"],
                "display_name": row["display_name"],
                "canonical_rank": row["canonical_rank"],
                "historical_rank": row["historical_rank"],
                "retained_score_token": row["historical_score"]["exact_empirical_crps"],
                "publication_score_token": row["score_precision"]["publication_token"],
                "score_precision": row["score_precision"],
                "historical_score_verified": row["historical_score_verified"],
                "verification_status": row["verification_status"],
            }
        )
    return {
        "result_kind": "retained_score_evidence",
        "is_new_execution": False,
        "reproduced": False,
        "experiment_id": ledger["score_evidence"]["experiment_id"],
        "metric": ledger["score_evidence"]["metric"],
        "aggregation": ledger["score_evidence"]["aggregation"],
        "retained_score_artifact": {
            "relative_path": artifact["relative_path"],
            "observed_sha256": artifact["observed_sha256"],
            "declared_sha256": artifact["declared_sha256"],
            "digest_status": artifact["digest_status"],
            "rows": artifact["rows"],
        },
        "rows": output_rows,
    }


__all__ = ["retained_score_report"]
