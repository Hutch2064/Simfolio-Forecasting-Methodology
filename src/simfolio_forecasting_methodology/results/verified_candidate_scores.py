"""Rank one verified new run beside retained score evidence without merging it."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from importlib.resources import files
from typing import Any

from ..candidate_registry import EVIDENCE_RESOURCE, load_candidate_evidence
from ..catalogue import load_canonical_ledger

_RESOURCE_ROOT = EVIDENCE_RESOURCE.rsplit("/", 1)[0]


def _verified_bytes(name: str, evidence: dict[str, Any]) -> bytes:
    resource = files("simfolio_forecasting_methodology").joinpath(
        f"{_RESOURCE_ROOT}/{name}"
    )
    value = resource.read_bytes()
    expected = evidence["packaged_evidence"].get(name)
    observed = hashlib.sha256(value).hexdigest()
    if expected != observed:
        raise ValueError(f"verified candidate evidence digest mismatch: {name}")
    return value


def verified_candidate_score_report() -> dict[str, Any]:
    """Return the expanded canonical score ranking without duplicate candidates."""

    evidence = load_candidate_evidence()
    audit = json.loads(
        _verified_bytes("independent_final_audit.json", evidence).decode("utf-8")
    )
    _verified_bytes("paired_cell_losses.npz", evidence)
    tasks = json.loads(_verified_bytes("candidate_tasks.json", evidence).decode("utf-8"))
    replay = json.loads(
        _verified_bytes("full_replay_receipt.json", evidence).decode("utf-8")
    )
    candidate = evidence["published_candidate"]
    run = evidence["paired_run"]
    ledger = load_canonical_ledger()

    if (
        audit.get("passed") is not True
        or audit.get("scope") != "full_canonical_paired"
        or audit.get("manifest_sha256") != evidence["source_run"]["original_manifest_sha256"]
        or audit.get("results_sha256") != evidence["source_run"]["original_results_sha256"]
        or audit.get("paired_cell_losses_sha256")
        != evidence["source_run"]["paired_cell_losses_sha256"]
        or len(tasks.get("tasks", ())) != run["tasks_per_model"]
        or tasks.get("model_id") != candidate["model_id"]
        or replay.get("schema_version") != "verified_candidate_replay_receipt_v1"
        or replay.get("candidate_model_id") != candidate["model_id"]
        or replay.get("tasks") != run["tasks_per_model"]
        or replay.get("cells_per_model") != run["cells_per_model"]
        or replay.get("simulations_per_origin") != run["simulations"]
        or replay.get("every_loss_vector_byte_equal") is not True
        or replay.get("exact_empirical_crps") != candidate["exact_empirical_crps"]
    ):
        raise ValueError("verified candidate audit/task evidence is inconsistent")
    for model in run["model_rankings"]:
        audited = audit["scores"].get(model["model_id"])
        if (
            audited is None
            or repr(float(audited)) != model["exact_empirical_crps"]
            or audit["tasks_per_model"] != model["task_count"]
            or audit["cells_per_model"] != model["cell_count"]
        ):
            raise ValueError("verified candidate score does not match its audit")

    rows: list[dict[str, Any]] = []
    for model in ledger["models"]:
        rows.append(
            {
                "public_model_id": model["public_model_id"],
                "display_name": model["display_name"],
                "exact_empirical_crps": model["historical_score"]["exact_empirical_crps"],
                "canonical_rank": model["canonical_rank"],
                "historical_rank": model["historical_rank"],
                "display_rank": None,
                "score_evidence_origin": (
                    "retained_historical_evidence" if model["historical_rank"] is not None
                    else ("unscored_canonical_candidate" if model["verification_status"].startswith('unscored_canonical_candidate_')
                          else "partial_canonical_execution" if model["historical_score"]["exact_empirical_crps"] is None
                          else "validated_full_canonical_execution")
                ),
                "is_new_execution": False,
                "historical_score_verified": model["historical_score_verified"],
                "verification_status": model["verification_status"],
            }
        )
    rows.sort(
        key=lambda row: (
            Decimal(row["exact_empirical_crps"]) if row["exact_empirical_crps"] is not None
            else Decimal("Infinity"),
            row["canonical_rank"] is None,
            row["canonical_rank"] or 0,
            row["public_model_id"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["display_rank"] = rank if row["exact_empirical_crps"] is not None else None

    paired_rows = [
        {
            "paired_run_rank": model["paired_run_rank"],
            "public_model_id": model["model_id"],
            "exact_empirical_crps": model["exact_empirical_crps"],
            "task_count": model["task_count"],
            "cell_count": model["cell_count"],
            "evidence_origin": model["evidence_origin"],
            "is_new_execution_in_candidate_run": (
                model["evidence_origin"] == "newly_executed_in_filtered_candidate_run"
            ),
        }
        for model in run["model_rankings"]
    ]
    return {
        "result_kind": "combined_ranked_score_evidence",
        "ranking_rule": "one-based numeric sort position by exact empirical CRPS across mixed evidence origins",
        "row_count": len(rows),
        "retained_historical_row_count": 175,
        "validated_full_panel_row_count": sum(m["historical_score_verified"] for m in ledger["models"]),
        "partial_panel_row_count": sum(m["verification_status"].startswith('partial_canonical_score_') for m in ledger["models"]),
        "unscored_candidate_row_count": sum(m["verification_status"].startswith('unscored_canonical_candidate_') for m in ledger["models"]),
        "independently_audited_new_execution_row_count": 0,
        "canonical_membership_digest": ledger["membership"]["membership_digest"],
        "canonical_membership_unchanged": True,
        "full_176_model_reproduction": False,
        "selection_and_inference": evidence["selection_and_inference"],
        "full_candidate_replay": {
            "scope": replay["scope"],
            "tasks": replay["tasks"],
            "cells_per_model": replay["cells_per_model"],
            "simulations_per_origin": replay["simulations_per_origin"],
            "every_loss_vector_byte_equal": replay["every_loss_vector_byte_equal"],
            "exact_empirical_crps": replay["exact_empirical_crps"],
            "elapsed_seconds": replay["elapsed_seconds"],
            "execution_context": replay["execution_context"],
            "replay_driver": replay["replay_driver"],
            "validated_numerical_kernels": replay["validated_numerical_kernels"],
            "cross_runtime_marginal_parity": replay["cross_runtime_marginal_parity"],
        },
        "retained_historical_score_linkage": ledger["score_evidence"][
            "retained_score_artifact"
        ]["digest_status"],
        "source_run": evidence["source_run"],
        "paired_run_models": paired_rows,
        "rows": rows,
    }


__all__ = ["verified_candidate_score_report"]
