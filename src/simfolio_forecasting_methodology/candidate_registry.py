"""Fail-closed registry for separately verified non-ledger candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Callable

from .catalogue import load_canonical_ledger
from .models.asset_level.filtered_innovation_moment_sv import (
    FILTERED_INNOVATION_FIXED_MEAN_MODEL_ID,
    FilteredInnovationFixedMeanMomentSV,
)

EVIDENCE_RESOURCE = (
    "resources/verified_candidates/filtered_innovation_fixed_mean/score_evidence.json"
)


@dataclass(frozen=True)
class CandidateRegistration:
    model_id: str
    evidence_status: str
    factory: Callable[[], Any]


def candidate_ids() -> tuple[str, ...]:
    return (FILTERED_INNOVATION_FIXED_MEAN_MODEL_ID,)


def load_candidate_evidence() -> dict[str, Any]:
    path = files("simfolio_forecasting_methodology").joinpath(EVIDENCE_RESOURCE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "verified_candidate_score_evidence_v1":
        raise ValueError("verified candidate evidence schema is unsupported")
    if payload.get("published_candidate", {}).get("model_id") != FILTERED_INNOVATION_FIXED_MEAN_MODEL_ID:
        raise ValueError("verified candidate evidence has an unexpected model ID")
    return payload


def candidate_registration(model_id: str) -> CandidateRegistration:
    if model_id != FILTERED_INNOVATION_FIXED_MEAN_MODEL_ID:
        raise ValueError(f"unknown verified candidate ID: {model_id}")
    evidence = load_candidate_evidence()
    if evidence["published_candidate"].get("is_new_execution") is not True:
        raise ValueError("candidate is not backed by newly executed score evidence")
    return CandidateRegistration(
        model_id=model_id,
        evidence_status="independently_audited_full_canonical_paired_execution",
        factory=FilteredInnovationFixedMeanMomentSV,
    )


def build_candidate_model(model_id: str):
    return candidate_registration(model_id).factory()


def candidate_execution_record(model_id: str) -> dict[str, Any]:
    """Return source/data identity for a new run without changing the 175 ledger."""

    registration = candidate_registration(model_id)
    evidence = load_candidate_evidence()
    ledger = load_canonical_ledger()
    identity = ledger["identity_policy"]
    model = evidence["published_candidate"]
    revision = evidence["source_run"]["research_repository_revision"]
    membership_payload = {
        "scope": "verified_candidate_only",
        "model_ids": [model_id],
        "source_revision": revision,
    }
    membership_digest = hashlib.sha256(
        json.dumps(membership_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_hashes = evidence["source_run"]["candidate_source_artifacts_sha256"]
    source_digest = hashlib.sha256(
        "".join(sorted(source_hashes.values())).encode("ascii")
    ).hexdigest()
    specification_payload = {
        "model_id": model_id,
        "filtered_innovations": "eps_x * exp(-0.5 * state_path)",
        "mean": "fixed_historical_mean",
        "predictive_distribution": "Gaussian_moment_matched",
        "copula_seed": [
            "copula_alternatives",
            "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced",
            "origin_date",
            "horizon",
            "simulations",
        ],
    }
    specification_fingerprint = hashlib.sha256(
        json.dumps(specification_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "public_model_id": model_id,
        "display_name": model["display_name"],
        "experiment_id": "filtered-innovation-fixed-mean-full-canonical-2026-10-02",
        "source_revision": revision,
        "source_artifact_digest": {"source_code": {"sha256": source_digest}},
        "specification_fingerprint": {"value": specification_fingerprint},
        "required_dependencies": evidence["source_run"]["environment"],
        "protocol_fingerprint": identity["protocol_fingerprint"],
        "dataset_fingerprint": identity["dataset_fingerprint"],
        "panel_fingerprint": identity["panel_fingerprint"],
        "membership_digest": membership_digest,
        "verification_status": registration.evidence_status,
    }


__all__ = [
    "CandidateRegistration",
    "build_candidate_model",
    "candidate_execution_record",
    "candidate_ids",
    "candidate_registration",
    "load_candidate_evidence",
]
