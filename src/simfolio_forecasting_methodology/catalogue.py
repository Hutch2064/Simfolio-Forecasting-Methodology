"""Authoritative canonical-175 ledger loading and evidence validation.

The ledger owns membership and retained score evidence.  Its per-model
contract is deliberately flat at the top level so later verified source
implementations can advance individual records without changing the public
identity or falling back to a generic model.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from importlib.abc import Traversable
from importlib.resources import files
from pathlib import Path
from typing import Any

LEDGER_RESOURCE = "resources/canonical_175/ledger.json"
EXPECTED_CANONICAL_COUNT = 175
EXPECTED_SOURCE_RANKS = tuple(range(12, 187))
EXPECTED_CELLS_PER_MODEL = 701_280
EXPECTED_MEMBERSHIP_DIGEST = "c93ea4fca270f7343924a11ec6c3574d1693c07592d9f203b650915548b1d9b5"

REQUIRED_MODEL_FIELDS = (
    "public_model_id",
    "display_name",
    "historical_model_ids",
    "seed_identity",
    "model_family",
    "structured_specification",
    "specification_fingerprint",
    "source_revision",
    "source_artifact_digest",
    "source_location_reference",
    "implementation_factory",
    "required_dependencies",
    "historical_rank",
    "historical_score",
    "score_precision",
    "experiment_id",
    "protocol_fingerprint",
    "dataset_fingerprint",
    "panel_fingerprint",
    "identity_recovered",
    "specification_recovered",
    "source_reference_verified",
    "implementation_available",
    "instantiation_validated",
    "forecast_smoke_tested",
    "source_parity_checked",
    "historical_score_verified",
    "verification_status",
    "notes",
)
_ALLOWED_MODEL_FIELDS = set(REQUIRED_MODEL_FIELDS) | {"canonical_rank"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _resource_path() -> Traversable:
    return files("simfolio_forecasting_methodology").joinpath(LEDGER_RESOURCE)


def _ledger_path(root: Path | None) -> Traversable | Path:
    if root is None:
        return _resource_path()
    path = Path(root)
    if path.is_file():
        return path
    candidates = (
        path / "ledger.json",
        path / "canonical_175" / "ledger.json",
        path / "resources" / "canonical_175" / "ledger.json",
        path.parent / "canonical_175" / "ledger.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"canonical ledger not found below {path}; canonical CSV files are not a fallback"
    )


def canonical_membership_digest(models: list[Mapping[str, Any]]) -> str:
    """Hash only canonical rank, historical rank, and immutable public ID."""

    lines = "\n".join(
        f"{model['canonical_rank']}|{model['historical_rank']}|{model['public_model_id']}"
        for model in models
    )
    return hashlib.sha256((lines + "\n").encode("utf-8")).hexdigest()


def _as_decimal(value: Any, field: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"{field} must retain its source numeric token as a string")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not a decimal source token: {value!r}") from exc


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None


def _validate_optional_sha(value: Any, field: str) -> None:
    if value is not None and not _is_sha256(value):
        raise ValueError(f"{field} must be null or a lowercase SHA-256 digest")


def _assert_public_paths(value: Any, field: str = "ledger") -> None:
    """Reject absolute local paths from the public ledger and its references."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_public_paths(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_paths(item, f"{field}[{index}]")
    elif isinstance(value, str) and (
        value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError(f"{field} contains an absolute local path")


def _validate_score(model: Mapping[str, Any], root_score: Mapping[str, Any]) -> None:
    score = model["historical_score"]
    precision = model["score_precision"]
    if not isinstance(score, Mapping) or not isinstance(precision, Mapping):
        raise TypeError(f"{model['public_model_id']}: historical score metadata is malformed")
    retained = score["exact_empirical_crps"]
    retained_decimal = _as_decimal(retained, "historical_score.exact_empirical_crps")
    publication = precision["publication_token"]
    _as_decimal(publication, "score_precision.publication_token")
    if precision["retained_source_token"] != retained:
        raise ValueError(f"{model['public_model_id']}: retained score token was changed")
    if format(float(retained_decimal), ".9g") != publication:
        raise ValueError(
            f"{model['public_model_id']}: publication score is not nine-significant-digit rendering"
        )
    if precision["publication_precision"]["kind"] != (
        "reconciled_9_significant_digits_variable_decimal_places"
    ):
        raise ValueError(f"{model['public_model_id']}: publication precision policy drifted")


def _validate_artifacts(model: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    digest = model["source_artifact_digest"]
    for name in ("retained_score_artifact", "source_code", "parameter_dictionary"):
        if not isinstance(digest.get(name), Mapping):
            raise TypeError(f"{model['public_model_id']}: {name} artifact digest is malformed")
    if digest["retained_score_artifact"].get("root_reference") != (
        "score_evidence.retained_score_artifact"
    ):
        raise ValueError(f"{model['public_model_id']}: retained score artifact root reference drifted")

    score_artifact = payload["score_evidence"]["retained_score_artifact"]
    if not _is_sha256(score_artifact.get("observed_sha256")):
        raise ValueError("retained score artifact observed digest is not SHA-256")
    if not _is_sha256(score_artifact.get("declared_sha256")):
        raise ValueError("retained score artifact declared digest is not SHA-256")
    if score_artifact["observed_sha256"] == score_artifact["declared_sha256"]:
        if score_artifact.get("digest_status") == "mismatch_observed_vs_declared":
            raise ValueError("retained score artifact declares a mismatch that is not present")
    elif score_artifact.get("digest_status") != "mismatch_observed_vs_declared":
        raise ValueError("retained score artifact digest mismatch was not declared")

    for name in ("source_code", "parameter_dictionary"):
        _validate_optional_sha(digest[name].get("sha256"), f"{name}.sha256")


def _validate_factory(model: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    factory = model["implementation_factory"]
    if not isinstance(factory, Mapping):
        raise TypeError(f"{model['public_model_id']}: implementation_factory must be an object")
    if not isinstance(factory.get("callable"), bool):
        raise TypeError(f"{model['public_model_id']}: implementation_factory.callable must be boolean")
    name = factory.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{model['public_model_id']}: implementation_factory.name must be string or null")
    status = factory.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError(f"{model['public_model_id']}: implementation_factory.status must be non-empty")
    factory_map = payload.get("implementation_factory_map", {})
    if not isinstance(factory_map, Mapping):
        raise TypeError("implementation_factory_map must be an object")
    mapped = factory_map.get(model["public_model_id"])
    verified_claim = "verified" in status.lower() and "unverified" not in status.lower()
    if name is not None or factory["callable"] or verified_claim:
        if not isinstance(mapped, Mapping):
            raise ValueError(
                f"{model['public_model_id']}: factory claim requires an explicit implementation map entry"
            )
        if mapped.get("factory_name") != name:
            raise ValueError(f"{model['public_model_id']}: factory map name does not match ledger")
        if mapped.get("status") != "verified_explicit_factory":
            raise ValueError(f"{model['public_model_id']}: factory map entry is not verified")
    elif mapped is not None:
        raise ValueError(f"{model['public_model_id']}: unmapped factory metadata is not allowed")


def _validate_flags(model: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    bool_fields = (
        "identity_recovered",
        "specification_recovered",
        "source_reference_verified",
        "implementation_available",
        "instantiation_validated",
        "forecast_smoke_tested",
        "source_parity_checked",
        "historical_score_verified",
    )
    if any(not isinstance(model[field], bool) for field in bool_fields):
        raise ValueError(f"{model['public_model_id']}: verification flags must be boolean")
    if model["identity_recovered"] and not model["source_reference_verified"]:
        raise ValueError(f"{model['public_model_id']}: recovered identity lacks a verified source reference")
    if model["specification_recovered"]:
        fingerprint = model["specification_fingerprint"]
        if not isinstance(fingerprint, Mapping) or not _is_sha256(fingerprint.get("value")):
            raise ValueError(f"{model['public_model_id']}: recovered specification lacks a valid fingerprint")
        if not model["structured_specification"]:
            raise ValueError(f"{model['public_model_id']}: recovered specification is empty")
    if model["implementation_available"]:
        source_digest = model["source_artifact_digest"]["source_code"].get("sha256")
        if not _is_sha256(source_digest):
            raise ValueError(f"{model['public_model_id']}: implementation availability lacks source digest")
    if model["instantiation_validated"]:
        if not model["implementation_available"]:
            raise ValueError(f"{model['public_model_id']}: instantiation lacks implementation availability")
        if not model["implementation_factory"].get("callable"):
            raise ValueError(f"{model['public_model_id']}: instantiation lacks a mapped callable factory")
    if model["forecast_smoke_tested"] and not model["instantiation_validated"]:
        raise ValueError(f"{model['public_model_id']}: smoke test lacks validated instantiation")
    if model["source_parity_checked"] and (
        not model["implementation_available"]
        or not model["forecast_smoke_tested"]
        or not model["source_reference_verified"]
    ):
        raise ValueError(f"{model['public_model_id']}: source parity lacks implementation, smoke, or source evidence")
    if model["historical_score_verified"] and not model["source_reference_verified"]:
        raise ValueError(f"{model['public_model_id']}: verified score lacks a verified source reference")

    status = model["verification_status"]
    if not isinstance(status, str) or not status.strip():
        raise ValueError(f"{model['public_model_id']}: verification_status must be a non-empty string")
    normalized_status = status.lower()
    if (
        "verified" in normalized_status and "unverified" not in normalized_status
        and not (model["source_reference_verified"] or model["historical_score_verified"])
    ):
        raise ValueError(f"{model['public_model_id']}: verified status lacks evidence flags")


def validate_canonical_ledger(payload: Mapping[str, Any]) -> None:
    """Validate immutable membership and evidence-consistent advancement rules."""

    if payload.get("scope") != "canonical_175_only":
        raise ValueError("canonical ledger scope must be canonical_175_only")
    if payload.get("source_of_truth") != "models":
        raise ValueError("canonical ledger source_of_truth must be models")
    if list(payload.get("required_model_fields", ())) != list(REQUIRED_MODEL_FIELDS):
        raise ValueError("canonical ledger required-field contract drifted")
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != EXPECTED_CANONICAL_COUNT:
        raise ValueError(f"canonical ledger must contain {EXPECTED_CANONICAL_COUNT} models")
    membership = payload.get("membership")
    if not isinstance(membership, Mapping):
        raise TypeError("canonical ledger membership metadata is missing")
    if membership.get("count") != EXPECTED_CANONICAL_COUNT:
        raise ValueError("canonical ledger membership count drifted")
    if tuple(membership.get("canonical_rank_range", ())) != (1, EXPECTED_CANONICAL_COUNT):
        raise ValueError("canonical ledger canonical-rank range drifted")
    if tuple(membership.get("source_rank_range", ())) != (12, 186):
        raise ValueError("canonical ledger historical-rank range drifted")

    for index, model in enumerate(models, start=1):
        if not isinstance(model, Mapping):
            raise TypeError(f"canonical ledger model {index} is not an object")
        missing = [field for field in REQUIRED_MODEL_FIELDS if field not in model]
        if missing:
            raise ValueError(f"{model.get('public_model_id')}: missing required fields {missing}")
        unexpected = sorted(set(model) - _ALLOWED_MODEL_FIELDS)
        if unexpected:
            raise ValueError(f"{model.get('public_model_id')}: unexpected fields {unexpected}")

    ids = [model["public_model_id"] for model in models]
    if any(not isinstance(model_id, str) or not model_id for model_id in ids):
        raise ValueError("canonical ledger contains an empty public model ID")
    if len(ids) != len(set(ids)):
        raise ValueError("canonical ledger contains duplicate public model IDs")
    factory_map = payload.get("implementation_factory_map", {})
    if not isinstance(factory_map, Mapping):
        raise TypeError("implementation_factory_map must be an object")
    unknown_factory_ids = sorted(set(factory_map) - set(ids))
    if unknown_factory_ids:
        raise ValueError(f"implementation factory map contains unknown IDs: {unknown_factory_ids}")
    canonical_ranks = [model["canonical_rank"] for model in models]
    historical_ranks = [model["historical_rank"] for model in models]
    if canonical_ranks != list(range(1, EXPECTED_CANONICAL_COUNT + 1)):
        raise ValueError("canonical ranks must be exactly 1..175")
    if historical_ranks != list(EXPECTED_SOURCE_RANKS):
        raise ValueError("historical ranks must be exactly 12..186")
    digest = canonical_membership_digest(models)
    if digest != EXPECTED_MEMBERSHIP_DIGEST:
        raise ValueError("canonical membership digest does not match the immutable expected digest")
    if membership.get("membership_digest") != EXPECTED_MEMBERSHIP_DIGEST:
        raise ValueError("canonical ledger membership digest drifted")

    score_evidence = payload.get("score_evidence")
    if not isinstance(score_evidence, Mapping):
        raise TypeError("score_evidence metadata is missing")
    for field in ("cells_per_model", "horizon_count", "portfolio_count"):
        if not isinstance(score_evidence.get(field), int) or score_evidence[field] <= 0:
            raise ValueError(f"score_evidence {field} must be a positive integer")
    retained_artifact = score_evidence.get("retained_score_artifact")
    if not isinstance(retained_artifact, Mapping):
        raise TypeError("score_evidence retained artifact metadata is missing")
    source_provenance = payload.get("source_provenance")
    if not isinstance(source_provenance, Mapping):
        raise TypeError("source_provenance metadata is missing")
    if not _is_git_sha(source_provenance.get("pinned_source_revision")):
        raise ValueError("source_provenance pinned source revision is not a full commit SHA")
    for field in ("observed_sha256", "declared_sha256"):
        if not _is_sha256(retained_artifact.get(field)):
            raise ValueError(f"score_evidence retained artifact {field} is not SHA-256")
    _validate_optional_sha(payload.get("identity_policy", {}).get("protocol_fingerprint"), "identity protocol fingerprint")
    _validate_optional_sha(payload.get("identity_policy", {}).get("dataset_fingerprint"), "identity data fingerprint")
    _validate_optional_sha(payload.get("identity_policy", {}).get("panel_fingerprint"), "identity panel fingerprint")
    _assert_public_paths(payload)

    previous_score: Decimal | None = None
    for model in models:
        historical_ids = model["historical_model_ids"]
        if (
            not isinstance(historical_ids, list)
            or not historical_ids
            or model["public_model_id"] not in historical_ids
            or len(historical_ids) != len(set(historical_ids))
            or any(not isinstance(item, str) or not item for item in historical_ids)
        ):
            raise ValueError(f"{model['public_model_id']}: historical source identity drifted")
        if not _is_git_sha(model["source_revision"]):
            raise ValueError(f"{model['public_model_id']}: source revision is not a full commit SHA")
        source_location = model["source_location_reference"]
        if not isinstance(source_location, Mapping) or not source_location:
            raise ValueError(f"{model['public_model_id']}: source location reference is missing")
        score_precision = model["score_precision"]
        if not isinstance(score_precision, Mapping):
            raise TypeError(f"{model['public_model_id']}: score precision metadata is malformed")
        if model["historical_rank"] != model["canonical_rank"] + 11:
            raise ValueError(f"{model['public_model_id']}: rank mapping drifted")
        _validate_score(model, score_evidence)
        _validate_artifacts(model, payload)
        _validate_factory(model, payload)
        _validate_flags(model, payload)
        for field in ("protocol_fingerprint", "dataset_fingerprint", "panel_fingerprint"):
            _validate_optional_sha(model[field], f"{model['public_model_id']}.{field}")
        score = _as_decimal(model["historical_score"]["exact_empirical_crps"], "historical_score")
        if previous_score is not None and score < previous_score:
            raise ValueError("historical scores are not monotonically nondecreasing")
        previous_score = score

    full_specs = sum(model["specification_recovered"] for model in models)
    if payload.get("identity_policy", {}).get("full_statistical_specifications_confirmed") != full_specs:
        raise ValueError("identity policy full-specification count does not match model flags")


def load_canonical_ledger(root: Path | None = None) -> dict[str, Any]:
    """Load and validate the packaged ledger independently of the cwd."""

    path = _ledger_path(root)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_canonical_ledger(payload)
    return payload


def load_canonical_models(root: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Return defensive copies of the authoritative canonical records."""

    return tuple(deepcopy(model) for model in load_canonical_ledger(root)["models"])


def canonical_model(model_id: str, root: Path | None = None) -> dict[str, Any]:
    """Return one canonical record or fail closed for an unknown ID."""

    for model in load_canonical_models(root):
        if model["public_model_id"] == model_id:
            return model
    raise ValueError(f"unknown canonical model ID: {model_id}")


def frontier_model_id(root: Path | None = None) -> str:
    """Return the immutable source ID at canonical rank one."""

    return load_canonical_models(root)[0]["public_model_id"]


__all__ = [
    "EXPECTED_CANONICAL_COUNT",
    "EXPECTED_CELLS_PER_MODEL",
    "EXPECTED_MEMBERSHIP_DIGEST",
    "EXPECTED_SOURCE_RANKS",
    "LEDGER_RESOURCE",
    "REQUIRED_MODEL_FIELDS",
    "canonical_membership_digest",
    "canonical_model",
    "frontier_model_id",
    "load_canonical_ledger",
    "load_canonical_models",
    "validate_canonical_ledger",
]
