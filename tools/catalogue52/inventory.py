"""Freeze a source-backed inventory of the 263 canonical catalogue rows.

This is a static audit only: it reads the ledger, source files, retained score
JSON, and adapter map. It never imports or executes model implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER_REL = Path("src/simfolio_forecasting_methodology/resources/canonical_175/ledger.json")
ADAPTER_REL = Path("tools/catalogue52/adapters.py")
BRIDGE_REL = Path("tools/catalogue52/single_asset_bridge.py")
EXCLUDED = {
    257: "M256 sibling: changes only the mean-curve EWMA timescale selector to centered conditional lag-one regression.",
    258: "M256 sibling: changes only the mean-curve EWMA timescale selector to a one-rate predictive-proxy likelihood.",
    259: "M256 sibling: changes only the mean-curve timescale-selection branch to forward BIC-selected rates and component count; retained as the closest adaptive-count sibling, but outside the one requested campaign slot.",
    260: "M256 sibling: changes only the mean-curve EWMA timescale selector to a stationary AR(1) likelihood on the existing latent history.",
}
SAVED_CONTROLS = {261: "Retained saved-result control for the M256 data-selected-timescale comparison; not a fresh candidate run."}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def git_show(repo: Path, revision: str, relative_path: str) -> bytes | None:
    if not revision or not relative_path:
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def normalize_engine_path(path: str | None) -> str | None:
    if not path:
        return None
    prefix = "source-research/scripts/"
    return "scripts/" + path[len(prefix):] if path.startswith(prefix) else path


def factory_path(factory_name: str | None) -> str | None:
    if not factory_name or ":" not in factory_name:
        return None
    module = factory_name.split(":", 1)[0]
    if not module.startswith("simfolio_forecasting_methodology."):
        return None
    return "src/" + module.replace(".", "/") + ".py"


def score_record(panel: dict, model_id: str) -> object | None:
    scores = panel.get("scores")
    if isinstance(scores, dict) and model_id in scores:
        entry = scores[model_id]
        if isinstance(entry, dict):
            return entry.get("exact_empirical_crps")
        return entry
    for key in ("models", "full_panel_scores"):
        rows = panel.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("model_id") == model_id:
                    return row.get("exact_empirical_crps")
        elif isinstance(rows, dict):
            entry = rows.get(model_id)
            if isinstance(entry, dict):
                return entry.get("exact_empirical_crps")
            if entry is not None:
                return entry
    if panel.get("model_id") == model_id:
        return panel.get("exact_empirical_crps")
    if panel.get("exact_empirical_crps") is not None:
        return panel["exact_empirical_crps"]
    return None


def score_counts(panel: dict, evidence: dict, model_id: str) -> dict:
    model_row = None
    rows = panel.get("models")
    if isinstance(rows, list):
        model_row = next((r for r in rows if isinstance(r, dict) and r.get("model_id") == model_id), None)
    complete = evidence.get("complete_portfolios")
    tasks = evidence.get("tasks_per_model")
    cells = evidence.get("cells_per_model")
    if model_row:
        complete = model_row.get("complete_portfolios", model_row.get("completed_portfolios", complete))
        tasks = model_row.get("completed_tasks_per_model", model_row.get("completed_tasks", tasks))
        cells = model_row.get("completed_cells_per_model", model_row.get("completed_cells", cells))
    complete = panel.get("completed_portfolios", complete)
    tasks = panel.get("completed_tasks_per_model", tasks)
    cells = panel.get("completed_portfolio_cells_per_model", cells)
    if evidence.get("scope", "").startswith("full_"):
        complete = evidence.get("complete_portfolios", 80) or 80
        tasks = evidence.get("tasks_per_model", 4080)
        cells = evidence.get("cells_per_model", 701280)
    return {
        "completed_portfolios": complete,
        "required_portfolios": evidence.get("required_portfolios", panel.get("required_portfolios", 80)),
        "completed_tasks": tasks,
        "required_tasks": evidence.get("required_tasks_per_model", 4080),
        "completed_cells": cells,
        "required_cells": evidence.get("required_cells_per_model", 701280),
        "all_denominator_gates_passed": evidence.get("all_denominator_gates_passed"),
    }


def match_numeric(left: object, right: object) -> bool | None:
    if left is None or right is None:
        return None
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return None


def load_adapter_map(root: Path) -> tuple[dict[int, dict], str | None]:
    adapter_file = root / ADAPTER_REL
    if not adapter_file.is_file():
        return {}, None
    namespace = runpy.run_path(str(adapter_file))
    specs = namespace["model_specs"]()
    return {int(row["canonical_rank"]): row for row in specs}, file_sha256(adapter_file)


def source_snapshot(root: Path, engine_root: Path, row: dict, adapter_file: Path | None) -> dict:
    rank = row["canonical_rank"]
    digest = row.get("source_artifact_digest") or {}
    source_code = digest.get("source_code") or {}
    raw_path = source_code.get("relative_path") or source_code.get("path")
    location = row.get("source_location_reference") or {}
    implementation_ref = location.get("implementation")
    factory = row.get("implementation_factory") or {}
    module_rel = factory_path(factory.get("name"))

    if rank <= 175:
        history_repo = engine_root
        history_path = normalize_engine_path(raw_path)
        if history_path is None and implementation_ref and implementation_ref.startswith("models/"):
            history_repo = root
            history_path = "src/simfolio_forecasting_methodology/" + implementation_ref
        current_repo = history_repo
        current_path = history_path
    else:
        history_repo = root
        history_path = raw_path or implementation_ref
        current_repo = root
        current_path = raw_path or implementation_ref

    # For package factories, report the actual method module even when the
    # historical row points to an older external research entrypoint.
    implementation_path = module_rel or current_path
    implementation_file = root / implementation_path if implementation_path else None
    original_file = current_repo / current_path if current_path else None
    current_digest = file_sha256(original_file) if original_file else None
    implementation_digest = file_sha256(implementation_file) if implementation_file else None

    recovered_name = {
        177: "filtered_innovation_sv_candidates.py",
        178: "filtered_innovation_sv_candidates.py",
        179: "fixed_mean_sv_candidates.py",
    }.get(rank)
    recovery_root = root.parent / "setup" / "recovered-legacy-source"
    recovered_file = recovery_root / recovered_name if recovered_name else None
    recovered_digest = file_sha256(recovered_file) if recovered_file else None
    recovery_receipt_path = root.parent / "setup" / "legacy-source-restoration.json"
    recovery_receipt = {}
    if recovery_receipt_path.is_file():
        try:
            recovery_receipt = json.loads(recovery_receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            recovery_receipt = {}
    recovered_expected = None
    if recovered_name:
        for item in recovery_receipt.get("recovered", {}).values():
            if Path(item.get("path", "")).name == recovered_name:
                recovered_expected = item.get("sha256")
                break

    revision = row.get("source_revision")
    historical_bytes = git_show(history_repo, revision, history_path) if history_path else None
    expected = source_code.get("sha256")
    historical_digest = sha256_bytes(historical_bytes) if historical_bytes is not None else None
    if historical_bytes is None:
        history_status = "not_row_specific_or_not_found"
    elif expected and historical_digest == expected:
        history_status = "historical_blob_sha256_verified"
    else:
        history_status = "historical_blob_present_digest_differs_or_is_not_row_digest"

    original_available = bool(original_file and original_file.is_file())
    adapter_available = bool(adapter_file and adapter_file.is_file())
    if recovered_file and recovered_file.is_file() and recovered_digest == recovered_expected:
        availability = "available_exact_recovered_source"
        history_status = "historical_blob_absent_external_recovered_source_verified"
    elif original_available:
        availability = "available"
    elif implementation_file and implementation_file.is_file():
        availability = "available_via_package_factory"
    elif adapter_available:
        # Keep missing historical bytes distinct from an executable recovered
        # adapter. Admission remains a separate, pending evidence gate.
        availability = "available_via_recovered_adapter"
    else:
        availability = "not_present_in_checkout"

    return {
        "source_revision": revision,
        "source_artifact_digest": expected,
        "source_artifact_digest_status": source_code.get("status"),
        "ledger_source_reference": raw_path or implementation_ref,
        "current_original_source_path": current_path,
        "current_original_source_available": original_available,
        "current_original_source_sha256": current_digest,
        "historical_blob_sha256": historical_digest,
        "historical_blob_status": history_status,
        "recovered_original_source_path": (
            "../setup/recovered-legacy-source/" + recovered_name if recovered_name else None
        ),
        "recovered_original_source_sha256": recovered_digest,
        "recovered_original_source_expected_sha256": recovered_expected,
        "recovered_original_source_sha256_matches_receipt": (
            recovered_digest == recovered_expected if recovered_expected else None
        ),
        "source_restoration_receipt": "../setup/legacy-source-restoration.json" if recovered_name else None,
        "implementation_path": implementation_path,
        "implementation_available": bool(implementation_file and implementation_file.is_file()),
        "implementation_sha256": implementation_digest,
        "campaign_adapter_source_available": adapter_available,
        "availability": availability,
    }


def model_entry(root: Path, engine_root: Path, row: dict, adapter: dict | None, bridge_sha: str | None) -> dict:
    rank = row["canonical_rank"]
    model_id = row["public_model_id"]
    adapter = adapter or {}
    adapter_rel = adapter.get("adapter")
    adapter_file = root / adapter_rel if adapter_rel else None
    adapter_hash = file_sha256(adapter_file) if adapter_file else None
    original = source_snapshot(root, engine_root, row, adapter_file)
    evidence = (row.get("source_artifact_digest") or {}).get("retained_score_artifact") or {}
    evidence_path = evidence.get("relative_path")
    evidence_file = root / evidence_path if evidence_path else None
    evidence_hash = file_sha256(evidence_file) if evidence_file else None
    expected_hash = evidence.get("sha256")
    artifact_match = bool(expected_hash and evidence_hash == expected_hash)
    panel = {}
    if evidence_file and evidence_file.is_file() and evidence_file.suffix == ".json":
        try:
            value = json.loads(evidence_file.read_text())
            panel = value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            panel = {}
    observed_score = score_record(panel, model_id)
    historical_score = (row.get("historical_score") or {}).get("exact_empirical_crps")
    counts = score_counts(panel, evidence, model_id) if panel else {
        "completed_portfolios": None,
        "required_portfolios": 80,
        "completed_tasks": None,
        "required_tasks": 4080,
        "completed_cells": None,
        "required_cells": 701280,
        "all_denominator_gates_passed": evidence.get("all_denominator_gates_passed"),
    }
    if rank <= 175:
        score_status = "aggregate_only_historical_record"
    elif rank <= 179 or evidence_file is None or not evidence_file.is_file() or not artifact_match:
        score_status = "pending_recovery"
    elif evidence.get("scope", "").startswith("full_"):
        score_status = "full_80_artifact_verified"
    elif evidence.get("scope") == "partial_canonical":
        score_status = "partial_artifact_verified"
    elif evidence.get("scope") == "unscored_canonical":
        score_status = "unscored_artifact_verified"
    else:
        score_status = "artifact_present_scope_unclassified"

    score_value_matches = match_numeric(historical_score, observed_score)
    spec_fingerprint = row.get("specification_fingerprint") or {}
    factory = row.get("implementation_factory") or {}
    status_flags = {
        key: row.get(key)
        for key in (
            "identity_recovered", "specification_recovered", "source_reference_verified",
            "implementation_available", "instantiation_validated", "forecast_smoke_tested",
            "source_parity_checked", "historical_score_verified",
        )
    }
    if rank in EXCLUDED:
        campaign_role = "excluded_near_duplicate"
        reason = EXCLUDED[rank]
    elif rank in SAVED_CONTROLS:
        campaign_role = "saved_result_control"
        reason = SAVED_CONTROLS[rank]
    else:
        campaign_role = "fresh_candidate"
        reason = None

    entry = {
        "public_model_id": model_id,
        "display_name": row["display_name"],
        "canonical_rank": rank,
        "historical_rank": row.get("historical_rank"),
        "family": row.get("model_family"),
        "specification": {
            "fingerprint": spec_fingerprint.get("value"),
            "fingerprint_kind": spec_fingerprint.get("kind"),
            "fingerprint_status": spec_fingerprint.get("status"),
            "ledger_record": f"{LEDGER_REL.as_posix()}#/models/{rank - 1}/structured_specification",
        },
        "factory": {
            "name": factory.get("name"),
            "status": factory.get("status"),
            "callable": factory.get("callable"),
            "path": original["implementation_path"],
            "file_available": original["implementation_available"],
            "file_sha256": original["implementation_sha256"],
        },
        "source": original,
        "historical_80": {
            "ledger_score_token": historical_score,
            "ledger_verification_status": row.get("verification_status"),
            "ledger_historical_score_verified": row.get("historical_score_verified"),
            "evidence_status": score_status,
            "artifact_path": evidence_path,
            "artifact_root_reference": evidence.get("root_reference"),
            "artifact_ledger_status": evidence.get("status"),
            "artifact_expected_sha256": expected_hash,
            "artifact_observed_sha256": evidence_hash,
            "artifact_sha256_matches_ledger": artifact_match if expected_hash else None,
            "artifact_scope": evidence.get("scope"),
            "artifact_manifest_sha256": evidence.get("manifest_sha256"),
            "artifact_tasks_per_model": evidence.get("tasks_per_model"),
            "artifact_cells_per_model": evidence.get("cells_per_model"),
            "artifact_simulations": evidence.get("simulations"),
            "artifact_all_denominator_gates_passed": evidence.get("all_denominator_gates_passed"),
            "observed_tasks": counts["completed_tasks"],
            "required_tasks": counts["required_tasks"],
            "observed_portfolios": counts["completed_portfolios"],
            "required_portfolios": counts["required_portfolios"],
            "observed_cells": counts["completed_cells"],
            "required_cells": counts["required_cells"],
            "observed_artifact_score": observed_score,
            "observed_score_matches_ledger_token": score_value_matches,
        },
        "adapter": {
            "path": adapter_rel,
            "file_available": bool(adapter_file and adapter_file.is_file()) if adapter_rel else True,
            "file_sha256": adapter_hash,
            "environment": adapter.get("environment", {}),
            "adapter_registry_path": ADAPTER_REL.as_posix(),
        },
        "ledger_flags": status_flags,
        "campaign": {
            "role": campaign_role,
            "included": campaign_role != "excluded_near_duplicate",
            "fresh_evaluation": campaign_role == "fresh_candidate",
            "reason": reason,
        },
    }
    if 180 <= rank <= 192:
        bridge_file = root / BRIDGE_REL
        entry["statistical_extension"] = {
            "extension": "one_asset_identity_copula_limit",
            "preserves_original_multivariate_source": True,
            "bridge_path": BRIDGE_REL.as_posix(),
            "bridge_file_available": bridge_file.is_file(),
            "bridge_file_sha256": bridge_sha,
            "validation_test_path": "tests/test_catalogue52_adapters.py",
            "validation_test_file_available": (root / "tests/test_catalogue52_adapters.py").is_file(),
            "validation_receipt_status": "test_source_present; durable_receipt_pending",
        }
    return entry


def build(root: Path, engine_root: Path) -> dict:
    ledger_path = root / LEDGER_REL
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    rows = ledger["models"]
    adapter_map, adapter_hash = load_adapter_map(root)
    bridge_sha = file_sha256(root / BRIDGE_REL)

    membership_text = "".join(
        f"{row['canonical_rank']}|{row.get('historical_rank')}|{row['public_model_id']}\n"
        for row in rows
    )
    membership_digest = sha256_bytes(membership_text.encode("utf-8"))
    entries = [model_entry(root, engine_root, row, adapter_map.get(row["canonical_rank"]), bridge_sha) for row in rows]
    included = [item for item in entries if item["campaign"]["included"]]
    candidates = [item for item in included if item["campaign"]["fresh_evaluation"]]

    family_counts = Counter(row["family"] for row in entries)
    included_family_counts = Counter(row["family"] for row in included)
    score_counts_summary = Counter(row["historical_80"]["evidence_status"] for row in entries)
    source_counts = Counter(row["source"]["availability"] for row in entries)
    adapter_missing = [row["canonical_rank"] for row in entries if not row["adapter"]["file_available"]]
    pending_sources = [row["canonical_rank"] for row in entries if row["source"]["availability"] == "not_present_in_checkout"]
    pending_scores = [row["canonical_rank"] for row in entries if row["historical_80"]["evidence_status"] == "pending_recovery"]
    score_evidence = ledger.get("score_evidence", {})
    shared_score = score_evidence.get("retained_score_artifact", {})
    shared_path = shared_score.get("relative_path")
    pinned_engine_revision = (ledger.get("source_provenance") or {}).get("pinned_source_revision")
    shared_pinned_bytes = git_show(engine_root, pinned_engine_revision, shared_path) if shared_path else None
    shared_current_file = engine_root / shared_path if shared_path else None

    return {
        "schema_version": "catalogue52-inventory-v1",
        "audit_mode": "static_inventory_no_model_runs",
        "frozen": True,
        "membership": {
            "canonical_count": len(rows),
            "canonical_rank_range": [min(r["canonical_rank"] for r in rows), max(r["canonical_rank"] for r in rows)],
            "ledger_scope_label": ledger.get("scope"),
            "ledger_scope_label_note": "canonical_175_only is a legacy label; membership count, rank range and validated digest describe all 263 rows.",
            "ledger_sha256": sha256_bytes(ledger_bytes),
            "ledger_relative_path": LEDGER_REL.as_posix(),
            "declared_membership_digest": ledger["membership"]["membership_digest"],
            "computed_membership_digest": membership_digest,
            "membership_digest_matches": membership_digest == ledger["membership"]["membership_digest"],
            "validation_policy_required_digest_note": "The ledger validation_policy membership_digest_required value is stale; it is not used as a membership validation failure.",
        },
        "checkout": {
            "repository": "simfolio-forecasting-methodology",
            "revision": git_value(root, "rev-parse", "HEAD"),
            "tree": git_value(root, "rev-parse", "HEAD^{tree}"),
            "adapter_registry_path": ADAPTER_REL.as_posix(),
            "adapter_registry_sha256": adapter_hash,
        "legacy_source_restoration_receipt_sha256": file_sha256(root.parent / "setup/legacy-source-restoration.json"),
            "engine_source_repository": "retained_source_checkout",
            "engine_worktree_revision": git_value(engine_root, "rev-parse", "HEAD"),
            "engine_pinned_source_revision": pinned_engine_revision,
        },
        "population": {
            "canonical_rows": len(entries),
            "excluded_count": len(EXCLUDED),
            "included_count_including_saved_control": len(included),
            "fresh_candidate_count": len(candidates),
            "saved_control_count": sum(row["campaign"]["role"] == "saved_result_control" for row in included),
            "excluded_ranks": sorted(EXCLUDED),
            "excluded_models": [
                {"canonical_rank": rank, "public_model_id": entries[rank - 1]["public_model_id"], "reason": reason}
                for rank, reason in sorted(EXCLUDED.items())
            ],
            "family_counts_canonical": dict(sorted(family_counts.items())),
            "family_counts_included": dict(sorted(included_family_counts.items())),
        },
        "evidence_summary": {
            "historical_80_status_counts": dict(sorted(score_counts_summary.items())),
            "source_availability_counts": dict(sorted(source_counts.items())),
            "source_paths_not_present_in_checkout_ranks": pending_sources,
            "pending_score_recovery_ranks": pending_scores,
            "missing_adapter_ranks": adapter_missing,
            "pending_one_asset_validation_ranks": [
                rank for rank in range(180, 193)
                if not (root / "docs/results/catalogue52/single-asset-bridge-receipt.json").is_file()
            ],
            "shared_historical_score_artifact": {
                "path": shared_path,
                "observed_sha256": shared_score.get("observed_sha256"),
                "declared_sha256": shared_score.get("declared_sha256"),
                "pinned_revision_blob_sha256": sha256_bytes(shared_pinned_bytes) if shared_pinned_bytes is not None else None,
                "current_worktree_sha256": file_sha256(shared_current_file) if shared_current_file else None,
                "note": "Ledger records a declared-versus-observed digest mismatch for the shared legacy score artifact; this does not erase the row score tokens, and the separate qualification is preserved.",
            },
            "interpretation": "Pending saved-score recovery or admission is not a final blocker decision. No rank-based pruning is applied.",
        },
        "models": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/catalogue52/inventory.json")
    args = parser.parse_args()
    inventory = build(args.root.resolve(), args.engine_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "canonical": inventory["membership"]["canonical_count"],
        "included": inventory["population"]["included_count_including_saved_control"],
        "fresh_candidates": inventory["population"]["fresh_candidate_count"],
        "excluded": inventory["population"]["excluded_count"],
        "score_evidence": inventory["evidence_summary"]["historical_80_status_counts"],
        "source_paths_not_present": inventory["evidence_summary"]["source_paths_not_present_in_checkout_ranks"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
