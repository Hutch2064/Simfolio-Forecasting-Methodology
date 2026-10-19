from __future__ import annotations

import copy

import pytest

from simfolio_forecasting_methodology.refinement52_results import (
    load_refinement52_results,
    validate_refinement52_index,
)


def test_packaged_refinement52_results_keep_immutable_ids_and_panels_separate():
    payload = load_refinement52_results()
    entries = {entry["research_id"]: entry for entry in payload["entries"]}
    assert "R52-ORIGINAL-MOMENT-VARIANCE-CURVE" in entries
    assert entries["R52-DETERMINISTIC-RMS-ABLATION"]["panel_results"]["historical80"] == {
        "status": "not_run"
    }
    assert payload["panels"]["canonical52"]["cells"] == 455832
    assert payload["panels"]["historical80"]["cells"] == 701280
    assert payload["catalogue_identity"]["m_ids_assigned"] is False


def test_duplicate_research_ids_are_rejected():
    payload = copy.deepcopy(load_refinement52_results())
    payload["entries"].append(copy.deepcopy(payload["entries"][0]))
    with pytest.raises(ValueError, match="unique immutable"):
        validate_refinement52_index(payload)


def test_unrun_historical80_score_is_rejected():
    payload = copy.deepcopy(load_refinement52_results())
    rms = next(entry for entry in payload["entries"]
               if entry["research_id"] == "R52-DETERMINISTIC-RMS-ABLATION")
    rms["panel_results"]["historical80"]["exact_empirical_crps"] = 0.4
    with pytest.raises(ValueError, match="cannot carry a score"):
        validate_refinement52_index(payload)


def test_unrun_direct_student_historical80_score_is_rejected():
    payload = copy.deepcopy(load_refinement52_results())
    row = next(entry for entry in payload["entries"]
               if entry["research_id"] == "R52-DIRECT-STUDENT-VARIANCE-ORIGINAL-MEAN")
    row["panel_results"]["historical80"] = {
        "status": "complete_full_panel", "exact_empirical_crps": 0.01,
    }
    with pytest.raises(ValueError, match="unrun historical80 results cannot carry a score"):
        validate_refinement52_index(payload)


def test_stopped_identity_cannot_carry_a_score():
    payload = copy.deepcopy(load_refinement52_results())
    row = next(entry for entry in payload["entries"]
               if entry["research_id"] == "R52-COHERENT-GROWTH-IDENTITY-CONTROL")
    row["panel_results"]["canonical52"]["exact_empirical_crps"] = 0.01
    with pytest.raises(ValueError, match="stopped identity controls must remain unscored"):
        validate_refinement52_index(payload)


def test_malformed_verified_archive_receipt_is_rejected():
    payload = copy.deepcopy(load_refinement52_results())
    malformed = {
        "status": "verified_private_online_archive",
        "verified_online_archive": True,
    }
    payload["archive_evidence"] = malformed
    payload["additional_diagnostics"]["private_archive"] = malformed
    with pytest.raises(ValueError, match="not checksum-bound"):
        validate_refinement52_index(payload)


def _attach_verified_smoke_receipt(payload):
    hashes = payload["source_hashes"]
    receipt_path, manifest_path, smoke_path, *oracle_paths = list(hashes)[:7]
    archive = {
        "status": "verified_private_online_archive",
        "format": "refinement52-archive-receipt-v1",
        "verified_online_archive": True,
        "repo": "private/example",
        "release": "refinement52-test",
        "manifest": {"sha256": hashes[manifest_path], "bytes": 1, "url": "https://example.test/manifest"},
        "assets": {"part.tar.gz": {"sha256": hashes[receipt_path], "bytes": 1,
                                     "url": "https://example.test/part"}},
        "member_count": 1,
        "member_bytes": 1,
        "independent_download_and_member_verification": True,
        "receipt_url": "https://example.test/receipt",
        "receipt_source_path": receipt_path,
        "receipt_sha256": hashes[receipt_path],
        "manifest_source_path": manifest_path,
        "manifest_sha256": hashes[manifest_path],
        "restore_command": "restore --verify",
        "external_dependencies": [],
        "source_paths": [receipt_path, manifest_path, smoke_path, *oracle_paths],
        "scientific_restore_smoke": "passed",
        "scientific_restore_smoke_receipt": {
            "status": "verified_restored_scientific_smoke",
            "variant": "node_spread",
            "complete_path_parity": True,
            "case_count": 4,
            "cases": [{"case": case, "full_path_sha256": "a" * 64,
                       "oracle_sha256": "b" * 64} for case in
                      ("SPYSIM_8766", "TLTSIM_8766", "SPXUSIM_8766", "XLESIM_126")],
            "receipt_source_path": smoke_path,
            "receipt_sha256": hashes[smoke_path],
            "source_paths": [smoke_path, *oracle_paths],
        },
        "deletion_authorized": False,
    }
    payload["archive_evidence"] = archive
    payload["additional_diagnostics"]["private_archive"] = archive
    return payload


def test_verified_restore_smoke_requires_a_hash_bound_four_case_receipt():
    payload = _attach_verified_smoke_receipt(copy.deepcopy(load_refinement52_results()))
    validate_refinement52_index(payload)
    payload["archive_evidence"]["scientific_restore_smoke_receipt"]["cases"][0][
        "full_path_sha256"] = "invalid"
    payload["additional_diagnostics"]["private_archive"] = payload["archive_evidence"]
    with pytest.raises(ValueError, match="scientific smoke receipt is not checksum-bound"):
        validate_refinement52_index(payload)


def test_unbound_source_path_is_rejected():
    payload = copy.deepcopy(load_refinement52_results())
    path = payload["entries"][0]["source_paths"][0]
    payload["source_hashes"].pop(path)
    with pytest.raises(ValueError, match="source hashes are missing"):
        validate_refinement52_index(payload)


def _with_selected80_result(payload):
    from simfolio_forecasting_methodology.refinement52_results import (
        SELECTED80_HASH_ROLES,
        SELECTED80_ID,
        SELECTED80_LABEL,
    )

    selected = next(row for row in payload["entries"]
                    if row["research_id"] == SELECTED80_ID)
    canonical52_before = copy.deepcopy(selected["panel_results"]["canonical52"])
    rough = next(row for row in payload["entries"]
                 if row["research_id"] == "R52-ROUGH-OFF")["panel_results"]["historical80"]
    m261 = payload["baseline_result"]["historical80"]["exact_empirical_crps"]
    score = sum(rough["portfolio_scores"].values()) / 80 * 0.95
    paths = {}
    for index, role in enumerate(sorted(SELECTED80_HASH_ROLES)):
        path = f"short-horizon-mechanisms/refinement52/selected80/test/{role}"
        digest = f"{index + 1:064x}"
        payload["source_hashes"][path] = digest
        selected["source_paths"].append(path)
        paths[role] = {"path": path, "sha256": digest}
    archived_roles = SELECTED80_HASH_ROLES - {"archive_receipt", "archive_manifest"}
    evidence = {
        "schema": "selected80-evidence-v1",
        "research_id": SELECTED80_ID,
        "candidate": SELECTED80_LABEL,
        "run_identity": "selected80-0123456789abcdef",
        "manifest_digest": "f" * 64,
        "panel": {"portfolios": 80, "origins": 4080, "cells": 701280,
                  "simulations": 240},
        "complete_origins": 4080,
        "required_origins": 4080,
        "complete_portfolios": 80,
        "failures": 0,
        "full80_runtime_parity_claimed": False,
        "independent_audit": {
            "status": "complete_independent_checkpoint_cell_rebuild",
            "complete_origins": 4080,
            "required_origins": 4080,
            "failures": 0,
            "all_original_identities_match": True,
            "all_bound_source_control_hashes_match": True,
            "all_selected_cells_and_counts_byte_equal_rebuilt_checkpoints": True,
            "saved_M261_and_rough_off_control_cells_byte_equal": True,
            "manifest_digest": "f" * 64,
        },
        "artifact_hashes": paths,
        "archive": {
            "status": "verified_new_selected80_private_archive",
            "format": "selected80-archive-receipt-v1",
            "repo": "Hutch2064/simfolio-engine",
            "release": "selected80-test",
            "identity": "a" * 20,
            "selected80_run_identity": "selected80-0123456789abcdef",
            "receipt_url": "https://example.test/receipt",
            "manifest_url": "https://example.test/manifest",
            "receipt_sha256": paths["archive_receipt"]["sha256"],
            "manifest_sha256": paths["archive_manifest"]["sha256"],
            "member_count": 100,
            "member_bytes": 1000,
            "task_checkpoint_members": 4080,
            "task_metadata_members": 4080,
            "independent_download_and_member_verification": True,
            "prior_receipt_fresh_download_verified": True,
            "deletion_authorized": False,
            "assets": {"part.tar.gz": {"bytes": 20, "sha256": "b" * 64,
                                       "url": "https://example.test/part"}},
            "archived_members": {
                role: {"path": f"selected80/selected80-test/{role}",
                       "sha256": paths[role]["sha256"]}
                for role in archived_roles
            },
        },
    }
    selected["panel_results"]["historical80"] = {
        "status": "complete_full_panel",
        "exact_empirical_crps": score,
        "production_m261_crps": m261,
        "percent_better_than_m261": 100 * (1 - score / m261),
        "percent_better_than_rough_off": 100 * (1 - score / rough["exact_empirical_crps"]),
        "portfolio_scores": {name: value * 0.95
                             for name, value in rough["portfolio_scores"].items()},
        "cells_sha256": paths["cells"]["sha256"],
        "evidence": evidence,
    }
    return payload, canonical52_before


def test_selected80_result_requires_complete_audit_archive_and_preserves_canonical52():
    payload, canonical52_before = _with_selected80_result(
        copy.deepcopy(load_refinement52_results()))
    validate_refinement52_index(payload)
    selected = next(row for row in payload["entries"]
                    if row["research_id"] == "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION")
    assert selected["panel_results"]["canonical52"] == canonical52_before
    assert selected["panel_results"]["historical80"]["evidence"]["panel"] == {
        "portfolios": 80, "origins": 4080, "cells": 701280, "simulations": 240,
    }


def test_selected80_rejects_partial_or_unverified_evidence():
    payload, _ = _with_selected80_result(copy.deepcopy(load_refinement52_results()))
    selected = next(row for row in payload["entries"]
                    if row["research_id"] == "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION")
    selected["panel_results"]["historical80"]["evidence"]["failures"] = 1
    with pytest.raises(ValueError, match="incomplete or wrong panel"):
        validate_refinement52_index(payload)

    payload, _ = _with_selected80_result(copy.deepcopy(load_refinement52_results()))
    selected = next(row for row in payload["entries"]
                    if row["research_id"] == "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION")
    selected["panel_results"]["historical80"]["portfolio_scores"].pop(
        next(iter(selected["panel_results"]["historical80"]["portfolio_scores"])))
    with pytest.raises(ValueError, match="portfolio scores are incomplete"):
        validate_refinement52_index(payload)

    payload, _ = _with_selected80_result(copy.deepcopy(load_refinement52_results()))
    selected = next(row for row in payload["entries"]
                    if row["research_id"] == "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION")
    selected["panel_results"]["historical80"]["evidence"]["archive"][
        "independent_download_and_member_verification"] = False
    with pytest.raises(ValueError, match="archive receipt is not verified"):
        validate_refinement52_index(payload)


def test_selected80_cannot_be_attached_to_another_research_id():
    payload, _ = _with_selected80_result(copy.deepcopy(load_refinement52_results()))
    selected = next(row for row in payload["entries"]
                    if row["research_id"] == "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION")
    other = next(row for row in payload["entries"]
                 if row["research_id"] == "R52-DIRECT-STUDENT-VARIANCE-ORIGINAL-MEAN")
    other["panel_results"]["historical80"] = copy.deepcopy(
        selected["panel_results"]["historical80"])
    other["source_paths"].extend(selected["source_paths"][-10:])
    with pytest.raises(ValueError, match="unrun historical80 results cannot carry a score"):
        validate_refinement52_index(payload)
