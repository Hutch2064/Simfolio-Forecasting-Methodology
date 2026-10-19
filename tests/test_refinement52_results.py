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
