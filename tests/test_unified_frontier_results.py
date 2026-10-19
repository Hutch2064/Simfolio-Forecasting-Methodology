import copy
import hashlib
import json

import pytest

from simfolio_forecasting_methodology.refinement52_results import load_refinement52_results
from simfolio_forecasting_methodology.unified_frontier_results import (
    load_unified_frontier_studies,
    validate_unified_frontier_index,
)


def test_closed_factorial_loads_without_private_raw_files():
    study = load_unified_frontier_studies()["studies"][0]
    assert study["study_id"] == "UF-FACTORIAL-f560a3f6b73353ed"
    assert len(study["variants"]) == 5
    assert study["panels"]["52"]["scores"]["RMS_corrected_nodes_saved_control"] == 0.5957716608071161
    assert study["panels"]["52"]["raw_results_scores"]["RMS_corrected_nodes_saved_control"] == 0.5957716608071159
    assert study["panels"]["80"]["score_aggregation"]["per_label"]["RMS_corrected_nodes_saved_control"]["ulp_difference"] == 3
    assert study["numerical_source_identity"]["source_snapshot_files"] == 590
    assert study["runtime_evidence"]["status"] == "not_measured"
    assert study["selection"]["joint_objective_met"] is False


def test_existing_24_rows_and_selected80_stay_available():
    legacy = load_refinement52_results()
    assert len(legacy["entries"]) == 24
    selected = next(row for row in legacy["entries"] if row["research_id"] == "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION")
    assert selected["panel_results"]["historical80"]["status"] == "complete_full_panel"
    assert selected["panel_results"]["historical80"]["exact_empirical_crps"] == 0.23457102422540146


@pytest.mark.parametrize(("change", "message"), [
    (lambda study: study["panels"].pop("80"), "separate full52"),
    (lambda study: study["variants"].pop(), "variant/control scores missing"),
    (lambda study: study["panels"]["52"]["denominators"].update({"cells_per_model": 455831}), "denominators differ"),
    (lambda study: study["sources"][0].update({"sha256": "bad"}), "source hash"),
    (lambda study: study["validation"].update({"production": "validated"}), "validation evidence types"),
    (lambda study: study["runtime_evidence"].update({"production_browser_ctv": True}), "not browser CTV"),
    (lambda study: study["selection"].update({"joint_objective_met": True}), "one law winning both"),
    (lambda study: study["panels"]["80"]["score_aggregation"].update({"gamma_n": 0.01}), "derived binary64"),
    (lambda study: study["panels"]["52"]["raw_results_scores"].update({"M261_saved_control": 0.1}), "preserve raw and canonical"),
])
def test_false_completeness_or_evidence_claims_fail_closed(change, message):
    payload = copy.deepcopy(load_unified_frontier_studies())
    change(payload["studies"][0])
    with pytest.raises(ValueError, match=message):
        validate_unified_frontier_index(payload)


def test_explicit_path_load_is_portable(tmp_path):
    payload = load_unified_frontier_studies()
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload))
    assert load_unified_frontier_studies(path) == payload


@pytest.mark.parametrize(("study_id", "variant_count", "label", "score52", "score80"), [
    ("UF-CONDITIONAL-1825f585d4e8d7d5", 7, "covariance_only", 0.5957716608071161, 0.23387900572210277),
    ("UF-MOMENTS-f4ef8715429fc6e6", 9, "total_covariance_only", 0.5957716608071161, 0.23239664339923255),
])
def test_closed_moment_studies_preserve_full_panels_and_typed_validation(study_id, variant_count, label, score52, score80):
    study = next(row for row in load_unified_frontier_studies()["studies"] if row["study_id"] == study_id)
    assert len(study["variants"]) == variant_count
    assert study["panels"]["52"]["scores"][label] == score52
    assert study["panels"]["80"]["scores"][label] == score80
    assert all(row["status"] == "complete_scored_and_audited" for row in study["panels"].values())
    assert study["runtime_evidence"]["status"] == "not_measured"
    assert study["runtime_evidence"]["production_browser_ctv"] is False
    assert study["validation"]["seed_robustness"]["status"] == "not_tested"
    assert study["validation"]["independent_post_selection"]["status"] == "not_tested"
    assert study["validation"]["production"] == "not_validated"
    assert study["selection"]["joint_objective_met"] is False
    assert study["archive_summary"]["restored_oracles"] == 2
    assert study["archive_summary"]["batch_restored_oracles"] == 4


def test_factorial_record_is_unchanged_when_moment_studies_append():
    studies = load_unified_frontier_studies()["studies"]
    assert len(studies) == 3
    factorial = studies[0]
    encoded = json.dumps(factorial, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == "9d2187fe9b832878f868d12837660c7df5f33c622f5b4dee97b61d8486e4ac34"
    assert len({row["archive_summary"]["normalized_closure_resource"] for row in studies}) == 3


def test_workspace_evidence_root_keeps_pinned_cleanup_receipt_separate_from_data_root():
    studies = {row["study_id"]: row for row in load_unified_frontier_studies()["studies"]}
    conditional = studies["UF-CONDITIONAL-1825f585d4e8d7d5"]
    followup = studies["UF-MOMENTS-f4ef8715429fc6e6"]
    assert conditional["evidence_path_base"] == "LAB"
    assert conditional["data_root_relative_to_evidence"] == "."
    assert followup["evidence_path_base"] == "workspace"
    assert followup["data_root_relative_to_evidence"] == "tmp/catalogue52-20261018"
    assert {"path": "storage-audits/2026-10-19/unified-factorial-2f0b153ea2accba072ba-cleanup.json",
            "sha256": "9dafa6df0da16df5784c0774aebb6e1091a26ea1ad6c63c470c8b2d81da51352"} in followup["sources"]


@pytest.mark.parametrize("study_id", ["UF-CONDITIONAL-1825f585d4e8d7d5", "UF-MOMENTS-f4ef8715429fc6e6"])
def test_appended_probes_cannot_claim_a_joint_winner(study_id):
    payload = copy.deepcopy(load_unified_frontier_studies())
    next(row for row in payload["studies"] if row["study_id"] == study_id)["selection"]["joint_objective_met"] = True
    with pytest.raises(ValueError, match="one law winning both"):
        validate_unified_frontier_index(payload)


def test_mean_and_combined_laws_keep_exact_scores_and_explicit_rejections():
    studies = {row["study_id"]: row for row in load_unified_frontier_studies()["studies"]}
    for study_id, label, full80 in [
        ("UF-CONDITIONAL-1825f585d4e8d7d5", "mean_only", 0.23385584849519025),
        ("UF-MOMENTS-f4ef8715429fc6e6", "conventional_mean_plus_covariance", 0.2331864906759),
    ]:
        study = studies[study_id]
        assert study["panels"]["52"]["scores"][label] == 0.5969980426397338
        assert study["panels"]["80"]["scores"][label] == full80
        variant = next(row for row in study["variants"] if row["label"] == label)
        assert variant["decision"] == "rejected"
        assert variant["methodology"] and variant["rationale"]
