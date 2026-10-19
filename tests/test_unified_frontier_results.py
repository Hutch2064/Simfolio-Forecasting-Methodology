import copy
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
