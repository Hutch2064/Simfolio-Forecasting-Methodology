import copy
import hashlib
import json
from pathlib import Path

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
    assert len(studies) == 12
    factorial = studies[0]
    encoded = json.dumps(factorial, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == "9d2187fe9b832878f868d12837660c7df5f33c622f5b4dee97b61d8486e4ac34"
    assert len({row["archive_summary"]["normalized_closure_resource"] for row in studies}) == 12


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


_NEW_RESOURCES = Path(__file__).resolve().parents[1] / 'src/simfolio_forecasting_methodology/resources/results'


def _new_resource(name):
    return json.loads((_NEW_RESOURCES / name).read_text())


@pytest.mark.parametrize(('study_id', 'label', 'score52', 'score80', 'variants'), [
    ('UF-INTERACTION-0c07f7cefb83d7ed', 'total_kernel_existing_conventional_mean',
     0.5969980426397338, 0.23171189271702697, 10),
    ('UF-RESERVE-27c9c8aa0fafac4f', 'conditional_rough_no_conventional_original_nodes',
     0.605745843721052, 0.22869542481735233, 11),
])
def test_interaction_and_reserve_full_panels_keep_exact_scores_controls_and_decisions(study_id, label, score52, score80, variants):
    study = next(s for s in load_unified_frontier_studies()['studies'] if s['study_id'] == study_id)
    assert len(study['variants']) == variants
    assert study['panels']['52']['scores'][label] == score52
    assert study['panels']['80']['scores'][label] == score80
    assert next(v for v in study['variants'] if v['label'] == label)['decision'] == 'rejected'
    assert study['selection']['joint_objective_met'] is False
    assert study['selection']['stopping_condition_met'] is False
    assert 'publication_readiness' not in study


@pytest.mark.parametrize('study_id', ['UF-INTERACTION-0c07f7cefb83d7ed', 'UF-RESERVE-27c9c8aa0fafac4f'])
def test_verified_archive_restoration_keeps_validation_types_and_does_not_establish_a_new_winner(study_id):
    study = next(s for s in load_unified_frontier_studies()['studies'] if s['study_id'] == study_id)
    archive = study['archive_summary']
    assert archive['status'] == 'verified_restored_scientific_smoke'
    assert archive['restored_oracles'] == 2 and archive['batch_restored_oracles'] == 5
    assert archive['all_path_and_CRPS_oracles_byte_equal'] is True
    assert archive['M261_all240_saved_reconstruction'] == 'passed'
    assert archive['scientific_proof']['sha256'] == '12f0a407913c41006622a2cf7f4ca15a203357ad3bb12962ad6db9d1f27f6517'
    assert archive['lineage_receipt']['sha256'] == '7002b26cffae99b465fde9e38246153b22076723b7c216aa25aeb80fa3c2f557'
    assert study['runtime_evidence']['status'] == 'not_measured'
    assert study['runtime_evidence']['production_browser_ctv'] is False
    assert study['validation'] == {'canonical_use': 'exploratory_reused_panels',
        'seed_robustness': {'status': 'not_tested'}, 'independent_post_selection': {'status': 'not_tested'},
        'production': 'not_validated'}


@pytest.mark.parametrize('study_id', ['UF-INTERACTION-0c07f7cefb83d7ed', 'UF-RESERVE-27c9c8aa0fafac4f'])
def test_an_unpassed_restoration_gate_remains_unpublishable(study_id):
    study = copy.deepcopy(next(s for s in load_unified_frontier_studies()['studies'] if s['study_id'] == study_id))
    study['archive_summary']['status'] = 'verified_bytes_scientific_restore_pending'
    study['archive_summary']['restored_oracles'] = 0
    with pytest.raises(ValueError, match='verified private archive references missing'):
        validate_unified_frontier_index({'schema': 'unified-frontier-studies-v1', 'studies': [study]})


def test_all_three_prior_study_records_are_unchanged():
    studies = load_unified_frontier_studies()['studies']
    expected = ['9d2187fe9b832878f868d12837660c7df5f33c622f5b4dee97b61d8486e4ac34',
        '186c48defe679768a40838a4447e6887804b21ee79e2a355ae03928ba252114f',
        'b5141e510910429964763fcbee8aaa258d3a358e6f2f636e12e1b2734b3dea20']
    assert [hashlib.sha256(json.dumps(s, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for s in studies[:3]] == expected


def test_targeted_M261_reference_has_no_new_rank_or_fabricated_panel_score():
    diagnostic = _new_resource('unified_frontier_m261_long_diagnostic.json')
    assert diagnostic['ranked_model'] is False
    assert not {'study_id', 'panels', 'scores', 'variants', 'rank'} & set(diagnostic)
    assert diagnostic['scope'] == {'panel': 'original80_targeted_long_contexts', 'observed_contexts': 240,
        'portfolios': 80, 'horizons': [8766, 5844, 2922], 'simulations': 240, 'full4080_origin_rerun': False,
        'new_candidate': False, 'new_full_panel_score': False, 'zero_cost_scored': False}
    assert diagnostic['independent_saved_data_audit']['status'] == 'passed'
    assert diagnostic['historical_observer_plain_parity']['actual_contexts'] == 3
    assert diagnostic['historical_observer_plain_parity']['indices'] == [48, 49, 50]
    restoration = diagnostic['restoration_validation']
    assert restoration['status'] == 'passed'
    assert restoration['observed_context_executed'] is True
    assert restoration['all240_saved_data_reconstruction_executed'] is True
    assert restoration['observed_context']['index'] == 48
    assert restoration['saved_reconstruction']['saved_records'] == 240
    assert restoration['saved_reconstruction']['portfolio_reconstructions'] == 80
    assert restoration['saved_reconstruction']['financial_forecasts'] == 0


def test_workspace_diagnostic_source_index_preserves_outside_root_evidence():
    source = _new_resource('unified_frontier_m261_long_diagnostic_sources.json')
    assert source['evidence_root'] == 'workspace' and len(source['files']) == 1367
    assert any(p.endswith('m261_long_full_audit/full-audit.json') for p in source['files'])
    assert any(p.endswith('diagnostics/m261-relative-long-controls.json') for p in source['files'])
    assert all(not Path(p).is_absolute() and '..' not in Path(p).parts and len(h) == 64 for p, h in source['files'].items())


def test_reserve_historical_parity_and_normalized_analysis_scope_remain_exact():
    study = next(s for s in load_unified_frontier_studies()['studies'] if s['study_id'] == 'UF-RESERVE-27c9c8aa0fafac4f')
    assert study['historical_scope']['historical_saved_path_oracles_available'] is False
    assert study['historical_scope']['retained_historical_full52_cells_and_counts_byte_equal'] is True
    assert study['analysis_normalization']['added_only'] == ['simulations', 'origin_horizon_contributions']
    assert 'reserve_publication_audit/normalized-analysis.json' in study['panels']['80']['evidence']['analysis']['path']


@pytest.mark.parametrize('study_id', ['UF-INTERACTION-0c07f7cefb83d7ed', 'UF-RESERVE-27c9c8aa0fafac4f'])
def test_normalized_archive_closure_binds_every_source_and_actual_scientific_proof(study_id):
    study = next(s for s in load_unified_frontier_studies()['studies'] if s['study_id'] == study_id)
    closure = _new_resource(study['archive_summary']['normalized_closure_resource'])
    assert closure['status'] == 'verified' and closure['source_input_seed_environment_closure'] is True
    assert closure['scientific_restore']['status'] == 'passed'
    assert closure['scientific_restore']['actual_batch_forecast_oracles'] == 5
    assert closure['scientific_restore']['actual_saved_M261_reconstruction'] == 'passed'
    assert all(closure['closed_artifacts'][s['path']] == s['sha256'] for s in study['sources'])


def test_corrected_nodes_and_bridge_keep_freshness_and_runtime_scopes_distinct():
    studies = {s['study_id']: s for s in load_unified_frontier_studies()['studies']}
    off = studies['UF-ROUGH-CORRECTED-OFF-021da156f7455bce']
    on = studies['UF-ROUGH-CORRECTED-ON-9953676cd480585f']
    bridge = studies['UF-ROUGH-CONVENTIONAL-BRIDGE-4ce66184d7db04fd']
    assert off['panels']['52']['scores']['conditional_rough_no_conventional_preserve_mean_continuous_nodes'] == 0.6048042045569166
    assert off['panels']['80']['scores']['conditional_rough_no_conventional_preserve_mean_continuous_nodes'] == 0.23045095600899104
    assert on['panels']['52']['scores']['conditional_rough_conventional_preserve_mean_continuous_nodes'] == 0.6095093947133338
    assert on['panels']['80']['scores']['conditional_rough_conventional_preserve_mean_continuous_nodes'] == 0.22648879635922456
    assert bridge['panels']['52']['status'] == 'not_evaluated'
    assert 'scores' not in bridge['panels']['52']
    assert bridge['catalogue_review_annotations']['historical52']['CRPS'] == 0.6022154610545161
    assert bridge['catalogue_review_annotations']['historical52']['new_forecasts'] == 0
    assert bridge['panels']['80']['scores']['rough_only_covariance_on_sampled_conventional_SV_corrected_nodes'] == 0.22904705025144675
    original = on['related_historical_experiments'][0]
    assert original['full52']['CRPS'] == 0.6105233636157572
    assert original['full52']['new_forecasts'] == 0
    assert original['full80']['status'] == 'not_run'
    assert original['full80']['CRPS'] is None
    for study in (off, on, bridge):
        assert not study['selection']['joint_objective_met']
        assert study['validation']['production'] == 'not_validated'
        assert study['validation']['seed_robustness']['status'] == 'not_tested'
        assert study['validation']['independent_post_selection']['status'] == 'not_tested'
        assert study['runtime_evidence']['production_browser_ctv'] is False
        assert study['archive_summary']['batch_restored_oracles'] == 6
        assert study['archive_summary']['original_runtime_reused'] is True
        assert study['archive_summary']['standalone_environment_restore'] is False


def test_bridge_inherited_panel_cannot_be_presented_as_fresh_scores():
    payload = copy.deepcopy(load_unified_frontier_studies())
    bridge = next(s for s in payload['studies'] if s['study_id'] == 'UF-ROUGH-CONVENTIONAL-BRIDGE-4ce66184d7db04fd')
    bridge['panels']['52']['scores'] = {'fake': 0.6022154610545161}
    with pytest.raises(ValueError, match='unscored panels cannot carry scores'):
        validate_unified_frontier_index(payload)


def test_final_IID_panels_keep_saved_scores_and_dependency_restore_scope():
    studies = {row['study_id']: row for row in load_unified_frontier_studies()['studies']}
    expected = [('UF-ROUGH-IID-b67722c7aaebc226', 'rough_only_IID_on_sampled_conventional_SV_corrected_nodes', .6053046527780706, .22835434759659093), ('UF-ORIGINAL-NODES-IID-52d4c26b608ac1ac', 'iid_conditional_unit_second_moment_scale_original_nodes', .600668345431752, .22653257917294423)]
    for identity, label, score52, score80 in expected:
        row = studies[identity]
        assert row['panels']['52']['scores'][label] == score52
        assert row['panels']['80']['scores'][label] == score80
        assert row['archive_summary']['restoration_scope'] == 'reused_dependency_only_not_IID_law_replay'
        assert row['archive_summary']['later_historical_source_closure'] is False
        assert row['selection']['joint_objective_met'] is False
        assert row['validation']['production'] == 'not_validated'
