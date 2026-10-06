from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from simfolio_forecasting_methodology.catalog import load_canonical_175
from simfolio_forecasting_methodology.catalogue import (
    EXPECTED_CANONICAL_COUNT,
    EXPECTED_MEMBERSHIP_DIGEST,
    EXPECTED_SOURCE_RANKS,
    REQUIRED_MODEL_FIELDS,
    canonical_membership_digest,
    canonical_model,
    load_canonical_ledger,
    load_canonical_models,
    validate_canonical_ledger,
)
from simfolio_forecasting_methodology.models.registry import build_model, registration


def _resolved_specification_ids() -> set[str]:
    return {
        row["public_model_id"]
        for row in load_canonical_ledger()["models"]
        if row["specification_recovered"]
    }


def test_membership_is_exact_and_digest_is_immutable():
    payload = load_canonical_ledger()
    models = list(load_canonical_models())

    assert payload["membership"]["count"] == EXPECTED_CANONICAL_COUNT == 236
    assert [model["canonical_rank"] for model in models] == list(range(1, 237))
    assert [model["historical_rank"] for model in models] == list(EXPECTED_SOURCE_RANKS)
    assert canonical_membership_digest(models) == EXPECTED_MEMBERSHIP_DIGEST
    assert payload["membership"]["membership_digest"] == EXPECTED_MEMBERSHIP_DIGEST


def test_map_predecessor_dynamic_rough_retains_complete_scored_evidence():
    import hashlib

    import numpy as np

    row = canonical_model('asset_map_predecessor_dynamic_rough_map')
    root = Path(__file__).resolve().parents[1]
    artifact = row['source_artifact_digest']['retained_score_artifact']
    path = root / artifact['relative_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
    result = json.loads(path.read_text())
    assert result['audit']['passed'] and result['audit']['all_cell_vectors_byte_equal']
    assert all(r['every_loss_byte_equal'] for r in result['smoke_parity'])
    cells = root / 'docs/results/rough-bayesian/map-predecessor-dynamic-cells.npz'
    assert hashlib.sha256(cells.read_bytes()).hexdigest() == result['paired_cells_sha256']
    with np.load(cells) as values:
        assert len(values['model_0']) == 701280
        assert len(set(values['portfolio'])) == 80
        assert repr(float(values['model_0'].mean())) == row['historical_score']['exact_empirical_crps']
    assert len(row['historical_score']['portfolio_scores']) == 80
    definition = row['structured_specification']['resolved_definition']
    assert definition['parameter_estimation']['chains'] == 0
    assert definition['parameter_estimation']['fixed_posterior_representatives'] is None
    assert definition['baseline']['model_id'] == 'experimental_filtered_innovation_moment_sv_fixed_mean'


def test_rehashing_a_mutated_id_is_rejected():
    payload = deepcopy(load_canonical_ledger())
    original_id = payload["models"][0]["public_model_id"]
    payload["models"][0]["public_model_id"] += "_tampered"
    payload["implementation_factory_map"][payload["models"][0]["public_model_id"]] = payload[
        "implementation_factory_map"
    ].pop(original_id)
    payload["models"][0]["historical_model_ids"] = [payload["models"][0]["public_model_id"]]
    payload["membership"]["membership_digest"] = canonical_membership_digest(payload["models"])

    with pytest.raises(ValueError, match="immutable expected digest"):
        validate_canonical_ledger(payload)


def test_each_row_uses_the_exact_flat_contract_and_resolved_spec_flags_are_scoped():
    payload = load_canonical_ledger()
    assert tuple(payload["required_model_fields"]) == REQUIRED_MODEL_FIELDS
    resolved_ids = _resolved_specification_ids()
    assert len(resolved_ids) == 236

    for model in payload["models"]:
        assert set(model) == set(REQUIRED_MODEL_FIELDS) | {"canonical_rank"}
        assert model["historical_model_ids"] == [model["public_model_id"]]
        assert model["identity_recovered"] is True
        assert model["specification_recovered"] is (model["public_model_id"] in resolved_ids)
        assert model["source_reference_verified"] is True
        executable = registration(model["public_model_id"]).factory is not None
        assert model["implementation_available"] is True
        assert model["instantiation_validated"] is executable
        assert model["forecast_smoke_tested"] is executable
        assert model["source_parity_checked"] is executable
        assert model["historical_score_verified"] is (model["historical_rank"] is None
                and model["historical_score"]["exact_empirical_crps"] is not None)
        assert model["protocol_fingerprint"] == payload["identity_policy"]["protocol_fingerprint"]
        assert model["dataset_fingerprint"] == payload["identity_policy"]["dataset_fingerprint"]
        assert model["panel_fingerprint"] == payload["identity_policy"]["panel_fingerprint"]
        assert model["implementation_factory"]["callable"] is executable
        if executable:
            assert build_model(model["public_model_id"]).model_id == model["public_model_id"]
        else:
            assert model["implementation_factory"]["name"] is None

    assert (
        payload["identity_policy"]["full_statistical_specifications_confirmed"]
        == len(resolved_ids)
    )
    assert len(resolved_ids) == sum(
        row["specification_recovered"] for row in payload["models"]
    )
    artifact = payload["score_evidence"]["retained_score_artifact"]
    assert (
        artifact["observed_sha256"]
        == "1244270ed1e637f55d776efab2d1c3ad8f498d63cac16fc808b22cf4343d061a"
    )
    assert (
        artifact["declared_sha256"]
        == "1d1a7cbaa2a990a43a67fc1b73640aeec877c05730e49d43931983b03cdedcc4"
    )
    assert artifact["observed_sha256"] != artifact["declared_sha256"]
    assert artifact["digest_status"] == "mismatch_observed_vs_declared"
    assert "/Users/" not in json.dumps(payload)
    assert "/tmp/" not in json.dumps(payload)
    assert "full_current_catalog" not in json.dumps(payload)


def test_retained_lexical_scores_and_public_precision_reconcile():
    models = [m for m in load_canonical_models()
              if m["historical_score"]["exact_empirical_crps"] is not None]
    assert all(
        isinstance(model["historical_score"]["exact_empirical_crps"], str) for model in models
    )
    assert any(
        model["score_precision"]["retained_source_token"]
        != model["score_precision"]["publication_token"]
        for model in models
    )

    for model in models:
        retained = model["historical_score"]["exact_empirical_crps"]
        publication = model["score_precision"]["publication_token"]
        assert format(float(Decimal(retained)), ".9g") == publication
        assert model["score_precision"]["retained_source_token"] == retained
        assert model["score_precision"]["retained_source_precision"]["kind"] == (
            "serialized_source_json_numeric_token"
        )


def test_compatibility_loader_reads_the_same_ledger_rows():
    payload = load_canonical_ledger()
    rows = load_canonical_175()
    assert len(rows) == len(payload["models"]) == 236
    for row, model in zip(rows, payload["models"]):
        assert row.canonical_rank == model["canonical_rank"]
        assert row.source_rank == model["historical_rank"]
        assert row.model_id == model["public_model_id"]
        assert row.cells == payload["score_evidence"]["cells_per_model"]
        assert row.exact_empirical_crps_text == model["historical_score"]["exact_empirical_crps"]
        assert row.publication_empirical_crps_text == model["score_precision"]["publication_token"]


def test_packaged_resource_load_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = load_canonical_models()
    assert len(models) == 236
    assert canonical_model(models[0]["public_model_id"])["canonical_rank"] == 1


def test_registry_is_fail_closed_for_known_and_unknown_ids(monkeypatch):
    from simfolio_forecasting_methodology.models import registry

    first_id = load_canonical_models()[0]["public_model_id"]
    monkeypatch.delitem(registry._EXPLICIT_FACTORIES, first_id)
    item = registration(first_id)
    assert item.model_id == first_id
    assert item.fidelity == canonical_model(first_id)["verification_status"]
    assert item.implementation == "blocked"
    assert item.factory is None
    with pytest.raises(ValueError, match="not executable: no verified explicit factory"):
        build_model(first_id)

    for unknown_id in ("frontier", "frontier_model", "definitely-not-a-canonical-id"):
        with pytest.raises(ValueError, match="unknown canonical model ID"):
            build_model(unknown_id)


def test_future_verified_specification_record_can_validate_without_factory_claim():
    payload = deepcopy(load_canonical_ledger())
    model = payload["models"][0]
    was_recovered = model["specification_recovered"]
    model["specification_recovered"] = True
    model["structured_specification"]["verified_numerical_defaults"] = "source-backed-test-fixture"
    model["specification_fingerprint"] = {
        "value": "a" * 64,
        "kind": "verified_full_statistical_kernel",
        "status": "verified",
    }
    model["implementation_available"] = True
    model["source_artifact_digest"]["source_code"] = {
        "root_reference": "source_provenance.source_code",
        "sha256": "b" * 64,
        "status": "verified",
    }
    model["protocol_fingerprint"] = "c" * 64
    model["dataset_fingerprint"] = "d" * 64
    model["panel_fingerprint"] = "e" * 64
    model["verification_status"] = "specification_evidence_verified"
    payload["identity_policy"]["full_statistical_specifications_confirmed"] += int(not was_recovered)
    validate_canonical_ledger(payload)


def test_partially_verified_record_can_remain_blocked():
    payload = deepcopy(load_canonical_ledger())
    model = payload["models"][0]
    was_recovered = model["specification_recovered"]
    model["specification_recovered"] = True
    model["specification_fingerprint"] = {
        "value": "a" * 64,
        "kind": "verified_partial_statistical_kernel",
        "status": "partially_verified",
    }
    model["verification_status"] = "specification_partially_verified_blocked"
    payload["identity_policy"]["full_statistical_specifications_confirmed"] += int(not was_recovered)
    validate_canonical_ledger(payload)


def test_source_parity_requires_implementation_smoke_and_source_but_not_score_reproduction():
    payload = deepcopy(load_canonical_ledger())
    model = payload["models"][0]
    factory_name = "simfolio_forecasting_methodology.future:factory"
    model["implementation_factory"] = {
        "name": factory_name,
        "status": "verified_explicit_factory",
        "callable": True,
    }
    payload["implementation_factory_map"][model["public_model_id"]] = {
        "factory_name": factory_name,
        "status": "verified_explicit_factory",
    }
    model["implementation_available"] = True
    model["source_artifact_digest"]["source_code"] = {
        "root_reference": "source_provenance.source_code",
        "sha256": "b" * 64,
        "status": "verified",
    }
    model["instantiation_validated"] = True
    model["forecast_smoke_tested"] = True
    model["source_parity_checked"] = True
    model["verification_status"] = "source_parity_verified_blocked"
    assert model["historical_score_verified"] is False
    validate_canonical_ledger(payload)


def test_factory_claim_without_explicit_map_is_rejected():
    payload = deepcopy(load_canonical_ledger())
    payload["implementation_factory_map"].pop(payload["models"][1]["public_model_id"], None)
    payload["models"][1]["implementation_factory"] = {
        "name": "simfolio_forecasting_methodology.models.future:factory",
        "status": "verified_explicit_factory",
        "callable": True,
    }
    with pytest.raises(ValueError, match="requires an explicit implementation map entry"):
        validate_canonical_ledger(payload)


def test_generated_reference_is_current_and_scoped_to_the_ledger():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/generate_model_reference.py", "--check"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    reference = (repository_root / "docs/canonical-model-reference.md").read_text(encoding="utf-8")
    assert reference.count("retained_score_evidence_only_blocked") == 0
    assert EXPECTED_MEMBERSHIP_DIGEST in reference
    assert "master_369" not in reference


def test_validated_asset_level_additions_include_all_frontier_generations():
    ledger = load_canonical_ledger()
    by_id = {row["public_model_id"]: row for row in ledger["models"]}
    assert "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced" in by_id
    expected = {
        "sv_parameter_mcmc_twochain_sixteen_node_moment_mixture": "0.25246784959071183",
        "experimental_filtered_innovation_moment_sv_fixed_mean": "0.25439860867855635",
        "experimental_filtered_innovation_moment_sv_dlm": "0.2547972662964723",
        "experimental_gaussian_moment_matched_sv_empirical_fixed_mean": "0.2588375710788695",
    }
    for model_id, score in expected.items():
        row = by_id[model_id]
        assert row["historical_score"]["exact_empirical_crps"] == score
        assert row["historical_rank"] is None
        assert row["historical_score_verified"] is True
        assert row["structured_specification"]["resolved_definition"]["forecast_level"] == "asset_daily_log_return"
        evidence = row["source_artifact_digest"]["retained_score_artifact"]
        assert evidence["tasks_per_model"] == 4080
        assert evidence["cells_per_model"] == 701280
        assert evidence["simulations"] == 240
    current = by_id["sv_parameter_mcmc_twochain_sixteen_node_moment_mixture"]
    sampler = current["structured_specification"]["resolved_definition"]["parameter_mcmc"]
    assert sampler["chains"] == 2
    assert sampler["retained_draws"] == 4096
    assert sampler["posterior_nodes"] == 16


def test_partial_candidates_keep_full_and_unfinished_portfolio_scores_blank():
    payload = load_canonical_ledger()
    partial = [m for m in payload['models'] if m['verification_status'].startswith('partial_canonical_score_')]
    for model in partial:
        assert model['historical_score']['exact_empirical_crps'] is None
        assert model['historical_score_verified'] is False
        rows = model['historical_score']['portfolio_scores']
        assert len(rows) == 80
        completed, origins = ((3, 180) if model['public_model_id'] ==
                             'asset_rough_volterra_sv_accuracy_lift_bayesian' else (53, 2769))
        assert sum(r['exact_empirical_crps'] is not None for r in rows) == completed
        assert sum(r['completed_origins'] for r in rows) == origins
    altered = deepcopy(payload)
    rows = next(m for m in altered['models'] if m['public_model_id'] ==
                'asset_rough_volterra_sv_accuracy_lift_bayesian')['historical_score']['portfolio_scores']
    next(r for r in rows if r['completed_origins'] < 51)['exact_empirical_crps'] = 0.1
    with pytest.raises(ValueError, match='unfinished portfolio score must be blank'):
        validate_canonical_ledger(altered)
    altered = deepcopy(payload)
    next(m for m in altered['models'] if m['public_model_id'] ==
         'asset_rough_volterra_sv_accuracy_lift_bayesian')['historical_score_verified'] = True
    with pytest.raises(ValueError, match='blank score requires'):
        validate_canonical_ledger(altered)


def test_rough_candidate_completed_panel_score_and_portfolio_coverage():
    row = canonical_model('asset_rough_volterra_sv_eight_factor')
    assert row['historical_score_verified'] is True
    assert row['historical_score']['exact_empirical_crps'] == '0.2511998559613309'
    assert len(row['historical_score']['portfolio_scores']) == 80
    assert all(r['completed_origins'] == 51 for r in row['historical_score']['portfolio_scores'])
    assert row['source_artifact_digest']['retained_score_artifact']['all_denominator_gates_passed'] is True


def test_new_bayesian_candidates_are_wired_without_fabricated_scores():
    payload = load_canonical_ledger()
    candidates = [m for m in payload['models']
                  if m['verification_status'].startswith('unscored_canonical_candidate_')]
    assert len(candidates) == 6
    for model in candidates:
        definition = model['structured_specification']['resolved_definition']
        assert definition['forecast_level'] == 'asset_daily_log_return'
        assert definition['parameter_mcmc']['kept_per_chain'] == 8192
        assert definition['volatility']['production_variance_anchor'] is (definition['family'] == 'asset_level_rough_volterra_overlay_upgrade')
        assert model['historical_score']['exact_empirical_crps'] is None
        assert all(row['completed_origins'] == 0 and row['exact_empirical_crps'] is None
                   for row in model['historical_score']['portfolio_scores'])
        assert model['historical_score_verified'] is False
        assert model['source_artifact_digest']['retained_score_artifact']['scope'] == 'unscored_canonical'


def test_dynamic_rough_completed_panel_score_and_diagnostics_are_retained():
    row = canonical_model('asset_rough_volterra_sv_dynamic_lift_bayesian')
    assert row['historical_score_verified'] is True
    assert row['historical_score']['exact_empirical_crps'] == '0.25074679212444156'
    assert len(row['historical_score']['portfolio_scores']) == 80
    assert all(p['completed_origins'] == 51 for p in row['historical_score']['portfolio_scores'])
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / 'docs/results/rough-bayesian/dynamic-full-audit.json').read_text())
    assert audit['all_cell_vectors_independently_reconstructed_byte_exact'] is True
    assert audit['kernel_validation']['all_daily_lag_certificates_passed'] is True
    assert len(audit['kernel_validation']['flagged_asset_fits']) == 6
