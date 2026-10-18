"""Publication evidence for the full-path mixture on the unchanged canonical80 block."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from simfolio_forecasting_methodology.catalogue import canonical_model, load_canonical_ledger

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "equal_path_pool_m261_and_bayesian_ewma_sbb"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pool_append_preserves_every_prior_score_token_rank_and_precision():
    models = load_canonical_ledger()["models"]
    prior = [
        (row["public_model_id"], row["canonical_rank"], row["historical_rank"],
         row["historical_score"], row["score_precision"])
        for row in models[:261]
    ]
    assert _digest(prior) == "3ccd48f0332f8226b693db36aa33a0a5927d83207e56b731de55884386aa9950"
    assert models[261]["public_model_id"] == MODEL_ID
    assert models[261]["canonical_rank"] == 262
    assert models[261]["historical_rank"] is None


def test_pool_canonical_score_has_all_original80_cells_and_manifest_bound_source():
    row = canonical_model(MODEL_ID)
    artifact = row["source_artifact_digest"]["retained_score_artifact"]
    result_path = ROOT / artifact["relative_path"]
    assert _file_digest(result_path) == artifact["sha256"]
    result = json.loads(result_path.read_text())
    assert (result["block"], result["portfolios"], result["tasks"], result["cells"]) == (
        "original80", 80, 4080, 701280
    )
    assert result["all_denominator_gates_passed"] is True
    assert row["historical_score"]["exact_empirical_crps"] == "0.2201357869781349"
    assert repr(result["exact_empirical_crps"]) == row["historical_score"]["exact_empirical_crps"]
    cells_path = ROOT / artifact["paired_cells"]["relative_path"]
    assert _file_digest(cells_path) == artifact["paired_cells"]["sha256"] == result["cells_sha256"]
    with np.load(cells_path) as cells:
        assert cells["losses"].shape == cells["counts"].shape == (80, 8766)
        assert np.isfinite(cells["losses"]).all()
        assert (cells["counts"] > 0).all()
        assert len(set(cells["portfolio"])) == 80
        assert float(cells["losses"].mean()) == result["exact_empirical_crps"]
        assert dict(zip(cells["portfolio"], cells["losses"].mean(axis=1).tolist())) == result["portfolio_scores"]
    definition = row["structured_specification"]["resolved_definition"]
    source = definition["source_reference"]
    binding = source["tested_source_binding"]
    manifest_path = ROOT / binding["manifest_path"]
    assert _file_digest(manifest_path) == binding["manifest_file_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert binding["manifest_sha256"] == result["manifest_sha256"]
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == result["manifest_sha256"]
    assert manifest["source_hashes"] == binding["recorded_source_hashes"]
    assert manifest["requested_model_id"] == MODEL_ID
    assert manifest["simulations"] == 240
    assert len(manifest["task_identities"]) == 4080
    assert manifest["source_revision"] == binding["manifest_base_revision"]
    assert row["source_revision"] == source["revision"] == binding["implementation_revision"] == (
        "574e1b34d478f74a0397905ac3ddbd558e1c118f"
    )
    assert binding["manifest_base_revision"] != binding["implementation_revision"]
    assert _file_digest(ROOT / source["path"]) == source["sha256"] == manifest["source_hashes"][source["path"]]
    audit_path = ROOT / artifact["independent_audit"]["relative_path"]
    assert _file_digest(audit_path) == artifact["independent_audit"]["sha256"]
    audit = json.loads(audit_path.read_text())
    assert audit["passed"] is True
    assert audit["blocks"]["original80"]["independent_canonical_accumulator_all_cells_and_counts_byte_equal"] is True
    assert audit["blocks"]["original80"]["exact_empirical_crps"] == result["exact_empirical_crps"]
    assert all(case["component_120_path_halves_byte_equal"] and case["loss_replay_byte_equal"] for case in audit["replays"])


def test_pool_spec_links_complete_component_laws_and_records_new_120_node_law():
    row = canonical_model(MODEL_ID)
    resource = json.loads((ROOT / "src/simfolio_forecasting_methodology/resources/specifications/canonical_statistical_specifications.json").read_text())
    definition = resource["resolved_definitions"][MODEL_ID]
    assert row["structured_specification"]["resolved_definition"] == definition
    assert _digest(definition) == row["specification_fingerprint"]["value"]
    assert definition["mixture"]["simulations"] == 240
    assert definition["mixture"]["component_simulations"] == 120
    assert not any(definition["mixture"][field] for field in (
        "daily_return_averaging", "parameter_averaging", "score_averaging", "fitted_weights"
    ))
    assert [component["model_id"] for component in definition["component_definitions"]] == [
        "asset_m256_data_timescales_existing_sv_rate", "bayesian_sbb_overlay_ewma_absolute"
    ]
    for component in definition["component_definitions"]:
        component_id = component["model_id"]
        full = resource["resolved_definitions"][component_id]
        assert component["resolved_definition_ref"] == f"resolved_definitions.{component_id}"
        assert component["definition_fingerprint"] == _digest(full)
        assert component["weight"] == 0.5 and component["simulations"] == 120
        assert component["factory_seed_contract"] == full["factory_seed_contract"]
    assert "120 nodes" in definition["fit_contract"]["empirical_quantile_grid"]
    assert "training-only" in definition["fit_contract"]["observations"]


def test_ou_pool_append_preserves_m262_and_all_prior_metadata():
    models = load_canonical_ledger()["models"]
    assert _digest(models[:262]) == "e6be763a3f41f76b50faee49c7c867008edd9835ed8e729d1cb800b7ef7805f7"
    row = models[262]
    assert row["public_model_id"] == "equal_path_pool_stationary_ou_and_bayesian_ewma_sbb"
    assert row["canonical_rank"] == 263 and row["historical_rank"] is None
    assert row["display_name"] == "Equal-weight stationary OU and Bayesian EWMA forecast mixture"
    assert row["historical_score"]["exact_empirical_crps"] == "0.22745132492949452"
    assert row["source_revision"] == "3aee75cab77d152b3311acf5da061dc07d339082"
    artifact = row["source_artifact_digest"]["retained_score_artifact"]
    result_path = ROOT / artifact["relative_path"]
    assert _file_digest(result_path) == artifact["sha256"]
    result = json.loads(result_path.read_text())
    assert (result["block"], result["portfolios"], result["tasks"], result["cells"]) == (
        "original80", 80, 4080, 701280
    )
    assert result["all_denominator_gates_passed"] is True
    cells_path = ROOT / artifact["paired_cells"]["relative_path"]
    assert _file_digest(cells_path) == result["cells_sha256"] == artifact["paired_cells"]["sha256"]
    with np.load(cells_path) as cells:
        assert cells["losses"].shape == cells["counts"].shape == (80, 8766)
        assert (cells["counts"] > 0).all() and np.isfinite(cells["losses"]).all()
        assert repr(float(cells["losses"].mean())) == row["historical_score"]["exact_empirical_crps"]
        assert dict(zip(cells["portfolio"], cells["losses"].mean(axis=1).tolist())) == result["portfolio_scores"]
    definition = row["structured_specification"]["resolved_definition"]
    source = definition["source_reference"]
    binding = source["tested_source_binding"]
    manifest_path = ROOT / binding["manifest_path"]
    assert _file_digest(manifest_path) == binding["manifest_file_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == (
        result["manifest_sha256"]
    ) == binding["manifest_sha256"]
    assert manifest["source_hashes"] == binding["recorded_source_hashes"]
    assert _file_digest(ROOT / source["path"]) == source["sha256"] == manifest["source_hashes"][source["path"]]
    assert len(manifest["task_identities"]) == 4080 and manifest["simulations"] == 240
    audit_path = ROOT / artifact["independent_audit"]["relative_path"]
    assert _file_digest(audit_path) == artifact["independent_audit"]["sha256"]
    audit = json.loads(audit_path.read_text())
    assert all(audit[key] for key in (
        "all_checks_passed", "independent_reaggregation_exact", "all_horizons_aligned",
        "all_losses_finite", "all_accepted_ou_optimizers_successful", "source_hash_guard_verified"
    ))
    assert audit["actual_execution_git_sha"] == source["revision"] == row["source_revision"]
    assert audit["ou_diagnostic_records"] == 24480 and audit["component_paths_each"] == 120
    resource = json.loads((ROOT / "src/simfolio_forecasting_methodology/resources/specifications/canonical_statistical_specifications.json").read_text())
    assert [component["model_id"] for component in definition["component_definitions"]] == [
        "asset_map_stationary_ar1_ou_residual_volatility", "bayesian_sbb_overlay_ewma_absolute"
    ]
    for component in definition["component_definitions"]:
        assert component["definition_fingerprint"] == _digest(resource["resolved_definitions"][component["model_id"]])
        assert component["weight"] == 0.5 and component["simulations"] == 120
    assert _digest(definition) == row["specification_fingerprint"]["value"]
