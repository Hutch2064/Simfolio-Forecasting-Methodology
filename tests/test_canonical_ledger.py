from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from decimal import Decimal
from importlib.resources import files
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
    resource = files("simfolio_forecasting_methodology").joinpath(
        "resources/specifications/canonical_statistical_specifications.json"
    )
    return set(json.loads(resource.read_text(encoding="utf-8"))["accepted_model_ids"])


def test_membership_is_exact_and_digest_is_immutable():
    payload = load_canonical_ledger()
    models = list(load_canonical_models())

    assert payload["membership"]["count"] == EXPECTED_CANONICAL_COUNT == 175
    assert [model["canonical_rank"] for model in models] == list(range(1, 176))
    assert [model["historical_rank"] for model in models] == list(EXPECTED_SOURCE_RANKS)
    assert canonical_membership_digest(models) == EXPECTED_MEMBERSHIP_DIGEST
    assert payload["membership"]["membership_digest"] == EXPECTED_MEMBERSHIP_DIGEST


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

    for model in payload["models"]:
        assert set(model) == set(REQUIRED_MODEL_FIELDS) | {"canonical_rank"}
        assert model["historical_model_ids"] == [model["public_model_id"]]
        assert model["identity_recovered"] is True
        assert model["specification_recovered"] is (model["public_model_id"] in resolved_ids)
        assert model["source_reference_verified"] is True
        executable = (model["canonical_rank"] == 1 or model["model_family"] == "base"
                      or (model["model_family"] == "bayesian_sbb_full_mcmc_sv_overlay")
                      or model["public_model_id"] in {'gas_score_driven_skewt', 'bayesian_mcmc_stochastic_volatility_sbb', 'dp_mixture_sv_sbb', 'observable_markov_state_sbb', 'stochastic_volatility_ar1_empirical', 'stochastic_volatility_ar1_empirical_sbb', 'zero_mean_gaussian_vol_only', 'constant_mean_gaussian', 'naive_iid_historical_portfolio_bootstrap', 'constant_mean_student_t'})
        assert model["implementation_available"] is executable
        assert model["instantiation_validated"] is executable
        assert model["forecast_smoke_tested"] is executable
        assert model["source_parity_checked"] is executable
        assert model["historical_score_verified"] is False
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
        == 125
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
    models = load_canonical_models()
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
    assert len(rows) == len(payload["models"]) == 175
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
    assert len(models) == 175
    assert canonical_model(models[0]["public_model_id"])["canonical_rank"] == 1


def test_registry_is_fail_closed_for_known_and_unknown_ids():
    first_id = load_canonical_models()[1]["public_model_id"]
    item = registration(first_id)
    assert item.model_id == first_id
    assert item.fidelity == "retained_score_evidence_only_blocked"
    assert item.implementation == "blocked"
    assert item.factory is None
    with pytest.raises(ValueError, match="not executable: no verified explicit factory"):
        build_model(first_id)

    for unknown_id in ("frontier", "frontier_model", "definitely-not-a-canonical-id"):
        with pytest.raises(ValueError, match="unknown canonical model ID"):
            build_model(unknown_id)


def test_future_verified_specification_record_can_validate_without_factory_claim():
    payload = deepcopy(load_canonical_ledger())
    model = next(item for item in payload["models"] if not item["specification_recovered"])
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
    payload["identity_policy"]["full_statistical_specifications_confirmed"] += 1
    validate_canonical_ledger(payload)


def test_partially_verified_record_can_remain_blocked():
    payload = deepcopy(load_canonical_ledger())
    model = next(item for item in payload["models"] if not item["specification_recovered"])
    model["specification_recovered"] = True
    model["specification_fingerprint"] = {
        "value": "a" * 64,
        "kind": "verified_partial_statistical_kernel",
        "status": "partially_verified",
    }
    model["verification_status"] = "specification_partially_verified_blocked"
    payload["identity_policy"]["full_statistical_specifications_confirmed"] += 1
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
    assert reference.count("retained_score_evidence_only_blocked") == 40
    assert EXPECTED_MEMBERSHIP_DIGEST in reference
    assert "master_369" not in reference
