from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from importlib.resources import files

from simfolio_forecasting_methodology.catalogue import load_canonical_ledger

RESOURCE = "resources/specifications/canonical_statistical_specifications.json"
MCMC_MANIFEST = "resources/catalogs/canonical_40_full_mcmc_sv_specs.json"


def _load_json(resource: str) -> dict:
    return json.loads(files("simfolio_forecasting_methodology").joinpath(resource).read_text())


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_resolved_spec_scope_and_family_membership_are_exact():
    ledger = load_canonical_ledger()
    resource = _load_json(RESOURCE)
    model_ids = {row["public_model_id"] for row in ledger["models"]}
    expected_base = {
        row["public_model_id"] for row in ledger["models"] if row["model_family"] == "base"
    }
    expected_mcmc = {
        row["public_model_id"]
        for row in ledger["models"]
        if row["model_family"] == "bayesian_sbb_full_mcmc_sv_overlay"
    }
    frontier_id = "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced"
    expected = expected_base | expected_mcmc | {frontier_id}

    assert len(expected_base) == 84
    assert len(expected_mcmc) == 40
    assert len(resource["accepted_model_ids"]) == 125
    assert set(resource["accepted_model_ids"]) == expected
    assert set(resource["resolved_definitions"]) == expected
    assert set(resource["bindings"]) == expected
    assert set(resource["resolved_definitions"]) <= model_ids


def test_each_resolved_definition_has_a_stable_full_fingerprint_and_source_identity():
    ledger = load_canonical_ledger()
    resource = _load_json(RESOURCE)
    rows = {row["public_model_id"]: row for row in ledger["models"]}

    assert resource["shared_components"]["seed_identity"]["base_and_full_mcmc"][
        "horizon_values"
    ] == ("dense daily tuple 1..H, inclusive")
    assert resource["shared_components"]["seed_identity"]["frontier"]["dependence"]["alias"] == (
        "copula_alternatives"
    )
    for model_id, definition in resource["resolved_definitions"].items():
        binding = resource["bindings"][model_id]
        row = rows[model_id]
        assert binding["definition_fingerprint"] == _digest(definition)
        assert row["specification_fingerprint"]["value"] == _digest(definition)
        assert row["specification_fingerprint"]["kind"] == "resolved_statistical_definition_sha256"
        assert row["structured_specification"]["resolved_definition"] == definition
        source = definition["source_reference"]
        assert not source["path"].startswith(("/", "~"))
        assert len(source["revision"]) == 40
        assert len(source["sha256"]) == 64
        assert definition["factory_seed_contract"] == (
            "origin_task.seed_to_forecast_context.seed.v1"
        )


def test_definition_digest_covers_nested_parameterization():
    resource = _load_json(RESOURCE)
    model_id = resource["accepted_model_ids"][0]
    definition = deepcopy(resource["resolved_definitions"][model_id])
    original = _digest(definition)
    if definition["family"] == "frontier":
        definition["marginal"]["candidate"]["bdes_multiscale_k_star"] += 1
    else:
        definition["failure_semantics"]["mutation_test"] = True
    assert _digest(definition) != original


def test_mcmc_uses_exact_source_descriptor_and_separate_resolved_defaults():
    resource = _load_json(RESOURCE)
    manifest = _load_json(MCMC_MANIFEST)
    manifest_by_id = {entry["id"]: entry for entry in manifest["entries"]}
    mcmc = {
        model_id: definition
        for model_id, definition in resource["resolved_definitions"].items()
        if definition["family"] == "full_mcmc_sv"
    }

    assert len(manifest_by_id) == 40
    assert (
        "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_full_inla_laplace_quadrature_centered_multiscale"
        not in manifest_by_id
    )
    assert (
        "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_optimal_block_adaptive_metropolis_proposal_mcmc"
        in manifest_by_id
    )
    for model_id, definition in mcmc.items():
        source_candidate = definition["source_candidate"]
        assert source_candidate == manifest_by_id[model_id]
        assert set(source_candidate) <= set(definition["resolved_candidate"])
        assert set(definition["resolved_candidate"]) - set(source_candidate)
        assert definition["source_seed_context_ref"] == "seed_identity.base_and_full_mcmc"


def test_unaccepted_rows_remain_unresolved():
    ledger = load_canonical_ledger()
    accepted = set(_load_json(RESOURCE)["accepted_model_ids"])
    for row in ledger["models"]:
        if row["public_model_id"] in accepted:
            assert row["specification_recovered"] is True
        else:
            assert row["specification_recovered"] is False
            fingerprint = row["specification_fingerprint"]
            assert (
                fingerprint is None
                or fingerprint["kind"] != "resolved_statistical_definition_sha256"
            )
