from __future__ import annotations

import json
from importlib.resources import files

from simfolio_forecasting_methodology.models.portfolio import (
    bayesian_vol,
    canonical_stack_reference,
    factor_residual,
    gas_reference,
    gjr_reference,
    reference_families,
    sv_extensions,
    sv_mcmc_reference,
)

RESOURCE = "resources/specifications/canonical_statistical_specifications.json"


def _load_resource() -> dict:
    return json.loads(files("simfolio_forecasting_methodology").joinpath(RESOURCE).read_text())


def test_source_backed_extension_batch_is_exactly_50_and_keeps_raw_descriptors_separate():
    resource = _load_resource()
    expected = (
        set(bayesian_vol.BAYESIAN_VOL_MODEL_IDS)
        | set(factor_residual.FACTOR_RESIDUAL_MODEL_IDS)
        | set(reference_families.REFERENCE_MODEL_IDS)
        | {
            "stochastic_volatility_ar1_empirical",
            "stochastic_volatility_ar1_empirical_sbb",
            "stochastic_volatility_ar1_student_t",
        }
        | set(sv_extensions.REFERENCE_MODEL_IDS)
        | set(sv_mcmc_reference.REFERENCE_MODEL_IDS)
        | set(gas_reference.REFERENCE_MODEL_IDS)
        | set(gjr_reference.REFERENCE_MODEL_IDS)
        | set(canonical_stack_reference.REFERENCE_MODEL_IDS)
        | {
            "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced",
            "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_conditional_sharpe_full_inla_laplace_quadrature_centered_multiscale",
        }
    )
    assert len(expected) == 50
    assert set(resource["portfolio_model_ids"]) == expected
    assert expected <= set(resource["resolved_definitions"])
    assert expected <= set(resource["accepted_model_ids"])
    assert "stochastic_volatility_ar1_student_t" in expected

    raw_factor = {
        model_id: resource["resolved_definitions"][model_id]["source_candidate"]
        for model_id in factor_residual.FACTOR_RESIDUAL_MODEL_IDS
    }
    assert all(set(candidate) == {"id", "type"} for candidate in raw_factor.values())
    for model_id in expected:
        definition = resource["resolved_definitions"][model_id]
        assert definition["source_candidate"]["id"] == model_id
        if "resolved_candidate" in definition:
            resolved_candidate = definition["resolved_candidate"]
            assert set(definition["source_candidate"]) <= set(resolved_candidate)
            assert definition["source_candidate"] != resolved_candidate
        else:
            assert definition.get("resolved_defaults") is not None
            assert definition["source_candidate"] != definition["resolved_defaults"]
        if "source_seed_descriptor" in definition:
            assert definition["source_seed_descriptor"]["source_model_key"] == model_id
        assert definition["factory"].get("exact_id_dispatch", True) is True


def test_portfolio_shared_closures_are_hashed_and_portable():
    resource = _load_resource()
    closures = resource["shared_components"]["portfolio"]["source_closures"]
    assert set(closures) == {
        "bayesian_vol",
        "factor_residual",
        "gas_reference",
        "gjr_reference",
        "reference_families",
        "sv_extensions",
        "sv_mcmc_reference",
        "sv_reference",
        "canonical_stack",
    }
    for closure in closures.values():
        assert len(closure["source_functions_sha256"]) == 64
        assert not closure["module"].startswith(("/", "~"))
        assert not closure["primary_artifact"]["path"].startswith(("/", "~"))
        for artifact in closure["dependency_artifacts"]:
            assert not artifact["path"].startswith(("/", "~"))

    for model_id in resource["portfolio_model_ids"]:
        definition = resource["resolved_definitions"][model_id]
        refs = definition["shared_component_refs"]
        if "source_component_ref" in definition:
            assert "portfolio.contract" in refs
            assert "portfolio.seed_identity" in refs
            assert definition["source_component_ref"] in refs
        else:
            assert any(ref.startswith(("asset_exact.", "inla.")) for ref in refs)
        assert set(definition["shared_component_digests"]) == set(refs)
        assert all(len(value) == 64 for value in definition["shared_component_digests"].values())


def test_canonical_stack_factor_proxy_is_complete_and_frozen():
    resource = _load_resource()
    contract = resource["shared_components"]["portfolio"]["source_closures"][
        "canonical_stack"
    ]["resolved_parameter_contract"]["factor_proxy"]

    assert contract["status"] == "complete_frozen_manifest_whitelisted_source_snapshot"
    assert contract["dispatch"] == "_fit_factor_drift_prior"
    assert contract["expected_factor_count"] == 9
    assert contract["minimum_history"] == 252
    assert contract["manifest"] == {
        "path": "resources/data/canonical_snapshot_manifest.json",
        "sha256": "3c2e8218b03ca7a435d145168c97a2f302be427e862170183078bc4917699c12",
        "validation": "content hash and every manifest series entry are checked before loading",
    }
    assert len(contract["series"]) == contract["expected_factor_count"]
    assert all(len(series["sha256"]) == 64 for series in contract["series"])
