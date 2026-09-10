from __future__ import annotations

import json
from importlib.resources import files

from simfolio_forecasting_methodology.models.portfolio import (
    bayesian_vol,
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


def test_portfolio_batch_is_exactly_34_and_keeps_raw_descriptors_separate():
    resource = _load_resource()
    expected = (
        set(bayesian_vol.BAYESIAN_VOL_MODEL_IDS)
        | set(factor_residual.FACTOR_RESIDUAL_MODEL_IDS)
        | set(reference_families.REFERENCE_MODEL_IDS)
        | {"stochastic_volatility_ar1_empirical", "stochastic_volatility_ar1_empirical_sbb"}
        | set(sv_extensions.REFERENCE_MODEL_IDS)
        | set(sv_mcmc_reference.REFERENCE_MODEL_IDS)
        | set(gas_reference.REFERENCE_MODEL_IDS)
        | set(gjr_reference.REFERENCE_MODEL_IDS)
    )
    assert len(expected) == 34
    assert set(resource["portfolio_model_ids"]) == expected
    assert expected <= set(resource["resolved_definitions"])
    assert expected <= set(resource["accepted_model_ids"])
    assert "stochastic_volatility_ar1_student_t" not in expected

    raw_factor = {
        model_id: resource["resolved_definitions"][model_id]["source_candidate"]
        for model_id in factor_residual.FACTOR_RESIDUAL_MODEL_IDS
    }
    assert all(set(candidate) == {"id", "type"} for candidate in raw_factor.values())
    for model_id in expected:
        definition = resource["resolved_definitions"][model_id]
        assert definition["source_candidate"]["id"] == model_id
        assert set(definition["source_candidate"]) <= set(definition["resolved_candidate"])
        assert definition["source_candidate"] != definition["resolved_candidate"]
        assert definition["source_seed_descriptor"]["source_model_key"] == model_id
        assert definition["factory"]["exact_id_dispatch"] is True


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
        assert "portfolio.contract" in refs
        assert "portfolio.seed_identity" in refs
        assert definition["source_component_ref"] in refs
        assert set(definition["shared_component_digests"]) == set(refs)
        assert all(len(value) == 64 for value in definition["shared_component_digests"].values())
