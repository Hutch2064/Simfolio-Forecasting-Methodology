"""Bounded source-to-extracted parity for the 20 Bayesian SBB overlays."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.portfolio.bayesian_vol import (
    BAYESIAN_VOL_MODEL_IDS,
    RAW_CANDIDATE_SPECS,
    RAW_SEED_DESCRIPTORS,
    RESOLVED_STATISTICAL_SPECS,
    SOURCE_CANDIDATE_SPECS,
    SOURCE_FUNCTIONS_SHA256,
    BayesianVolOverlayModel,
    make_bayesian_vol_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

_FIXTURE_RESOURCE = resources.files("simfolio_forecasting_methodology").joinpath(
    "resources", "test_fixtures", "portfolio", "bayesian_vol_source_parity.json"
)


def _load_fixture() -> dict[str, Any]:
    with _FIXTURE_RESOURCE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _array_record(values: Any) -> dict[str, Any] | None:
    if values is None:
        return None
    array = np.ascontiguousarray(np.asarray(values))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _fit_summary(fit: dict[str, Any]) -> dict[str, Any]:
    base = dict(fit.get("base_fit", {}) or {})
    curve = dict(fit.get("curve_fit", {}) or {})
    meta = dict(fit.get("meta", {}) or {})
    base_keys = (
        "sample_mu",
        "sigma",
        "posterior_mean",
        "posterior_sd",
        "mu_cap",
        "sample_mu_days",
        "nonnegative_drift",
        "posterior_mu_draws",
    )
    curve_keys = (
        "curve_type",
        "persistence",
        "last_variance_x",
        "long_variance_x",
        "last_log_variance_x",
        "long_log_variance_x",
        "last_log_variance_error_variance",
        "fit_status",
        "objective",
    )
    meta_keys = (
        "method",
        "overlay_model",
        "overlay_persistence",
        "overlay_half_life_days",
        "overlay_fit_status",
        "ml_model",
        "feature_set",
        "feature_count",
        "training_rows",
    )
    summary: dict[str, Any] = {
        "base": {key: base[key] for key in base_keys if key in base},
        "curve": {key: curve[key] for key in curve_keys if key in curve},
        "meta": {key: meta[key] for key in meta_keys if key in meta},
        "standardized_residuals": _array_record(base.get("standardized_residuals")),
    }
    if "arch_params" in curve:
        summary["arch_params"] = {
            str(key): float(value) for key, value in dict(curve["arch_params"]).items()
        }
    if "sigma_clip_bounds_x" in curve:
        summary["sigma_clip_bounds_x"] = [float(value) for value in curve["sigma_clip_bounds_x"]]
    if "vol_anchor" in fit:
        anchor = dict(fit["vol_anchor"])
        summary["vol_anchor"] = {
            key: anchor[key]
            for key in (
                "model",
                "target",
                "anchor_log_variance_x",
                "anchor_log_variance_error_variance",
                "predicted_sigma_x",
                "training_rows",
                "feature_count",
                "factor_model",
                "factor_cols",
                "anchor_weight",
                "pre_anchor_log_variance_x",
                "pre_anchor_log_variance_error_variance",
            )
            if key in anchor
        }
    return summary


def _assert_summary_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for section in ("base", "curve", "meta"):
        assert actual[section].keys() == expected[section].keys()
        for key, expected_value in expected[section].items():
            actual_value = actual[section][key]
            if isinstance(expected_value, (bool, str, int)):
                assert actual_value == expected_value, (section, key)
            else:
                assert actual_value == pytest.approx(expected_value, rel=2e-10, abs=2e-12), (section, key)
    assert actual["standardized_residuals"] == expected["standardized_residuals"]
    for key in ("arch_params", "sigma_clip_bounds_x", "vol_anchor"):
        if key in expected:
            assert key in actual
            if key == "arch_params":
                assert actual[key].keys() == expected[key].keys()
                for name, expected_value in expected[key].items():
                    assert actual[key][name] == pytest.approx(expected_value, rel=2e-10, abs=2e-12)
            elif key == "sigma_clip_bounds_x":
                np.testing.assert_allclose(actual[key], expected[key], rtol=2e-10, atol=2e-12)
            else:
                assert actual[key].keys() == expected[key].keys()
                for name, expected_value in expected[key].items():
                    if isinstance(expected_value, (str, int, list)):
                        assert actual[key][name] == expected_value
                    else:
                        assert actual[key][name] == pytest.approx(
                            expected_value, rel=2e-10, abs=2e-12
                        )


def test_exact_bayesian_vol_factory_domain_and_unknown_rejection() -> None:
    fixture = _load_fixture()
    assert len(BAYESIAN_VOL_MODEL_IDS) == 20
    assert set(BAYESIAN_VOL_MODEL_IDS) == {case["model_id"] for case in fixture["cases"]}
    assert sum("ml_vol_overlay" in SOURCE_CANDIDATE_SPECS[mid]["type"] for mid in BAYESIAN_VOL_MODEL_IDS) == 1
    assert SOURCE_CANDIDATE_SPECS[
        "bayesian_sbb_overlay_gjr_garch_1_1_empirical_bayes_sharpe"
    ]["source_descriptor_status"] == "historical_recovered"
    with pytest.raises(ValueError, match="unknown Bayesian volatility model"):
        BayesianVolOverlayModel("bayesian_sbb_overlay_unknown")
    with pytest.raises(ValueError, match="no source-backed Bayesian volatility factory"):
        make_bayesian_vol_model("bayesian_sbb_overlay_unknown")


def test_raw_seed_descriptors_and_resolved_defaults_are_separate() -> None:
    assert set(RAW_CANDIDATE_SPECS) == set(BAYESIAN_VOL_MODEL_IDS)
    assert all("source_descriptor_status" not in spec for spec in RAW_CANDIDATE_SPECS.values())
    assert set(RAW_SEED_DESCRIPTORS) == set(BAYESIAN_VOL_MODEL_IDS)
    assert all(descriptor["panel_seed"] == 20260528 for descriptor in RAW_SEED_DESCRIPTORS.values())
    assert set(RESOLVED_STATISTICAL_SPECS) == set(BAYESIAN_VOL_MODEL_IDS)
    for model_id in BAYESIAN_VOL_MODEL_IDS:
        resolved = RESOLVED_STATISTICAL_SPECS[model_id]
        assert "mean_dispatch" in resolved
        assert "innovation_and_simulation" in resolved
    anchor_id = "bayesian_sbb_overlay_sv_ar1_logvol_bias_corrected_hac_drift_uncertainty_harx_ff6_vol_anchor"
    assert RESOLVED_STATISTICAL_SPECS[anchor_id]["vol_anchor"]["factor_model"] == "ff6"
    assert RAW_CANDIDATE_SPECS[anchor_id]["vol_anchor_model"] == "ridge_harx_ff6"
    assert RAW_CANDIDATE_SPECS[anchor_id]["validation_status"].startswith("clean_rank33")
    assert "vol_anchor" not in RESOLVED_STATISTICAL_SPECS[
        "bayesian_sbb_overlay_sv_ar1_logvol_bias_corrected"
    ]
    assert RESOLVED_STATISTICAL_SPECS[
        "bayesian_sbb_ml_vol_overlay_rf_harx_ff6"
    ]["ml_descriptor"]["factor_model"] == "ff6"


def test_source_function_digest_and_fixture_identity() -> None:
    fixture = _load_fixture()
    assert fixture["schema"] == "bayesian-vol-source-parity-v1"
    assert fixture["source_function_digest"] == SOURCE_FUNCTIONS_SHA256
    assert fixture["source_artifacts"]["revision"] == "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
    assert len(fixture["source_function_names"]) == len(set(fixture["source_function_names"]))
    assert len(fixture["cases"]) == 20


def test_all_20_source_fit_states_and_daily_paths_match_fixture() -> None:
    fixture = _load_fixture()
    training_payload = fixture["training"]
    training = TrainingData(
        portfolio_log_returns=np.asarray(training_payload["values"], dtype=np.float64),
        training_dates=np.asarray(training_payload["dates"], dtype="datetime64[ns]"),
    )
    context_payload = fixture["context"]
    for case in fixture["cases"]:
        model_id = case["model_id"]
        context = ForecastContext(
            model_id=model_id,
            portfolio_id="fixture",
            origin_label=context_payload["origin_date"],
            origin_date=context_payload["origin_date"],
            horizon_days=int(context_payload["horizon_days"]),
            simulations=int(context_payload["simulations"]),
            seed=0,
        )
        model = make_bayesian_vol_model(model_id)
        fit = model.fit(training)
        _assert_summary_equal(_fit_summary(fit), case["fit"])
        paths = model.simulate_daily_log_returns(training, context)
        expected = np.asarray(case["paths"], dtype=np.float64)
        np.testing.assert_allclose(paths, expected, rtol=2e-10, atol=2e-12)
        assert _array_record(paths) == case["paths_array"]


def test_bayesian_vol_adapter_requires_real_dates_for_ml_and_anchor() -> None:
    values = np.linspace(-0.01, 0.01, 720, dtype=np.float64)
    training = TrainingData(values)
    ml = make_bayesian_vol_model("bayesian_sbb_ml_vol_overlay_rf_harx_ff6")
    anchor = make_bayesian_vol_model(
        "bayesian_sbb_overlay_sv_ar1_logvol_bias_corrected_hac_drift_uncertainty_harx_ff6_vol_anchor"
    )
    with pytest.raises(ValueError, match="requires training_dates"):
        ml.fit(training)
    with pytest.raises(ValueError, match="requires training_dates"):
        anchor.fit(training)
