"""Bounded source-to-extracted parity for the two FF6 residual SBB models."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from importlib import resources
from typing import Any

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.numerical import factors as factor_loader
from simfolio_forecasting_methodology.models.portfolio.factor_residual import (
    FACTOR_RESIDUAL_MODEL_IDS,
    RAW_FACTOR_RESIDUAL_SPECS,
    RESOLVED_FACTOR_RESIDUAL_SPECS,
    SOURCE_FUNCTIONS_SHA256,
    FactorResidualSBBModel,
    make_factor_residual_model,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData
from simfolio_forecasting_methodology.seeds import forecast_oos_candidate_seed

_FIXTURE_RESOURCE = resources.files("simfolio_forecasting_methodology").joinpath(
    "resources", "test_fixtures", "portfolio", "factor_residual_source_parity.json"
)
FIT_NUMERIC_RTOL = 0.0
FIT_NUMERIC_ATOL = 2e-12


def _load_fixture() -> dict[str, Any]:
    with _FIXTURE_RESOURCE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _array_record(values: Any, *, include_values: bool = False) -> dict[str, Any] | None:
    if values is None:
        return None
    array = np.ascontiguousarray(np.asarray(values))
    record: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    if include_values:
        record["values"] = array.tolist()
    return record


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _nested_fit_summary(
    fit: Mapping[str, Any] | None, *, include_values: bool = False
) -> dict[str, Any] | None:
    if fit is None:
        return None
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
        "base": {key: _plain(base[key]) for key in base_keys if key in base},
        "curve": {key: _plain(curve[key]) for key in curve_keys if key in curve},
        "meta": {key: _plain(meta[key]) for key in meta_keys if key in meta},
        "standardized_residuals": _array_record(
            base.get("standardized_residuals"), include_values=include_values
        ),
    }
    if "sigma_x" in curve:
        summary["curve_sigma_x"] = _array_record(
            curve.get("sigma_x"), include_values=include_values
        )
    if "arch_params" in curve:
        summary["arch_params"] = {
            str(key): float(value) for key, value in dict(curve["arch_params"]).items()
        }
    if "sigma_clip_bounds_x" in curve:
        summary["sigma_clip_bounds_x"] = [float(value) for value in curve["sigma_clip_bounds_x"]]
    return summary


def _fit_summary(fit: Mapping[str, Any], *, include_values: bool = False) -> dict[str, Any]:
    model = fit.get("model")
    steps = getattr(model, "named_steps", {})
    scaler = steps.get("standardscaler")
    ridge = steps.get("ridge")
    return {
        "factor_values": _array_record(fit.get("factor_values"), include_values=include_values),
        "rf_values": _array_record(fit.get("rf_values"), include_values=include_values),
        "residuals": _array_record(fit.get("residuals"), include_values=include_values),
        "block_length": int(fit.get("block_length", 0)),
        "meta": _plain(fit.get("meta", {})),
        "model": {
            "steps": list(steps),
            "scaler_mean": _array_record(
                getattr(scaler, "mean_", None), include_values=include_values
            ),
            "scaler_scale": _array_record(
                getattr(scaler, "scale_", None), include_values=include_values
            ),
            "ridge_coef": _array_record(
                getattr(ridge, "coef_", None), include_values=include_values
            ),
            "ridge_intercept": _plain(getattr(ridge, "intercept_", None)),
        },
        "residual_overlay_fit": _nested_fit_summary(
            fit.get("residual_overlay_fit"), include_values=include_values
        ),
    }


def _is_array_record(value: Any) -> bool:
    return isinstance(value, Mapping) and {"dtype", "shape", "sha256", "values"}.issubset(value)


def _assert_array_record_equal(
    actual: Mapping[str, Any], expected: Mapping[str, Any], path: str
) -> None:
    required = {"dtype", "shape", "sha256", "values"}
    assert set(actual) == required, f"{path}: unexpected actual array record fields"
    assert set(expected) == required, f"{path}: source array values are missing"
    assert actual["dtype"] == expected["dtype"], f"{path}: dtype changed"
    assert actual["shape"] == expected["shape"], f"{path}: shape changed"
    expected_dtype = np.dtype(expected["dtype"])
    actual_dtype = np.dtype(actual["dtype"])
    expected_values = np.asarray(expected["values"], dtype=expected_dtype)
    actual_values = np.asarray(actual["values"], dtype=actual_dtype)
    assert list(expected_values.shape) == expected["shape"], f"{path}: invalid source shape"
    assert list(actual_values.shape) == actual["shape"], f"{path}: invalid actual shape"
    source_digest = _array_record(expected_values)
    assert source_digest == {
        "dtype": expected["dtype"],
        "shape": expected["shape"],
        "sha256": expected["sha256"],
    }, f"{path}: stored source array digest does not match its values"
    actual_digest = _array_record(actual_values)
    assert actual_digest == {
        "dtype": actual["dtype"],
        "shape": actual["shape"],
        "sha256": actual["sha256"],
    }, f"{path}: actual array digest does not match its values"
    if np.issubdtype(expected_dtype, np.number):
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=FIT_NUMERIC_RTOL,
            atol=FIT_NUMERIC_ATOL,
            err_msg=f"{path}: numeric fit state differs from pinned source",
        )
    else:
        np.testing.assert_array_equal(actual_values, expected_values, err_msg=path)


def _assert_fit_value_equal(actual: Any, expected: Any, path: str) -> None:
    if _is_array_record(expected):
        assert isinstance(actual, Mapping), f"{path}: actual fit state is not an array record"
        _assert_array_record_equal(actual, expected, path)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{path}: actual fit state is not a mapping"
        assert actual.keys() == expected.keys(), f"{path}: fit metadata keys changed"
        for key in expected:
            _assert_fit_value_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: fit metadata list changed type"
        assert len(actual) == len(expected), f"{path}: fit metadata list length changed"
        for index, expected_item in enumerate(expected):
            _assert_fit_value_equal(actual[index], expected_item, f"{path}[{index}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=FIT_NUMERIC_RTOL,
            atol=FIT_NUMERIC_ATOL,
            err_msg=f"{path}: scalar fit state differs from pinned source",
        )
    else:
        assert actual == expected, f"{path}: non-numeric fit metadata changed"


def _assert_fit_summary_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    assert actual.keys() == expected.keys(), "fit summary fields changed"
    for key in expected:
        _assert_fit_value_equal(actual[key], expected[key], f"fit.{key}")


def test_factor_residual_factory_domain_and_specification() -> None:
    fixture = _load_fixture()
    assert FACTOR_RESIDUAL_MODEL_IDS == (
        "factor_ff6_ridge_residual_sbb_none",
        "factor_ff6_ridge_residual_sbb_absolute_ewma",
    )
    assert {case["model_id"] for case in fixture["cases"]} == set(FACTOR_RESIDUAL_MODEL_IDS)
    assert set(RESOLVED_FACTOR_RESIDUAL_SPECS) == set(FACTOR_RESIDUAL_MODEL_IDS)
    for model_id in FACTOR_RESIDUAL_MODEL_IDS:
        assert set(RAW_FACTOR_RESIDUAL_SPECS[model_id]) == {"id", "type"}
        assert dict(RAW_FACTOR_RESIDUAL_SPECS[model_id]) == next(
            case["source_specification"]
            for case in fixture["cases"]
            if case["model_id"] == model_id
        )
        resolved = RESOLVED_FACTOR_RESIDUAL_SPECS[model_id]
        assert resolved["factor_model"] == "ff6"
        assert resolved["residual_overlay"] == "none"
        assert "source function default" in resolved["factor_model_parameter_source"]
        assert "source function default" in resolved["residual_overlay_parameter_source"]
        assert isinstance(make_factor_residual_model(model_id), FactorResidualSBBModel)
    with pytest.raises(ValueError, match="unknown factor residual model"):
        FactorResidualSBBModel("factor_ff6_ridge_residual_sbb_unknown")
    with pytest.raises(ValueError, match="no source-backed factor residual factory"):
        make_factor_residual_model("factor_ff6_ridge_residual_sbb_unknown")


def test_factor_snapshot_hash_and_fixture_identity() -> None:
    fixture = _load_fixture()
    assert fixture["schema"] == "factor-residual-source-parity-v1"
    assert fixture["source_function_digest"] == SOURCE_FUNCTIONS_SHA256
    assert fixture["source_artifacts"]["revision"] == "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
    snapshot = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources",
        "data",
        "canonical_snapshot",
        "app",
        "factor_data",
        "french_daily.csv.gz",
    )
    with resources.as_file(snapshot) as path:
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == factor_loader._FACTOR_FILE_SHA256["ff6"]
        )
    assert fixture["training"]["factor_snapshot_sha256"] == factor_loader._FACTOR_FILE_SHA256["ff6"]
    assert fixture["training"]["source_loader_frame"]["frame_equals_packaged"] is True
    assert len(fixture["cases"]) == 2


def test_factor_snapshot_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = factor_loader._FACTOR_FILE_SHA256["ff6"]
    monkeypatch.setitem(factor_loader._FACTOR_FILE_SHA256, "ff6", "0" * 64)
    with pytest.raises(ValueError, match="factor snapshot is unavailable"):
        factor_loader.load_packaged_factor_frame("ff6")
    factor_loader._FACTOR_FILE_SHA256["ff6"] = expected


def test_both_factor_paths_match_source_fit_states_and_daily_paths() -> None:
    fixture = _load_fixture()
    training_payload = fixture["training"]
    training = TrainingData(
        portfolio_log_returns=np.asarray(training_payload["values"], dtype=np.float64),
        training_dates=np.asarray(training_payload["dates"], dtype="datetime64[ns]"),
    )
    context_payload = fixture["context"]
    for case in fixture["cases"]:
        model_id = case["model_id"]
        model = make_factor_residual_model(model_id)
        fit = model.fit(training)
        _assert_fit_summary_equal(_fit_summary(fit, include_values=True), case["fit"])
        context = ForecastContext(
            model_id=model_id,
            portfolio_id="fixture",
            origin_label=context_payload["origin_date"],
            origin_date=context_payload["origin_date"],
            horizon_days=int(context_payload["horizon_days"]),
            simulations=int(context_payload["simulations"]),
            seed=0,
        )
        paths = model.simulate_daily_log_returns(training, context)
        expected = np.asarray(case["paths"], dtype=np.float64)
        np.testing.assert_allclose(paths, expected, rtol=2e-10, atol=2e-12)
        assert _array_record(paths) == case["paths_array"]
        dispatcher = case["dispatcher"]
        _assert_fit_summary_equal(dispatcher["fit"], case["fit"])
        for horizon_key, terminal_record in dispatcher["terminals"].items():
            horizon = int(horizon_key)
            terminal = np.cumsum(paths, axis=1, dtype=np.float64)[:, horizon - 1]
            expected_terminal = np.asarray(terminal_record["values"], dtype=np.float64)
            np.testing.assert_array_equal(terminal, expected_terminal)
            assert _array_record(terminal) == terminal_record["array"]
        assert (
            forecast_oos_candidate_seed(
                context.origin_date,
                tuple(range(1, context.horizon_days + 1)),
                model_id,
                context.simulations,
            )
            == case["source_forecast_seed"]
        )


def test_factor_fit_fixture_rejects_numeric_perturbation() -> None:
    fixture = _load_fixture()
    expected = fixture["cases"][0]["fit"]
    perturbed = copy.deepcopy(expected)
    perturbed["model"]["scaler_mean"]["values"][0] += 1e-6
    with pytest.raises(AssertionError, match="actual array digest"):
        _assert_fit_summary_equal(perturbed, expected)


def test_factor_adapter_requires_real_training_dates() -> None:
    values = np.linspace(-0.01, 0.01, 720, dtype=np.float64)
    model = make_factor_residual_model("factor_ff6_ridge_residual_sbb_none")
    with pytest.raises(ValueError, match="requires training_dates"):
        model.fit(TrainingData(values))
