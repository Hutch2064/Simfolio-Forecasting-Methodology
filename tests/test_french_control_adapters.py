"""Training-only native parity and independent control-formula checks."""
import ast
import hashlib
import importlib
import textwrap
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from simfolio_forecasting_methodology.french_panel import control_adapters
from simfolio_forecasting_methodology.french_panel.control_adapters import (
    CONTROL_IDS,
    EWMA_QMLE_ID,
    TESTING_NAIVE_ID,
    WEBSITE_DEFAULT_ID,
    EwmaQmleControl,
    WebsiteDefaultControl,
    _ewma_objective,
    _ewma_variances,
    build_control_model,
    fit_ewma_qmle,
    frozen_control_manifest,
    native_control_arguments,
)
from simfolio_forecasting_methodology.french_panel.control_adapters import (
    TestingNaiveControl as NaiveControl,
)
from simfolio_forecasting_methodology.runner import (
    ForecastContext,
    PortfolioPolicy,
    TrainingData,
)

SOURCE_PACKAGE = "simfolio_forecasting_methodology.resources.french1926.control_sources"


def inputs(values=None, *, horizon=252, paths=240, seed=713):
    if values is None:
        rng = np.random.default_rng(409)
        innovations = rng.normal(size=420)
        values = .00035 + np.exp(np.linspace(-5.2, -4.0, 420)) * innovations
    values = np.asarray(values, dtype=np.float64)
    dates = pd.bdate_range("1926-07-01", periods=len(values))
    training = TrainingData(values, values.reshape(-1, 1),
                            PortfolioPolicy(("French",), (1.0,), "none"), dates.to_numpy())
    context = ForecastContext("unused", "French", "origin", horizon, paths, seed,
                              pd.bdate_range(dates[-1] + pd.offsets.BDay(),
                                             periods=horizon).to_numpy(), str(dates[-1].date()))
    return training, context


def _source_fragment(source, name):
    node, = [x for x in ast.walk(ast.parse(source))
             if isinstance(x, ast.FunctionDef) and x.name == name]
    start = min([node.lineno] + [x.lineno for x in node.decorator_list])
    return textwrap.dedent("".join(source.splitlines(keepends=True)[start-1:node.end_lineno])) + "\n"


def test_frozen_sources_bind_current_engine_and_exact_native_function_bodies():
    manifest = frozen_control_manifest()
    assert manifest["engine_revision"] == "8090a1142d57335b3afd266483ae156ff051a9e6"
    root = files(SOURCE_PACKAGE)
    original = root.joinpath("smart_iid_forecast.original.py.txt").read_text()
    executable = root.joinpath("smart_iid_forecast.py").read_text()
    restored = executable.replace("from .forecast_allocation_composition import",
                                  "from app.forecast_allocation_composition import")
    restored = restored.replace("from .engine_helpers import", "from app.engine import")
    assert restored == original
    assert hashlib.sha256(original.encode()).hexdigest() == manifest[
        "upstream_source_sha256"]["app/smart_iid_forecast.py"]
    assert hashlib.sha256(root.joinpath("forecast_allocation_composition.py").read_bytes()
                          ).hexdigest() == manifest["upstream_source_sha256"][
                              "app/forecast_allocation_composition.py"]
    assert {item.name for item in root.iterdir() if item.is_file()
            and item.name != "manifest.json"} == set(manifest["files"])
    helpers = root.joinpath("engine_helpers.py").read_text()
    for name, expected in manifest["extracted_engine_functions"].items():
        source = (root.joinpath("engine_testing_route.py.txt").read_text()
                  if name == "_simulate_smart_iid_forecast_log_paths" else helpers)
        assert hashlib.sha256(_source_fragment(source, name).encode()).hexdigest() == expected
    testing = root.joinpath("model_testing.original.py.txt").read_text()
    assert hashlib.sha256(testing.encode()).hexdigest() == manifest[
        "upstream_source_sha256"]["app/model_testing.py"]
    assert '"naive_benchmark": True' in _source_fragment(testing, "run_naive_reference")
    assert manifest["historical_ewma_comparison"]["scores_inherited"] is False


@pytest.mark.parametrize("model_id", [WEBSITE_DEFAULT_ID, TESTING_NAIVE_ID])
@pytest.mark.parametrize("horizon", [1260, 19738])
def test_product_controls_match_the_frozen_native_kernel_at_all_horizons(model_id, horizon):
    training, context = inputs(horizon=horizon)
    model = build_control_model(model_id)
    kernel = importlib.import_module(f"{SOURCE_PACKAGE}.smart_iid_forecast")
    arguments = native_control_arguments(training, context)
    expected, meta = kernel.simulate_smart_iid_log_returns(
        **arguments, _naive_stationary_bootstrap=model_id == TESTING_NAIVE_ID)
    actual, actual_meta = model.simulate_with_metadata(training, context)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.cumsum(actual, axis=1), np.cumsum(expected, axis=1))
    assert actual.shape == (240, horizon)
    assert actual_meta["simulation_method"] == model_id
    assert actual_meta["rejoin_implementation"] == "numba_fused_no_rebalance_wealth64"
    assert actual_meta["source_date_end"] == context.origin_date
    assert actual_meta["mean_model"] == meta["mean_model"]


def test_default_matches_independent_training_t_stat_shrinkage_and_iid_draws():
    training, context = inputs(horizon=91)
    actual, meta = WebsiteDefaultControl().simulate_with_metadata(training, context)
    history = np.log1p(np.expm1(training.portfolio_log_returns))
    mean = history.mean()
    t_squared = (mean / (history.std(ddof=1) / np.sqrt(history.size))) ** 2
    shrink = t_squared / (1.0 + t_squared)
    adjusted = np.log1p(np.expm1(history + mean * shrink - mean))
    indices = np.random.default_rng(context.seed).integers(
        0, len(history), size=(context.simulations, context.horizon_days), dtype=np.int64)
    np.testing.assert_allclose(actual, adjusted[indices], atol=2e-15, rtol=2e-12)
    assert meta["mean_shrink_weights"] == pytest.approx([shrink], abs=1e-15)
    assert meta["innovation_resampling"] == "iid"


def test_testing_reference_uses_unshifted_means_and_joint_plugin_stationary_blocks():
    training, context = inputs(horizon=115)
    actual, meta = NaiveControl().simulate_with_metadata(training, context)
    history = np.log1p(np.expm1(training.portfolio_log_returns))
    helpers = importlib.import_module(f"{SOURCE_PACKAGE}.engine_helpers")
    block = min(len(history), max(helpers._politis_white_block_length(history),
                                 helpers._politis_white_block_length((history-history.mean())**2)))
    assert meta["block_length"] == block
    assert block > 1  # Fixture distinguishes this control from joint IID.
    assert meta["mean_model"] == "unshifted_historical_log_returns"
    rng = np.random.default_rng(context.seed)
    initial = rng.integers(0, len(history), size=context.simulations, dtype=np.int32)
    flags = rng.random((context.simulations, context.horizon_days-1)) < 1 / block
    points = rng.integers(0, len(history), size=flags.shape, dtype=np.int32)
    indices = np.empty((context.simulations, context.horizon_days), dtype=int)
    indices[:, 0] = initial
    for day in range(1, context.horizon_days):
        indices[:, day] = np.where(flags[:, day-1], points[:, day-1],
                                   (indices[:, day-1] + 1) % len(history))
    np.testing.assert_allclose(actual, history[indices], atol=2e-15, rtol=2e-12)


@pytest.mark.parametrize("model_id", CONTROL_IDS)
def test_forecast_has_no_access_to_held_out_returns_and_is_label_independent(model_id):
    training, context = inputs(horizon=31)
    model = build_control_model(model_id)
    before = training.portfolio_log_returns.copy()
    expected = model.simulate_daily_log_returns(training, context)
    # These are the only contextual fields besides calendar/seed/horizon. An
    # arbitrary held-out identity cannot change either a fit or its paths.
    different_labels = replace(context, model_id="other", portfolio_id="other",
                               origin_label="held_out_target_changed")
    actual = model.simulate_daily_log_returns(training, different_labels)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(training.portfolio_log_returns, before)
    assert np.isfinite(actual).all()
    with pytest.raises(ValueError, match="training boundary"):
        model.simulate_daily_log_returns(training, replace(context,
            future_dates=training.training_dates[-31:]))


@pytest.mark.parametrize("model_id", CONTROL_IDS)
@pytest.mark.parametrize("constant", [0.0, .0002, -.0002])
def test_cash_constant_and_zero_variance_inputs(model_id, constant):
    training, context = inputs(np.full(60, constant), horizon=63)
    paths = build_control_model(model_id).simulate_daily_log_returns(training, context)
    assert np.isfinite(paths).all()
    assert paths.shape == (240, 63)
    np.testing.assert_array_equal(paths, np.broadcast_to(paths[:1], paths.shape))
    if model_id in (EWMA_QMLE_ID, TESTING_NAIVE_ID):
        np.testing.assert_allclose(paths, constant, atol=5e-16, rtol=1e-12)
    if constant == 0:
        np.testing.assert_array_equal(paths, np.zeros_like(paths))


@pytest.mark.parametrize("model_id", CONTROL_IDS)
@pytest.mark.parametrize("horizon", [19738, 25200])
def test_standard_240_cash_paths_retain_the_full_horizon(model_id, horizon):
    training, context = inputs(np.zeros(60), horizon=horizon)
    output = build_control_model(model_id).simulate_daily_log_returns(training, context)
    assert output.shape == (240, horizon)
    assert np.isfinite(output).all()
    assert np.count_nonzero(output) == 0


def test_ewma_fit_is_predictive_gaussian_qmle_with_no_fixed_decay():
    training, _ = inputs()
    history = training.portfolio_log_returns
    fit = fit_ewma_qmle(history)
    assert fit.fit_status == "complete"
    assert fit.decay is not None and 0 <= fit.decay <= 1
    assert abs(fit.decay - .94) > 1e-5
    residuals = history - history.mean()
    variance = np.var(history, ddof=1)
    expected_objective = 0.0
    predictive = []
    for residual in residuals:
        predictive.append(variance)
        expected_objective += np.log(variance) + residual * residual / variance
        variance = fit.decay * variance + (1 - fit.decay) * residual * residual
    assert fit.objective == pytest.approx(expected_objective, rel=2e-14)
    assert fit.forecast_variance == pytest.approx(variance, rel=2e-14)
    assert fit.mean == history.mean()
    actual_predictive, actual_next = _ewma_variances(residuals, fit.initial_variance, fit.decay)
    np.testing.assert_allclose(actual_predictive, predictive, rtol=2e-14)
    assert actual_next == pytest.approx(variance, rel=2e-14)
    # Training likelihood, including endpoints, selects decay; no held-out CRPS.
    for candidate in np.linspace(0, 1, 101):
        assert fit.objective <= _ewma_objective(candidate, residuals, fit.initial_variance) + 1e-7


def test_ewma_paths_recursively_update_variance_from_their_simulated_shocks():
    training, context = inputs(horizon=73)
    fit = fit_ewma_qmle(training.portfolio_log_returns)
    actual = EwmaQmleControl().simulate_daily_log_returns(training, context)
    variance = np.full(context.simulations, fit.forecast_variance)
    rng = np.random.default_rng(context.seed)
    expected = np.empty_like(actual)
    for day in range(context.horizon_days):
        shocks = np.sqrt(variance) * rng.standard_normal(context.simulations)
        expected[:, day] = fit.mean + shocks
        variance = fit.decay * variance + (1-fit.decay) * shocks**2
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-17)
    # No deterministic common volatility overlay: future shocks change the
    # variance differently for each path after the first simulated observation.
    assert np.ptp(variance) > 0


def test_ewma_constant_fit_reports_unidentified_decay_and_new_identity():
    fit = fit_ewma_qmle(np.full(421, .0003))
    assert fit.mean == .0003 and fit.decay is None
    assert fit.fit_status == "deterministic_constant_history"
    assert fit.forecast_variance == fit.initial_variance == 0
    assert EwmaQmleControl.model_id == "FRENCH1926-EWMA-QMLE"
    assert EwmaQmleControl.dependency_identity["historical_scores_inherited"] is False
    with pytest.raises(ValueError, match="unknown French control"):
        build_control_model("bayesian_sbb_overlay_ewma_qmle")


@pytest.mark.parametrize("values", [[], [np.nan], [0., np.inf]])
def test_ewma_rejects_invalid_training(values):
    with pytest.raises(ValueError, match="finite nonempty"):
        fit_ewma_qmle(values)


def test_ewma_full_held_out_horizon_is_stochastic_and_finite():
    training, context = inputs(horizon=19738)
    paths = EwmaQmleControl().simulate_daily_log_returns(training, context)
    assert paths.shape == (240, 19738)
    assert np.isfinite(paths).all()
    assert np.any(paths[0] != paths[1])


def test_ewma_fit_failure_raises_without_fixed_decay_fallback(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(control_adapters, "minimize_scalar",
                        lambda *args, **kwargs: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="decay fit failed"):
        fit_ewma_qmle(np.array([0., .01, -.01]))


@pytest.mark.parametrize("module_name, kernel_name", [
    ("smart_iid_forecast", "_fused_no_rebalance_terminal_wealth"),
    ("engine_helpers", "_stationary_bootstrap_indices_from_draws_numba"),
])
def test_native_controls_reject_missing_compiled_kernels(monkeypatch, module_name, kernel_name):
    module = importlib.import_module(f"{SOURCE_PACKAGE}.{module_name}")
    monkeypatch.setattr(module, kernel_name, None)
    control_adapters._native_kernel.cache_clear()
    try:
        training, context = inputs(horizon=2)
        with pytest.raises(ImportError, match="numba dependency"):
            WebsiteDefaultControl().simulate_daily_log_returns(training, context)
    finally:
        control_adapters._native_kernel.cache_clear()
