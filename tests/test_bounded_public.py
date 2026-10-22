"""New bounded-bootstrap law, frozen wealth parity, and input-boundary checks."""
import hashlib
import importlib
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simfolio_forecasting_methodology.french_panel import control_adapters
from simfolio_forecasting_methodology.runner import (
    ForecastContext,
    PortfolioPolicy,
    TrainingData,
)

# CI runs from outside the checkout against the installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bounded_public import MODEL_ID, bounded_block_indices, build_bounded_public


def inputs(values=None, horizon=17, paths=240, seed=714):
    if values is None:
        values = np.random.default_rng(204).normal(.0002, .01, 512)
    values = np.asarray(values, dtype=np.float64)
    dates = pd.bdate_range("1926-07-01", periods=len(values))
    training = TrainingData(values, values[:, None],
                            PortfolioPolicy(("French",), (1.,), "none"), dates.to_numpy())
    context = ForecastContext(MODEL_ID, "French", "origin", horizon, paths, seed,
                              pd.bdate_range(dates[-1] + pd.offsets.BDay(),
                                             periods=horizon).to_numpy(), str(dates[-1].date()))
    return training, context


def independent_indices(n, horizon, paths, rng):
    rows = []
    for _ in range(paths):
        selected = []
        while len(selected) < horizon:
            length = rng.integers(1, min(n, horizon) + 1)
            start = rng.integers(0, n)
            for offset in range(length):
                if len(selected) == horizon:
                    break
                selected.append((start + offset) % n)
        rows.append(selected)
    return np.array(rows, dtype=np.int64)


@pytest.mark.parametrize("n,horizon", [(1, 1), (1, 7), (5, 1), (5, 4), (3, 11), (29, 83)])
def test_indices_match_independent_hand_loop_and_rng_consumption(n, horizon):
    actual_rng = np.random.default_rng(172)
    expected_rng = np.random.default_rng(172)
    actual = bounded_block_indices(n, horizon, 4, actual_rng)
    expected = independent_indices(n, horizon, 4, expected_rng)
    np.testing.assert_array_equal(actual, expected)
    assert actual_rng.integers(0, 2**31) == expected_rng.integers(0, 2**31)
    assert actual.dtype == np.int64
    assert np.all((actual >= 0) & (actual < n))


class RecordedDraws:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.count = 0

    def integers(self, low, high):
        expected_low, expected_high, value = next(self.calls)
        assert (low, high) == (expected_low, expected_high)
        assert low <= value < high
        self.count += 1
        return value


def test_fixed_horizon_cap_circular_wrap_replacement_and_last_block_truncation():
    # The last length draw still has high=5, despite only one day remaining.
    # Identical historical starts are allowed across independently drawn blocks.
    rng = RecordedDraws([(1, 5, 3), (0, 5, 4), (1, 5, 4), (0, 5, 4)])
    np.testing.assert_array_equal(bounded_block_indices(5, 4, 1, rng), [[4, 0, 1, 4]])
    assert rng.count == 4


def test_history_cap_is_fixed_when_forecast_is_longer_than_history():
    rng = RecordedDraws([(1, 4, 3), (0, 3, 2)] * 3)
    np.testing.assert_array_equal(bounded_block_indices(3, 8, 1, rng),
                                  [[2, 0, 1, 2, 0, 1, 2, 0]])
    assert rng.count == 6


def test_seeded_first_block_length_and_start_marginals_are_uniform():
    class Recorder:
        def __init__(self):
            self.rng = np.random.default_rng(617)
            self.draws = []

        def integers(self, low, high):
            value = self.rng.integers(low, high)
            self.draws.append(value)
            return value

    recorder = Recorder()
    lengths, starts = [], []
    for _ in range(12000):
        recorder.draws.clear()
        bounded_block_indices(11, 7, 1, recorder)
        lengths.append(recorder.draws[0])
        starts.append(recorder.draws[1])
    length_counts = np.bincount(lengths, minlength=8)[1:]
    start_counts = np.bincount(starts, minlength=11)
    np.testing.assert_allclose(length_counts, np.full(7, 12000 / 7), rtol=.1)
    np.testing.assert_allclose(start_counts, np.full(11, 12000 / 11), rtol=.1)


@pytest.mark.parametrize("n,horizon,paths", [(0, 1, 1), (1, 0, 1), (1, 1, 0),
                                              (-1, 4, 2), (3.5, 4, 2), (True, 2, 1)])
def test_invalid_index_dimensions_fail_closed(n, horizon, paths):
    with pytest.raises(ValueError, match="positive integers"):
        bounded_block_indices(n, horizon, paths, np.random.default_rng(1))


def test_paths_match_frozen_public_kernel_exactly_when_only_indices_are_replaced(monkeypatch):
    training, context = inputs(horizon=97)
    model = build_bounded_public()
    actual, metadata = model.simulate_with_metadata(training, context)
    indices = independent_indices(len(training.portfolio_log_returns), context.horizon_days,
                                  context.simulations, np.random.default_rng(context.seed))
    helpers = importlib.import_module(control_adapters._SOURCE_PACKAGE + ".engine_helpers")
    monkeypatch.setattr(helpers, "_stationary_bootstrap_indices_from_draws_fast",
                        lambda **kwargs: indices)
    expected = control_adapters.TestingNaiveControl().simulate_daily_log_returns(training, context)
    np.testing.assert_array_equal(actual, expected)
    assert metadata["block_length_maximum"] == 97
    assert metadata["mean_model"] == "unshifted_historical_log_returns"
    assert metadata["rejoin_implementation"] == "numba_fused_no_rebalance_wealth64"


def test_paths_match_independent_single_asset_circular_wealth_accounting():
    training, context = inputs([-.05, .03, -.01, .008], horizon=31, paths=5)
    actual = build_bounded_public().simulate_daily_log_returns(training, context)
    indices = independent_indices(4, 31, 5, np.random.default_rng(context.seed))
    log_history = np.log1p(np.clip(np.expm1(training.portfolio_log_returns), -.999999,
                                  np.finfo(np.float64).max / 10.))
    simple = np.clip(np.expm1(log_history), -.999999, np.finfo(np.float64).max / 10.)
    expected = np.empty_like(actual)
    for path in range(5):
        wealth = 1.
        previous_log_wealth = 0.
        for day in range(31):
            wealth *= 1. + simple[indices[path, day]]
            log_wealth = math.log(max(wealth, 1e-300))
            expected[path, day] = log_wealth - previous_log_wealth
            previous_log_wealth = log_wealth
    np.testing.assert_array_equal(actual, expected)


def test_no_politis_white_or_geometric_stationary_kernel_is_called(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("old block calibration must not be called")
    helpers = importlib.import_module(control_adapters._SOURCE_PACKAGE + ".engine_helpers")
    monkeypatch.setattr(helpers, "_politis_white_block_length", forbidden)
    monkeypatch.setattr(helpers, "_stationary_bootstrap_indices_from_draws_fast", forbidden)
    training, context = inputs()
    assert build_bounded_public().simulate_daily_log_returns(training, context).shape == (240, 17)


def test_training_only_and_labels_do_not_change_paths_or_mutate_inputs():
    training, context = inputs()
    before = training.portfolio_log_returns.copy()
    model = build_bounded_public()
    expected = model.simulate_daily_log_returns(training, context)
    changed = replace(context, model_id="held_out_model", portfolio_id="other_series",
                      origin_label="target_returns_or_scores_changed", origin_date="2099-01-01")
    actual = model.simulate_daily_log_returns(training, changed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(training.portfolio_log_returns, before)
    with pytest.raises(ValueError, match="training boundary"):
        model.simulate_daily_log_returns(training, replace(context,
            future_dates=training.training_dates[-context.horizon_days:]))


@pytest.mark.parametrize("values", [[0., np.nan], [0., np.inf], [0., -np.inf]])
def test_nonfinite_training_fails_closed(values):
    training, context = inputs(values)
    with pytest.raises(ValueError, match="finite"):
        build_bounded_public().simulate_daily_log_returns(training, context)


def test_overflowed_preparation_fails_closed():
    training, context = inputs([0., 1000.])
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite prepared"):
        build_bounded_public().simulate_daily_log_returns(training, context)


def test_invalid_native_wealth_output_fails_closed(monkeypatch):
    native = control_adapters._native_kernel()
    monkeypatch.setattr(native, "_fused_no_rebalance_terminal_wealth",
                        lambda *args: (np.full((240, 17), np.nan), None))
    training, context = inputs()
    with pytest.raises(ValueError, match="invalid native wealth"):
        build_bounded_public().simulate_daily_log_returns(training, context)


@pytest.mark.parametrize("cash", [False, True])
def test_full_19738_day_240_path_output(cash):
    training, context = inputs(np.zeros(512) if cash else None, horizon=19738)
    actual = build_bounded_public().simulate_daily_log_returns(training, context)
    assert actual.shape == (240, 19738)
    assert np.isfinite(actual).all()
    if cash:
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
    else:
        assert np.any(actual[0] != actual[1])


def test_distinct_identity_declares_horizon_conditional_law_and_closure():
    model = build_bounded_public()
    identity = model.dependency_identity
    assert model.model_id == MODEL_ID
    assert model.model_id != control_adapters.TESTING_NAIVE_ID
    assert identity["horizon_conditional"] is True
    assert identity["prefix_coherence_across_requested_horizons"] is False
    assert identity["historical_scores_inherited"] is False
    assert identity["politis_white"] is False
    assert identity["source_sha256"] == hashlib.sha256(
        Path(importlib.import_module("bounded_public").__file__).read_bytes()).hexdigest()
    assert len(model.implementation_digest) == 64
    assert model.implementation_digest == build_bounded_public().implementation_digest
    from simfolio_forecasting_methodology.results.checkpoint import stable_digest
    assert model.implementation_digest == stable_digest(model.implementation_identity)
    assert model.implementation_identity["factory"] == "bounded_public:build_bounded_public"
    assert model.implementation_identity["model_id"] == MODEL_ID
    assert model.implementation_identity["source_files"]["scripts/bounded_public.py"] == identity[
        "source_sha256"]
    assert control_adapters.TestingNaiveControl.model_id == "model_testing_joint_stationary_bootstrap_v1"
