"""Observation-level cycle exhaustion, exact RNG law, and native wealth parity."""
import hashlib
import importlib
from dataclasses import replace
from pathlib import Path

import bounded_public
import numpy as np
import pytest
from bounded_public_without_replacement import (
    MODEL_ID,
    bounded_cycle_indices,
    build_bounded_cycle_public,
)
from test_bounded_public import RecordedDraws, inputs

from simfolio_forecasting_methodology.french_panel import control_adapters
from simfolio_forecasting_methodology.results.checkpoint import stable_digest


def independent_cycle_indices(n, horizon, paths, rng):
    output = np.empty((paths, horizon), dtype=np.int64)
    for path in range(paths):
        unused = np.ones(n, dtype=bool)
        position = 0
        while position < horizon:
            if not unused.any():
                unused[:] = True
            longest = 0
            for start in range(n):
                length = 0
                while length < n and unused[(start + length) % n]:
                    length += 1
                longest = max(longest, length)
            length = int(rng.integers(1, min(n, horizon, longest) + 1))
            feasible = [start for start in range(n)
                        if all(unused[(start + offset) % n] for offset in range(length))]
            start = feasible[int(rng.integers(0, len(feasible)))]
            for offset in range(min(length, horizon - position)):
                row = (start + offset) % n
                assert unused[row]
                output[path, position] = row
                unused[row] = False
                position += 1
    return output


@pytest.mark.parametrize("n,horizon", [(1, 1), (1, 13), (7, 1), (7, 4), (7, 7),
                                          (7, 8), (7, 21), (19, 51), (31, 127)])
def test_interval_sampler_matches_independent_mask_hand_loop_and_rng_state(n, horizon):
    actual_rng = np.random.default_rng(707)
    reference_rng = np.random.default_rng(707)
    actual = bounded_cycle_indices(n, horizon, 6, actual_rng)
    expected = independent_cycle_indices(n, horizon, 6, reference_rng)
    np.testing.assert_array_equal(actual, expected)
    assert actual_rng.integers(0, 2**31) == reference_rng.integers(0, 2**31)


def test_small_exhaustive_unused_mask_oracle_and_cycle_invariants():
    checked = 0
    for n in range(1, 13):
        for horizon in range(1, 2 * n + 3):
            for seed in range(20):
                actual_rng = np.random.default_rng(seed)
                reference_rng = np.random.default_rng(seed)
                actual = bounded_cycle_indices(n, horizon, 3, actual_rng)
                expected = independent_cycle_indices(n, horizon, 3, reference_rng)
                np.testing.assert_array_equal(actual, expected)
                assert actual_rng.integers(0, 2**31) == reference_rng.integers(0, 2**31)
                for path in actual:
                    for start in range(0, horizon, n):
                        cycle = path[start:start + n]
                        assert len(np.unique(cycle)) == len(cycle)
                        if len(cycle) == n:
                            np.testing.assert_array_equal(np.sort(cycle), np.arange(n))
                checked += 1
    assert checked == 3600


@pytest.mark.parametrize("n,horizon", [(1, 13), (3, 11), (31, 127), (504, 1520)])
def test_every_completed_history_cycle_is_a_permutation_and_partial_cycle_has_no_reuse(n, horizon):
    actual = bounded_cycle_indices(n, horizon, 12, np.random.default_rng(224))
    for path in actual:
        for start in range(0, horizon, n):
            cycle = path[start:start + n]
            assert len(np.unique(cycle)) == len(cycle)
            if len(cycle) == n:
                np.testing.assert_array_equal(np.sort(cycle), np.arange(n))


def test_circular_wrap_history_exhaustion_refill_and_final_truncation():
    rng = RecordedDraws([(1, 6, 3), (0, 5, 4), (1, 3, 2), (0, 1, 0),
                         (1, 6, 5), (0, 5, 1)])
    np.testing.assert_array_equal(bounded_cycle_indices(5, 8, 1, rng),
                                  [[4, 0, 1, 2, 3, 1, 2, 3]])
    assert rng.count == 6


def test_fragmentation_caps_length_and_sorted_feasible_starts_include_circular_runs():
    rng = RecordedDraws([
        (1, 9, 2), (0, 8, 1),  # Remove 1,2.
        (1, 7, 2), (0, 5, 2),  # Starts 3,4,5,6,7: choose 5; remove 5,6.
        (1, 3, 1), (0, 4, 0),  # Starts 0,3,4,7: choose 0.
        (1, 3, 2), (0, 1, 0),  # Only run 3,4 admits length 2.
        (1, 2, 1), (0, 1, 0),  # Consume final unused observation 7.
        (1, 9, 8), (0, 8, 2),  # New cycle; draw 8, truncate at remaining H=4.
    ])
    np.testing.assert_array_equal(bounded_cycle_indices(8, 12, 1, rng),
                                  [[1, 2, 5, 6, 0, 3, 4, 7, 2, 3, 4, 5]])
    assert rng.count == 12


@pytest.mark.parametrize("n,horizon,paths", [(0, 1, 1), (1, 0, 1), (1, 1, 0),
                                              (-1, 4, 2), (3.5, 4, 2), (True, 2, 1)])
def test_invalid_index_dimensions_fail_closed(n, horizon, paths):
    with pytest.raises(ValueError, match="positive integers"):
        bounded_cycle_indices(n, horizon, paths, np.random.default_rng(1))


def test_seeded_daily_index_marginals_are_rotation_symmetric():
    actual = bounded_cycle_indices(7, 15, 7000, np.random.default_rng(619))
    for day in range(15):
        counts = np.bincount(actual[:, day], minlength=7)
        np.testing.assert_allclose(counts, np.full(7, 1000.), rtol=.12)


def test_paths_match_frozen_public_native_wealth_with_only_cycle_indices_substituted(monkeypatch):
    training, context = inputs(horizon=173)
    context = replace(context, model_id=MODEL_ID)
    actual, metadata = build_bounded_cycle_public().simulate_with_metadata(training, context)
    indices = bounded_cycle_indices(len(training.portfolio_log_returns), context.horizon_days,
                                    context.simulations, np.random.default_rng(context.seed))
    helpers = importlib.import_module(control_adapters._SOURCE_PACKAGE + ".engine_helpers")
    monkeypatch.setattr(helpers, "_stationary_bootstrap_indices_from_draws_fast",
                        lambda **kwargs: indices)
    expected = control_adapters.TestingNaiveControl().simulate_daily_log_returns(training, context)
    np.testing.assert_array_equal(actual, expected)
    assert metadata["simulation_method"] == MODEL_ID
    assert metadata["mean_model"] == "unshifted_historical_log_returns"
    assert metadata["replacement"] == "no_observation_reuse_until_complete_history_cycle_refill"


def test_no_fitted_block_or_stationary_resampling_kernel_is_called(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("old block calibration must not be called")
    helpers = importlib.import_module(control_adapters._SOURCE_PACKAGE + ".engine_helpers")
    monkeypatch.setattr(helpers, "_politis_white_block_length", forbidden)
    monkeypatch.setattr(helpers, "_stationary_bootstrap_indices_from_draws_fast", forbidden)
    training, context = inputs(horizon=83)
    assert build_bounded_cycle_public().simulate_daily_log_returns(training, context).shape == (240, 83)


def test_training_only_and_calendar_boundary():
    training, context = inputs(horizon=23)
    before = training.portfolio_log_returns.copy()
    model = build_bounded_cycle_public()
    expected = model.simulate_daily_log_returns(training, context)
    actual = model.simulate_daily_log_returns(training, replace(context,
        model_id="held_out_model_changed", portfolio_id="other", origin_label="target_changed"))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(training.portfolio_log_returns, before)
    with pytest.raises(ValueError, match="training boundary"):
        model.simulate_daily_log_returns(training, replace(context,
            future_dates=training.training_dates[-context.horizon_days:]))


def test_complete_history_cycles_have_fixed_terminal_log_return_up_to_wealth_roundoff():
    training, context = inputs(np.random.default_rng(918).normal(.0001, .01, 31), horizon=93)
    output = build_bounded_cycle_public().simulate_daily_log_returns(training, context)
    history = np.log1p(np.clip(np.expm1(training.portfolio_log_returns), -.999999,
                              np.finfo(np.float64).max / 10.))
    source = np.clip(np.expm1(history), -.999999, np.finfo(np.float64).max / 10.)
    expected = np.sum(np.log1p(source)) * np.arange(1, 4)
    cycle_terminals = np.cumsum(output, axis=1)[:, 30::31]
    np.testing.assert_allclose(cycle_terminals, np.broadcast_to(expected, (240, 3)),
                               atol=4e-15, rtol=2e-13)
    assert np.any(output[0] != output[1])  # Intermediate paths still differ.


@pytest.mark.parametrize("values", [[0., np.nan], [0., np.inf], [0., -np.inf]])
def test_nonfinite_training_fails_closed(values):
    training, context = inputs(values)
    with pytest.raises(ValueError, match="finite"):
        build_bounded_cycle_public().simulate_daily_log_returns(training, context)


def test_overflowed_preparation_fails_closed():
    training, context = inputs([0., 1000.])
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite prepared"):
        build_bounded_cycle_public().simulate_daily_log_returns(training, context)


def test_invalid_native_wealth_output_fails_closed(monkeypatch):
    native = control_adapters._native_kernel()
    monkeypatch.setattr(native, "_fused_no_rebalance_terminal_wealth",
                        lambda *args: (np.full((240, 17), np.nan), None))
    training, context = inputs()
    with pytest.raises(ValueError, match="invalid native wealth"):
        build_bounded_cycle_public().simulate_daily_log_returns(training, context)


@pytest.mark.parametrize("cash", [False, True])
def test_full_19738_day_240_path_output(cash):
    values = np.zeros(5040) if cash else np.random.default_rng(840).normal(.0001, .012, 5040)
    training, context = inputs(values, horizon=19738)
    output = build_bounded_cycle_public().simulate_daily_log_returns(training, context)
    assert output.shape == (240, 19738)
    assert np.isfinite(output).all()
    if cash:
        np.testing.assert_array_equal(output, np.zeros_like(output))
    else:
        assert np.any(output[0] != output[1])


def test_separate_identity_binds_immutable_with_replacement_base_and_new_sampler():
    model = build_bounded_cycle_public()
    identity = model.implementation_identity
    assert identity["model_id"] == MODEL_ID
    assert identity["factory"] == "bounded_public_without_replacement:build_bounded_cycle_public"
    assert len(identity["source_files"]) == 15
    assert identity["source_files"]["scripts/bounded_public.py"] == hashlib.sha256(
        Path(bounded_public.__file__).read_bytes()).hexdigest()
    assert model.implementation_digest == stable_digest(identity)
    assert model.dependency_identity["horizon_conditional"] is True
    assert model.dependency_identity["prefix_coherence_across_requested_horizons"] is False
    assert bounded_public.build_bounded_public().implementation_digest == (
        "2c3c3c8f2ce9864eb3e6a1b473047aff9dacb83486f89b5e1f01299fe8ca686c")
