"""Horizon-bounded circular blocks without observation reuse inside each cycle.

Exhaust every historical observation once per path before refilling the pool.
Given unused runs, draw length uniformly through min(n, H, longest unused run),
then choose uniformly among all historical starts feasible for that length.
Final forecast truncation follows the draw. Requested H conditions this law.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

import numpy as np
from simfolio_forecasting_methodology.french_panel import control_adapters
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

import bounded_public

MODEL_ID = "model_testing_joint_uniform_bounded_cycle_bootstrap_v1"


def bounded_cycle_indices(n: int, horizon: int, paths: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Draw path-major blocks with valid starts ordered by historical index.

Unused circular runs are stored as (start, length), avoiding an n-element mask
scan at every block. Deleting a block can split only its selected unused run.
"""
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) or x < 1
           for x in (n, horizon, paths)):
        raise ValueError("training rows, horizon and paths must be positive integers")
    maximum = min(n, horizon)
    indices = np.empty((paths, horizon), dtype=np.int64)
    for path in range(paths):
        position = 0
        remaining = n
        runs = [(0, n)]
        while position < horizon:
            if remaining == 0:
                remaining, runs = n, [(0, n)]
            cap = min(maximum, max(length for _, length in runs))
            length = int(rng.integers(1, cap + 1))
            if remaining == n:
                # A wholly unused circle admits every start for every L <= n.
                start = int(rng.integers(0, n))
                selected_run, offset = 0, 0
                run_start, run_length = start, n
            else:
                # Feasible starts inside a run form a circular interval. Split
                # only at historical zero, then rank all starts in index order.
                segments = []
                for run_index, (run_start, run_length) in enumerate(runs):
                    count = run_length - length + 1
                    if count < 1:
                        continue
                    first = min(count, n - run_start)
                    segments.append((run_start, first, run_index))
                    if first < count:
                        segments.append((0, count - first, run_index))
                segments.sort()
                rank = int(rng.integers(0, sum(count for _, count, _ in segments)))
                for segment_start, count, selected_run in segments:
                    if rank < count:
                        start = segment_start + rank
                        break
                    rank -= count
                run_start, run_length = runs[selected_run]
                offset = (start - run_start) % n
            copied = min(length, horizon - position)
            indices[path, position:position + copied] = (start + np.arange(copied)) % n
            runs.pop(selected_run)
            if offset:
                runs.append((run_start, offset))
            after = run_length - offset - copied
            if after:
                runs.append(((start + copied) % n, after))
            remaining -= copied
            position += copied
    return indices


class BoundedCyclePublicControl(bounded_public.BoundedPublicControl):
    model_id: ClassVar[str] = MODEL_ID

    @property
    def dependency_identity(self) -> dict:
        identity = super().dependency_identity
        identity.update({
            "method": MODEL_ID,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "with_replacement_base_sha256": hashlib.sha256(
                Path(bounded_public.__file__).read_bytes()).hexdigest(),
            "resampling": "joint_rows_circular_blocks_without_observation_reuse_per_history_cycle",
            "block_length": "integer_uniform_1_to_min_training_rows_requested_horizon_longest_unused_run",
            "start_selection": "uniform_among_all_length_feasible_unused_starts_in_history_index_order",
            "pool_refill": "only_after_all_training_observations_have_been_consumed_once_per_path",
            "last_block": "truncate_only_after_length_and_feasible_start_draw",
            "draw_order": "path_major_then_per_block_length_then_ranked_feasible_historical_start",
        })
        return identity

    @property
    def implementation_identity(self) -> dict:
        identity = super().implementation_identity
        sources = identity["source_files"]
        sources["scripts/bounded_public.py"] = self.dependency_identity[
            "with_replacement_base_sha256"]
        sources["scripts/bounded_public_without_replacement.py"] = self.dependency_identity[
            "source_sha256"]
        identity["model_id"] = MODEL_ID
        identity["factory"] = "bounded_public_without_replacement:build_bounded_cycle_public"
        return identity

    def simulate_with_metadata(self, training: TrainingData, context: ForecastContext):
        arguments = control_adapters.native_control_arguments(training, context)
        simple = arguments["asset_returns"].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError("bounded cycle comparator requires finite prepared simple returns")
        lower, upper = -0.999999, np.finfo(np.float64).max / 10.0
        log_history = np.log1p(np.clip(simple, lower, upper))
        source_simple = np.clip(np.expm1(log_history), lower, upper)
        indices = bounded_cycle_indices(len(log_history), context.horizon_days,
                                        context.simulations, arguments["rng"])
        kernel = control_adapters._native_kernel()
        unit_weight = np.ones(1, dtype=np.float64)
        output, _ = kernel._fused_no_rebalance_terminal_wealth(
            source_simple, indices, unit_weight, unit_weight,
            np.ones(context.horizon_days, dtype=np.float64))
        if output.shape != (context.simulations, context.horizon_days) or not np.isfinite(output).all():
            raise ValueError("bounded cycle comparator produced invalid native wealth paths")
        return output, {
            "simulation_method": MODEL_ID,
            "innovation_resampling": "uniform_bounded_unused_circular_blocks_with_cycle_refill",
            "block_length_minimum": 1,
            "block_length_maximum": min(len(log_history), context.horizon_days),
            "effective_block_cap": "minimum_of_original_bound_and_longest_unused_circular_run",
            "replacement": "no_observation_reuse_until_complete_history_cycle_refill",
            "mean_model": "unshifted_historical_log_returns",
            "rejoin_implementation": "numba_fused_no_rebalance_wealth64",
            "source_rows": len(log_history),
            "source_date_end": str(arguments["historical_dates"][-1].date()),
        }


def build_bounded_cycle_public() -> BoundedCyclePublicControl:
    return BoundedCyclePublicControl()
