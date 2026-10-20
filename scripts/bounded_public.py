"""New horizon-conditional comparator; existing Testing Naive stays unchanged.

Each block draws length uniformly from 1..min(training rows, requested horizon),
then draws a historical start uniformly with replacement. Consecutive joint rows
wrap circularly. Only the last block is truncated. The bound is fixed for the
whole requested horizon, so changing that horizon can change its prefix law.
There is no Politis-White estimate, fitted block parameter, or mean adjustment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
from simfolio_forecasting_methodology.french_panel import control_adapters
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData

MODEL_ID = "model_testing_joint_uniform_bounded_block_bootstrap_v1"


def bounded_block_indices(n: int, horizon: int, paths: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Path-major draws: block length, historical start, then the next block."""
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) or x < 1
           for x in (n, horizon, paths)):
        raise ValueError("training rows, horizon and paths must be positive integers")
    maximum = min(n, horizon)
    indices = np.empty((paths, horizon), dtype=np.int64)
    for path in range(paths):
        position = 0
        while position < horizon:
            length = int(rng.integers(1, maximum + 1))
            start = int(rng.integers(0, n))
            copied = min(length, horizon - position)
            indices[path, position:position + copied] = (start + np.arange(copied)) % n
            position += copied
    return indices


class BoundedPublicControl:
    model_id: ClassVar[str] = MODEL_ID

    @property
    def dependency_identity(self) -> dict:
        return {
            "method": MODEL_ID,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_adapter_sha256": hashlib.sha256(
                Path(control_adapters.__file__).read_bytes()).hexdigest(),
            "training_validation_sha256": hashlib.sha256(
                (Path(control_adapters.__file__).parents[1] / "runner.py").read_bytes()).hexdigest(),
            "frozen_native_control": control_adapters.frozen_control_manifest(),
            "resampling": "joint_rows_circular_blocks_with_replacement",
            "block_length": "integer_uniform_inclusive_1_to_min_training_rows_requested_horizon",
            "draw_order": "path_major_then_per_block_length_then_historical_start",
            "last_block": "truncate_only_after_drawing_from_fixed_full_horizon_bound",
            "mean": "unshifted_historical_log_returns",
            "horizon_conditional": True,
            "prefix_coherence_across_requested_horizons": False,
            "politis_white": False,
            "fitted_block_calibration": False,
            "adaptation": "one_French_series_as_unit_weight_asset_no_costs_no_rebalancing",
            "historical_scores_inherited": False,
        }

    @property
    def implementation_identity(self) -> dict:
        dependencies = self.dependency_identity
        package = Path(control_adapters.__file__).parents[1]
        prefix = "src/simfolio_forecasting_methodology/"
        sources = {"scripts/bounded_public.py": dependencies["source_sha256"]}
        for relative in ("french_panel/control_adapters.py", "runner.py", "__init__.py",
                         "french_panel/__init__.py"):
            sources[prefix + relative] = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        control_root = "resources/french1926/control_sources/"
        for name in ["manifest.json", *dependencies["frozen_native_control"]["files"]]:
            relative = control_root + name
            sources[prefix + relative] = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        return {"schema": "french-external-model-implementation-v1", "model_id": MODEL_ID,
                "factory": "bounded_public:build_bounded_public", "source_files": sources,
                "numerical_contract": dependencies}

    @property
    def implementation_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.implementation_identity, ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def simulate_with_metadata(self, training: TrainingData, context: ForecastContext):
        arguments = control_adapters.native_control_arguments(training, context)
        # Preserve the frozen public kernel's simple/log/simple round trips and
        # clipping exactly. Inputs that overflow that preparation fail closed.
        simple = arguments["asset_returns"].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError("bounded comparator requires finite prepared simple returns")
        lower, upper = -0.999999, np.finfo(np.float64).max / 10.0
        log_history = np.log1p(np.clip(simple, lower, upper))
        source_simple = np.clip(np.expm1(log_history), lower, upper)
        indices = bounded_block_indices(len(log_history), context.horizon_days,
                                        context.simulations, arguments["rng"])
        kernel = control_adapters._native_kernel()
        unit_weight = np.ones(1, dtype=np.float64)
        output, _ = kernel._fused_no_rebalance_terminal_wealth(
            source_simple, indices, unit_weight, unit_weight,
            np.ones(context.horizon_days, dtype=np.float64))
        if output.shape != (context.simulations, context.horizon_days) or not np.isfinite(output).all():
            raise ValueError("bounded comparator produced invalid native wealth paths")
        return output, {
            "simulation_method": MODEL_ID,
            "innovation_resampling": "uniform_bounded_circular_blocks",
            "block_length_minimum": 1,
            "block_length_maximum": min(len(log_history), context.horizon_days),
            "block_bound_fixed_for_requested_horizon": True,
            "mean_model": "unshifted_historical_log_returns",
            "rejoin_implementation": "numba_fused_no_rebalance_wealth64",
            "source_rows": len(log_history),
            "source_date_end": str(arguments["historical_dates"][-1].date()),
        }

    def simulate_daily_log_returns(self, training: TrainingData,
                                   context: ForecastContext) -> np.ndarray:
        return self.simulate_with_metadata(training, context)[0]


def build_bounded_public() -> BoundedPublicControl:
    return BoundedPublicControl()
