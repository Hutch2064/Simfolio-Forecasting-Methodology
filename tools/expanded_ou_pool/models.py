"""Equal complete-path pool of exact OU residual volatility and Bayesian EWMA.

This is a new forecast law, not a parity-preserving speed optimization.
Each component retains its original fit, seed contract, and portfolio rejoin.
"""
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "equal_ou_pool_private_M244", ROOT / "tools/endpoint_residual_volatility/models.py"
)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)

from simfolio_forecasting_methodology.models.registry import build_model

FIRST = next(c for c in parent.CANDIDATES
             if c.model_id == "asset_map_stationary_ar1_ou_residual_volatility")
SECOND = build_model("bayesian_sbb_overlay_ewma_absolute")
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
MODEL_ID = "equal_path_pool_stationary_ou_and_bayesian_ewma_sbb"


@dataclass(frozen=True)
class Candidate:
    model_id: str = MODEL_ID

    def simulate_daily_log_returns(self, training, context):
        if context.simulations < 2 or context.simulations % 2:
            raise ValueError("equal-stratified pool requires an even simulation count")
        half = context.simulations // 2
        first = FIRST.simulate_daily_log_returns(
            training, replace(context, model_id=FIRST.model_id, simulations=half)
        )
        second = SECOND.simulate_daily_log_returns(
            training, replace(context, model_id=SECOND.model_id, simulations=half)
        )
        return np.concatenate((first, second), axis=0)


initialize = parent.initialize
clear_path_cache = parent.clear_path_cache
CANDIDATES = (Candidate(),)
