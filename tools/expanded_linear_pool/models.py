"""Equal linear opinion pool of two distinct coherent forecast models.

Gneiting and Ranjan (2013), Combining Predictive Distributions,
https://arxiv.org/abs/1106.1638 . This is a density mixture, not averaging
daily returns, fitted parameters, or scores. Each complete future path comes
from one component. Equal weights are specified before this candidate's test;
no test-period observations or fitted OOS weights enter either component.

The 240-path panel uses120 draws from each component. The total remains240;
this is a new forecast law, not a parity-preserving speed optimization.
"""
from dataclasses import dataclass, replace
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ["SIMFOLIO_TIMESCALE_VARIANT"] = "existing_sv_rate"
spec = importlib.util.spec_from_file_location(
    "equal_pool_private_M261", ROOT / "tools/data_selected_timescales/models.py"
)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)

from simfolio_forecasting_methodology.models.registry import build_model

FIRST = parent.CANDIDATES[0]
SECOND = build_model("bayesian_sbb_overlay_ewma_absolute")
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
MODEL_ID = "equal_path_pool_m261_and_bayesian_ewma_sbb"


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
