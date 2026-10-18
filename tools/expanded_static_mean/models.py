"""Exact M261 residual paths translated to the raw historical asset log mean.

Only location changes. The unchanged M261 simulation completes first; each
asset receives its deterministic horizon-specific translation before the
unchanged portfolio rejoin. No variance formula or RNG stream is replaced.
"""

import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
    moment_return_curves,
)

ROOT = Path(__file__).resolve().parents[2]
os.environ["SIMFOLIO_TIMESCALE_VARIANT"] = "existing_sv_rate"
spec = importlib.util.spec_from_file_location(
    "static_mean_private_M261", ROOT / "tools/data_selected_timescales/models.py"
)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)

MODEL_ID = "asset_m261_raw_historical_log_mean"
shell = parent.shell
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
TIMINGS = parent.TIMINGS


def translate_asset_paths_inplace(paths, translation):
    """Add one deterministic location vector without another path tensor."""
    np.add(paths, translation[None, :, :], out=paths)


def mean_translation(assets, horizon):
    translation = np.empty((horizon, assets.shape[1]))
    for asset, history in enumerate(assets.T):
        data = np.ascontiguousarray(history).tobytes()
        original_mean, _ = moment_return_curves(
            parent.parent.body.learned.student.original_fit(data), horizon
        )
        translation[:, asset] = history.mean() - original_mean
    return translation


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self, training, context):
        original_rejoin = shell.rejoin

        def translated_rejoin(paths, weights, rebalance_mask):
            started = time.perf_counter()
            assets = np.asarray(training.asset_log_returns, np.float64)
            translation = mean_translation(assets, context.horizon_days)
            translate_asset_paths_inplace(paths, translation)
            shell.controls.timed("asset_mean_translation", started)
            return original_rejoin(paths, weights, rebalance_mask)

        # This research adapter is selected alone in its process. The M261
        # asset workers finish before this synchronous rejoin is called.
        shell.rejoin = translated_rejoin
        try:
            result = super().simulate_daily_log_returns(training, context)
        finally:
            shell.rejoin = original_rejoin
        for history in np.asarray(training.asset_log_returns, np.float64).T:
            identity = hashlib.sha256(np.ascontiguousarray(history).tobytes()).hexdigest()
            FIT_DIAGNOSTICS[self.model_id, identity].update(
                return_mean_curve="constant_raw_historical_asset_log_mean",
                historical_asset_log_mean=float(history.mean()),
                mean_change="deterministic_asset_location_translation_before_portfolio_rejoin",
                predictive_variance="unchanged_M261_centered_asset_residuals",
            )
        return result


initialize = parent.initialize
CANDIDATES = (Candidate(model_id=MODEL_ID),)
MODEL_IDS = (parent.MODEL_IDS[0], MODEL_ID)
clear_path_cache = parent.clear_path_cache
source_hashes = parent.source_hashes
overlay = parent.overlay
load_paths = parent.load_paths
