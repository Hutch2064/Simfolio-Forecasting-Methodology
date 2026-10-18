"""Raw stationary bootstrap with automatic PPW block-length selection.

    Politis and Romano (1994), DOI 10.1080/01621459.1994.10476870.
    Patton, Politis and White (2009), DOI 10.1080/07474930802459016.

No fitted drift, volatility multiplier, residual distribution or copula. Shared
historical row indices preserve cross-asset observations and local serial
dependence. The original policy rejoin is used for the asset-level candidate.
"""
from dataclasses import dataclass
import os

from arch.bootstrap import optimal_block_length
import numpy as np

from simfolio_forecasting_methodology.models.asset_level.frontier import (
    _historical_rebalance_dates, _validate_calendar,
)
from simfolio_forecasting_methodology.models.numerical.base_models import (
    _stationary_bootstrap_indices_from_draws,
)
from streamed_paths import rejoin

TIMINGS = {}
FIT_DIAGNOSTICS = {}
MODEL_IDS = ("asset_joint_raw_stationary_bootstrap_ppw", "portfolio_raw_stationary_bootstrap_ppw")


def block_length(history):
    centered = history - history.mean(axis=0)
    estimate = optimal_block_length(np.column_stack((history, centered * centered)))
    values = estimate["stationary"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ArithmeticError("nonfinite automatic stationary-bootstrap block length")
    return max(1.0, float(values.max()))


def indices(n, length, simulations, horizon, seed):
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, n, size=simulations, dtype=np.int32)
    flags = rng.random((simulations, horizon - 1)) < 1.0 / length
    points = rng.integers(0, n, size=flags.shape, dtype=np.int32)
    return _stationary_bootstrap_indices_from_draws(
        n=n, initial_indices=initial, restart_flags=flags, restart_points=points
    )


@dataclass(frozen=True)
class Candidate:
    model_id: str

    def simulate_daily_log_returns(self, training, context):
        training.validate()
        past, future = _validate_calendar(training, context)
        asset_level = self.model_id == MODEL_IDS[0]
        history = (np.asarray(training.asset_log_returns, dtype=float) if asset_level else
                   np.asarray(training.portfolio_log_returns, dtype=float)[:, None])
        length = block_length(history)
        sampled = history[indices(len(history), length, context.simulations,
                                  context.horizon_days, context.seed)]
        if asset_level:
            dates = _historical_rebalance_dates(past.append(future), training.policy.rebalance)
            weights = training.policy.weights
            mask = np.array([date in dates for date in future])
        else:
            weights = (1.0,)
            mask = np.zeros(context.horizon_days, dtype=bool)
        FIT_DIAGNOSTICS[self.model_id] = {"block_length": length,
            "selector": "arch.bootstrap.optimal_block_length stationary PPW2009",
            "selector_inputs": "returns and centered squared returns, all columns",
            "training_rows": len(history), "asset_level": asset_level,
            "mean": "unchanged empirical history", "seed": int(context.seed)}
        return rejoin(sampled, weights, mask)


selected = os.environ.get("SIMFOLIO_RAW_SBB", MODEL_IDS[0])
if selected not in MODEL_IDS:
    raise ValueError("unknown raw stationary-bootstrap candidate")
CANDIDATES = (Candidate(selected),)


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    rejoin(np.zeros((2, 3, 2)), (0.5, 0.5), np.array([False, True, False]))


def clear_path_cache():
    pass
