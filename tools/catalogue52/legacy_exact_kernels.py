"""Scoped compiler-only adapter for original catalogue models M001, M008 and M009.

No fit cache, inference settings, seed, sampling budget or dependence changes.
The context always restores every patched global, including after exceptions.
"""
from __future__ import annotations

import hashlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/mcmc_runtime"))
from compiled_smoothers import make as make_smoothers
from exact_path_kernels import fast_rejoin
from research_jit_pilot import make_accelerators

from simfolio_forecasting_methodology.models.asset_level import exact_kalman, frontier
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg
from simfolio_forecasting_methodology.models.numerical import inla_bdes as ib
from simfolio_forecasting_methodology.models.numerical import mcmc_sv as mc
from simfolio_forecasting_methodology.models.portfolio import inla

CANDIDATES = ()
_REPLACEMENTS = None


def source_version():
    paths = [Path(module.__file__) for module in (bd, dg, ib, mc, frontier, exact_kalman, inla)]
    paths += [ROOT / "tools/mcmc_runtime" / name for name in
              ("compiled_smoothers.py", "exact_path_kernels.py", "research_jit_pilot.py")]
    paths += [Path(__file__)]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _rejoin(log_paths, weights, rebalance_mask, *, cost_per_turnover_bps=15.0):
    # Match all original input validation before the retained exact recurrence.
    paths = np.asarray(log_paths, dtype=np.float64)
    if paths.ndim != 3 or not np.all(np.isfinite(paths)):
        raise ValueError("frontier_invalid_rejoin_paths")
    target = np.asarray(weights, dtype=np.float64)
    if target.ndim != 1 or target.size != paths.shape[2] or not np.all(np.isfinite(target)):
        raise ValueError("frontier_invalid_rejoin_weights")
    if float(np.maximum(target, 0.0).sum()) <= 0.0:
        raise ValueError("frontier_nonpositive_rejoin_weights")
    mask = np.asarray(rebalance_mask, dtype=bool)
    if mask.shape != (paths.shape[1],):
        raise ValueError("frontier_invalid_rebalance_mask")
    return fast_rejoin(paths, target, mask, cost_per_turnover_bps=cost_per_turnover_bps)


def prepare():
    global _REPLACEMENTS
    if _REPLACEMENTS is not None:
        return
    _, _, dlm, sv = make_accelerators()
    rts, ewma = make_smoothers()
    mc_dlm = njit(cache=False, fastmath=False)(mc._dlm_ar1_loglik)
    x = np.array([0.1, 0.2, -0.3])
    rts(x, 0.0, 0.9, 0.2)
    ewma(x, 0.1)
    mc_dlm(x, 1.0, 0.9, 0.01)
    _rejoin(np.zeros((2, 2, 1)), (1.0,), np.array([False, True]))
    _REPLACEMENTS = [(bd, "_dlm_ar1_loglik", dlm),
                     (mc, "_dlm_ar1_loglik", mc_dlm),
                     (bd, "_sv_kalman_filter", sv), (ib, "_sv_kalman_filter", sv),
                     (bd, "_sv_kalman_rts_smoother_mean", rts),
                     (ib, "_sv_kalman_rts_smoother_mean", rts),
                     (bd, "_bdes_ewma", ewma),
                     (dg, "rebalanced_portfolio_log_paths", _rejoin),
                     (frontier, "rebalanced_portfolio_log_paths", _rejoin),
                     (exact_kalman, "rebalanced_portfolio_log_paths", _rejoin)]


@contextmanager
def accelerated():
    prepare()
    saved = [(module, name, getattr(module, name)) for module, name, _ in _REPLACEMENTS]
    try:
        for module, name, replacement in _REPLACEMENTS:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


@dataclass(frozen=True)
class ExactKernelModel:
    reference: object

    @property
    def model_id(self):
        return self.reference.model_id

    def simulate_daily_log_returns(self, training, context):
        with accelerated():
            return self.reference.simulate_daily_log_returns(training, context)


def initialize(_data=None, _workers=1, _simulations=1):
    global CANDIDATES
    prepare()
    CANDIDATES = (ExactKernelModel(frontier.HistoricalFrontierModel()),
                  ExactKernelModel(exact_kalman.ExactKalmanDynamicGaussianFactorModel()),
                  ExactKernelModel(inla.FullINLABDESModel()))
