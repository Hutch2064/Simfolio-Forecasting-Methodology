"""Isolated public SBB refinement. Never imported by the running catalogue."""
from dataclasses import asdict
import math
import numpy as np
from numba import njit
from simfolio_forecasting_methodology.french_panel.control_adapters import (
    fit_ewma_qmle, _ewma_variances,
)
from simfolio_forecasting_methodology.resources.french1926.control_sources.engine_helpers import (
    _politis_white_block_length, _stationary_bootstrap_indices_from_draws_fast,
)


def native_history(history):
    return np.log1p(np.clip(np.expm1(history), -0.999999,
                            np.finfo(np.float64).max / 10.0))


def public_indices(history, paths, horizon, seed):
    mean = np.mean(history)
    block = min(len(history), max(_politis_white_block_length(history),
                                  _politis_white_block_length((history-mean)**2)))
    rng = np.random.default_rng(seed)
    indices = _stationary_bootstrap_indices_from_draws_fast(
        n=len(history),
        initial_indices=rng.integers(0, len(history), size=paths, dtype=np.int32),
        restart_flags=rng.random((paths, max(horizon-1, 0))) < 1.0/block,
        restart_points=rng.integers(0, len(history), size=(paths, max(horizon-1, 0)),
                                    dtype=np.int32),
    ).astype(np.int64)
    return indices, block


@njit(cache=True, fastmath=False)
def conditional_paths(z, indices, mean, decay, variance):
    paths, horizon = indices.shape
    out = np.empty((paths, horizon))
    for p in range(paths):
        v, wealth, previous = variance, 1.0, 0.0
        for h in range(horizon):
            residual = math.sqrt(v)*z[indices[p, h]]
            r = mean+residual
            simple = min(max(math.expm1(r), -0.999999),
                         np.finfo(np.float64).max/10.0)
            wealth *= 1.0+simple
            current = math.log(max(wealth, 1.0e-300))
            out[p, h] = current-previous
            previous = current
            v = decay*v+(1.0-decay)*residual*residual
    return out


def simulate(history, indices):
    fit = fit_ewma_qmle(history)
    if fit.decay is None:
        return np.full(indices.shape, fit.mean), asdict(fit)
    variances, _ = _ewma_variances(history-fit.mean, fit.initial_variance, fit.decay)
    if np.any(variances <= 0):
        raise ValueError('nonpositive training conditional variance')
    z = (history-fit.mean)/np.sqrt(variances)
    z -= z.mean()
    z /= np.sqrt(np.mean(z*z))
    daily = conditional_paths(z, indices, fit.mean, fit.decay, fit.forecast_variance)
    return daily, {**asdict(fit), 'residual_mean': float(z.mean()),
                   'residual_second_moment': float(np.mean(z*z))}
