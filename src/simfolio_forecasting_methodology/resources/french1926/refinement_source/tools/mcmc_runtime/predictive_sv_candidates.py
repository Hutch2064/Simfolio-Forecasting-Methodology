"""Experimental asset-level predictive-SV models, outside the canonical catalogue.

The Gaussian-state candidate integrates the multiscale volatility and DLM mean
at each horizon by quadrature. It changes the forecast distribution: replacing
lagged empirical leverage/block shocks with Gaussian state shocks is an explicit
approximation, not a parity optimization or a claim to implement full INLA.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg


@njit(cache=True, fastmath=False)
def gaussian_uniform_kernel(initial, draws, loading, mean, phi, q_sd, r_sd):
    days = draws.shape[0]
    simulations, factors = initial.shape
    assets = loading.shape[0]
    out = np.empty((simulations, days, assets))
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    for sim in range(simulations):
        state = initial[sim].copy()
        for day in range(days):
            start = sim * factors
            state = state * phi + draws[day, start:start + factors] * q_sd
            offset = simulations * factors + sim * assets
            values = mean + draws[day, offset:offset + assets] * r_sd + state @ loading.T
            for asset in range(assets):
                value = 0.5 * (1.0 + math.erf(values[asset] * inv_sqrt_two))
                out[sim, day, asset] = min(max(value, 1e-8), 1.0 - 1e-8)
    return out


def fast_uniforms(model, simulations, horizon, rng):
    posterior_mean, posterior_covariance = dg.kalman_terminal_posterior(model)
    initial = rng.multivariate_normal(posterior_mean, posterior_covariance,
                                     size=simulations, check_valid="raise")
    draws = rng.normal(size=(horizon, simulations * (model["factor_count"] + model["asset_count"])))
    return gaussian_uniform_kernel(initial, draws, model["loading"], model["mean"],
                                   model["phi"], np.sqrt(model["innovation_variance"]),
                                   np.sqrt(model["residual_variance"]))


def fast_rebalanced(paths, weights, mask, *, cost_per_turnover_bps=15.0):
    """Same NumPy row operations as the reference, batched across simulations."""
    target = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    target /= target.sum()
    holdings = np.broadcast_to(target, (paths.shape[0], len(target))).copy()
    out = np.empty(paths.shape[:2])
    cost = max(float(cost_per_turnover_bps), 0.0) / 10000.0
    for day in range(paths.shape[1]):
        previous = holdings.sum(axis=1)
        holdings *= np.exp(np.clip(paths[:, day], -745.0, 50.0))
        ending = holdings.sum(axis=1)
        if mask[day]:
            turnover = np.abs(target - holdings / np.maximum(ending, 1e-300)[:, None]).sum(axis=1)
            ending = np.maximum(ending - ending * 0.5 * turnover * cost, 0.0)
            holdings = ending[:, None] * target
        # math.log has a few last-bit differences from np.log; preserve it.
        for simulation in range(paths.shape[0]):
            out[simulation, day] = math.log(max(ending[simulation], 1e-300) /
                                          max(previous[simulation], 1e-300))
    return out
