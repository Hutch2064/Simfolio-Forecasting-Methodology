"""Pure moment calculations shared by experimental predictive-SV models."""

from __future__ import annotations

import math

import numpy as np


def predictive_state_moments(fit, horizon):
    """Exact multiscale moments under independent Gaussian state innovations.

    The residual and the four EWMA components share a common shock, so all
    cross-covariances are included. The fitted initial state is deterministic,
    matching the current multiscale forecast initialization.
    """
    q = fit["bdes_multiscale_vol"]
    days = np.arange(1, horizon + 1, dtype=np.float64)
    phi = np.r_[q["phis"], q["residual_phi"]]
    loading = np.r_[q["b"], 1.0]
    initial = np.r_[q["q_last"], q["residual_last"]]
    sd = np.r_[np.sqrt(np.maximum(q["q_var"], 1e-10)), q["residual_innovation_sd"]]
    common = np.r_[sd[:-1], sd[-1] * q["residual_common_loading"]]
    noise_cov = np.outer(common, common)
    noise_cov[-1, -1] = sd[-1] ** 2
    powers = phi[:, None] ** days[None, :]
    mean = q["ell"] + (loading * initial) @ powers
    variance = np.zeros(horizon)
    for left in range(len(phi)):
        for right in range(len(phi)):
            product = phi[left] * phi[right]
            variance += (loading[left] * loading[right] * noise_cov[left, right] *
                         (1.0 - product ** days) / (1.0 - product))
    base = fit["base_fit"]
    drift_phi = float(base["dlm_state_transition_phi"])
    drift_power = drift_phi ** days
    drift_mean = base["dlm_long_run_anchor_mean"] + base["dlm_state_posterior_deviation_mean"] * drift_power
    drift_var = (base["dlm_state_posterior_deviation_var"] * drift_power ** 2 +
                 base["dlm_state_noise_var"] * (1.0 - drift_power ** 2) / (1.0 - drift_phi ** 2))
    return mean, np.maximum(variance, 0.0), drift_mean, np.maximum(drift_var, 0.0)


def moment_return_curves(fit, horizon, *, shrink=0.0):
    mean_h, var_h, mean_mu, var_mu = predictive_state_moments(fit, horizon)
    base = fit["base_fit"]
    anchor = float(base["dlm_long_run_anchor_mean"])
    se = math.sqrt(fit["mean_meta"]["hac_long_run_variance"] / fit["n_obs"])
    t_squared = (anchor / max(se, 1e-15)) ** 2
    reliability = t_squared / (1.0 + t_squared)
    mean_mu -= shrink * (1.0 - reliability) * anchor
    # Match the first two return moments, including the nonlinear Jensen terms
    # and mean-state variance, under the candidate's Gaussian latent-state law.
    expected_sigma = np.exp(0.5 * mean_h + 0.125 * var_h) / 100.0
    expected_sigma_squared = np.exp(mean_h + 0.5 * var_h) / 10000.0
    mean_return = mean_mu * expected_sigma / base["sigma"]
    variance_return = (expected_sigma_squared *
                       (1.0 + (var_mu + mean_mu * mean_mu) / base["sigma"] ** 2) -
                       mean_return * mean_return)
    return mean_return, np.sqrt(np.maximum(variance_return, 0.0))


__all__ = ["moment_return_curves", "predictive_state_moments"]
