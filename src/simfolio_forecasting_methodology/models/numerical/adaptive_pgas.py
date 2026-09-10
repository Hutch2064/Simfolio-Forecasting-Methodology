"""Source-derived adaptive-PGAS dynamic-factor dependence closure.

This packaged closure is limited to the historical exact-Kalman asset-level
dependence dispatcher.  It keeps the source numerical call order and fails
closed when the required numerical diagnostics are unavailable.

Model
-----
The retained exact-Kalman dispatcher uses the Gaussian special case of the
dynamic factor state-space model.  It maps pseudo-observations to normal
scores, samples the exact Kalman terminal-state posterior, and generates
future joint asset uniforms for the source marginal-path rejoin.

The retained dispatcher explicitly disables the Student-t scale-mixture
branch.  This package therefore rejects that configuration instead of
shipping an unassigned PGAS candidate.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit, prange
from scipy.special import ndtri

Array = np.ndarray

SOURCE_PGAS_WRAPPER_PATH = "tmp/asset_level_full_panel_20260823/simfolio_adaptive_pgas.py"
SOURCE_PGAS_WRAPPER_SHA256 = "f7988a6cfbdf674c1efeb1ca6836b34e5f34c1ee241e97ef44551ded7ba87c69"


def _fit_dynamic_factor_model(
    history: Array,
    *,
    pseudo_observations: Callable[[Array], Array],
    student_copula_fit: Callable[[Array], tuple[Array, float]],
    stable_correlation: Callable[[Array], Array],
    student_scale_mixture: bool,
) -> dict[str, Any]:
    if bool(student_scale_mixture):
        raise ValueError("pgas_student_scale_mixture_is_not_authorized_for_exact_kalman_dispatch")
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 80 or values.shape[1] < 2:
        raise ValueError("pgas_requires_at_least_80_observations_and_two_assets")
    if not np.all(np.isfinite(values)):
        raise ValueError("pgas_nonfinite_asset_history")

    uniforms = np.asarray(pseudo_observations(values), dtype=np.float64)
    if uniforms.shape != values.shape or not np.all(np.isfinite(uniforms)):
        raise ValueError("pgas_invalid_pseudo_observations")
    observations = ndtri(np.clip(uniforms, 1e-8, 1.0 - 1e-8))
    if not np.all(np.isfinite(observations)):
        raise ValueError("pgas_nonfinite_student_t_scores")
    mean = np.mean(observations, axis=0)
    centered = observations - mean[None, :]
    covariance = np.cov(centered, rowvar=False, ddof=1)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != values.shape[1]:
        raise ValueError("pgas_invalid_asset_covariance")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= -1e-7:
        raise ValueError("pgas_non_psd_asset_covariance")
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    cumulative = np.cumsum(eigenvalues) / max(float(np.sum(eigenvalues)), 1e-12)
    factor_count = int(np.searchsorted(cumulative, 0.80) + 1)
    factor_count = max(1, min(factor_count, 3, int(values.shape[1])))
    loading = eigenvectors[:, :factor_count] * np.sqrt(eigenvalues[:factor_count])[None, :]
    residual_variance = np.diag(covariance) - np.sum(loading * loading, axis=1)
    if not np.all(np.isfinite(residual_variance)) or np.any(residual_variance <= 1e-7):
        raise ValueError("pgas_nonpositive_factor_residual_variance")

    factor_scores = centered @ loading @ np.linalg.inv(loading.T @ loading)
    phi = np.empty(factor_count, dtype=np.float64)
    innovation_variance = np.empty(factor_count, dtype=np.float64)
    for factor in range(factor_count):
        left = factor_scores[:-1, factor]
        right = factor_scores[1:, factor]
        denominator = float(np.dot(left, left))
        fitted = float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
        phi[factor] = float(np.clip(fitted, -0.995, 0.995))
        variance = float(np.var(right - phi[factor] * left, ddof=1))
        innovation_variance[factor] = max(variance, 1e-7)
    initial_variance = innovation_variance / np.maximum(1.0 - phi * phi, 1e-7)
    return {
        "mean": mean,
        "observations": observations,
        "loading": loading,
        "residual_variance": residual_variance,
        "phi": phi,
        "innovation_variance": innovation_variance,
        "initial_variance": initial_variance,
        "degrees_of_freedom": None,
        "student_scale_mixture": False,
        "factor_count": factor_count,
        "observation_count": int(values.shape[0]),
        "asset_count": int(values.shape[1]),
    }


def _kalman_terminal_posterior_numba_dense(
    observations: Array,
    loading: Array,
    mean: Array,
    phi: Array,
    initial_variance: Array,
    innovation_variance: Array,
    residual_variance: Array,
) -> tuple[Array, Array]:
    """JIT kernel for the exact linear-Gaussian terminal filter."""
    observation_count = observations.shape[0]
    asset_count = observations.shape[1]
    factor_count = loading.shape[1]
    state_mean = np.zeros(factor_count, dtype=np.float64)
    state_covariance = np.zeros((factor_count, factor_count), dtype=np.float64)
    innovation_covariance = np.zeros((factor_count, factor_count), dtype=np.float64)
    observation_covariance = np.zeros((asset_count, asset_count), dtype=np.float64)
    for factor in range(factor_count):
        state_covariance[factor, factor] = initial_variance[factor]
        innovation_covariance[factor, factor] = innovation_variance[factor]
    for asset in range(asset_count):
        observation_covariance[asset, asset] = residual_variance[asset]
    for time_index in range(observation_count):
        if time_index > 0:
            state_mean = phi * state_mean
            state_covariance = (
                phi[:, None] * state_covariance * phi[None, :] + innovation_covariance
            )
        innovation = observations[time_index] - mean - loading @ state_mean
        observation_covariance_t = loading @ state_covariance @ loading.T + observation_covariance
        state_observation_covariance = state_covariance @ loading.T
        kalman_gain = np.linalg.solve(
            observation_covariance_t,
            state_observation_covariance.T,
        ).T
        state_mean = state_mean + kalman_gain @ innovation
        state_covariance = state_covariance - kalman_gain @ observation_covariance_t @ kalman_gain.T
        state_covariance = 0.5 * (state_covariance + state_covariance.T)
    return state_mean, state_covariance


@njit(cache=True)
def _kalman_terminal_posterior_numba_information(
    observations: Array,
    loading: Array,
    mean: Array,
    phi: Array,
    initial_variance: Array,
    innovation_variance: Array,
    residual_variance: Array,
) -> tuple[Array, Array]:
    """Exact Kalman filter in factor-space information form.

    The observation covariance is ``H P H' + R`` with diagonal ``R``.  The
    dense implementation above solves an asset-sized system for every
    observation even though the latent state has at most three dimensions.
    Conditioning on the observation in information form instead requires
    only a factor-sized solve while preserving the same linear-Gaussian model.
    The dense kernel remains available as a numerical reference for parity
    tests in the research harness.
    """
    observation_count = observations.shape[0]
    asset_count = observations.shape[1]
    factor_count = loading.shape[1]
    state_mean = np.zeros(factor_count, dtype=np.float64)
    state_covariance = np.zeros((factor_count, factor_count), dtype=np.float64)
    observation_information = np.zeros((factor_count, factor_count), dtype=np.float64)
    for factor in range(factor_count):
        state_covariance[factor, factor] = initial_variance[factor]
    for factor_left in range(factor_count):
        for factor_right in range(factor_count):
            value = 0.0
            for asset in range(asset_count):
                value += (
                    loading[asset, factor_left]
                    * loading[asset, factor_right]
                    / residual_variance[asset]
                )
            observation_information[factor_left, factor_right] = value

    identity = np.eye(factor_count, dtype=np.float64)
    for time_index in range(observation_count):
        if time_index > 0:
            state_mean = phi * state_mean
            for factor_left in range(factor_count):
                for factor_right in range(factor_count):
                    state_covariance[factor_left, factor_right] = (
                        phi[factor_left]
                        * state_covariance[factor_left, factor_right]
                        * phi[factor_right]
                    )
                state_covariance[factor_left, factor_left] += innovation_variance[factor_left]

        prior_precision = np.linalg.inv(state_covariance)
        posterior_precision = prior_precision + observation_information
        centered_observation = observations[time_index] - mean
        information_vector = prior_precision @ state_mean
        for factor in range(factor_count):
            value = 0.0
            for asset in range(asset_count):
                value += (
                    loading[asset, factor] * centered_observation[asset] / residual_variance[asset]
                )
            information_vector[factor] += value
        state_covariance = np.linalg.solve(posterior_precision, identity)
        state_covariance = 0.5 * (state_covariance + state_covariance.T)
        state_mean = np.linalg.solve(posterior_precision, information_vector)
    return state_mean, state_covariance


@njit(cache=True)
def _small_factor_inverse_numba(matrix: Array, output: Array) -> None:
    """Invert a positive-definite factor-space matrix with dimension 1--3.

    The exact-Kalman candidate caps its latent factor dimension at three.  A
    generic ``np.linalg.inv`` is materially more expensive than the closed
    form SPD inverses at these dimensions and is called once per historical
    observation.  The principal-minor checks keep this an exact all-or-error
    operation: an invalid posterior is reported rather than repaired or
    replaced by a fallback method.
    """
    factor_count = matrix.shape[0]
    if factor_count == 1:
        value = matrix[0, 0]
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("pgas_nonpositive_factor_precision")
        output[0, 0] = 1.0 / value
        return
    if factor_count == 2:
        a00 = matrix[0, 0]
        a01 = matrix[0, 1]
        a11 = matrix[1, 1]
        determinant = a00 * a11 - a01 * a01
        if not np.isfinite(determinant) or not np.isfinite(a00) or a00 <= 0.0 or determinant <= 0.0:
            raise ValueError("pgas_nonpositive_factor_precision")
        reciprocal = 1.0 / determinant
        output[0, 0] = a11 * reciprocal
        output[0, 1] = -a01 * reciprocal
        output[1, 0] = -a01 * reciprocal
        output[1, 1] = a00 * reciprocal
        return
    if factor_count == 3:
        a00 = matrix[0, 0]
        a01 = matrix[0, 1]
        a02 = matrix[0, 2]
        a11 = matrix[1, 1]
        a12 = matrix[1, 2]
        a22 = matrix[2, 2]
        cofactor00 = a11 * a22 - a12 * a12
        cofactor01 = a02 * a12 - a01 * a22
        cofactor02 = a01 * a12 - a02 * a11
        cofactor11 = a00 * a22 - a02 * a02
        cofactor12 = a01 * a02 - a00 * a12
        cofactor22 = a00 * a11 - a01 * a01
        determinant = a00 * cofactor00 + a01 * cofactor01 + a02 * cofactor02
        if (
            not np.isfinite(determinant)
            or not np.isfinite(a00)
            or not np.isfinite(cofactor00)
            or a00 <= 0.0
            or cofactor00 <= 0.0
            or determinant <= 0.0
        ):
            raise ValueError("pgas_nonpositive_factor_precision")
        reciprocal = 1.0 / determinant
        output[0, 0] = cofactor00 * reciprocal
        output[0, 1] = cofactor01 * reciprocal
        output[0, 2] = cofactor02 * reciprocal
        output[1, 0] = cofactor01 * reciprocal
        output[1, 1] = cofactor11 * reciprocal
        output[1, 2] = cofactor12 * reciprocal
        output[2, 0] = cofactor02 * reciprocal
        output[2, 1] = cofactor12 * reciprocal
        output[2, 2] = cofactor22 * reciprocal
        return
    raise ValueError("pgas_unsupported_factor_dimension")


@njit(cache=True)
def _kalman_terminal_posterior_numba_information_small(
    observations: Array,
    loading: Array,
    mean: Array,
    phi: Array,
    initial_variance: Array,
    innovation_variance: Array,
    residual_variance: Array,
) -> tuple[Array, Array]:
    """Exact information-form Kalman filter using small factor-space solves."""
    observation_count = observations.shape[0]
    asset_count = observations.shape[1]
    factor_count = loading.shape[1]
    if factor_count < 1 or factor_count > 3:
        raise ValueError("pgas_unsupported_factor_dimension")
    state_mean = np.zeros(factor_count, dtype=np.float64)
    state_covariance = np.zeros((factor_count, factor_count), dtype=np.float64)
    observation_information = np.zeros((factor_count, factor_count), dtype=np.float64)
    prior_precision = np.empty((factor_count, factor_count), dtype=np.float64)
    posterior_precision = np.empty((factor_count, factor_count), dtype=np.float64)
    information_vector = np.empty(factor_count, dtype=np.float64)
    next_state_mean = np.empty(factor_count, dtype=np.float64)
    for factor in range(factor_count):
        state_covariance[factor, factor] = initial_variance[factor]
    for factor_left in range(factor_count):
        for factor_right in range(factor_count):
            value = 0.0
            for asset in range(asset_count):
                value += (
                    loading[asset, factor_left]
                    * loading[asset, factor_right]
                    / residual_variance[asset]
                )
            observation_information[factor_left, factor_right] = value

    for time_index in range(observation_count):
        if time_index > 0:
            for factor in range(factor_count):
                state_mean[factor] = phi[factor] * state_mean[factor]
            for factor_left in range(factor_count):
                for factor_right in range(factor_count):
                    state_covariance[factor_left, factor_right] = (
                        phi[factor_left]
                        * state_covariance[factor_left, factor_right]
                        * phi[factor_right]
                    )
                state_covariance[factor_left, factor_left] += innovation_variance[factor_left]

        _small_factor_inverse_numba(state_covariance, prior_precision)
        for factor_left in range(factor_count):
            for factor_right in range(factor_count):
                posterior_precision[factor_left, factor_right] = (
                    prior_precision[factor_left, factor_right]
                    + observation_information[factor_left, factor_right]
                )
        for factor in range(factor_count):
            value = 0.0
            for prior_factor in range(factor_count):
                value += prior_precision[factor, prior_factor] * state_mean[prior_factor]
            for asset in range(asset_count):
                value += (
                    loading[asset, factor]
                    * (observations[time_index, asset] - mean[asset])
                    / residual_variance[asset]
                )
            information_vector[factor] = value

        _small_factor_inverse_numba(posterior_precision, state_covariance)
        for factor in range(factor_count):
            value = 0.0
            for posterior_factor in range(factor_count):
                value += (
                    state_covariance[factor, posterior_factor]
                    * information_vector[posterior_factor]
                )
            next_state_mean[factor] = value
        for factor in range(factor_count):
            state_mean[factor] = next_state_mean[factor]
    return state_mean, state_covariance


@njit(parallel=True, cache=True)
def _simulate_future_gaussian_uniforms_numba(
    factor_state: Array,
    random_draws: Array,
    phi: Array,
    innovation_scale: Array,
    residual_scale: Array,
    loading: Array,
    mean: Array,
) -> tuple[Array, Array]:
    """JIT the Gaussian factor recurrence, projection, and normal CDF."""
    simulation_count = factor_state.shape[0]
    forecast_horizon = random_draws.shape[0]
    factor_count = factor_state.shape[1]
    asset_count = loading.shape[0]
    output = np.empty((simulation_count, forecast_horizon, asset_count), dtype=np.float64)
    final_state = np.empty_like(factor_state)
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    for simulation in prange(simulation_count):
        state = np.empty(factor_count, dtype=np.float64)
        for factor in range(factor_count):
            state[factor] = (
                factor_state[simulation, factor] * phi[factor]
                + random_draws[0, simulation, factor] * innovation_scale[factor]
            )
        for day in range(forecast_horizon):
            if day > 0:
                for factor in range(factor_count):
                    state[factor] = (
                        state[factor] * phi[factor]
                        + random_draws[day, simulation, factor] * innovation_scale[factor]
                    )
            for asset in range(asset_count):
                value = (
                    mean[asset]
                    + random_draws[day, simulation, factor_count + asset] * residual_scale[asset]
                )
                for factor in range(factor_count):
                    value += state[factor] * loading[asset, factor]
                output[simulation, day, asset] = 0.5 * (1.0 + math.erf(value * inverse_sqrt_two))
        for factor in range(factor_count):
            final_state[simulation, factor] = state[factor]
    return output, final_state


@njit(parallel=True, cache=True)
def _simulate_future_gaussian_uniforms_flat_numba(
    factor_state: Array,
    random_draws: Array,
    phi: Array,
    innovation_scale: Array,
    residual_scale: Array,
    loading: Array,
    mean: Array,
) -> tuple[Array, Array]:
    """Exact Gaussian projection from day-major contiguous random draws."""
    simulation_count = factor_state.shape[0]
    forecast_horizon = random_draws.shape[0]
    factor_count = factor_state.shape[1]
    asset_count = loading.shape[0]
    output = np.empty((simulation_count, forecast_horizon, asset_count), dtype=np.float64)
    final_state = np.empty_like(factor_state)
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    for simulation in prange(simulation_count):
        state = np.empty(factor_count, dtype=np.float64)
        for factor in range(factor_count):
            state[factor] = (
                factor_state[simulation, factor] * phi[factor]
                + random_draws[0, simulation * factor_count + factor] * innovation_scale[factor]
            )
        for day in range(forecast_horizon):
            if day > 0:
                for factor in range(factor_count):
                    state[factor] = (
                        state[factor] * phi[factor]
                        + random_draws[day, simulation * factor_count + factor]
                        * innovation_scale[factor]
                    )
            asset_draw_start = simulation_count * factor_count + simulation * asset_count
            for asset in range(asset_count):
                value = (
                    mean[asset]
                    + random_draws[day, asset_draw_start + asset] * residual_scale[asset]
                )
                for factor in range(factor_count):
                    value += state[factor] * loading[asset, factor]
                output[simulation, day, asset] = 0.5 * (1.0 + math.erf(value * inverse_sqrt_two))
        for factor in range(factor_count):
            final_state[simulation, factor] = state[factor]
    return output, final_state


def _kalman_terminal_posterior(
    observations: Array,
    model: dict[str, Any],
) -> tuple[Array, Array]:
    """Compute the exact terminal factor posterior for the Gaussian model."""
    observations = np.asarray(observations, dtype=np.float64)
    state_mean, state_covariance = _kalman_terminal_posterior_numba_information_small(
        observations,
        model["loading"],
        model["mean"],
        model["phi"],
        model["initial_variance"],
        model["innovation_variance"],
        model["residual_variance"],
    )
    if not np.all(np.isfinite(state_mean)) or not np.all(np.isfinite(state_covariance)):
        raise ValueError("pgas_nonfinite_kalman_terminal_posterior")
    try:
        np.linalg.cholesky(state_covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("pgas_nonpositive_kalman_terminal_posterior") from exc
    return state_mean, state_covariance


def _simulate_future_gaussian_uniforms(
    model: dict[str, Any],
    factor_state: Array,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
    block_callback: Callable[[int, Array], None] | None = None,
) -> Array:
    """Vectorized Gaussian future simulation with the original draw order.

    The dynamic factor recurrence still has to be evaluated sequentially in
    time, but all simulation paths and all asset innovations are generated in
    one contiguous draw.  The final normal CDF is likewise applied once to
    the complete tensor instead of once per forecast day.
    """
    factor_state = np.asarray(factor_state, dtype=np.float64)
    simulation_count = int(n_sims)
    forecast_horizon = int(horizon)
    factor_count = int(model["factor_count"])
    asset_count = int(model["asset_count"])
    # Generate and consume bounded day blocks in the same day-major draw
    # order as the former full-horizon buffer.  This preserves seeded output
    # while avoiding a second full forecast tensor for random draws.
    uniforms = (
        np.empty((simulation_count, forecast_horizon, asset_count), dtype=np.float64)
        if block_callback is None
        else None
    )
    try:
        block_size = max(64, int(os.environ.get("SIMFOLIO_FASTMAP_DEPENDENCE_BLOCK_SIZE", "512")))
    except (TypeError, ValueError):
        block_size = 512
    for start in range(0, forecast_horizon, block_size):
        end = min(start + block_size, forecast_horizon)
        block_horizon = end - start
        # A single contiguous draw has the same NumPy RNG sequence as the
        # former per-day factor-then-asset calls, while avoiding Python-level
        # calls and the strided 3-D random-draw layout.
        random_draws = rng.normal(
            size=(block_horizon, simulation_count * (factor_count + asset_count)),
        )
        block_uniforms, factor_state = _simulate_future_gaussian_uniforms_flat_numba(
            factor_state,
            random_draws,
            model["phi"],
            np.sqrt(model["innovation_variance"]),
            np.sqrt(model["residual_variance"]),
            model["loading"],
            model["mean"],
        )
        if not np.all(np.isfinite(block_uniforms)):
            raise ValueError("pgas_nonfinite_future_uniforms")
        # The callback consumes the block immediately.  Clip in place so the
        # exact seeded values are unchanged without allocating a second full
        # block beside the Numba output.
        np.clip(block_uniforms, 1e-8, 1.0 - 1e-8, out=block_uniforms)
        if block_callback is None:
            uniforms[:, start:end, :] = block_uniforms
        else:
            block_callback(int(start), block_uniforms)
    if block_callback is not None:
        return None
    if not np.all(np.isfinite(uniforms)):
        raise ValueError("pgas_nonfinite_future_uniforms")
    return uniforms


def adaptive_pgas_uniform_paths(
    history: Array,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
    *,
    pseudo_observations: Callable[[Array], Array],
    student_copula_fit: Callable[[Array], tuple[Array, float]],
    stable_correlation: Callable[[Array], Array],
    config: dict[str, Any] | None = None,
    uniform_callback: Callable[[int, Array], None] | None = None,
) -> tuple[Array | None, dict[str, Any]]:
    """Fit the source Gaussian factor model and simulate joint asset uniforms."""
    settings = {
        "min_particles": 32,
        "max_particles": 128,
        "pilot_replicates": 3,
        "target_loglik_variance": 2.0,
        "ess_fraction": 0.50,
        "pgas_iterations": 12,
        "pgas_burn_in": 6,
        "student_scale_mixture": False,
    }
    if config:
        settings.update({key: value for key, value in config.items() if value is not None})
    if bool(settings["student_scale_mixture"]):
        raise ValueError("pgas_student_scale_mixture_is_not_authorized_for_exact_kalman_dispatch")
    if (
        int(settings["pgas_iterations"]) <= int(settings["pgas_burn_in"])
        or int(settings["pgas_burn_in"]) < 1
    ):
        raise ValueError("pgas_invalid_iteration_schedule")
    model = _fit_dynamic_factor_model(
        history,
        pseudo_observations=pseudo_observations,
        student_copula_fit=student_copula_fit,
        stable_correlation=stable_correlation,
        student_scale_mixture=False,
    )
    observations = model["observations"]
    terminal_mean, terminal_covariance = _kalman_terminal_posterior(observations, model)
    factor_state = rng.multivariate_normal(
        terminal_mean,
        terminal_covariance,
        size=int(n_sims),
        check_valid="raise",
    )
    output = _simulate_future_gaussian_uniforms(
        model,
        factor_state,
        n_sims,
        horizon,
        rng,
        block_callback=uniform_callback,
    )
    return output, {
        "copula": "adaptive dynamic Gaussian factor dependence",
        "inference": "exact Kalman terminal-state posterior sampling",
        "factor_count": int(model["factor_count"]),
        "degrees_of_freedom": None,
        "student_scale_mixture": False,
        "observation_count": int(model["observation_count"]),
        "asset_count": int(model["asset_count"]),
        "pgas_iterations": 0,
        "pgas_burn_in": 0,
        "selected_particles": None,
        "ess_fraction": float(settings["ess_fraction"]),
        "pilot": None,
        "pgas_diagnostics": [],
        "terminal_state_draws": int(n_sims),
    }
