"""Daily-grid fractional OU and rough Heston kernels, with an error budget.

The fractional integral has alpha=H+1/2. Its exact daily cell averages are
((j+1)**alpha-j**alpha)/Gamma(alpha+1). Quadrature approximates these averages,
not the singular point value at zero. The OU high-pass transformation turns
the fractional integral into a signed fractional OU impulse response.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import gamma, hyp1f1

H_BOUNDS = (0.03, 0.49)
KAPPA_BOUNDS = (1 / 2520, 0.5)
MAX_SPAN = 25200
GAUSS_NODES, GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(3)


def fractional_cells(hurst, lags):
    alpha = hurst + 0.5
    lags = np.asarray(lags, dtype=float)
    out = np.empty_like(lags)
    zero = lags == 0
    out[zero] = 1 / gamma(alpha + 1)
    # Stable subtraction at large lag.
    out[~zero] = (lags[~zero] ** alpha * np.expm1(alpha * np.log1p(1 / lags[~zero]))
                  / gamma(alpha + 1))
    return out


def quadrature(hurst, edges):
    alpha = hurst + 0.5
    power = 1 - alpha
    nodes, gauss = GAUSS_NODES, GAUSS_WEIGHTS
    log_edges = np.log(edges)
    widths = np.diff(log_edges) / 2
    locations = (log_edges[1:] + log_edges[:-1]) / 2
    rates = np.exp((locations[:, None] + widths[:, None] * nodes).ravel())
    mass = (widths[:, None] * gauss * rates.reshape(-1, 3) ** power).ravel()
    mass /= gamma(alpha) * gamma(1 - alpha)
    weights = mass * (-np.expm1(-rates)) / rates
    # The unresolved low-frequency integral contributes an almost constant
    # factor. Keep it explicitly rather than losing persistent variance.
    low_mass = edges[0] ** power / power / (gamma(alpha) * gamma(1 - alpha))
    low_rate = edges[0] / 2
    rates, weights = np.r_[low_rate, rates], np.r_[low_mass, weights]
    # Daily-grid rates this large are white at machine precision. Aggregating
    # their common-driver contribution avoids redundant states; the first
    # nonzero-lag absolute impulse error is bounded by 1e-15 * sum(weights).
    fast = rates >= -np.log(1e-15)
    if np.any(fast):
        rates, weights = np.r_[rates[~fast], 1e9], np.r_[weights[~fast], weights[fast].sum()]
    return rates, weights


def fou_cells(hurst, kappa, count):
    alpha = hurst + .5
    times = np.arange(count + 1, dtype=float)
    accumulated = times ** alpha / gamma(alpha + 1) * hyp1f1(1, alpha + 1, -kappa * times)
    return np.diff(accumulated)


@lru_cache(maxsize=4)
def grid(tolerance):
    """Choose a fixed grid valid over the entire parameter prior domain.

    Fixing the error-selected grid before MCMC avoids parameter-dependent
    latent dimensions. Factor count is an outcome, never a model parameter.
    """
    lags = np.unique(np.r_[np.arange(64), np.geomspace(64, MAX_SPAN, 512).astype(int)])
    worst = np.inf
    for count in (8, 12, 16, 24, 32, 48, 64):
        edges = np.geomspace(1e-11, 1e9, count + 1)
        worst = 0.
        for hurst in np.linspace(*H_BOUNDS, 17):
            rates, weights = quadrature(hurst, edges)
            approximate = np.exp(-lags[:, None] * rates) @ weights
            exact = fractional_cells(hurst, lags)
            worst = max(worst, float(np.max(np.abs(approximate / exact - 1))))
        if worst <= tolerance:
            return edges, {'fractional_cell_max_relative_error': worst,
                           'tolerance': tolerance, 'factors': len(rates),
                           'machine_precision_white_aggregation': 1e-15,
                           'maximum_checked_lag': MAX_SPAN, 'hurst_grid_points': 17}
    raise ValueError(f'fractional kernel did not meet error budget: {worst}')


@lru_cache(maxsize=4)
def stationary_root(tolerance):
    edges, _ = grid(tolerance)
    rates, _ = quadrature(.1, edges)
    innovation = np.sqrt(-np.expm1(-2 * rates))
    covariance = np.outer(innovation, innovation) / -np.expm1(-rates[:, None] - rates[None, :])
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-8:
        raise ValueError('fractional OU covariance is not positive semidefinite')
    active = values > 1e-11
    root = vectors * np.sqrt(np.maximum(values, 0))[None, :]
    inverse_root = np.zeros_like(root)
    inverse_root[:, active] = vectors[:, active] / np.sqrt(values[active])[None, :]
    return root, inverse_root, covariance


def coefficients(hurst, kappa, tolerance, with_root=True):
    edges, evidence = grid(tolerance)
    rates, weights = quadrature(hurst, edges)
    phi = np.exp(-rates)
    p = np.exp(-kappa)
    a = -np.expm1(-kappa) / kappa
    denominator = rates - kappa
    if np.min(np.abs(denominator)) < 1e-12:
        # This parameter set requires a repeated-pole representation, rather
        # than unstable cancellation. Its zero-measure boundary is excluded.
        raise ValueError('fractional OU repeated pole')
    # Exact CELL INTEGRALS of the continuous lifted fOU impulse response:
    # mass*x/(x-kappa)*exp(-x*t) - mass*kappa/(x-kappa)*exp(-kappa*t).
    # weights already include the first exponential's daily cell integral.
    cell_integral = -np.expm1(-rates) / rates
    mass = weights / cell_integral
    signed = weights * rates / denominator
    slow = -kappa * a * np.sum(mass / denominator)
    phi = np.r_[phi, p]
    signed = np.r_[signed, slow]
    innovation = np.sqrt(-np.expm1(-2 * np.r_[rates, kappa]))
    all_rates = np.r_[rates, kappa]
    covariance = np.outer(innovation, innovation) / -np.expm1(-all_rates[:, None] - all_rates[None, :])
    scaled = signed / innovation
    variance = scaled @ covariance @ scaled
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('invalid fractional OU stationary variance')
    scaled /= np.sqrt(variance)
    # Eigen root is stable for nearly collinear very slow factors. The
    # discarded eigenvalues are numerical roundoff, reported by covariance
    # reconstruction tests; no arbitrary independent factor noise is added.
    root = None
    if with_root:
        base, inverse_root, _ = stationary_root(tolerance)
        cross = inverse_root.T @ covariance[:-1, -1]
        residual = 1 - cross @ cross
        if residual < -1e-7:
            raise ValueError('fractional OU conditional covariance is negative')
        root = np.zeros_like(covariance)
        root[:-1, :-1] = base
        root[-1, :-1] = cross
        root[-1, -1] = np.sqrt(max(residual, 0))
    return phi, innovation, scaled, root, evidence
