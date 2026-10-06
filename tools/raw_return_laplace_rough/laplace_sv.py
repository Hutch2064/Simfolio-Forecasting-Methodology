"""Raw Gaussian returns, AR(1) log variance, sparse Laplace state integration.

Hyperparameters use MAP; this is not full INLA or exact posterior integration.
All latent Newton solves and inverse-diagonal calculations are linear in history.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def precision(n, phi, eta):
    q = eta * eta
    diagonal = np.full(n, (1 + phi * phi) / q)
    if n == 1:
        diagonal[0] = (1 - phi * phi) / q
    else:
        diagonal[0] = diagonal[-1] = 1 / q
    return diagonal, -phi / q


@njit(cache=True, nogil=True)
def multiply(diagonal, off, x):
    value = diagonal * x
    for i in range(len(x) - 1):
        value[i] += off * x[i + 1]
        value[i + 1] += off * x[i]
    return value


@njit(cache=True, nogil=True)
def factor(diagonal, off):
    pivots = diagonal.copy()
    for i in range(1, len(pivots)):
        pivots[i] -= off * off / pivots[i - 1]
    return pivots


@njit(cache=True, nogil=True)
def solve(pivots, off, rhs):
    value = rhs.copy()
    for i in range(1, len(value)):
        value[i] -= off * value[i - 1] / pivots[i - 1]
    value[-1] /= pivots[-1]
    for i in range(len(value) - 2, -1, -1):
        value[i] = (value[i] - off * value[i + 1]) / pivots[i]
    return value


@njit(cache=True, nogil=True)
def inverse_diagonal(pivots, off):
    variance = np.empty(len(pivots))
    adjacent = np.empty(len(pivots) - 1)
    variance[-1] = 1 / pivots[-1]
    for i in range(len(pivots) - 2, -1, -1):
        adjacent[i] = -off / pivots[i] * variance[i + 1]
        variance[i] = 1 / pivots[i] + (off / pivots[i]) ** 2 * variance[i + 1]
    return variance, adjacent


@njit(cache=True, nogil=True)
def objective(h, squared, level, diagonal, off):
    delta = h - level
    value = .5 * np.dot(delta, multiply(diagonal, off, delta))
    for i in range(len(h)):
        if h[i] < -700:
            return math.inf
        value += .5 * (h[i] + squared[i] * math.exp(-h[i]))
    return value


@njit(cache=True, nogil=True)
def latent_mode(squared, level, phi, eta):
    """Stop on Newton decrement; iteration ceiling reports failure explicitly."""
    diagonal, off = precision(len(squared), phi, eta)
    h = np.full(len(squared), level)
    converged = False
    decrement = math.inf
    for iteration in range(100):
        curvature = .5 * squared * np.exp(-h)
        gradient = multiply(diagonal, off, h - level) + .5 - curvature
        pivots = factor(diagonal + curvature, off)
        step = solve(pivots, off, -gradient)
        decrement = -np.dot(gradient, step)
        if decrement <= 1e-14 and np.max(np.abs(step)) <= 1e-10:
            converged = True
            break
        current = objective(h, squared, level, diagonal, off)
        rounding = 32 * np.finfo(np.float64).eps * max(1., abs(current))
        scale = 1.
        accepted = False
        for _ in range(60):
            proposed = h + scale * step
            if objective(proposed, squared, level, diagonal, off) <= current - 1e-4 * scale * decrement + rounding:
                h = proposed
                accepted = True
                break
            scale *= .5
        if not accepted:
            break
    curvature = .5 * squared * np.exp(-h)
    pivots = factor(diagonal + curvature, off)
    return h, pivots, diagonal, off, curvature, converged, iteration + 1, decrement


@njit(cache=True, nogil=True)
def likelihood_gradient(squared, level, phi, eta):
    """Analytic derivatives include the implicit change of latent mode/curvature."""
    h, pivots, diagonal, off, curvature, success, iterations, decrement = latent_mode(squared, level, phi, eta)
    variance, adjacent = inverse_diagonal(pivots, off)
    n = len(h)
    delta = h - level
    qdelta = multiply(diagonal, off, delta)
    logdet_covariance = n * math.log(eta * eta) - math.log(1 - phi * phi)
    likelihood = (-.5 * n * math.log(2 * math.pi) - .5 * logdet_covariance
                  - objective(h, squared, level, diagonal, off) - .5 * np.log(pivots).sum())
    gradient = np.empty(3)
    d_phi = np.full(n, 2 * phi / (eta * eta))
    if n == 1:
        d_phi[0] = -2 * phi / (eta * eta)
    else:
        d_phi[0] = d_phi[-1] = 0.
    for parameter in range(3):
        if parameter == 0:
            d_gradient = -multiply(diagonal, off, np.ones(n))
            d_energy = -qdelta.sum()
            d_logdet = 0.
            trace = 0.
        elif parameter == 1:
            d_gradient = multiply(d_phi, -1 / (eta * eta), delta)
            d_energy = .5 * np.dot(delta, d_gradient)
            d_logdet = 2 * phi / (1 - phi * phi)
            trace = np.dot(variance, d_phi) - 2 * adjacent.sum() / (eta * eta)
        else:
            d_gradient = -2 * qdelta
            d_energy = -np.dot(delta, qdelta)
            d_logdet = 2. * n
            trace = -2 * (np.dot(variance, diagonal) + 2 * off * adjacent.sum())
        d_mode = solve(pivots, off, -d_gradient)
        trace -= np.dot(variance * curvature, d_mode)
        gradient[parameter] = -.5 * d_logdet - d_energy - .5 * trace
    return likelihood, gradient, h, variance, success, iterations, decrement


def fit(squared, observed_proxy, initial):
    """Original conventional priors/support, raw-return Laplace marginal MAP."""
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    center = float(observed_proxy.mean())
    lo, hi = np.quantile(observed_proxy, [.01, .99])
    bounds = [(lo - 4, hi + 4), (-7., 7.), (math.log(.02), math.log(2.5))]
    def target(theta):
        level, p, e = theta
        phi, eta = expit(p), math.exp(e)
        if phi >= .999:
            return 1e100, np.zeros(3)
        likelihood, gradient, _, _, success, _, _ = likelihood_gradient(squared, level, phi, eta)
        if not success or not np.isfinite(likelihood):
            return 1e100, np.zeros(3)
        prior = (-.5 * ((level - center) / 4) ** 2 - .5 * ((phi - .94) / .20) ** 2
                 - .5 * (e - math.log(.35)) ** 2 + math.log(phi * (1 - phi)) + e)
        gradient[1] *= phi * (1 - phi)
        gradient += np.array([-(level - center) / 16,
                              -(phi - .94) / .04 * phi * (1 - phi) + 1 - 2 * phi,
                              -(e - math.log(.35)) + 1])
        return -float(likelihood + prior), -gradient
    start = [initial[0], logit(initial[1]), math.log(initial[2])]
    result = minimize(target, start, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-10, 'gtol': 1e-5, 'maxiter': 200, 'maxls': 100})
    if not result.success:
        result = minimize(lambda t: target(t)[0], result.x, method='Powell', bounds=bounds,
                          options={'ftol': 1e-10, 'xtol': 1e-6, 'maxiter': 200})
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e99:
        raise ArithmeticError('raw-return Laplace conventional SV MAP failed')
    level, p, e = result.x
    theta = (float(level), float(expit(p)), math.exp(e))
    likelihood, _, h, variance, success, iterations, decrement = likelihood_gradient(squared, *theta)
    if not success:
        raise ArithmeticError('raw-return conditional latent mode did not converge')
    return theta, h, variance, {'success': True, 'evaluations': int(result.nfev),
                               'objective': float(result.fun), 'loglikelihood': float(likelihood),
                               'latent_iterations': int(iterations), 'newton_decrement': float(decrement),
                               'latent_integration': 'Laplace approximation, tridiagonal Hessian',
                               'posterior_parameter_uncertainty': False}
