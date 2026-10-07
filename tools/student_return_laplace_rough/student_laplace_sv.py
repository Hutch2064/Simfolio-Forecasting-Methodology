"""Raw unit-variance Student returns, AR(1) log variance, sparse Laplace state integration.

Hyperparameters use MAP; this is not full INLA or exact posterior integration.
All latent Newton solves and inverse-diagonal calculations are linear in history.
"""
import math

import numpy as np
from numba import njit
from scipy.special import betaln, digamma


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


def normalizer(u):
    """Stable Gaussian endpoint and derivative; series error is O(u**4)."""
    if u < 1e-5:
        return (-.5 * math.log(2 * math.pi) + .75 * u + u*u + 11/8 * u**3,
                .75 + 2*u + 33/8 * u*u)
    value = -betaln(.5, .5/u) - .5 * (math.log1p(-2*u) - math.log(u))
    derivative = -.5 / (u*u) * (digamma((1/u + 1)/2) - digamma(1/(2*u))) + 1/(u*(1-2*u)) / 2
    return value, derivative


@njit(cache=True, nogil=True)
def observation_terms(h, squared, u):
    a = squared * np.exp(-h)
    b = 1 - 2*u
    denominator = b + a*u
    g = .5 - .5*(1+u)*a/denominator
    curvature = .5*(1+u)*a*b/(denominator*denominator)
    h_derivative = curvature*(a*u-b)/denominator
    g_u = -.5*a/denominator + .5*(1+u)*a*(a-2)/(denominator*denominator)
    curvature_u = curvature*(1/(1+u)-2/b-2*(a-2)/denominator)
    energy = np.empty(len(h)); energy_u = np.empty(len(h))
    for i in range(len(h)):
        z = a[i]*u/b
        if abs(z) < .001:
            f = 0.; derivative = 0.
            for k in range(8):
                f += (-z)**k/(k+1)
                if k:
                    derivative += k*(-1.)**k*z**(k-1)/(k+1)
        else:
            f = math.log1p(z)/z
            derivative = (z/(1+z)-math.log1p(z))/(z*z)
        energy[i] = .5*h[i] + .5*(1+u)*a[i]/b*f
        energy_u[i] = .5*a[i]/b*f + (1+u)*a[i]/(b*b)*f + .5*(1+u)*a[i]*a[i]/(b**3)*derivative
    return energy, g, curvature, h_derivative, energy_u, g_u, curvature_u


@njit(cache=True, nogil=True)
def state_terms(h, squared, u):
    a = squared * np.exp(-h)
    b = 1 - 2*u
    denominator = b + a*u
    return (.5 - .5*(1+u)*a/denominator,
            .5*(1+u)*a*b/(denominator*denominator))


@njit(cache=True, nogil=True)
def observation_energy(h, squared, u):
    a = squared * np.exp(-h)
    b = 1 - 2*u
    energy = np.empty(len(h))
    for i in range(len(h)):
        z = a[i]*u/b
        if abs(z) < .001:
            f = 0.
            for k in range(8):
                f += (-z)**k/(k+1)
        else:
            f = math.log1p(z)/z
        energy[i] = .5*h[i] + .5*(1+u)*a[i]/b*f
    return energy


@njit(cache=True, nogil=True)
def objective(h, squared, level, diagonal, off, u):
    if np.min(h) < -700:
        return math.inf
    delta = h-level
    return .5*np.dot(delta, multiply(diagonal, off, delta)) + observation_energy(h, squared, u).sum()


@njit(cache=True, nogil=True)
def latent_mode(squared, level, phi, eta, u):
    """Stop on Newton decrement; iteration ceiling reports failure explicitly."""
    diagonal, off = precision(len(squared), phi, eta)
    h = np.full(len(squared), level)
    converged = False
    decrement = math.inf
    current = objective(h, squared, level, diagonal, off, u)
    for iteration in range(100):
        observation_gradient, curvature = state_terms(h, squared, u)
        gradient = multiply(diagonal, off, h - level) + observation_gradient
        pivots = factor(diagonal + curvature, off)
        step = solve(pivots, off, -gradient)
        decrement = -np.dot(gradient, step)
        if decrement <= 1e-14:
            converged = True
            break
        rounding = 32 * np.finfo(np.float64).eps * max(1., abs(current))
        scale = 1.
        accepted = False
        for _ in range(60):
            proposed = h + scale * step
            proposed_value = objective(proposed, squared, level, diagonal, off, u)
            if proposed_value <= current - 1e-4 * scale * decrement + rounding:
                h = proposed
                current = proposed_value
                accepted = True
                break
            scale *= .5
        if not accepted:
            break
    if not converged:
        curvature = state_terms(h, squared, u)[1]
        pivots = factor(diagonal + curvature, off)
    return h, pivots, diagonal, off, curvature, converged, iteration + 1, decrement


@njit(cache=True, nogil=True)
def likelihood_kernel(squared, level, phi, eta, u, logc, logc_derivative):
    """Analytic derivatives include the implicit change of latent mode/curvature."""
    h, pivots, diagonal, off, _curvature, success, iterations, decrement = latent_mode(squared, level, phi, eta, u)
    variance, adjacent = inverse_diagonal(pivots, off)
    n = len(h)
    delta = h - level
    qdelta = multiply(diagonal, off, delta)
    logdet_covariance = n * math.log(eta * eta) - math.log(1 - phi * phi)
    likelihood = (n * logc - .5 * logdet_covariance
                  - objective(h, squared, level, diagonal, off, u) - .5 * np.log(pivots).sum())
    gradient = np.empty(4)
    _, _, _, h_derivative, energy_u, g_u, curvature_u = observation_terms(h, squared, u)
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
        trace += np.dot(variance * h_derivative, d_mode)
        gradient[parameter] = -.5 * d_logdet - d_energy - .5 * trace
    d_mode = solve(pivots, off, -g_u)
    gradient[3] = n*logc_derivative - energy_u.sum() - .5*np.dot(variance, curvature_u + h_derivative*d_mode)
    return likelihood, gradient, h, variance, success, iterations, decrement


def likelihood_gradient(squared, level, phi, eta, u):
    return likelihood_kernel(squared, level, phi, eta, u, *normalizer(u))


def fit(squared, observed_proxy, initial):
    """Original conventional priors/support, raw-return Laplace marginal MAP."""
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    center = float(observed_proxy.mean())
    lo, hi = np.quantile(observed_proxy, [.01, .99])
    bounds = [(lo - 4, hi + 4), (-7., float(logit(np.nextafter(.999, 0.)))), (math.log(.02), math.log(2.5)), (0., .5-np.sqrt(np.finfo(float).eps))]
    def target(theta):
        level, p, e, u = theta
        phi, eta = expit(p), math.exp(e)
        if phi >= .999:
            return 1e100, np.zeros(4)
        likelihood, gradient, _, _, success, _, _ = likelihood_gradient(squared, level, phi, eta, u)
        if not success or not np.isfinite(likelihood):
            return 1e100, np.zeros(4)
        prior = (-.5 * ((level - center) / 4) ** 2 - .5 * ((phi - .94) / .20) ** 2
                 - .5 * (e - math.log(.35)) ** 2 + math.log(phi * (1 - phi)) + e)
        gradient[1] *= phi * (1 - phi)
        gradient += np.array([-(level - center) / 16,
                              -(phi - .94) / .04 * phi * (1 - phi) + 1 - 2 * phi,
                              -(e - math.log(.35)) + 1, 0.])
        return -float(likelihood + prior) / len(squared), -gradient / len(squared)
    start = [initial[0], logit(initial[1]), math.log(initial[2]), 0.]
    def projected_gradient(theta):
        _, gradient = target(theta)
        for i, (lower, upper) in enumerate(bounds):
            if theta[i] <= lower and gradient[i] > 0 or theta[i] >= upper and gradient[i] < 0:
                gradient[i] = 0.
        return float(np.max(np.abs(gradient)))
    result = minimize(target, start, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 200, 'maxls': 100})
    evaluations = int(result.nfev)
    if not result.success or projected_gradient(result.x) > 1e-6 or result.fun >= 1e99:
        result = minimize(target, result.x if result.fun < 1e99 else start, jac=True,
                          method='SLSQP', bounds=bounds,
                          options={'ftol': 1e-12, 'maxiter': 200})
        evaluations += int(result.nfev)
    pg = projected_gradient(result.x)
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e99 or pg > 1e-6:
        raise ArithmeticError(f'Student-return Laplace MAP failed: {result.message}; projected gradient={pg}; theta={result.x}')
    level, p, e, u = result.x
    theta = (float(level), float(expit(p)), math.exp(e), float(u))
    likelihood, _, h, variance, success, iterations, decrement = likelihood_gradient(squared, *theta)
    if not success:
        raise ArithmeticError('raw-return conditional latent mode did not converge')
    return theta, h, variance, {'success': True, 'evaluations': evaluations,
                               'objective': float(result.fun * len(squared)), 'max_projected_mean_gradient': pg, 'loglikelihood': float(likelihood),
                               'latent_iterations': int(iterations), 'newton_decrement': float(decrement),
                               'latent_integration': 'Laplace approximation, tridiagonal Hessian',
                               'posterior_parameter_uncertainty': False, 'inverse_df': float(u),
                               'degrees_of_freedom': None if u==0 else float(1/u),
                               'tail_prior': 'uniform inverse_df on [0, 1/2), Gaussian limit at zero'}
