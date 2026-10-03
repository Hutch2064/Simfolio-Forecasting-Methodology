"""Analytic derivatives of the incumbent bounded Gaussian SV parameter target."""
import math

import numpy as np
from numba import njit
from steady_kalman_target import target as scalar_target


@njit(cache=True, nogil=True)
def theta_target_gradient(y, theta):
    level, a, b = theta
    phi = 1.0 / (1.0 + math.exp(-a))
    eta = math.exp(b)
    if phi >= .999 or eta <= 1e-6:
        return -math.inf, np.zeros(3)
    dp = phi * (1.0 - phi)
    q = eta * eta
    den = max(1.0 - phi * phi, 1e-4)
    v = max(q / den, 1e-6)
    dv = np.zeros(3)
    if q / den > 1e-6:
        dv[2] = 2.0 * q / den
        if 1.0 - phi * phi > 1e-4:
            dv[1] = 2.0 * q * phi * dp / (den * den)
    m = level
    dm = np.array([1.0, 0.0, 0.0])
    value = 0.0
    grad = np.zeros(3)
    for obs in y:
        f = max(v + 4.934802200544679, 1e-8)
        error = obs - m
        value -= .5 * (math.log(2.0 * math.pi * f) + error * error / f)
        gain = v / f
        post_m = m + gain * error
        raw = (1.0 - gain) * v
        post_v = max(raw, 1e-8)
        for j in range(3):
            df = dv[j] if v + 4.934802200544679 > 1e-8 else 0.
            grad[j] += error * dm[j] / f + .5 * (error * error / (f * f) - 1.0 / f) * df
            dg = (dv[j] * f - v * df) / (f * f)
            post_dm = (1.0 - gain) * dm[j] + dg * error
            post_dv = (1.0 - gain) * dv[j] - dg * v if raw > 1e-8 else 0.
            dm[j] = phi * post_dm
            dv[j] = phi * phi * post_dv
        m = level + phi * (post_m - level)
        dm[0] += 1.0 - phi
        dm[1] += dp * (post_m - level)
        v = phi * phi * post_v + q
        dv[1] += 2.0 * phi * dp * post_v
        dv[2] += 2.0 * q
    mean = y.mean()
    value += -.5 * ((level - mean) / 4.0) ** 2
    value += -.5 * ((phi - .94) / .20) ** 2
    value += -.5 * (b - math.log(.35)) ** 2
    value += math.log(phi * (1.0 - phi)) + b
    grad[0] -= (level - mean) / 16.0
    grad[1] += -(phi - .94) * dp / .04 + 1.0 - 2.0 * phi
    grad[2] += -(b - math.log(.35)) + 1.0
    return value, grad


@njit(cache=True, nogil=True)
def unconstrained_target_gradient(y, u, lower, upper):
    # This additional logistic transform preserves the original hard support.
    s = np.empty(3)
    for j in range(3):
        if u[j] >= 0:
            s[j] = 1.0 / (1.0 + math.exp(-u[j]))
        else:
            z = math.exp(u[j])
            s[j] = z / (1.0 + z)
    theta = lower + (upper - lower) * s
    value, gradient = theta_target_gradient(y, theta)
    jac = (upper - lower) * s * (1.0 - s)
    if np.any(jac <= 0) or not np.isfinite(value):
        return -math.inf, np.zeros(3)
    value += np.log(jac).sum()
    return value, gradient * jac + 1.0 - 2.0 * s


@njit(cache=True, nogil=True)
def batch_target(y, draws, lower, upper):
    values = np.empty(len(draws))
    mean = y.mean()
    for i in range(len(draws)):
        theta = np.empty(3)
        logjac = 0.
        valid = True
        for j in range(3):
            u = draws[i, j]
            if u >= 0:
                s = 1. / (1. + math.exp(-u))
            else:
                e = math.exp(u); s = e / (1. + e)
            theta[j] = lower[j] + (upper[j] - lower[j]) * s
            jac = (upper[j] - lower[j]) * s * (1. - s)
            if jac <= 0: valid = False
            else: logjac += math.log(jac)
        values[i] = scalar_target(y, theta[0], theta[1], theta[2], mean) + logjac if valid else -math.inf
    return values


@njit(cache=True, nogil=True)
def variational_objective(y, params, z, lower, upper):
    L = np.zeros((3, 3))
    L[0, 0] = math.exp(params[3])
    L[1, 0] = params[4]
    L[1, 1] = math.exp(params[5])
    L[2, 0] = params[6]
    L[2, 1] = params[7]
    L[2, 2] = math.exp(params[8])
    value = params[3] + params[5] + params[8]
    gm = np.zeros(3)
    gL = np.zeros((3, 3))
    for i in range(len(z)):
        u = params[:3] + L @ z[i]
        lp, gu = unconstrained_target_gradient(y, u, lower, upper)
        if not np.isfinite(lp):
            return 1e100, np.zeros(9)
        value += lp / len(z)
        gm += gu / len(z)
        gL += np.outer(gu, z[i]) / len(z)
    grad = np.empty(9)
    grad[:3] = gm
    grad[3] = gL[0, 0] * L[0, 0] + 1.0
    grad[4] = gL[1, 0]
    grad[5] = gL[1, 1] * L[1, 1] + 1.0
    grad[6] = gL[2, 0]
    grad[7] = gL[2, 1]
    grad[8] = gL[2, 2] * L[2, 2] + 1.0
    return -value, -grad


@njit(cache=True, nogil=True)
def hmc_chain(y, mode, L, lower, upper, burn, kept, seed):
    np.random.seed(seed)
    z = np.random.normal(0., 1.5, 3)
    lp, gu = unconstrained_target_gradient(y, mode + L @ z, lower, upper)
    g = L.T @ gu
    trace = np.empty((kept, 3))
    accepted = 0
    energy_errors = 0
    epsilon = .15
    mu = math.log(10.0 * epsilon)
    log_average = math.log(epsilon)
    hbar = 0.0
    for i in range(burn + kept):
        momentum = np.random.normal(0., 1., 3)
        proposal = z.copy()
        p = momentum + .5 * epsilon * g
        new_lp = lp
        new_g = g.copy()
        for step in range(np.random.randint(4, 13)):
            proposal += epsilon * p
            new_lp, new_gu = unconstrained_target_gradient(y, mode + L @ proposal, lower, upper)
            new_g = L.T @ new_gu
            p += epsilon * new_g
        p -= .5 * epsilon * new_g
        error = new_lp - lp + .5 * (np.dot(momentum, momentum) - np.dot(p, p))
        probability = math.exp(min(0., error)) if np.isfinite(error) else 0.
        if np.random.random() < probability:
            z = proposal
            lp = new_lp
            g = new_g
            if i >= burn:
                accepted += 1
        if i >= burn:
            trace[i - burn] = mode + L @ z
            if not np.isfinite(error) or abs(error) > 1000:
                energy_errors += 1
        else:
            n = i + 1
            weight = 1.0 / (n + 10.0)
            hbar = (1.0 - weight) * hbar + weight * (.8 - probability)
            log_epsilon = mu - math.sqrt(n) * hbar / .05
            epsilon = min(1., max(1e-4, math.exp(min(0., log_epsilon))))
            average_weight = n ** (-.75)
            log_average = average_weight * math.log(epsilon) + (1.0 - average_weight) * log_average
            if n == burn:
                epsilon = math.exp(log_average)
    return trace, accepted / kept, energy_errors, epsilon
