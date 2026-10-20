"""Variance-targeted GARCH fitted by training-only conditional Gaussian CRPS."""
from time import perf_counter
import math
import numpy as np
from numba import njit
from scipy.optimize import minimize
from expanded_public_smooth_mean_live import module as parent
gaussian = parent.retained.FHS.base


@njit(cache=True, fastmath=False)
def conditional_loss(y, a, b):
    n = len(y)
    v = np.empty(n)
    v[0] = 1.
    da = db = ga = gb = loss = 0.
    for t in range(n):
        if t:
            old = v[t-1]
            v[t] = 1.-a-b+a*y[t-1]**2+b*old
            da = -1.+y[t-1]**2+b*da
            db = -1.+old+b*db
        value = v[t]
        if value <= 0 or not np.isfinite(value):
            raise ValueError('Invalid conditional training variance')
        sigma = np.sqrt(value)
        z = y[t]/sigma
        phi = np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        loss += y[t]*math.erf(z/np.sqrt(2.))+sigma*(2*phi-1/np.sqrt(np.pi))
        slope = (phi-.5/np.sqrt(np.pi))/sigma
        ga += slope*da
        gb += slope*db
    next_v = 1.-a-b+a*y[-1]**2+b*v[-1]
    return loss/n, np.array([ga/n, gb/n]), v, next_v


def objective(theta, y):
    a, b, z = gaussian.parameters(theta)
    loss, gradient, _, _ = conditional_loss(y, a, b)
    scale = 1.-gaussian.STATIONARITY_MARGIN
    jac = np.array([[scale*z[1]*(1.-z[1]), -scale*z[1]*z[2]],
                    [-scale*z[1]*z[2], scale*z[2]*(1.-z[2])]])
    return loss, jac.T@gradient


def fit_crps(history):
    started = perf_counter()
    mu = float(np.mean(history))
    centered = history-mu
    target = float(np.mean(centered*centered))
    if not np.isfinite(target) or target <= 0:
        raise ValueError('Positive finite training variance required')
    y = centered/np.sqrt(target)
    attempts, eligible = [], []
    for a, b in ((.05, .90), (.10, .50)):
        theta = np.log(np.array([a, b])/(1.-a-b))
        fit = minimize(objective, theta, args=(y,), jac=True, method='L-BFGS-B',
            options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
        value, gradient = objective(fit.x, y)
        valid = bool(fit.success and np.isfinite(value) and np.isfinite(gradient).all()
            and np.max(np.abs(gradient)) <= 1e-4)
        attempts.append({'success': bool(fit.success), 'eligible': valid, 'objective': float(value),
            'gradient_max': float(np.max(np.abs(gradient))), 'iterations': int(fit.nit)})
        if valid:
            eligible.append((float(value), fit.x))
    if not eligible:
        raise ArithmeticError('Both fixed CRPS training fits failed convergence')
    value, theta = min(eligible, key=lambda x: x[0])
    a, b, _ = gaussian.parameters(theta)
    null = conditional_loss(y, 0., 0.)[0]
    if value >= null:
        a = b = 0.
    value, _, variances, next_v = conditional_loss(y, a, b)
    pool = y/np.sqrt(variances)
    pool = pool-pool.mean()
    pool = pool/np.sqrt(np.mean(pool*pool))
    return {'mean': mu, 'target_variance': target, 'alpha': a, 'beta': b, 'omega': 1.-a-b,
        'next_variance_normalized': float(next_v), 'pool': pool, 'objective': float(value),
        'constant_variance_objective': null, 'fixed_start_results': attempts,
        'fit_seconds': perf_counter()-started, 'training_n': len(history),
        'training_objective': 'conditional_Gaussian_CRPS_in_fixed_training_SD_units',
        'forecast_innovations': 'empirical_filtered_residuals_not_Gaussian'}
