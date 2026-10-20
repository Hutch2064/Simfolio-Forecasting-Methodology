"""Variance-targeted HARCH(1,5,22), training Gaussian QMLE and empirical shocks."""
import numpy as np
from numba import njit
from scipy.optimize import minimize

LAGS = (1, 5, 22)
MARGIN = 1e-8
STARTS = ((.05, .15, .70), (.10, .10, .30))


@njit(cache=True)
def lag_features(y):
    out = np.empty((len(y), 3))
    for t in range(len(y)):
        for j in range(3):
            lag = LAGS[j]
            total = 0.
            for k in range(1, lag+1):
                total += y[t-k]**2 if t >= k else 1.
            out[t, j] = total/lag
    return out


def parameters(theta):
    z = np.r_[0., theta]
    z = np.exp(z-z.max())
    z /= z.sum()
    return (1-MARGIN)*z[1:], z


def objective(theta, y, features):
    a, z = parameters(theta)
    variance = 1+np.dot(features-1, a)
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError('Invalid conditional HARCH variance')
    q = .5*np.mean(np.log(variance)+y*y/variance)
    gradient_a = .5*np.mean((1/variance-y*y/variance**2)[:, None]*(features-1), axis=0)
    jacobian = (1-MARGIN)*(np.diag(z[1:])-np.outer(z[1:], z[1:]))
    return float(q), jacobian.T@gradient_a


def fit(history):
    history = np.asarray(history, dtype=float)
    if history.ndim != 1 or len(history) < max(LAGS) or not np.isfinite(history).all():
        raise ValueError('Finite native training history with at least22 observations required')
    mean = float(history.mean())
    target = float(np.mean((history-mean)**2))
    if target <= 0 or not np.isfinite(target):
        raise ValueError('Positive training variance required')
    y = (history-mean)/np.sqrt(target)
    features = lag_features(y)
    reports = []
    eligible = []
    for start in STARTS:
        theta = np.log(np.asarray(start)/(1-sum(start)))
        result = minimize(objective, theta, args=(y, features), jac=True, method='L-BFGS-B',
                          options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
        value, gradient = objective(result.x, y, features)
        valid = bool(result.success and np.isfinite(value) and np.max(abs(gradient)) <= 1e-4)
        reports.append({'success': bool(result.success), 'eligible': valid, 'objective': value,
                        'gradient_max': float(np.max(abs(gradient))), 'iterations': int(result.nit)})
        if valid:
            eligible.append((value, result.x))
    if not eligible:
        raise ArithmeticError('Neither fixed HARCH fit converged; no failed-fit fallback')
    value, theta = min(eligible, key=lambda x: x[0])
    a, _ = parameters(theta)
    null = .5*float(np.mean(y*y))
    boundary = value >= null
    if boundary:
        a = np.zeros(3)
        value = null
    variance = 1+np.dot(features-1, a)
    pool = y/np.sqrt(variance)
    pool -= pool.mean()
    pool /= np.sqrt(np.mean(pool**2))
    return {'mean': mean, 'target_variance': target, 'alpha': a, 'omega': float(1-a.sum()),
            'lags': list(LAGS), 'pool': pool, 'last_squared_normalized': (y[-22:]**2).copy(),
            'objective': value, 'constant_variance_objective': null,
            'constant_variance_selected': bool(boundary), 'fixed_start_results': reports,
            'training_n': len(history)}


@njit(cache=True)
def native_paths(pool, indices, mean, target, alpha, last_squared, delta):
    paths, horizon = indices.shape
    out = np.empty((paths, horizon))
    clips = floors = 0
    upper = np.finfo(np.float64).max/10
    max_log = np.log1p(upper)
    omega = 1-alpha.sum()
    for p in range(paths):
        ring = last_squared.copy()
        cursor = 0
        wealth = 1.
        previous = 0.
        for t in range(horizon):
            variance = omega
            for j in range(3):
                lag = LAGS[j]
                total = 0.
                for k in range(1, lag+1):
                    total += ring[(cursor-k) % 22]
                variance += alpha[j]*total/lag
            innovation = np.sqrt(variance)*pool[indices[p, t]]
            raw = mean+delta[p]+np.sqrt(target)*innovation
            if raw > max_log:
                simple = upper
                clips += 1
            else:
                simple = np.expm1(raw)
                if simple < -.999999:
                    simple = -.999999
                    clips += 1
            wealth *= 1+simple
            if wealth < 1e-300:
                floors += 1
            current = np.log(max(wealth, 1e-300))
            out[p, t] = current-previous
            previous = current
            ring[cursor] = innovation*innovation
            cursor = (cursor+1) % 22
    return out, clips, floors
