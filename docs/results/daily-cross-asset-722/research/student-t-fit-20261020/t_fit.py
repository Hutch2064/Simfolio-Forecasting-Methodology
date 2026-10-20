"""Variance-targeted Student-t likelihood fit; empirical forecast shocks remain empirical."""
from pathlib import Path
import sys

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
sys.path.insert(0, str(ROOT / 'research/variance_targeted_fhs'))
import fhs_candidate as gaussian

# Standardized Student-t support and numerical bounds used by arch.StudentsT.
DF_BOUNDS = (2.05, 500.0)


@njit(cache=True, fastmath=False)
def conditional_loss(y, a, b, nu):
    v = np.empty(len(y))
    v[0] = 1.0
    da = db = ga = gb = gn = loss = 0.0
    for t in range(len(y)):
        if t:
            old = v[t-1]
            v[t] = 1-a-b + a*y[t-1]**2 + b*old
            da = -1 + y[t-1]**2 + b*da
            db = -1 + old + b*db
        u = y[t]**2 / ((nu-2)*v[t])
        logterm = np.log1p(u)
        loss += .5*np.log(v[t]) + .5*(nu+1)*logterm
        slope = (.5-.5*(nu+1)*u/(1+u))/v[t]
        ga += slope*da
        gb += slope*db
        gn += .5*logterm - .5*(nu+1)*u/((nu-2)*(1+u))
    return loss/len(y), np.array([ga, gb, gn])/len(y), v, 1-a-b+a*y[-1]**2+b*v[-1]


def objective(theta, y):
    a, b, z = gaussian.parameters(theta[:2])
    nu = float(theta[2])
    value, gradient, _, _ = conditional_loss(y, a, b, nu)
    value += .5*np.log((nu-2)*np.pi) + gammaln(nu/2)-gammaln((nu+1)/2)
    gradient[2] += .5/(nu-2)+.5*digamma(nu/2)-.5*digamma((nu+1)/2)
    margin = 1-gaussian.STATIONARITY_MARGIN
    jac = margin*np.array([[z[1]*(1-z[1]), -z[1]*z[2]],
                           [-z[1]*z[2], z[2]*(1-z[2])]])
    return float(value), np.r_[jac.T@gradient[:2], gradient[2]]


def fit_student(history):
    history = np.asarray(history, dtype=float)
    base = gaussian.fit_variance_target(history)
    y = (history-base['mean'])/np.sqrt(base['target_variance'])
    starts = [(0.05, .90, 8.0), (.10, .50, 30.0)]
    if base['alpha'] > 0 and base['beta'] > 0:
        starts.append((base['alpha'], base['beta'], 8.0))
    attempts = []
    eligible = []
    for a, b, nu in starts:
        theta = np.r_[np.log(np.array([a, b])/(1-a-b)), nu]
        fit = minimize(objective, theta, args=(y,), jac=True, method='L-BFGS-B',
                       bounds=[(None, None), (None, None), DF_BOUNDS],
                       options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
        value, gradient = objective(fit.x, y)
        projected = gradient.copy()
        if fit.x[2] <= DF_BOUNDS[0]+1e-8 and gradient[2] > 0:
            projected[2] = 0
        if fit.x[2] >= DF_BOUNDS[1]-1e-8 and gradient[2] < 0:
            projected[2] = 0
        valid = bool(fit.success and np.isfinite(value) and np.isfinite(gradient).all()
                     and np.max(np.abs(projected)) <= 1e-4)
        attempts.append({'eligible': valid, 'success': bool(fit.success),
                         'objective': value, 'projected_gradient_max': float(np.max(np.abs(projected))),
                         'nu': float(fit.x[2]), 'iterations': int(fit.nit)})
        if valid:
            eligible.append((value, fit.x))
    if not eligible:
        raise ArithmeticError('No converged Student-t training fit; no failed-fit fallback')
    value, theta = min(eligible, key=lambda pair: pair[0])
    gaussian_value = base['objective'] + .5*np.log(2*np.pi)
    if value >= gaussian_value:
        return {**base, 'likelihood_selected': 'Gaussian_boundary', 'student_df': None,
                'student_attempts': attempts, 'complete_gaussian_nll': gaussian_value}
    a, b, _ = gaussian.parameters(theta[:2])
    _, _, variances, next_v = conditional_loss(y, a, b, theta[2])
    pool = y/np.sqrt(variances)
    pool -= pool.mean()
    pool /= np.sqrt(np.mean(pool*pool))
    return {**base, 'alpha': a, 'beta': b, 'omega': 1-a-b,
            'next_variance_normalized': float(next_v), 'pool': pool,
            'likelihood_selected': 'Student_t', 'student_df': float(theta[2]),
            'student_nll': float(value), 'complete_gaussian_nll': gaussian_value,
            'student_attempts': attempts}
