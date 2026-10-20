"""Conditional MA1-GARCH QMLE: explicit zero pre-sample shock, variance targeting."""
from time import perf_counter
from pathlib import Path
import sys
import numpy as np
from numba import njit
from scipy.optimize import minimize
from expanded_public_smooth_mean_live import module as parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ma1_qualification'))
from ma1 import fit as fit_MA_seed
gaussian = parent.retained.FHS.base
MARGIN = 1e-8


@njit(cache=True, fastmath=False)
def innovations(y, theta):
    errors = np.empty(len(y))
    derivative = np.empty(len(y))
    errors[0], derivative[0] = y[0], 0.
    for t in range(1, len(y)):
        errors[t] = y[t]-theta*errors[t-1]
        derivative[t] = -errors[t-1]-theta*derivative[t-1]
    return errors, derivative


@njit(cache=True, fastmath=False)
def filter_loss(y, a, b, theta):
    errors, de = innovations(y, theta)
    target = np.mean(errors*errors)
    dtarget = np.mean(2*errors*de)
    variances = np.empty(len(y))
    variances[0] = target
    da = db = ga = gb = gt = loss = 0.
    dt = dtarget
    for t in range(len(y)):
        if t:
            old = variances[t-1]
            variances[t] = target*(1.-a-b)+a*errors[t-1]**2+b*old
            da = -target+errors[t-1]**2+b*da
            db = -target+old+b*db
            dt = dtarget*(1.-a-b)+2*a*errors[t-1]*de[t-1]+b*dt
        v = variances[t]
        if v <= 0 or not np.isfinite(v):
            raise ValueError('Invalid conditional MA1-GARCH training variance')
        loss += np.log(v)+errors[t]**2/v
        slope = 1./v-errors[t]**2/v**2
        ga += slope*da
        gb += slope*db
        gt += slope*dt+2*errors[t]*de[t]/v
    next_v = target*(1.-a-b)+a*errors[-1]**2+b*variances[-1]
    return .5*loss/len(y), np.array([.5*ga/len(y), .5*gb/len(y), .5*gt/len(y)]), errors, variances, next_v


def objective(parameters, y):
    a, b, z = gaussian.parameters(parameters[:2])
    theta = (1.-MARGIN)*np.tanh(parameters[2])
    value, gradient, _, _, _ = filter_loss(y, a, b, theta)
    scale = 1.-gaussian.STATIONARITY_MARGIN
    jac = np.array([[scale*z[1]*(1.-z[1]), -scale*z[1]*z[2]],
                    [-scale*z[1]*z[2], scale*z[2]*(1.-z[2])]])
    return value, np.r_[jac.T@gradient[:2], gradient[2]*(1.-MARGIN)*(1.-np.tanh(parameters[2])**2)]


def fit_joint(history):
    started = perf_counter()
    mu = float(np.mean(history))
    centered = history-mu
    observed_target = float(np.mean(centered*centered))
    if not np.isfinite(observed_target) or observed_target <= 0:
        raise ValueError('Finite positive training variance required')
    y = centered/np.sqrt(observed_target)
    boundary = gaussian.fit_variance_target(history)
    seed_theta = fit_MA_seed(history)['theta']
    starts = [(.05, .90, 0.), (.10, .50, 0.), (boundary['alpha'], boundary['beta'], seed_theta)]
    # If the explicit constant-volatility boundary was selected, use its first
    # fixed interior start instead of taking log(0) for a numerical start.
    if starts[-1][0] == 0 or starts[-1][1] == 0:
        starts[-1] = (.05, .90, seed_theta)
    eligible, attempts = [], []
    for a, b, theta in starts:
        initial = np.r_[np.log(np.array([a, b])/(1.-a-b)), np.arctanh(theta/(1.-MARGIN))]
        fit = minimize(objective, initial, args=(y,), jac=True, method='L-BFGS-B',
            options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
        value, gradient = objective(fit.x, y)
        valid = bool(fit.success and np.isfinite(value) and np.isfinite(gradient).all()
                     and np.max(np.abs(gradient)) <= 1e-4)
        attempts.append({'success': bool(fit.success), 'eligible': valid, 'objective': float(value),
                         'gradient_max': float(np.max(np.abs(gradient))), 'iterations': int(fit.nit)})
        if valid:
            eligible.append((float(value), fit.x))
    if not eligible:
        raise ArithmeticError('All fixed joint MA1-GARCH training fits failed convergence')
    value, parameters = min(eligible, key=lambda x: x[0])
    boundary_value = filter_loss(y, boundary['alpha'], boundary['beta'], 0.)[0]
    if value >= boundary_value:
        return {**boundary, 'theta': 0., 'last_innovation': float(centered[-1]),
                'joint_attempts': attempts, 'joint_boundary_selected': True,
                'joint_objective': boundary_value, 'joint_fit_seconds': perf_counter()-started}
    a, b, _ = gaussian.parameters(parameters[:2])
    theta = float((1.-MARGIN)*np.tanh(parameters[2]))
    value, _, e, v, next_v = filter_loss(y, a, b, theta)
    target = float(np.mean(e*e))
    pool = e/np.sqrt(v)
    pool -= pool.mean()
    pool /= np.sqrt(np.mean(pool*pool))
    return {'mean': mu, 'target_variance': target*observed_target, 'alpha': a, 'beta': b,
            'omega': 1.-a-b, 'next_variance_normalized': float(next_v/target), 'pool': pool,
            'theta': theta, 'last_innovation': float(e[-1]*np.sqrt(observed_target)),
            'joint_attempts': attempts, 'joint_boundary_selected': False, 'joint_objective': float(value),
            'boundary_objective': boundary_value, 'training_n': len(history),
            'fit_seconds': perf_counter()-started, 'joint_fit_seconds': perf_counter()-started,
            'pre_sample_innovation': 0., 'coefficient_uncertainty_integrated': False}


@njit(cache=True, fastmath=False)
def recursive_paths(pool, indices, center, target, a, b, next_v, theta, last_innovation):
    paths, horizon = indices.shape
    out = np.empty((paths, horizon))
    clips = floors = 0
    omega = 1.-a-b
    upper = np.finfo(np.float64).max/10.
    max_log = np.log1p(upper)
    for p in range(paths):
        v = next_v
        previous_epsilon = last_innovation
        wealth = 1.
        previous = 0.
        for h in range(horizon):
            normalized = np.sqrt(v)*pool[indices[p, h]]
            epsilon = np.sqrt(target)*normalized
            raw = center+epsilon+theta*previous_epsilon
            if raw > max_log:
                simple = upper
                clips += 1
            else:
                simple = np.expm1(raw)
                if simple < -.999999:
                    simple = -.999999
                    clips += 1
            wealth *= 1.+simple
            if wealth < 1e-300:
                floors += 1
            current = np.log(max(wealth, 1e-300))
            out[p, h] = current-previous
            previous = current
            previous_epsilon = epsilon
            v = omega+a*normalized*normalized+b*v
    return out, clips, floors


def parameter_paths(fit, indices, delta):
    if fit['theta'] == 0.:
        return parent.retained.FHS.parameter_paths(fit, indices, delta)
    paths, horizon = indices.shape
    if np.asarray(delta).shape != (paths,) or not np.isfinite(delta).all():
        raise ValueError('Invalid independent mean draws')
    out = np.empty((paths, horizon))
    clips = floors = 0
    for p in range(paths):
        daily, c, f = recursive_paths(fit['pool'], indices[p:p+1], fit['mean']+delta[p],
            fit['target_variance'], fit['alpha'], fit['beta'], fit['next_variance_normalized'],
            fit['theta'], fit['last_innovation'])
        out[p] = daily[0]
        clips += c
        floors += f
    return out, clips, floors
