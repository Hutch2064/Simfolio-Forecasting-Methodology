"""Frozen earlier GJR estimator, applied per constituent with current log inputs."""
from time import perf_counter
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit
from candidates.asset_prior_best import fit as symmetric
MARGIN=symmetric.STATIONARITY_MARGIN

def parameters(theta):
    p, m, fraction = expit(np.asarray(theta, dtype=float))
    beta = (1.-MARGIN)*fraction
    budget = 1.-MARGIN-beta
    return budget*p, budget*m, beta, np.array([p, m, fraction])

def coordinates(positive, negative, beta):
    budget = 1.-MARGIN-beta
    fractions = np.array([positive/budget, negative/budget, beta/(1.-MARGIN)])
    if not np.all((fractions > 0) & (fractions < 1)):
        raise ValueError('Interior stationary starting values required')
    return np.log(fractions/(1.-fractions))

@njit(cache=True, fastmath=False)
def filter_loss(y, positive, negative, beta, k_positive, k_negative):
    omega = 1.-beta-positive*k_positive-negative*k_negative
    variance = np.empty(len(y))
    variance[0] = 1.
    dp = dm = db = gp = gm = gb = loss = 0.
    for t in range(len(y)):
        if t:
            old = variance[t-1]
            square = y[t-1]*y[t-1]
            plus = square if y[t-1] >= 0 else 0.
            minus = square if y[t-1] < 0 else 0.
            variance[t] = omega+positive*plus+negative*minus+beta*old
            dp = -k_positive+plus+beta*dp
            dm = -k_negative+minus+beta*dm
            db = -1.+old+beta*db
        value = variance[t]
        if value <= 0 or not np.isfinite(value):
            raise ValueError('Invalid conditional asymmetric training variance')
        square = y[t]*y[t]
        loss += np.log(value)+square/value
        slope = 1./value-square/(value*value)
        gp += slope*dp
        gm += slope*dm
        gb += slope*db
    square = y[-1]*y[-1]
    next_v = omega+(positive if y[-1] >= 0 else negative)*square+beta*variance[-1]
    return .5*loss/len(y), .5*np.array([gp, gm, gb])/len(y), variance, next_v

def objective(theta, y, k_positive, k_negative):
    positive, negative, beta, weights = parameters(theta)
    value, gradient, _, _ = filter_loss(y, positive, negative, beta, k_positive, k_negative)
    p, m, fraction = weights
    budget = 1.-MARGIN-beta
    db = (1.-MARGIN)*fraction*(1.-fraction)
    transformed = np.array([gradient[0]*budget*p*(1.-p),
        gradient[1]*budget*m*(1.-m), db*(gradient[2]-gradient[0]*p-gradient[1]*m)])
    return value, transformed

def target_pool(y, variances, positive, negative, beta):
    pool = y/np.sqrt(variances)
    pool = pool-pool.mean()
    pool = pool/np.sqrt(np.mean(pool*pool))
    k_negative = float(np.mean(np.where(pool < 0, pool*pool, 0.)))
    k_positive = 1.-k_negative
    persistence = beta+positive*k_positive+negative*k_negative
    omega = 1.-persistence
    if not np.isfinite(pool).all() or not omega > 0 or persistence >= 1.:
        raise ValueError('Invalid empirical innovation pool or stationarity')
    return pool, omega, persistence, k_negative

def fit(history):
    started = perf_counter()
    x = np.asarray(history, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Finite one-dimensional training returns required')
    parent = symmetric.fit_variance_target(x)
    centered = x-parent['mean']
    y = centered/np.sqrt(parent['target_variance'])
    total = float(np.mean(y*y))
    k_negative = float(np.mean(np.where(y < 0, y*y, 0.))/total)
    k_positive = 1.-k_negative
    attempts, eligible = [], []
    # Same two symmetric initial points as the parent; an asymmetric start
    # prevents a coordinate saddle. These are fixed before field forecasts.
    for positive, negative, beta in [(.05, .05, .90), (.10, .10, .50), (.03, .12, .80)]:
        result = minimize(objective, coordinates(positive, negative, beta),
            args=(y, k_positive, k_negative), jac=True, method='L-BFGS-B',
            options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
        value, gradient = objective(result.x, y, k_positive, k_negative)
        valid = bool(result.success and np.isfinite(value) and np.isfinite(gradient).all()
                     and np.max(np.abs(gradient)) <= 1e-4)
        attempts.append({'success': bool(result.success), 'eligible': valid,
            'objective': float(value), 'gradient_max': float(np.max(np.abs(gradient))),
            'iterations': int(result.nit)})
        if valid:
            ap, am, b, _ = parameters(result.x)
            eligible.append((float(value), ap, am, b))
    if not eligible:
        raise ArithmeticError('All three fixed asymmetric training fits failed convergence')
    value, positive, negative, beta = min(eligible)
    if value >= parent['objective']:
        # The fitted class includes the exact nested symmetric parent, rather
        # than pretending a failed asymmetric optimization converged.
        positive = negative = parent['alpha']
        beta = parent['beta']
        variance = symmetric.filter_loss(y, positive, beta)[2]
        next_v = parent['next_variance_normalized']
        value = parent['objective']
        pool, omega = parent['pool'], parent['omega']
        pool_k_negative = float(np.mean(np.where(pool < 0, pool*pool, 0.)))
        persistence = positive+beta
        nested = True
    else:
        value, _, variance, next_v = filter_loss(y, positive, negative, beta, k_positive, k_negative)
        pool, omega, persistence, pool_k_negative = target_pool(y, variance, positive, negative, beta)
        nested = False
    return {'mean': parent['mean'], 'target_variance': parent['target_variance'],
        'alpha_positive': float(positive), 'alpha_negative': float(negative), 'beta': float(beta),
        'omega_forecast': float(omega), 'omega_training': float(1.-beta-positive*k_positive-negative*k_negative),
        'next_variance_normalized': float(next_v), 'pool': pool, 'objective': float(value),
        'nested_parent_objective': parent['objective'], 'nested_parent_selected': nested,
        'training_negative_second_moment_fraction': k_negative,
        'innovation_negative_second_moment': pool_k_negative,
        'innovation_persistence': float(persistence), 'fixed_start_results': attempts,
        'fit_seconds': perf_counter()-started, 'training_n': len(x)}
