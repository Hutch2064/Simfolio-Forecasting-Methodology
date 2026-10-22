"""Frozen GJR fit with an optional same-prefix symmetric parent; no optimizer change."""
from time import perf_counter
import numpy as np
from candidates.asset_gjr import fit as reference
symmetric=reference.symmetric;minimize=reference.minimize
parameters=reference.parameters;coordinates=reference.coordinates
filter_loss=reference.filter_loss;objective=reference.objective;target_pool=reference.target_pool

def fit(history,base_parent=None):
    started = perf_counter()
    x = np.asarray(history, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Finite one-dimensional training returns required')
    parent = symmetric.fit_variance_target(x) if base_parent is None else base_parent
    if base_parent is not None:
        mu=float(np.mean(x));target=float(np.mean((x-mu)*(x-mu)))
        if parent['training_n']!=len(x) or np.float64(parent['mean']).tobytes()!=np.float64(mu).tobytes() or np.float64(parent['target_variance']).tobytes()!=np.float64(target).tobytes():raise ValueError('Precomputed GJR parent must match the exact native training prefix')
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
