"""Conditional Gaussian empirical Bayes projection with stationary AR(1) errors.

REML integrates the intercept and Gaussian loading prior, then profiles noise.
All optimization uses cached small sufficient-statistic matrices, not T x T
covariances. This conditions on the estimated latent path, not raw returns.
"""
import numpy as np
from scipy.optimize import minimize


def statistics(y, z):
    design = np.column_stack((np.ones(len(y)), z, y))
    difference, previous = np.diff(design, axis=0), design[:-1]
    return (difference.T @ difference,
            difference.T @ previous + previous.T @ difference,
            previous.T @ previous, np.outer(design[0], design[0]), len(y))


def evaluate(parameters, stats, null=False):
    gd, gc, gp, first, n = stats
    phi = float(np.tanh(parameters[0]))
    delta = 1 - phi
    t = gd + delta * gc + delta ** 2 * gp + delta * (2 - delta) * first
    dt = -gc - 2 * delta * gp - 2 * phi * first
    r = np.array([0, len(t) - 1])
    trr = t[np.ix_(r, r)]
    if null:
        c = trr
        dc = [dt[np.ix_(r, r)]]
        determinant = 0.
        determinant_gradient = [2 * phi]
        v = np.zeros((len(t) - 2, 2))
        penalty = np.inf
    else:
        penalty = float(np.exp(parameters[1]))
        m = t[1:-1, 1:-1] + penalty * np.eye(len(t) - 2)
        tzr = t[1:-1, r]
        v = np.linalg.solve(m, tzr)
        c = trr - tzr.T @ v
        dm = dt[1:-1, 1:-1]
        dzr = dt[1:-1, r]
        dc = [dt[np.ix_(r, r)] - dzr.T @ v - v.T @ dzr + v.T @ dm @ v,
              penalty * v.T @ v]
        eigen = np.linalg.eigvalsh(t[1:-1, 1:-1])
        determinant = float(np.log1p(eigen / penalty).sum())
        determinant_gradient = [2 * phi + (1 - phi ** 2) * np.trace(np.linalg.solve(m, dm)),
                                -float(np.sum(eigen / (eigen + penalty)))]
    level = c[0, 1] / c[0, 0]
    rss = c[1, 1] - c[0, 1] * level
    if rss <= 0 or c[0, 0] <= 0:
        raise ArithmeticError('nonpositive REML residual variance or intercept precision')
    variance = rss / (n - 1)
    value = .5 * ((n - 1) * np.log(variance) - np.log1p(-phi ** 2)
                  + determinant + np.log(c[0, 0]))
    gradient = []
    for i, derivative in enumerate(dc):
        drss = derivative[1, 1] - 2 * level * derivative[0, 1] + level ** 2 * derivative[0, 0]
        scale = 1 - phi ** 2 if i == 0 else 1.
        gradient.append(.5 * ((n - 1) * drss / rss * scale
                             + determinant_gradient[i] + derivative[0, 0] / c[0, 0] * scale))
    return value / (n - 1), np.array(gradient) / (n - 1), (level, v[:, 1] - v[:, 0] * level, phi, variance, penalty)


def fit(y, z, initial_phi, initial_penalty):
    stats = statistics(y, z)
    # Numerical floating-point support bounds; neither is a scientific prior.
    support = np.arctanh(1 - np.finfo(float).eps)
    start = np.arctanh(np.clip(initial_phi, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps))
    bounds = [(-support, support), (-np.log(np.finfo(float).max) / 2, np.log(np.finfo(float).max) / 2)]
    results = []
    for null in (False, True):
        point = [start] if null else [start, np.log(max(initial_penalty, np.finfo(float).tiny))]
        result = minimize(lambda x, null=null: evaluate(x, stats, null)[:2], point, jac=True,
                          method='L-BFGS-B', bounds=bounds[:len(point)],
                          options={'gtol': 1e-7, 'ftol': 1e-12, 'maxiter': 200})
        value, gradient, estimate = evaluate(result.x, stats, null)
        if not np.isfinite(value) or np.max(np.abs(gradient)) > 1e-6:
            raise ArithmeticError(f'REML optimization failed: {result.message}, gradient={gradient}')
        results.append((value, estimate, result, null, gradient))
    value, estimate, result, null, gradient = min(results, key=lambda item: item[0])
    diagnostics = {'mean_restricted_negative_loglikelihood': float(value),
                   'null_loading_variance': null, 'penalty_ratio': None if null else float(estimate[-1]),
                   'residual_phi': float(estimate[2]), 'residual_innovation_variance': float(estimate[3]),
                   'optimizer_iterations': int(result.nit), 'optimizer_evaluations': int(result.nfev),
                   'gradient_inf_norm': float(np.max(np.abs(gradient))),
                   'finite_vs_null_mean_loglikelihood_gain': float(results[1][0] - results[0][0])}
    return estimate[:4], diagnostics
