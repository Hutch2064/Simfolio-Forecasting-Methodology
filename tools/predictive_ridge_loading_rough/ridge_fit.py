"""Conditional predictive ridge calibration, not iid generalized cross-validation.

One column needs only loading contraction. Multiple columns learn ridge and
contraction jointly with rates; zero ridge is an explicit boundary fit. Stable
SVD uses only the usual floating-point rank threshold, no scientific ridge floor.
"""
import math

import numpy as np
from numba import njit
from rate_fit import ewma
from scipy.optimize import minimize
from scipy.special import expit, logit


@njit(cache=True, nogil=True)
def projection(centered, phis, ridge):
    q = np.empty((len(centered), len(phis)))
    for j in range(len(phis)):
        col = ewma(centered, 1. - phis[j])
        q[:, j] = col - np.mean(col)
    u, s, vh = np.linalg.svd(q, full_matrices=False)
    multiplier = np.zeros(len(s))
    alpha = ridge * max(np.var(centered), 1e-8)
    cutoff = np.finfo(np.float64).eps * max(q.shape) * s[0]
    for j in range(len(s)):
        if alpha > 0.:
            multiplier[j] = s[j] / (s[j] * s[j] + alpha)
        elif s[j] > cutoff:
            multiplier[j] = 1. / s[j]
    return q, vh.T @ (multiplier * (u.T @ centered))


@njit(cache=True, nogil=True)
def errors(centered, phis, scale, ridge):
    q, raw = projection(centered, phis, ridge)
    b = raw * scale
    residual = centered - q @ b
    residual -= np.mean(residual)
    denom = residual[:-1] @ residual[:-1]
    rho = residual[:-1] @ residual[1:] / denom if denom > np.finfo(np.float64).tiny else 0.
    rho = min(.999999, max(-.999999, rho))
    return centered[1:] - (q[:-1] * phis) @ b - rho * residual[:-1]


def decode(z, k, positive):
    phis = np.clip(expit(z[:k]), np.nextafter(0., 1.), np.nextafter(1., 0.))
    scale = float(expit(z[k]))
    if positive:
        if not (math.log(np.nextafter(0., 1.)) <= z[k + 1] <= math.log(np.finfo(float).max)):
            return phis, scale, np.inf
        ridge = math.exp(float(z[k + 1]))
    else:
        ridge = 0.
    return phis, scale, ridge


def objective(z, centered, k, positive):
    phis, scale, ridge = decode(z, k, positive)
    if not np.isfinite(ridge): return 1e100
    error = errors(centered, phis, scale, ridge)
    variance = float(np.mean(error * error))
    if not np.isfinite(variance) or variance <= 0.: return 1e100
    return .5 * len(error) * (np.log(2. * np.pi * variance) + 1.)


def minimize_target(centered, rates, positive):
    k = len(rates)
    z = np.r_[logit(rates), 0., math.log(.05)] if positive else np.r_[logit(rates), 0.]
    initial = objective(z, centered, k, positive)
    result = minimize(objective, z, args=(centered, k, positive), method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-5})
    solver = 'L-BFGS-B'
    if not result.success or not np.isfinite(result.fun):
        result = minimize(objective, result.x if np.isfinite(result.x).all() else z,
                          args=(centered, k, positive), method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-7, 'ftol': 1e-10})
        solver = 'Powell_after_gradient_line_search_failure'
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e100 or result.fun > initial + 1e-7:
        raise ArithmeticError(f'predictive ridge optimization failed: {result.message}')
    phis, scale, ridge = decode(result.x, k, positive)
    _, raw = projection(centered, phis, ridge)
    return phis, {'estimator': 'conditional_Gaussian_predictive_QMLE_decay_contraction_and_ridge',
                      'negative_loglikelihood': float(result.fun), 'initial_negative_loglikelihood': float(initial),
                      'loading_scale': scale, 'ridge_scale': ridge, 'loading_coefficients': raw.tolist(),
                      'component_count': k, 'phis': phis.tolist(), 'success': True, 'solver': solver,
                      'evaluations': int(result.nfev), 'iterations': int(result.nit), 'optimizer_message': str(result.message)}


def fit_order(h, rates):
    centered = h - np.mean(h)
    zero_rates, zero = minimize_target(centered, rates, False)
    if len(rates) == 1:
        zero['ridge_identifiability'] = 'single-column ridge absorbed by fitted contraction; no redundant ridge parameter'
        return zero_rates, zero
    positive_rates, positive = minimize_target(centered, rates, True)
    result = (positive_rates, positive) if positive['negative_loglikelihood'] < zero['negative_loglikelihood'] else (zero_rates, zero)
    result[1]['ridge_identifiability'] = 'multiple columns: positive ridge fitted with explicit zero-ridge boundary'
    result[1]['boundary_negative_loglikelihoods'] = {'zero':zero['negative_loglikelihood'], 'positive':positive['negative_loglikelihood']}
    return result


def fit_adaptive(h, initial_phis):
    h = np.asarray(h, float); centered = h - np.mean(h)
    rho = float(centered[:-1] @ centered[1:] / (centered[:-1] @ centered[:-1]))
    rates, record = fit_order(h, np.array([np.clip(rho, np.finfo(float).eps, 1. - np.finfo(float).eps)]))
    def parameter_count(k): return 2 * k + 4 + int(k > 1)
    def bic(receipt): return 2 * receipt['negative_loglikelihood'] + parameter_count(receipt['component_count']) * np.log(len(h) - 1)
    score = bic(record); tested = [{'count':1, 'bic':float(score)}]
    while parameter_count(len(rates) + 1) < len(h) - 1:
        grid = np.sort(np.r_[0., np.log(-np.log(2.) / np.log(rates)), np.log(len(h))])
        gap = int(np.argmax(np.diff(grid)))
        extra = np.exp(-np.log(2.) / np.exp(.5 * (grid[gap] + grid[gap + 1])))
        proposed, receipt = fit_order(h, np.r_[rates, extra])
        next_score = bic(receipt); tested.append({'count':len(proposed), 'bic':float(next_score)})
        if next_score >= score: break
        rates, record, score = proposed, receipt, next_score
    record.update(count_selection='forward_conditional_BIC_predictive_ridge_first_nonimprovement',
                  selected_bic=float(score), tested_orders=tested, bic_parameter_count=parameter_count(len(rates)),
                  ridge_parameter_count=int(len(rates) > 1),
                  interpretation='conditional regularized prediction on Student latent proxy; not iid GCV or exact joint Bayesian evidence')
    return rates, record
