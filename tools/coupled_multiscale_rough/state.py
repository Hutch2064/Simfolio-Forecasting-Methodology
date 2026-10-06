"""Stable positive autoregression with jointly learned exponential memories.

The same h/q recursion defines conditional likelihood and future simulation.
History is the fitted Student latent proxy, not observed realized volatility.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax


@njit(cache=True, nogil=True)
def innovations(centered, phis, coefficients):
    q = np.empty((len(centered), len(phis)))
    for j in range(len(phis)):
        # Start at the centered long-run level; use the same future recursion.
        memory = 0.
        for t in range(len(centered)):
            memory = phis[j] * memory + (1. - phis[j]) * centered[t]
            q[t, j] = memory
    errors = centered[1:] - coefficients[0] * centered[:-1]
    for j in range(len(phis)):
        errors -= coefficients[j + 1] * q[:-1, j]
    return errors, q[-1]


def decode(z, k):
    phis = np.clip(expit(z[:k]), np.nextafter(0., 1.), np.nextafter(1., 0.))
    coefficients = softmax(np.r_[0., z[k:]])[1:]
    return phis, coefficients


def objective(z, centered, k):
    phis, coefficients = decode(z, k)
    if coefficients.sum() >= 1. or not np.isfinite(z).all(): return 1e100
    errors, _ = innovations(centered, phis, coefficients)
    variance = float(np.mean(errors * errors))
    if not np.isfinite(variance) or variance <= 0.: return 1e100
    return .5 * len(errors) * (np.log(2. * np.pi * variance) + 1.)


def fit_order(h, phis, coefficients):
    centered = h - h.mean(); k = len(phis)
    z = np.r_[logit(phis), np.log(coefficients / (1. - coefficients.sum()))]
    initial = objective(z, centered, k)
    result = minimize(objective, z, args=(centered, k), method='L-BFGS-B',
                      options={'maxiter':200, 'ftol':1e-10, 'gtol':1e-5})
    solver = 'L-BFGS-B'
    if not result.success or not np.isfinite(result.fun):
        result = minimize(objective, result.x if np.isfinite(result.x).all() else z,
                          args=(centered, k), method='Powell',
                          options={'maxiter':200, 'xtol':1e-7, 'ftol':1e-10})
        solver = 'Powell_after_gradient_line_search_failure'
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e100 or result.fun > initial + 1e-7:
        raise ArithmeticError(f'coupled memory optimization failed: {result.message}')
    phis, coefficients = decode(result.x, k)
    errors, q_last = innovations(centered, phis, coefficients)
    record = {'success':True, 'phis':phis.tolist(), 'coefficients':coefficients.tolist(),
              'innovation_sd':float(np.sqrt(np.mean(errors * errors))), 'level':float(h.mean()),
              'initial':np.r_[centered[-1], q_last].tolist(), 'component_count':k,
              'negative_loglikelihood':float(result.fun), 'initial_negative_loglikelihood':float(initial),
              'iterations':int(result.nit), 'evaluations':int(result.nfev), 'solver':solver,
              'optimizer_message':str(result.message),
              'estimator':'conditional_Gaussian_QMLE_positive_stable_autoregression_on_Student_latent_proxy',
              'support':'nonnegative autoregressive coefficients with sum below one; learned stationary memory decays'}
    return phis, coefficients, record


def fit(h):
    h = np.asarray(h, float); centered = h - h.mean()
    if np.ptp(h) == 0.:
        return {'success':True, 'phis':[], 'coefficients':[0.], 'innovation_sd':0.,
                'level':float(h.mean()), 'initial':[0.], 'component_count':0,
                'count_selection':'exact_constant_proxy_zero_variance_boundary',
                'selected_bic':None, 'tested_orders':[], 'bic_parameter_count':3}
    rho = centered[:-1] @ centered[1:] / (centered[:-1] @ centered[:-1])
    phis, coefficients, record = fit_order(h, np.empty(0), np.array([np.clip(rho, np.finfo(float).eps, 1. - np.finfo(float).eps)]))
    def bic(receipt): return 2 * receipt['negative_loglikelihood'] + (2 * receipt['component_count'] + 3) * np.log(len(h) - 1)
    score = bic(record); tested = [{'count':0, 'bic':float(score)}]
    while 2 * (len(phis) + 1) + 3 < len(h) - 1:
        grid = np.sort(np.r_[0., np.log(-np.log(2.) / np.log(phis)), np.log(len(h))])
        gap = int(np.argmax(np.diff(grid)))
        extra = np.exp(-np.log(2.) / np.exp(.5 * (grid[gap] + grid[gap + 1])))
        proposed, c, receipt = fit_order(h, np.r_[phis, extra], np.r_[coefficients * .9, .05])
        next_score = bic(receipt); tested.append({'count':len(proposed), 'bic':float(next_score)})
        if next_score >= score: break
        phis, coefficients, record, score = proposed, c, receipt, next_score
    record.update(count_selection='forward_conditional_BIC_from_zero_memories_first_nonimprovement',
                  selected_bic=float(score), tested_orders=tested, bic_parameter_count=2 * len(phis) + 3,
                  parameter_uncertainty=False)
    return record


def transition(phis, coefficients):
    k = len(phis); f = np.empty((k + 1, k + 1)); f[0] = coefficients
    for j in range(k):
        f[j + 1] = (1. - phis[j]) * coefficients
        f[j + 1, j + 1] += phis[j]
    return f


@njit(cache=True, nogil=True)
def moments(f, initial, sd, horizon):
    mean = initial.copy(); covariance = np.zeros((len(mean), len(mean)))
    q = np.outer(sd, sd); means = np.empty(horizon); variances = np.empty(horizon)
    for t in range(horizon):
        mean = f @ mean
        covariance = f @ covariance @ f.T + q
        means[t] = mean[0]; variances[t] = covariance[0, 0]
    return means, variances
