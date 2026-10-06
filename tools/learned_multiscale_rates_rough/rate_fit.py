"""Conditional Gaussian predictive QMLE for EWMA decay rates on a latent proxy.

Existing ridge/loading shrinkage and residual AR1 estimates stay fixed as
algorithms. This is conditional on a fitted history, not raw-return inference.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, logit


@njit(cache=True, nogil=True)
def ewma(x, alpha):
    out = np.empty_like(x)
    finite = x[np.isfinite(x)]
    level = np.median(finite) if finite.size else 0.
    for i in range(len(x)):
        value = x[i]
        if np.isfinite(value):
            level = alpha * value + (1. - alpha) * level
        out[i] = level
    return out


@njit(cache=True, nogil=True)
def residual_innovations(centered, phis, shrink_loadings=True, loading_scale=np.nan):
    n = len(centered); k = len(phis)
    q = np.empty((n, k))
    for j in range(k):
        col = ewma(centered, 1. - phis[j])
        q[:, j] = col - np.mean(col)
    ridge = .05 * max(np.var(centered), 1e-8) * np.eye(k)
    b = np.linalg.solve(q.T @ q + ridge, q.T @ centered)
    signal = np.empty(k)
    for j in range(k):
        signal[j] = abs(b[j]) * np.std(q[:, j])
    positive = signal[signal > 0.]
    if np.isfinite(loading_scale):
        b *= loading_scale
    elif shrink_loadings and positive.size:
        b *= signal / (signal + np.median(positive) + 1e-8)
    residual = centered - q @ b
    residual -= np.mean(residual)
    denominator = np.dot(residual[:-1], residual[:-1])
    rho = np.dot(residual[:-1], residual[1:]) / denominator if denominator > np.finfo(np.float64).tiny else 0.
    rho = min(.999999, max(-.999999, rho))
    error = centered[1:] - (q[:-1] * phis) @ b - rho * residual[:-1]
    return error


def objective(z, centered, shrink_loadings=True):
    phis = np.clip(expit(z), np.nextafter(0., 1.), np.nextafter(1., 0.))
    if not np.all((phis > 0.) & (phis < 1.)):
        return np.inf
    error = residual_innovations(centered, phis, shrink_loadings)
    variance = float(np.mean(error * error))
    if not np.isfinite(variance) or variance <= 0.:
        raise ArithmeticError('degenerate conditional multiscale variance')
    return .5 * len(error) * (np.log(2. * np.pi * variance) + 1.)


def fit(h, initial_phis, shrink_loadings=True):
    h = np.asarray(h, float)
    centered = h - np.mean(h)
    z = logit(initial_phis)
    initial = objective(z, centered, shrink_loadings)
    result = minimize(objective, z, args=(centered, shrink_loadings), method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-5})
    solver = 'L-BFGS-B'
    if not result.success or not np.isfinite(result.fun):
        start = result.x if np.isfinite(result.x).all() else z
        result = minimize(objective, start, args=(centered, shrink_loadings), method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-7, 'ftol': 1e-10})
        solver = 'Powell_after_gradient_line_search_failure'
    if not result.success or not np.isfinite(result.fun):
        raise ArithmeticError(f'learned decay optimization failed: {result.message}')
    if result.fun > initial + 1e-7:
        raise ArithmeticError('learned decay target worse than initial rates')
    phis = np.clip(expit(result.x), np.nextafter(0., 1.), np.nextafter(1., 0.))
    return phis, {'estimator': 'conditional_Gaussian_one_step_predictive_QMLE_on_Student_latent_mode',
                  'initial_negative_loglikelihood': float(initial),
                  'negative_loglikelihood': float(result.fun), 'success': bool(result.success),
                  'iterations': int(result.nit), 'evaluations': int(result.nfev),
                  'phis': phis.tolist(), 'component_count': len(phis),
                  'count_selection': 'fixed four; isolate decay locations first',
                  'optimizer_message': str(result.message), 'solver': solver}


def fit_adaptive(h, initial_phis, shrink_loadings=True):
    """Forward component-order selection by conditional BIC, not exact evidence.

Fit one component, then add a rate initialized in the largest log-timescale
interval resolvable by the history. Stop at the first BIC increase. This is a
local forward search, not a global guarantee over all possible orders.
"""
    h = np.asarray(h, float); centered = h - np.mean(h)
    rho = float(np.dot(centered[:-1], centered[1:]) / np.dot(centered[:-1], centered[:-1]))
    first = np.array([np.clip(rho, np.finfo(float).eps, 1. - np.finfo(float).eps)])
    rates, record = fit(h, first, shrink_loadings)
    def bic(receipt):
        return 2. * receipt['negative_loglikelihood'] + (2 * receipt['component_count'] + 3) * np.log(len(h) - 1)
    selected_bic = bic(record)
    tested = [{'count': 1, 'bic': float(selected_bic)}]
    while 2 * (len(rates) + 1) + 3 < len(h) - 1:
        log_times = np.log(-np.log(2.) / np.log(rates))
        grid = np.sort(np.r_[0., log_times, np.log(len(h))])
        gap = int(np.argmax(np.diff(grid)))
        extra_half = np.exp(.5 * (grid[gap] + grid[gap + 1]))
        extra_phi = np.exp(-np.log(2.) / extra_half)
        proposal, receipt = fit(h, np.r_[rates, extra_phi], shrink_loadings)
        score = bic(receipt)
        tested.append({'count': len(proposal), 'bic': float(score)})
        if score >= selected_bic:
            break
        rates, record, selected_bic = proposal, receipt, score
    record.update(count_selection='forward_conditional_BIC_first_nonimprovement',
                  selected_bic=float(selected_bic), tested_orders=tested,
                  bic_parameter_count=2 * len(rates) + 3,
                  bic_interpretation='conditional approximate model selection on fitted latent proxy; not joint raw-return Bayes evidence')
    return rates, record


def fit_one(h, initial_phis, shrink_loadings=True):
    """Matched one-component control using the adaptive search's initial fit."""
    h = np.asarray(h, float); centered = h - np.mean(h)
    rho = float(np.dot(centered[:-1], centered[1:]) / np.dot(centered[:-1], centered[:-1]))
    first = np.array([np.clip(rho, np.finfo(float).eps, 1. - np.finfo(float).eps)])
    rates, record = fit(h, first, shrink_loadings)
    record['count_selection'] = 'fixed_one_learned_decay_control'
    return rates, record


def objective_loading(z, centered):
    phis = np.clip(expit(z[:-1]), np.nextafter(0., 1.), np.nextafter(1., 0.))
    errors = residual_innovations(centered, phis, False, expit(z[-1]))
    variance = float(np.mean(errors * errors))
    return .5 * len(errors) * (np.log(2. * np.pi * variance) + 1.)


def fit_loading(h, initial_phis):
    h = np.asarray(h, float); centered = h - np.mean(h)
    z = np.r_[logit(initial_phis), 0.]
    initial = objective_loading(z, centered)
    result = minimize(objective_loading, z, args=(centered,), method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-5})
    solver = 'L-BFGS-B'
    if not result.success or not np.isfinite(result.fun):
        result = minimize(objective_loading, result.x if np.isfinite(result.x).all() else z,
                          args=(centered,), method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-7, 'ftol': 1e-10})
        solver = 'Powell_after_gradient_line_search_failure'
    if not result.success or not np.isfinite(result.fun) or result.fun > initial + 1e-7:
        raise ArithmeticError(f'predictive loading optimization failed: {result.message}')
    rates = np.clip(expit(result.x[:-1]), np.nextafter(0., 1.), np.nextafter(1., 0.))
    return rates, {'estimator': 'conditional_Gaussian_predictive_QMLE_decay_and_loading_shrinkage',
                      'success': True, 'initial_negative_loglikelihood': float(initial),
                      'negative_loglikelihood': float(result.fun), 'loading_scale': float(expit(result.x[-1])),
                      'component_count': len(rates), 'phis': rates.tolist(), 'iterations': int(result.nit),
                      'evaluations': int(result.nfev), 'solver': solver, 'optimizer_message': str(result.message)}


def fit_adaptive_loading(h, initial_phis):
    h = np.asarray(h, float); centered = h - np.mean(h)
    rho = float(centered[:-1] @ centered[1:] / (centered[:-1] @ centered[:-1]))
    rates, record = fit_loading(h, np.array([np.clip(rho, np.finfo(float).eps, 1. - np.finfo(float).eps)]))
    def bic(receipt):
        return 2. * receipt['negative_loglikelihood'] + (2 * receipt['component_count'] + 4) * np.log(len(h) - 1)
    score = bic(record); tested = [{'count': 1, 'bic': float(score)}]
    while 2 * (len(rates) + 1) + 4 < len(h) - 1:
        grid = np.sort(np.r_[0., np.log(-np.log(2.) / np.log(rates)), np.log(len(h))])
        gap = int(np.argmax(np.diff(grid)))
        extra = np.exp(-np.log(2.) / np.exp(.5 * (grid[gap] + grid[gap + 1])))
        proposed, receipt = fit_loading(h, np.r_[rates, extra])
        proposed_score = bic(receipt); tested.append({'count': len(proposed), 'bic': float(proposed_score)})
        if proposed_score >= score: break
        rates, record, score = proposed, receipt, proposed_score
    record.update(count_selection='forward_conditional_BIC_predictive_loading',
                  selected_bic=float(score), tested_orders=tested, bic_parameter_count=2 * len(rates) + 4)
    return rates, record
