"""Two-state, fixed-mean Gaussian HMM with stated proper empirical MAP priors."""
import numpy as np
from numba import njit
from scipy.optimize import minimize

STARTS = ((.5, 1.5, .95), (.25, 3., .8))


@njit(cache=True)
def smooth(y, variance, transition):
    n = len(y)
    emission = np.empty((n, 2))
    alpha = np.empty((n, 2))
    scales = np.empty(n)
    ll = 0.
    for t in range(n):
        a = -.5*(np.log(2*np.pi*variance[0])+y[t]*y[t]/variance[0])
        b = -.5*(np.log(2*np.pi*variance[1])+y[t]*y[t]/variance[1])
        offset = max(a, b)
        emission[t, 0] = np.exp(a-offset)
        emission[t, 1] = np.exp(b-offset)
        for j in range(2):
            prediction = .5 if t == 0 else alpha[t-1, 0]*transition[0, j]+alpha[t-1, 1]*transition[1, j]
            alpha[t, j] = prediction*emission[t, j]
        scales[t] = alpha[t, 0]+alpha[t, 1]
        alpha[t] /= scales[t]
        ll += np.log(scales[t])+offset
    gamma = np.empty_like(alpha)
    counts = np.zeros((2, 2))
    beta = np.ones(2)
    for t in range(n-1, -1, -1):
        gamma[t] = alpha[t]*beta
        gamma[t] /= gamma[t].sum()
        if t > 0:
            joint = np.empty((2, 2))
            for i in range(2):
                for j in range(2):
                    joint[i, j] = alpha[t-1, i]*transition[i, j]*emission[t, j]*beta[j]
            counts += joint/joint.sum()
            previous = np.empty(2)
            for i in range(2):
                previous[i] = (transition[i, 0]*emission[t, 0]*beta[0]+transition[i, 1]*emission[t, 1]*beta[1])/scales[t]
            beta = previous
    return ll, gamma, counts, alpha[-1].copy()


def prior(variance, transition):
    # IG(shape=2, scale=3), independent variances; Dirichlet(2,2) rows.
    return float(np.log(transition).sum()-3*np.log(variance).sum()-3*(1/variance).sum())


def mstep(y, gamma, counts):
    variance = ((gamma*y[:, None]**2).sum(axis=0)+6)/(gamma.sum(axis=0)+6)
    transition = (counts+1)/(counts.sum(axis=1, keepdims=True)+2)
    return variance, transition


def unpack(theta):
    variance = np.exp(theta[:2])
    from scipy.special import expit
    p = expit(theta[2:])
    return variance, np.column_stack((1-p, p))


def objective_gradient(theta, y):
    variance, transition = unpack(theta)
    if not np.isfinite(variance).all() or np.any(variance <= 0) or np.any(transition <= 0):
        return np.inf, np.zeros(4)
    ll, gamma, counts, _ = smooth(y, variance, transition)
    variance_gradient = .5*((gamma*y[:, None]**2).sum(axis=0)/variance-gamma.sum(axis=0))-3+3/variance
    transition_gradient = counts[:, 1]-transition[:, 1]*counts.sum(axis=1)+1-2*transition[:, 1]
    return -(ll+prior(variance, transition))/len(y), -np.r_[variance_gradient, transition_gradient]/len(y)


def fit(history):
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 1 or len(history) < 3 or not np.isfinite(history).all():
        raise ValueError('Finite one-dimensional training history of at least three observations required')
    mean = float(history.mean())
    residual = history-mean
    target = float(np.mean(residual**2))
    if target == 0:
        return {'mean': mean, 'variance': np.zeros(2), 'transition': np.full((2, 2), .5),
                'filtered': np.full(2, .5), 'constant_history': True, 'starts': [], 'target_variance': 0.}
    y = residual/np.sqrt(target)
    solutions = []
    reports = []
    for low, high, persistence in STARTS:
        variance = np.array([low, high])
        transition = np.array([[persistence, 1-persistence], [1-persistence, persistence]])
        ll, gamma, counts, _ = smooth(y, variance, transition)
        initial = ll+prior(variance, transition)
        # Monotone MAP EM warm start; direct observed-posterior optimization
        # then avoids arbitrarily slow EM convergence for overlapping regimes.
        for _ in range(20):
            variance, transition = mstep(y, gamma, counts)
            next_ll, gamma, counts, _ = smooth(y, variance, transition)
            updated = next_ll+prior(variance, transition)
            if updated < initial-1e-9*(1+abs(initial)):
                raise RuntimeError('MAP EM posterior decreased')
            initial = updated
        theta = np.r_[np.log(variance), np.log(transition[:, 1]/transition[:, 0])]
        result = minimize(objective_gradient, theta, args=(y,), jac=True, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 40})
        value, gradient = objective_gradient(result.x, y)
        variance, transition = unpack(result.x)
        ll, gamma, counts, filtered = smooth(y, variance, transition)
        vnext, pnext = mstep(y, gamma, counts)
        fixed_point = max(float(np.max(abs(np.log(vnext/variance)))), float(np.max(abs(pnext-transition))))
        valid = bool(result.success and np.isfinite(value) and np.max(abs(gradient)) <= 1e-6
                     and fixed_point <= 1e-4 and -value*len(y) >= initial-1e-7)
        report = {'success': valid, 'optimizer_success': bool(result.success), 'message': str(result.message),
                  'iterations': int(result.nit), 'gradient_per_observation': float(np.max(abs(gradient))),
                  'MAP_EM_fixed_point_error': fixed_point, 'log_posterior_normalized_data': -float(value)*len(y),
                  'expected_occupancy': gamma.sum(axis=0).tolist()}
        reports.append(report)
        if valid:
            solutions.append((value, variance, transition, filtered))
    if not solutions:
        raise RuntimeError('No converged stationary HMM MAP fit: '+str(reports))
    value, variance, transition, filtered = min(solutions, key=lambda x: x[0])
    order = np.argsort(variance)
    return {'mean': mean, 'variance': variance[order]*target,
            'transition': transition[np.ix_(order, order)], 'filtered': filtered[order],
            'constant_history': False, 'starts': reports, 'target_variance': target,
            'best_log_posterior_normalized_data': -float(value)*len(y)}


@njit(cache=True)
def native_paths(variance, transition, filtered, mean, delta, state_uniforms, shocks):
    paths, horizon = shocks.shape
    out = np.empty_like(shocks)
    clips = floors = 0
    high = np.finfo(np.float64).max/10
    max_log = np.log1p(high)
    for p in range(paths):
        state = 0 if state_uniforms[p, 0] < filtered[0] else 1
        wealth = 1.
        previous = 0.
        for t in range(horizon):
            state = 0 if state_uniforms[p, t+1] < transition[state, 0] else 1
            log_return = mean+delta[p]+np.sqrt(variance[state])*shocks[p, t]
            if log_return > max_log:
                simple = high
                clips += 1
            else:
                simple = np.expm1(log_return)
                if simple < -.999999:
                    simple = -.999999
                    clips += 1
            wealth *= 1+simple
            if wealth < 1e-300:
                floors += 1
            current = np.log(max(wealth, 1e-300))
            out[p, t] = current-previous
            previous = current
    return out, clips, floors
