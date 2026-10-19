"""MAP in standardized Gaussian-prior coordinates, with only H box constrained."""
import math
import time

import numpy as np
from scipy.optimize import minimize


def physical(u):
    return np.array([.5 * u[0], math.log(1 / 63) + 2 * u[1], math.log(.7) + 1.5 * u[2]])


def fit_map(target, max_seconds=10.):
    started = time.perf_counter()
    evaluations = 0

    def objective(u):
        nonlocal evaluations
        evaluations += 1
        if time.perf_counter() - started > max_seconds:
            raise RuntimeError('untruncated-prior MAP exceeded the resource ceiling')
        value = float(target(physical(u)))
        return -value if np.isfinite(value) else 1e100

    bounds = [(1e-10, 1 - 1e-10), (None, None), (None, None)]
    initial = [.2, 0., 0.]
    attempts = []
    mode = None
    for maxls in (20, 100):
        mode = minimize(objective, initial if mode is None else mode.x, method='L-BFGS-B', bounds=bounds,
                        options={'ftol': 1e-10, 'gtol': 1e-5, 'maxiter': 200, 'maxls': maxls})
        attempts.append({'method': 'L-BFGS-B', 'maxls': maxls, 'success': bool(mode.success), 'message': str(mode.message)})
        if mode.success:
            break
    if not mode.success or not np.isfinite(mode.fun) or mode.fun >= 1e100:
        raise ArithmeticError(f'untruncated-prior MAP failed: {mode.message}')
    return {'map': physical(mode.x), 'estimator': 'MAP_point', 'posterior_uncertainty': False,
            'evaluations': evaluations, 'seconds': time.perf_counter() - started,
            'optimizer': {'success': bool(mode.success), 'message': str(mode.message),
                          'iterations': int(mode.nit), 'ftol': 1e-10, 'gtol': 1e-5,
                          'maximum_iterations': 200, 'attempts': attempts}}
