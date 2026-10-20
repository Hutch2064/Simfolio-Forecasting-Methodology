"""Exact stationary Gaussian MA(1) innovations likelihood, training qualification only."""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar
MARGIN = 1e-8


@njit(cache=True, fastmath=False)
def innovations(y, theta):
    errors = np.empty(len(y))
    variances = np.empty(len(y))
    variances[0] = 1.+theta*theta
    errors[0] = y[0]
    for i in range(1, len(y)):
        variances[i] = 1.+theta*theta-theta*theta/variances[i-1]
        errors[i] = y[i]-theta*errors[i-1]/variances[i-1]
    return errors, variances


def concentrated_loss(theta, y):
    errors, variances = innovations(y, theta)
    scale2 = float(np.mean(errors*errors/variances))
    if not np.isfinite(scale2) or scale2 <= 0:
        raise ValueError('Finite positive concentrated innovation variance required')
    return .5*(np.log(scale2)+float(np.mean(np.log(variances))))


def fit(history):
    history = np.asarray(history, dtype=float)
    mu = float(history.mean())
    y = history-mu
    if len(y) < 5 or not np.isfinite(y).all() or not np.any(y):
        raise ValueError('Finite nonconstant history required')
    # Standardizing makes the single convergence tolerance independent of return units.
    sd = float(np.sqrt(np.mean(y*y)))
    normalized = y/sd
    result = minimize_scalar(concentrated_loss, args=(normalized,), method='bounded',
        bounds=(-1.+MARGIN, 1.-MARGIN), options={'xatol': 1e-9, 'maxiter': 500})
    if not result.success or not np.isfinite(result.fun):
        raise ArithmeticError('Exact MA1 training likelihood did not converge')
    theta = float(result.x)
    null = concentrated_loss(0., normalized)
    if result.fun >= null:
        theta = 0.
    errors, variances = innovations(y, theta)
    scale2 = float(np.mean(errors*errors/variances))
    return {'theta': theta, 'mean': mu, 'innovation_variance': scale2,
        'profile_objective_standardized': concentrated_loss(theta, normalized),
        'null_objective_standardized': null, 'converged': True,
        'iterations': int(result.nfev), 'errors': errors, 'variance_factors': variances,
        'last_latent_innovation_mean_Gaussian_working_model': float(errors[-1]/variances[-1]),
        'last_latent_innovation_variance_Gaussian_working_model': float(scale2*max(0., 1.-1./variances[-1])),
        'mean_fixed_at_training_sample_mean_not_jointly_estimated': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
