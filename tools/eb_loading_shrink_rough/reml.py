"""Gaussian loading evidence conditional on plug-in level and AR1 whitening."""
import numpy as np
from scipy.optimize import minimize


def statistics(y, z, phi):
    white_y = np.r_[np.sqrt(1 - phi ** 2) * y[0], y[1:] - phi * y[:-1]]
    white_z = np.vstack((np.sqrt(1 - phi ** 2) * z[0], z[1:] - phi * z[:-1]))
    a = white_z.T @ white_z
    return a, white_z.T @ white_y, float(white_y @ white_y), len(y), np.linalg.eigvalsh(a)


def evaluate(log_penalty, stats):
    a, cross, yy, n, eigen = stats
    penalty = float(np.exp(log_penalty))
    beta = np.linalg.solve(a + penalty * np.eye(len(a)), cross)
    rss = yy - cross @ beta
    if rss <= 0:
        raise ArithmeticError('nonpositive conditional loading evidence variance')
    value = .5 * (n * np.log(rss / n) + np.log1p(eigen / penalty).sum()) / n
    gradient = .5 * (n * penalty * (beta @ beta) / rss - np.sum(eigen / (eigen + penalty))) / n
    return float(value), float(gradient), beta


def fit(y, z, phi, initial_penalty):
    stats = statistics(y, z, phi)
    limit = np.log(np.finfo(float).max) / 2
    result = minimize(lambda x: evaluate(float(x[0]), stats)[:2], [np.log(initial_penalty)],
                      jac=True, method='L-BFGS-B', bounds=[(-limit, limit)],
                      options={'gtol': 1e-7, 'ftol': 1e-12, 'maxiter': 200})
    value, gradient, beta = evaluate(float(result.x[0]), stats)
    if abs(gradient) > 1e-6 or not np.isfinite(value):
        raise ArithmeticError(f'loading evidence optimization failed: {result.message}')
    null_value = .5 * np.log(stats[2] / stats[3])
    null = null_value <= value
    if null:
        beta = np.zeros_like(beta)
    return beta, {'conditional_mean_negative_loglikelihood': float(min(value, null_value)),
                  'null_loading_variance': bool(null), 'penalty_ratio': None if null else float(np.exp(result.x[0])),
                  'whitening_phi_from_original_fit': float(phi), 'gradient_inf_norm': abs(gradient),
                  'optimizer_iterations': int(result.nit), 'optimizer_evaluations': int(result.nfev)}
