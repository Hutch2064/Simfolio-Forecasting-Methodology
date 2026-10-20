"""One training-only variance-component root, no forecast-score inputs."""
import numpy as np
from scipy.optimize import brentq


def tau_squared(means, variances):
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if (means.ndim != 1 or means.shape != variances.shape or len(means) < 2
            or not np.isfinite(means).all() or not np.isfinite(variances).all()
            or np.any(variances < 0)):
        raise ValueError('Finite batch means and nonnegative sampling variances required')
    centered = means-means.mean()
    scale = float(max(np.max(variances), np.max(centered*centered)))
    if scale == 0:
        return 0.
    m, v = centered/np.sqrt(scale), variances/scale
    df = len(means)-1
    def equation(t):
        variance = v+t
        zero = variance == 0
        if np.any(zero):
            exact = m[zero]
            if np.any(exact != exact[0]):
                return np.inf
            mean = exact[0]
            return float(np.sum((m[~zero]-mean)**2/variance[~zero])-df)
        weights = 1/variance
        mean = np.dot(weights, m)/weights.sum()
        return float(np.dot(weights, (m-mean)**2)-df)
    if equation(0.) <= 0:
        return 0.
    upper = float(np.var(m, ddof=1)+np.max(v))
    while equation(upper) > 0:
        upper *= 2
        if not np.isfinite(upper):
            raise ArithmeticError('No finite variance-component bracket')
    root = brentq(equation, 0., upper, xtol=np.finfo(float).tiny, rtol=8*np.finfo(float).eps)
    if not np.isfinite(root) or abs(equation(root)) > 1e-9*df:
        raise ArithmeticError('Variance-component equation failed qualification')
    return float(root*scale)


def fit_training_batches(history):
    history = np.asarray(history, dtype=float)
    q = int(np.sqrt(len(history)))
    if history.ndim != 1 or q < 4 or not np.isfinite(history).all():
        raise ValueError('Finite history and at least four training batches required')
    groups = np.array_split(history, q)
    means = np.array([group.mean() for group in groups])
    variances = np.array([np.var(group, ddof=1)/len(group) for group in groups])
    return {'tau_squared': tau_squared(means, variances), 'batches': q,
            'batch_sizes': [len(group) for group in groups],
            'batch_means': means, 'within_batch_sampling_variances': variances}
