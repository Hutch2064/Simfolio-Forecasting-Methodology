"""Joint extension of the prior best's weighted batch mean uncertainty."""
import numpy as np
from scipy.stats import norm


def estimate(history):
    x = np.asarray(history, dtype=float)
    n, assets = x.shape
    q = int(np.sqrt(n))
    if q < 4 or not np.isfinite(x).all():
        raise ValueError('At least four finite training batches required')
    mu = x.mean(axis=0)
    residual = np.stack([np.sqrt(len(g))*(g.mean(axis=0)-mu) for g in np.array_split(x, q)])
    df = q-1
    # Every observation enters, including unequal final batch sizes. This is
    # covariance of mean uncertainty, not covariance of independent daily shocks.
    covariance = (residual.T@residual)/df/n*(df/(df-2))
    se = np.sqrt(np.diag(covariance))
    center = mu.copy()
    for a in range(assets):
        if se[a] == 0 or mu[a] == 0:
            continue
        if abs(mu[a]) >= se[a]:
            weight = 1/(1+(se[a]/mu[a])**2)
        else:
            ratio = (mu[a]/se[a])**2
            weight = ratio/(1+ratio)
        center[a] *= weight
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) < -1e-10*max(float(np.max(eigenvalues)), np.finfo(float).tiny):
        raise ArithmeticError('Batch covariance is not positive semidefinite')
    root = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))
    return {'sample_mean': mu, 'center': center, 'covariance': covariance,
            'root': root, 'batch_count': q, 'df': df}


def draws(fit, paths, seed):
    if paths < 2 or paths % 2:
        raise ValueError('Even ensemble required for antithetic mean uncertainty')
    rng = np.random.default_rng(seed)
    z = np.empty((paths, len(fit['sample_mean'])))
    for a in range(z.shape[1]):
        probabilities = (np.arange(paths//2)+1.-rng.random(paths//2))/paths
        lower = norm.ppf(probabilities)
        z[:, a] = np.concatenate((lower, -lower))[rng.permutation(paths)]
    return z@fit['root'].T+(fit['center']-fit['sample_mean'])
