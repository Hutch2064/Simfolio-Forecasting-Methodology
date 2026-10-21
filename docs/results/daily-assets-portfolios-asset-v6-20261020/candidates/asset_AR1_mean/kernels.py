"""Joint AR1-GARCH innovations with log-domain in-path portfolio accounting."""
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def advance(pool, indices, delta, mu, target, alpha, beta, variance, phi, previous,
            target_weights, current_weights, asset_paths, portfolio_paths):
    paths, width = indices.shape
    assets = pool.shape[1]
    result = np.empty((paths, width))
    for p in range(paths):
        for t in range(width):
            maximum = -np.inf
            for a in range(assets):
                normalized = np.sqrt(variance[p, a])*pool[indices[p, t], a]
                center = mu[a]+delta[p, a]
                if phi[a]==0.:
                    value = center+np.sqrt(target[a])*normalized
                else:
                    value = center+phi[a]*(previous[p,a]-center)+np.sqrt(target[a])*normalized
                previous[p,a]=value
                variance[p, a] = 1.-alpha[a]-beta[a]+alpha[a]*normalized*normalized+beta[a]*variance[p, a]
                asset_paths[p, a] += value
                current_weights[p, a] = target_weights[a]+value
                maximum = max(maximum, current_weights[p, a])
            total = 0.
            for a in range(assets):
                total += np.exp(current_weights[p, a]-maximum)
            step = maximum+np.log(total)
            portfolio_paths[p] += step
            for a in range(assets):
                current_weights[p, a] -= step
            result[p, t] = portfolio_paths[p]
    return result
