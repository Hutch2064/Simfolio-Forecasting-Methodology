"""Asset-level raw/FHS accounting; log arithmetic avoids wealth overflow."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def advance_raw(history, indices, delta, mu, scales,
                target_weights, current_weights, asset_paths, portfolio_paths):
    paths, width = indices.shape
    assets = history.shape[1]
    result = np.empty((paths, width))
    for p in range(paths):
        for t in range(width):
            maximum = -np.inf
            for a in range(assets):
                if scales[a,t]==1.:
                    value=history[indices[p,t],a]+delta[p,a]
                else:
                    value=mu[a]+scales[a,t]*(history[indices[p,t],a]-mu[a])+delta[p,a]
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
