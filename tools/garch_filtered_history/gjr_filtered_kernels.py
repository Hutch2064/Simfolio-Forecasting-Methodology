"""Stable import identity for the candidate's cached numerical recursion."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def forecast_paths(params, terminal_variance, terminal_residual, pool, indices, means):
    sims, horizon = indices.shape
    assets = pool.shape[1]
    paths = np.empty((sims, horizon, assets))
    for sim in range(sims):
        variance = terminal_variance.copy()
        residual = terminal_residual.copy()
        for day in range(horizon):
            for asset in range(assets):
                omega, alpha, gamma, beta = params[asset]
                square = residual[asset]*residual[asset]
                v = omega + alpha*square + gamma*(residual[asset] < 0.)*square + beta*variance[asset]
                if not np.isfinite(v) or v <= 0.:
                    raise ArithmeticError('invalid future GJR conditional variance')
                innovation = np.sqrt(v)*pool[indices[sim, day], asset]
                value = means[day, asset] + innovation
                if not np.isfinite(value):
                    raise ArithmeticError('invalid future GJR return')
                paths[sim, day, asset] = min(max(value, -1.), 1.)
                residual[asset] = innovation
                variance[asset] = v
    return paths

