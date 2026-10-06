"""One-step Gaussian innovations for the existing multiscale state law."""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def predict(y, level, phi, loading, covariance, observation_variance):
    state = np.zeros(len(phi))
    variance = covariance / (1 - phi[:, None] * phi[None, :])
    result = np.empty(len(y))
    for t in range(len(y)):
        result[t] = level + loading @ state
        projected = variance @ loading
        denominator = observation_variance + loading @ projected
        state = phi * (state + projected * ((y[t] - result[t]) / denominator))
        variance = (variance - np.outer(projected, projected) / denominator) * phi[:, None] * phi[None, :] + covariance
        variance = .5 * (variance + variance.T)
    return result
