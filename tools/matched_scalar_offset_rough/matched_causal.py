"""Exact one-step Gaussian proxy predictions under the fitted scalar AR1 law."""
import numpy as np
from numba import njit


@njit(cache=True,nogil=True)
def predict(y,level,rho,innovation_sd,measurement_variance):
    mean=level;variance=innovation_sd**2/(1.-rho*rho);result=np.empty(len(y))
    for t in range(len(y)):
        result[t]=mean
        gain=variance/(variance+measurement_variance)
        mean=level+rho*(mean+gain*(y[t]-mean)-level)
        # Product form avoids cancellation when the stationary prior is diffuse.
        variance=rho*rho*(variance*(measurement_variance/(variance+measurement_variance)))+innovation_sd**2
    return result
