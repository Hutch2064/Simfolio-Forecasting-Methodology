"""Importable compiled predictor; cached environments survive candidate aliases."""
import numpy as np
from numba import njit


@njit(cache=True,nogil=True)
def causal_predictor(y,level,phi,eta,noise):
    result=np.empty(y.size);mean=level
    variance=max(eta*eta/max(1-phi*phi,1e-4),1e-6)
    for t in range(y.size):
        result[t]=mean
        gain=variance/(variance+noise)
        mean=level+phi*(mean+gain*(y[t]-mean)-level)
        variance=phi*phi*max((1-gain)*variance,1e-8)+eta*eta
    return result

