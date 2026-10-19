"""Compile fixed-proposal postburn with the existing NumPy Generator stream."""
import math

import numpy as np
from numba import njit
from steady_kalman_target import target


@njit(cache=True,fastmath=False,nogil=True)
def postburn(y,t,lp,chol,rng,kept,lower,upper,ymean):
    trace=np.empty((kept,3));accept=np.zeros(kept)
    for i in range(kept):
        proposal=t+chol@rng.normal(size=3)
        value=-np.inf
        if (lower[0]<=proposal[0]<=upper[0] and lower[1]<=proposal[1]<=upper[1] and lower[2]<=proposal[2]<=upper[2]):
            value=target(y,proposal[0],proposal[1],proposal[2],ymean)
        if math.log(rng.random())<value-lp:
            t=proposal;lp=value;accept[i]=1.
        trace[i]=t
    return trace,accept
