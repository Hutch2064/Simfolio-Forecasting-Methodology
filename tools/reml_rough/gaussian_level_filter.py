"""One-pass Gaussian mean profiling and restricted covariance likelihood."""
import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def restricted_filter(y, phi, w, q, measurement_variance):
    n = phi.size
    p = q/(1-phi[:, None]*phi[None, :])
    state_y = np.zeros(n)
    state_one = np.zeros(n)
    predicted = np.empty((n,n))
    pw = np.empty(n)
    gain = np.empty(n)
    fixed = False
    syy = sy1 = s11 = logdet = 0.
    variance = 0.
    for t in range(y.size):
        if not fixed:
            for i in range(n):
                total = 0.
                for j in range(n):
                    predicted[i,j] = p[i,j]*phi[i]*phi[j]+q[i,j]
                    total += predicted[i,j]*w[j]
                pw[i] = total
            variance = measurement_variance+np.dot(w,pw)
            gain = pw/variance
        prediction_y = prediction_one = 0.
        for i in range(n):
            state_y[i] *= phi[i]
            state_one[i] *= phi[i]
            prediction_y += w[i]*state_y[i]
            prediction_one += w[i]*state_one[i]
        ey = y[t]-prediction_y
        e1 = 1.-prediction_one
        syy += ey*ey/variance
        sy1 += ey*e1/variance
        s11 += e1*e1/variance
        logdet += math.log(2*math.pi*variance)
        for i in range(n):
            state_y[i] += gain[i]*ey
            state_one[i] += gain[i]*e1
        if not fixed:
            fixed = True
            for i in range(n):
                for j in range(n):
                    updated = predicted[i,j]-pw[i]*pw[j]/variance
                    if updated != p[i,j]:
                        fixed = False
                    p[i,j] = updated
    level = sy1/s11
    level_variance = 1/s11
    # Flat-prior integration of the unknown constant level, equivalently
    # REML up to a parameter-independent constant. Return its uncertainty.
    restricted = -.5*(logdet+syy-sy1*sy1/s11+math.log(s11))
    state = state_y-level*state_one
    integrated_covariance = p+np.outer(state_one,state_one)*level_variance
    return restricted,level,level_variance,state,integrated_covariance
