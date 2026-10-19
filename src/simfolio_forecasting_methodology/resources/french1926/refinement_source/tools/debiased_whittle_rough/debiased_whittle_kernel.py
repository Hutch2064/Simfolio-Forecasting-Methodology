"""Exact finite-sample spectrum of a stationary OU mixture."""
import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def expected_periodogram(phi, stationary_mass, n):
    """Exact triangular-window covariance transform at positive DFT frequencies.

    Sum (1-l/n)*z**l analytically for each OU factor, z=phi*exp(-iw).
    At DFT frequencies z**n=phi**n. No frequency subsampling or factor cap.
    """
    result = np.zeros(n//2)
    for j in range(1, n//2+1):
        omega = 2*math.pi*j/n
        for k in range(phi.size):
            z = phi[k]*complex(math.cos(omega), -math.sin(omega))
            triangle = z/(1-z) - z*(1-phi[k]**n)/(n*(1-z)**2)
            result[j-1] += stationary_mass[k]*(1+2*triangle.real)
    return result

