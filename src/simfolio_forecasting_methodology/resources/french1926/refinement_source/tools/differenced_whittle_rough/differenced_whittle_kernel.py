"""Exact finite-sample spectrum of a differenced OU mixture."""
import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def differenced_expected_periodogram(phi, stationary_mass, n):
    """Exact finite-n expectation for differences, including zero frequency.

    Difference covariance: gamma_d(0)=2*m*(1-phi), and for lag >=1,
    gamma_d(lag)=-m*(1-phi)**2*phi**(lag-1). This also handles white noise.
    """
    result = np.zeros(n//2+1)
    for j in range(n//2+1):
        omega = 2*math.pi*j/n
        e = complex(math.cos(omega), -math.sin(omega))
        for k in range(phi.size):
            z = phi[k]*e
            triangle = e/(1-z)-e*(1-phi[k]**n)/(n*(1-z)**2)
            result[j] += stationary_mass[k]*(2*(1-phi[k])-2*(1-phi[k])**2*triangle.real)
    return result

