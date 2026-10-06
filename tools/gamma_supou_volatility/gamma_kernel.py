"""Accuracy-controlled Gaussian supOU covariance with Gamma-mixed rates.

C(t)=(1+kappa*t)**(-shape). Equivalently, stationary variance is distributed
over OU rates lambda~Gamma(shape,scale=kappa). This is the Gaussian specialization
of the Gamma-rate supOU covariance in Barndorff-Nielsen & Stelzer (2011), example
3.4, with their mixing shape equal to shape+1. It has polynomial memory decay.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.special import gammainc, roots_jacobi
from dynamic import CovarianceCheck


class GammaCovarianceCheck(CovarianceCheck):
    def values(self, lags):
        missing = [int(t) for t in lags if int(t) not in self.exact]
        if missing:
            values = np.exp(-self.hurst*np.log1p(self.kappa*np.asarray(missing)))
            self.exact.update(zip(missing, values))
            self.evaluations += len(missing)
        return np.array([self.exact[int(t)] for t in lags])


def quadrature(shape, kappa, order, tolerance):
    upper_rate = -math.log(tolerance/2)
    extent = math.log1p(upper_rate/kappa)
    nodes, mass = roots_jacobi(order, 0., shape-1)
    v = extent*(nodes+1)/2
    u = np.expm1(v)
    # Gauss-Jacobi handles v**(shape-1); retain the rest of the transformed
    # Gamma density in the positive weights. Split the same .001 budget as
    # the rough baseline. Fast tail mass becomes an independent white factor.
    log_density = (shape-1)*np.log(u/v)-u+v
    mass *= np.exp(log_density-log_density.max())
    retained = gammainc(shape, upper_rate/kappa)
    mass *= retained/mass.sum()
    return np.r_[np.exp(-kappa*u), 0.], np.r_[mass, 1-retained]


@lru_cache(maxsize=4096)
def selected_kernel(log_shape, kappa, maximum_lag, tolerance=.001):
    shape = math.exp(log_shape)
    check = GammaCovarianceCheck(shape, kappa, maximum_lag, tolerance)
    order = 1
    while True:
        phi, mass = quadrature(shape, kappa, order, tolerance)
        passed, error_bound = check.passes(phi, mass)
        if passed:
            return phi, mass, dict(factors=phi.size, quadrature_order=order,
                tolerance=tolerance, maximum_daily_lag=maximum_lag,
                autocorrelation_error_upper_bound=error_bound,
                exact_covariance_evaluations=check.evaluations,
                selection='first_passing_positive_integer_order',
                white_tail_error_upper_bound=tolerance/2,
                covariance='(1+kappa*lag)**(-shape)', shape=shape)
        order += 1


def configuration(theta, adaptive):
    maximum_lag = int(adaptive.split(':')[1])
    phi, mass, _ = selected_kernel(float(theta[0]), math.exp(theta[1]), maximum_lag)
    return phi, np.ones(phi.size), np.diag(math.exp(2*theta[2])*mass*(1-phi*phi))


def log_prior(theta):
    if not (math.log(.03)<theta[0]<math.log(10) and
            math.log(1/2520)<theta[1]<math.log(.5) and
            math.log(.05)<theta[2]<math.log(3)):
        return -np.inf
    return (-.5*(theta[0]/1.5)**2 - .5*((theta[1]-math.log(1/63))/2)**2
            - .5*((theta[2]-math.log(.7))/1.5)**2)
