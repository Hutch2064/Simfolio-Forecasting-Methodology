"""Asset-level, variable-order quadrature of the tempered rough covariance.

For alpha=H+1/2, the normalized stationary covariance has the exact
representation E[exp(-kappa*(1+Z)/(1-Z)*t)], Z~Beta(1-alpha,2*H).
Positive quadrature weights therefore define a valid stationary Gaussian
OU mixture. Resolution is selected for each (H,kappa) over the complete
requested daily-lag interval, without a fixed factor count or factor cap.
"""
from dataclasses import replace
from functools import lru_cache
import math

import numpy as np
from scipy.special import beta, betainc, eval_jacobi, roots_jacobi
from scipy.special._orthogonal import _gen_roots_and_weights

from overlay import Candidate, exact_covariance


class JacobiGeometry:
    """Reuse recurrence coefficients across orders for one parameter proposal.

    The pinned SciPy root refinement and weight normalization remain unchanged.
    Allocation grows geometrically; it does not restrict eligible factor counts.
    """
    def __init__(self, hurst):
        self.b = -(hurst + .5)
        self.mass = 2. ** (self.b + 1) * beta(1., self.b + 1)
        self.size = 0

    def roots(self, order):
        a, b = 0., self.b
        if order > self.size:
            self.size = 1 << (order - 1).bit_length()
            k = np.arange(self.size, dtype='d')
            self.diagonal = np.where(k == 0, (b-a)/(2+a+b),
                (b*b-a*a)/((2.*k+a+b)*(2.*k+a+b+2)))
            k = k[1:]
            self.off_diagonal = (2./(2.*k+a+b)
                * np.sqrt((k+a)*(k+b)/(2*k+a+b+1))
                * np.where(k == 1, 1., np.sqrt(k*(k+a+b)/(2.*k+a+b-1))))
        return _gen_roots_and_weights(order, self.mass,
            lambda k: self.diagonal[:k.size],
            lambda k: self.off_diagonal[:k.size],
            lambda n, x: eval_jacobi(n, a, b, x),
            lambda n, x: .5*(n+a+b+1)*eval_jacobi(n-1, a+1, b+1, x),
            False, False)


def quadrature(hurst, kappa, order, tolerance, geometry=None):
    alpha = hurst + .5
    # Split the numerical error budget equally. At positive daily lags,
    # replacing rates >= upper by white noise costs at most tolerance/2.
    upper = -math.log(tolerance / 2)
    extent = math.log(upper / kappa)
    nodes, mass = (roots_jacobi(order, 0., -alpha) if geometry is None
                   else geometry.roots(order))
    v = extent * (nodes + 1) / 2
    rates = kappa * np.exp(v)
    log_density = (np.log(rates) - alpha * np.log(kappa * np.expm1(v) / v)
                   - alpha * np.log(rates + kappa))
    mass *= np.exp(log_density - log_density.max())
    retained = betainc(.5 - hurst, 2 * hurst, (upper - kappa) / (upper + kappa))
    mass *= retained / mass.sum()
    phi, full_mass = np.empty(order + 1), np.empty(order + 1)
    phi[:-1], phi[-1] = np.exp(-rates), 0.
    full_mass[:-1], full_mass[-1] = mass, 1 - retained
    return phi, full_mass


@lru_cache(maxsize=128)
def initial_lags(limit):
    lags = np.unique(np.concatenate((2 ** np.arange(limit.bit_length()), [limit])))
    return lags[(lags >= 1) & (lags <= limit)]


class CovarianceCheck:
    """Monotonic interval enclosure covering every integer daily lag.

    Both covariances decrease with lag. On [a,b] their difference lies in
    [approx(b)-exact(a), approx(a)-exact(b)]. Refine unresolved intervals;
    adjacent integer endpoints exhaust their daily lags. Exact evaluations
    are shared across all candidate orders for this parameter setting.
    """
    def __init__(self, hurst, kappa, maximum_lag, tolerance):
        self.hurst, self.kappa = hurst, kappa
        self.maximum_lag, self.tolerance = maximum_lag, tolerance
        self.exact = {}
        self.evaluations = 0
        self.lags = initial_lags(maximum_lag)
        self.initial_exact = self.values(self.lags)

    def values(self, lags):
        missing = [int(t) for t in lags if int(t) not in self.exact]
        if missing:
            values = exact_covariance(self.hurst, self.kappa, np.asarray(missing, float))
            if not np.isfinite(values).all() or (values < 0).any():
                raise ArithmeticError('nonfinite exact rough covariance reference')
            self.exact.update(zip(missing, values))
            self.evaluations += len(missing)
        return np.array([self.exact[int(t)] for t in lags])

    def passes(self, phi, mass):
        lags, exact = self.lags, self.initial_exact
        def approximate(t):
            return (phi[None, :] ** np.asarray(t)[:, None]) @ mass
        actual = approximate(lags)
        margin = np.finfo(float).eps * phi.size * 8
        threshold = self.tolerance - margin
        if np.max(np.abs(actual - exact)) > threshold:
            return False, math.inf
        left, right = lags[:-1], lags[1:]
        el, er, al, ar = exact[:-1], exact[1:], actual[:-1], actual[1:]
        maximum_bound = float(np.max(np.abs(actual - exact)))
        while left.size:
            bound = np.maximum(np.abs(al - er), np.abs(ar - el))
            # For a positive, unit-mass exponential mixture, C''(t) <=
            # 4*exp(-2)/t**2. The difference of two such covariances has
            # twice that curvature bound. Linear interpolation therefore
            # bounds its interior error without evaluating every lag.
            endpoint = np.maximum(np.abs(al - el), np.abs(ar - er))
            curvature = endpoint + math.exp(-2) * ((right - left) / left) ** 2
            bound = np.minimum(bound, curvature)
            unresolved = (right - left > 1) & (bound > threshold)
            settled = ~unresolved
            if settled.any():
                certified = np.where(right - left > 1, bound, endpoint)
                maximum_bound = max(maximum_bound, float(certified[settled].max()))
            if not unresolved.any():
                break
            left, right, el, er, al, ar = [v[unresolved] for v in (left, right, el, er, al, ar)]
            middle = (left + right) // 2
            em, am = self.values(middle), approximate(middle)
            if np.max(np.abs(em - am)) > threshold:
                return False, math.inf
            left, right, el, er, al, ar = [np.concatenate(v) for v in (
                (left, middle), (middle, right), (el, em), (em, er), (al, am), (am, ar))]
        return True, maximum_bound + margin


@lru_cache(maxsize=4096)
def selected_kernel(hurst, kappa, maximum_lag, tolerance=.001):
    check = CovarianceCheck(hurst, kappa, maximum_lag, tolerance)
    geometry = JacobiGeometry(hurst)
    order = 1
    while True:
        phi, mass = quadrature(hurst, kappa, order, tolerance, geometry)
        passed, error_bound = check.passes(phi, mass)
        if passed:
            return phi, mass, {'factors': phi.size, 'quadrature_order': order,
                'tolerance': tolerance, 'maximum_daily_lag': maximum_lag,
                'autocorrelation_error_upper_bound': error_bound,
                'exact_covariance_evaluations': check.evaluations,
                'selection': 'first_passing_positive_integer_order',
                'white_tail_error_upper_bound': tolerance / 2}
        order += 1


def configuration(theta, maximum_lag):
    phi, mass, _ = selected_kernel(float(theta[0]), math.exp(theta[1]), maximum_lag)
    covariance = np.diag(math.exp(2 * theta[2]) * mass * (1 - phi * phi))
    return phi, np.ones(phi.size), covariance


class DynamicCandidate(Candidate):
    def simulate_daily_log_returns(self, training, context):
        # Requested horizon is known at the origin; no future observations
        # enter resolution selection. The lag bound is part of the fit key.
        maximum_lag = training.asset_log_returns.shape[0] + context.horizon_days - 1
        configured = replace(self, adaptive=f'dynamic:{maximum_lag}')
        return Candidate.simulate_daily_log_returns(configured, training, context)
