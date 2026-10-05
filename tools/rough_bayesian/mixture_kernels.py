"""Compiled exact-target mixture proposals and scalar-observation smoothing.

Omori et al. (2007) ten-normal table, reproduced in Hiraki/Chib/Omori
https://arxiv.org/html/2404.13986v2 Table 1. It defines only the proposal.
All kernels use strict floating point, without approximate covariance stopping.
"""
import math
import numpy as np
from numba import njit

PROBABILITY = np.array([.00609,.04775,.13057,.20674,.22715,.18842,.12047,.05591,.01575,.00115])
LOCATION = np.array([1.92677,1.34744,.73504,.02266,-.85173,-1.97278,-3.46788,-5.55246,-8.68384,-14.65])
VARIANCE = np.array([.11265,.17788,.26768,.40611,.62699,.98583,1.57469,2.54498,4.16591,7.33342])
LOG_CONSTANT = np.log(PROBABILITY) - .5*np.log(2*math.pi*VARIANCE)


@njit(cache=True, nogil=True)
def exact_return_loglik(squared, h):
    total = 0.
    for t in range(h.size):
        # Raw Gaussian return likelihood remains defined for exactly zero returns.
        if not np.isfinite(h[t]) or h[t] < -700:
            return -np.inf
        total -= .5*(h[t] + squared[t]*math.exp(-h[t]) + math.log(2*math.pi))
    return total


@njit(cache=True, nogil=True)
def mixture_terms(log_squared, h, active, uniforms):
    """Conditional indicators and exact/proposal density ratio in one pass."""
    offset, variance = np.empty(h.size), np.empty(h.size)
    correction = 0.
    terms = np.empty(PROBABILITY.size)
    for t in range(h.size):
        if not active[t]:
            offset[t], variance[t] = 0., np.inf
            correction -= .5*h[t]
            continue
        residual = log_squared[t]-h[t]
        maximum = -np.inf
        for j in range(terms.size):
            terms[j] = LOG_CONSTANT[j]-.5*(residual-LOCATION[j])**2/VARIANCE[j]
            maximum = max(maximum, terms[j])
        total = 0.
        for j in range(terms.size):
            terms[j] = math.exp(terms[j]-maximum)
            total += terms[j]
        cutoff, cumulative, chosen = uniforms[t]*total, 0., terms.size-1
        for j in range(terms.size):
            cumulative += terms[j]
            if cutoff < cumulative:
                chosen = j
                break
        offset[t], variance[t] = LOCATION[chosen], VARIANCE[chosen]
        exact = .5*(residual-math.exp(residual))- .5*math.log(2*math.pi) if residual < 700 else -np.inf
        correction += exact - (maximum+math.log(total))
    return offset, variance, correction


@njit(cache=True, nogil=True)
def gaussian_geometry(phi, q, length):
    """Zero-measurement-noise innovations representation of scalar log variance.

    Store T*k gains rather than T*k*k matrices. Reuse steady covariance only
    when its complete floating-point array stops changing exactly.
    """
    n = phi.size
    p = np.diag(q/(1-phi*phi))
    gains, sd = np.empty((length,n)), np.empty(length)
    fixed = False
    gain, pw = np.empty(n), np.empty(n)
    scale = 0.
    terminal = np.empty((n,n))
    for t in range(length):
        if not fixed:
            for i in range(n):
                pw[i] = 0.
                for j in range(n):
                    pw[i] += p[i,j]
            variance = pw.sum()
            scale = math.sqrt(variance)
            gain[:] = pw/variance
            if t == length-1:
                terminal = p-np.outer(gain,gain)*scale**2
            new_p = np.empty((n,n))
            fixed = True
            for i in range(n):
                for j in range(n):
                    value = (p[i,j]-pw[i]*pw[j]/variance)*phi[i]*phi[j]
                    if i == j:
                        value += q[i]
                    new_p[i,j] = value
                    if value != p[i,j]:
                        fixed = False
            p = new_p
        if fixed and t == length-1:
            terminal = p-np.outer(gain,gain)*scale**2
        gains[t] = gain
        sd[t] = scale
    return gains, sd, terminal


@njit(cache=True, nogil=True)
def whiten(h, phi, gains, sd, level):
    state, white = np.zeros(phi.size), np.empty(h.size)
    for t in range(h.size):
        error = h[t]-level-state.sum()
        white[t] = error/sd[t]
        state += gains[t]*error
        if t+1 < h.size:
            state *= phi
    return white, state


@njit(cache=True, nogil=True)
def unwhiten(white, phi, gains, sd, level):
    state, h = np.zeros(phi.size), np.empty(white.size)
    for t in range(white.size):
        error = sd[t]*white[t]
        h[t] = level+state.sum()+error
        state += gains[t]*error
        if t+1 < white.size:
            state *= phi
    return h


@njit(cache=True, nogil=True)
def smooth_mean(observations, phi, q, noise_variance):
    """Scalar Durbin-Koopman information smoother, O(T*k*k) work/O(T*k) storage."""
    length, n = observations.size, phi.size
    p = np.diag(q/(1-phi*phi))
    state = np.zeros(n)
    pws = np.empty((length,n))
    innovation_over_f, inverse_f, predictions = np.empty(length), np.empty(length), np.empty(length)
    for t in range(length):
        predictions[t] = state.sum()
        for i in range(n):
            pws[t,i] = 0.
            for j in range(n):
                pws[t,i] += p[i,j]
        variance = pws[t].sum()+noise_variance[t]
        inverse_f[t] = 1/variance
        innovation_over_f[t] = (observations[t]-predictions[t])*inverse_f[t]
        state += pws[t]*innovation_over_f[t]
        if t+1 < length:
            state *= phi
            for i in range(n):
                for j in range(n):
                    p[i,j] = (p[i,j]-pws[t,i]*pws[t,j]*inverse_f[t])*phi[i]*phi[j]
                    if i == j:
                        p[i,j] += q[i]
    result, r = np.empty(length), np.zeros(n)
    for t in range(length-1,-1,-1):
        next_r = phi*r
        c = innovation_over_f[t]-np.dot(pws[t],next_r)*inverse_f[t]
        result[t] = predictions[t]+np.dot(pws[t],next_r)+pws[t].sum()*c
        r = next_r+c
    return result


@njit(cache=True, nogil=True)
def simulation_smoother(observations, phi, q, noise_variance, state_normals, measurement_normals):
    state = state_normals[0]*np.sqrt(q/(1-phi*phi))
    innovation_sd = np.sqrt(q)
    hplus, residual = np.empty(observations.size), np.empty(observations.size)
    for t in range(observations.size):
        hplus[t] = state.sum()
        residual[t] = (observations[t]-hplus[t]-math.sqrt(noise_variance[t])*measurement_normals[t]
                       if np.isfinite(noise_variance[t]) else 0.)
        if t+1 < observations.size:
            state = phi*state+innovation_sd*state_normals[t+1]
    return hplus+smooth_mean(residual,phi,q,noise_variance)


@njit(cache=True, nogil=True)
def measurement_geometry(phi,q,noise_variance):
    """Reusable heteroskedastic Kalman geometry for marginalization and smoothing."""
    length,n = noise_variance.size,phi.size
    p = np.diag(q/(1-phi*phi))
    pws,inverse_f,log_f = np.empty((length,n)),np.empty(length),np.empty(length)
    for t in range(length):
        for i in range(n):
            pws[t,i] = 0.
            for j in range(n):
                pws[t,i] += p[i,j]
        variance = pws[t].sum()+noise_variance[t]
        inverse_f[t] = 1/variance
        log_f[t] = math.log(2*math.pi*variance) if np.isfinite(variance) else 0.
        if t+1 < length:
            for i in range(n):
                for j in range(n):
                    p[i,j] = (p[i,j]-pws[t,i]*pws[t,j]*inverse_f[t])*phi[i]*phi[j]
                p[i,i] += q[i]
    return pws,inverse_f,log_f


@njit(cache=True, nogil=True)
def marginal_likelihood(observations,phi,pws,inverse_f,log_f):
    state,total = np.zeros(phi.size),0.
    for t in range(observations.size):
        residual = observations[t]-state.sum()
        total -= .5*(log_f[t]+residual*residual*inverse_f[t])
        state += pws[t]*(residual*inverse_f[t])
        if t+1 < observations.size:
            state *= phi
    return total


@njit(cache=True, nogil=True)
def cached_smooth_mean(observations,phi,pws,inverse_f):
    length = observations.size
    state = np.zeros(phi.size)
    predictions,innovation_over_f = np.empty(length),np.empty(length)
    for t in range(length):
        predictions[t] = state.sum()
        innovation_over_f[t] = (observations[t]-predictions[t])*inverse_f[t]
        state += pws[t]*innovation_over_f[t]
        if t+1 < length:
            state *= phi
    result,r = np.empty(length),np.zeros(phi.size)
    for t in range(length-1,-1,-1):
        next_r = phi*r
        c = innovation_over_f[t]-np.dot(pws[t],next_r)*inverse_f[t]
        result[t] = predictions[t]+np.dot(pws[t],next_r)+pws[t].sum()*c
        r = next_r+c
    return result


@njit(cache=True, nogil=True)
def cached_simulation_smoother(observations,phi,q,noise_variance,pws,inverse_f,state_normals,measurement_normals):
    state = state_normals[0]*np.sqrt(q/(1-phi*phi))
    innovation_sd = np.sqrt(q)
    hplus,residual = np.empty(observations.size),np.empty(observations.size)
    for t in range(observations.size):
        hplus[t] = state.sum()
        residual[t] = (observations[t]-hplus[t]-math.sqrt(noise_variance[t])*measurement_normals[t]
                       if np.isfinite(noise_variance[t]) else 0.)
        if t+1 < observations.size:
            state = phi*state+innovation_sd*state_normals[t+1]
    return hplus+cached_smooth_mean(residual,phi,pws,inverse_f)


@njit(cache=True, nogil=True)
def marginalized_level(observations,phi,pws,inverse_f,log_f,prior_variance):
    """Integrate a Gaussian level prior; return its conditional mean and SD.

    Whiten both the observations and the constant regressor with one Kalman
    geometry. Truncation probability is applied by the caller in log space.
    """
    state_y,state_one = np.zeros(phi.size),np.zeros(phi.size)
    yy,yo,oo,normalization = 0.,0.,0.,0.
    for t in range(observations.size):
        residual_y = observations[t]-state_y.sum()
        residual_one = 1-state_one.sum()
        yy += residual_y*residual_y*inverse_f[t]
        yo += residual_y*residual_one*inverse_f[t]
        oo += residual_one*residual_one*inverse_f[t]
        normalization += log_f[t]
        state_y += pws[t]*(residual_y*inverse_f[t])
        state_one += pws[t]*(residual_one*inverse_f[t])
        if t+1 < observations.size:
            state_y *= phi
            state_one *= phi
    precision = oo+1/prior_variance
    mean,sd = yo/precision,1/math.sqrt(precision)
    return -.5*(normalization+yy-yo*yo/precision+math.log(prior_variance*precision)),mean,sd

