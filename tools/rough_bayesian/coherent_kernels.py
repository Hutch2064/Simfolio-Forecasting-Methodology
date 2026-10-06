"""Strict arithmetic kernels avoiding unused mixture draws and state allocations."""
import math

import numpy as np
from mixture_kernels import LOCATION, LOG_CONSTANT, VARIANCE
from numba import njit


@njit(cache=True,nogil=True)
def correction_only(log_squared,h,active):
    correction = 0.
    terms = np.empty(LOCATION.size)
    for t in range(h.size):
        if not active[t]:
            correction -= .5*h[t]
            continue
        residual,maximum = log_squared[t]-h[t],-np.inf
        for j in range(terms.size):
            terms[j] = LOG_CONSTANT[j]-.5*(residual-LOCATION[j])**2/VARIANCE[j]
            maximum = max(maximum,terms[j])
        total = 0.
        for j in range(terms.size):
            total += math.exp(terms[j]-maximum)
        exact = .5*(residual-math.exp(residual))-.5*math.log(2*math.pi) if residual < 700 else -np.inf
        correction += exact-(maximum+math.log(total))
    return correction


@njit(cache=True,nogil=True)
def marginalized_level(observations,phi,pws,inverse_f,log_f,prior_variance):
    state_y,state_one = np.zeros(phi.size),np.zeros(phi.size)
    yy,yo,oo,normalization = 0.,0.,0.,0.
    for t in range(observations.size):
        residual_y,residual_one = observations[t]-state_y.sum(),1-state_one.sum()
        yy += residual_y*residual_y*inverse_f[t]
        yo += residual_y*residual_one*inverse_f[t]
        oo += residual_one*residual_one*inverse_f[t]
        normalization += log_f[t]
        gain_y,gain_one = residual_y*inverse_f[t],residual_one*inverse_f[t]
        for j in range(phi.size):
            state_y[j] += pws[t,j]*gain_y
            state_one[j] += pws[t,j]*gain_one
            if t+1 < observations.size:
                state_y[j] *= phi[j]
                state_one[j] *= phi[j]
    precision = oo+1/prior_variance
    mean,sd = yo/precision,1/math.sqrt(precision)
    return -.5*(normalization+yy-yo*yo/precision+math.log(prior_variance*precision)),mean,sd


@njit(cache=True,nogil=True)
def cached_smooth_mean(observations,phi,pws,inverse_f):
    length,n = observations.size,phi.size
    state = np.zeros(n)
    predictions,innovation_over_f = np.empty(length),np.empty(length)
    for t in range(length):
        predictions[t] = state.sum()
        innovation_over_f[t] = (observations[t]-predictions[t])*inverse_f[t]
        for j in range(n):
            state[j] += pws[t,j]*innovation_over_f[t]
            if t+1 < length:
                state[j] *= phi[j]
    result,r,next_r = np.empty(length),np.zeros(n),np.empty(n)
    for t in range(length-1,-1,-1):
        for j in range(n):
            next_r[j] = phi[j]*r[j]
        c = innovation_over_f[t]-np.dot(pws[t],next_r)*inverse_f[t]
        result[t] = predictions[t]+np.dot(pws[t],next_r)+pws[t].sum()*c
        for j in range(n):
            r[j] = next_r[j]+c
    return result


@njit(cache=True,nogil=True)
def cached_simulation_smoother(observations,phi,q,noise_variance,pws,inverse_f,state_normals,measurement_normals):
    state = state_normals[0]*np.sqrt(q/(1-phi*phi))
    innovation_sd = np.sqrt(q)
    hplus,residual = np.empty(observations.size),np.empty(observations.size)
    for t in range(observations.size):
        hplus[t] = state.sum()
        residual[t] = (observations[t]-hplus[t]-math.sqrt(noise_variance[t])*measurement_normals[t]
                       if np.isfinite(noise_variance[t]) else 0.)
        if t+1 < observations.size:
            for j in range(phi.size):
                state[j] = phi[j]*state[j]+innovation_sd[j]*state_normals[t+1,j]
    return hplus+cached_smooth_mean(residual,phi,pws,inverse_f)


@njit(cache=True,nogil=True)
def conditional_terminal(h,phi,q,level):
    """Exact terminal state; avoid unused whitened history and T*k gain storage."""
    n = phi.size
    p,state = np.diag(q/(1-phi*phi)),np.zeros(n)
    fixed = False
    gain,pw = np.empty(n),np.empty(n)
    variance = 0.
    terminal = np.empty((n,n))
    for t in range(h.size):
        if not fixed:
            for i in range(n):
                pw[i] = 0.
                for j in range(n):
                    pw[i] += p[i,j]
            variance = pw.sum()
            scale = math.sqrt(variance)
            gain[:] = pw/variance
            if t == h.size-1:
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
        if fixed and t == h.size-1:
            terminal = p-np.outer(gain,gain)*scale**2
        error = h[t]-level-state.sum()
        for j in range(n):
            state[j] += gain[j]*error
            if t+1 < h.size:
                state[j] *= phi[j]
    return state,terminal
