"""Randomized strength-one Latin hypercubes for unchanged copula primitives."""
import numpy as np
from scipy.special import gammaincinv, ndtr, ndtri, stdtr


def stratified_uniforms(rng, days, sims, dimensions):
    shape=(days,sims,dimensions)
    strata=np.broadcast_to(np.arange(sims)[None,:,None],shape)
    # Independently permute each day/dimension slice, rather than sharing a
    # permutation across assets or dates. Jitters are independent as well.
    u=rng.random(shape)
    u+=rng.permuted(strata,axis=1)
    u/=sims
    np.clip(u,np.nextafter(0.,1.),np.nextafter(1.,0.),out=u)
    return u


def copula_uniforms(root, nu, sims, horizon, seed):
    rng=np.random.default_rng(seed)
    scale_rng=np.random.default_rng(np.random.SeedSequence([seed,0x54434f50]))
    result=np.empty((sims,horizon,len(root)))
    # Memory blocking matches the IID implementation; no path/day is omitted.
    for start in range(0,horizon,1024):
        stop=min(start+1024,horizon)
        z=stratified_uniforms(rng,stop-start,sims,len(root))
        ndtri(z,out=z)
        z=z@root.T
        if np.isinf(nu):
            ndtr(z,out=z)
        else:
            scale=stratified_uniforms(scale_rng,stop-start,sims,1)
            gammaincinv(nu/2,scale,out=scale)
            scale/=nu/2
            if np.any(scale<=0) or not np.isfinite(scale).all():
                raise ArithmeticError('unresolved stratified t-copula mixing scale')
            z/=np.sqrt(scale)
            stdtr(nu,z,out=z)
        if not np.isfinite(z).all():
            raise ArithmeticError('unresolved stratified copula probabilities')
        np.clip(z,1e-8,1.-1e-8,out=z)
        result[:,start:stop]=z.transpose(1,0,2)
    return result
