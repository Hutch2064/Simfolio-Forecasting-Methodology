"""Compile proposal steps between unchanged Python covariance adaptations."""
import math

import numpy as np
from compiled_postburn import postburn
from steady_kalman_target import target


def chain(y,theta,burn,kept,seed):
    rng=np.random.default_rng(seed)
    low,high=np.quantile(y,[.01,.99]);bounds=((low-4,high+4),(-7,7),(math.log(.02),math.log(2.5)))
    lower=np.asarray([b[0] for b in bounds]);upper=np.asarray([b[1] for b in bounds]);ymean=float(np.mean(y))
    t=np.asarray(theta).copy()
    lp=-np.inf if any(t[i]<lo or t[i]>hi for i,(lo,hi) in enumerate(bounds)) else target(y,*t,ymean)
    covariance=np.diag([.01,.05,.03])**2;chol=np.linalg.cholesky(covariance)
    trace=np.empty((burn+kept,3));accept=np.zeros(burn+kept)
    start=0
    while start<burn:
        stop=min(96 if start<96 else start+32,burn)
        values,accepted=postburn(y,t,lp,chol,rng,stop-start,lower,upper,ymean)
        trace[start:stop]=values;accept[start:stop]=accepted;t=values[-1].copy();lp=target(y,*t,ymean)
        if 64<=stop-1<burn and stop%32==0:
            covariance=(2.38**2/3)*(np.cov(trace[:stop],rowvar=False)+np.eye(3)*1e-6)
            chol=np.linalg.cholesky(covariance)
        start=stop
    values,accepted=postburn(y,t,lp,chol,rng,kept,lower,upper,ymean)
    trace[burn:]=values;accept[burn:]=accepted
    return trace[burn:],float(accept[burn:].mean())
