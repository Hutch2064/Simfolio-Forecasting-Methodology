"""Error-bounded Hermite evaluation of the same Student CDF.

The interpolation is in theta=atan(x/sqrt(nu)). Its fourth derivative has
an analytic global bound on the existing, unclipped probability domain.
The expensive direct CDF remains the reference and the small-array fallback.
"""
from functools import lru_cache

import numpy as np
from numba import njit
from scipy.special import betaln, stdtr, stdtrit

ERROR=1e-12


def resolution(nu):
    limit=np.arctan(stdtrit(nu,1-1e-8)/np.sqrt(nu))
    m=nu-1.;constant=np.exp(-betaln(nu/2,.5))
    def maximum(k):
        theta=limit if m<=k else min(limit,np.arctan(np.sqrt(k/(m-k))))
        return np.exp(k*np.log(np.sin(theta))+(m-k)*np.log(np.cos(theta)))
    fourth=constant*(abs(3*m*m-2*m)*maximum(1)+abs(m*(m-1)*(m-2))*maximum(3))
    intervals=max(1,int(np.ceil(2*limit/(384*ERROR/fourth)**.25)))
    return limit,intervals,constant


@lru_cache(maxsize=16)
def grid(nu):
    limit,n,constant=resolution(nu);theta=np.linspace(-limit,limit,n+1)
    values=stdtr(nu,np.sqrt(nu)*np.tan(theta))
    derivatives=constant*np.exp((nu-1)*np.log(np.cos(theta)))
    return limit,2*limit/n,values,derivatives


@njit(cache=True)
def interpolate(x,nu,limit,h,values,derivatives):
    flat=x.reshape(-1);root=np.sqrt(nu);n=len(values)-1
    for j in range(len(flat)):
        theta=np.arctan(flat[j]/root)
        if theta<=-limit:flat[j]=1e-8;continue
        if theta>=limit:flat[j]=1-1e-8;continue
        location=(theta+limit)/h;i=min(int(location),n-1);t=location-i
        t2=t*t;t3=t2*t
        flat[j]=(2*t3-3*t2+1)*values[i]+(t3-2*t2+t)*h*derivatives[i]+(-2*t3+3*t2)*values[i+1]+(t3-t2)*h*derivatives[i+1]


def probabilities(x,nu,capacity=None):
    _,n,_=resolution(nu)
    if n>=(x.size if capacity is None else capacity):
        stdtr(nu,x,out=x)
    else:
        interpolate(x,nu,*grid(nu))
