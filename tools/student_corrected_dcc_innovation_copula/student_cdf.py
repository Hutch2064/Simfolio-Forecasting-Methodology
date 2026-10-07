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


def resolution(nu,error=ERROR):
    limit=np.arctan(stdtrit(nu,1-1e-8)/np.sqrt(nu))
    m=nu-1.;constant=np.exp(-betaln(nu/2,.5))
    def maximum(k):
        theta=limit if m<=k else min(limit,np.arctan(np.sqrt(k/(m-k))))
        return np.exp(k*np.log(np.sin(theta))+(m-k)*np.log(np.cos(theta)))
    fourth=constant*(abs(3*m*m-2*m)*maximum(1)+abs(m*(m-1)*(m-2))*maximum(3))
    intervals=max(1,int(np.ceil(2*limit/(384*error/fourth)**.25)))
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
    if nu>1e5:
        stdtr(nu,x,out=x);return
    _,n,_=resolution(nu)
    if n>=(x.size if capacity is None else capacity):
        stdtr(nu,x,out=x)
    else:
        interpolate(x,nu,*grid(nu))


QUANTILE_CDF_ERROR=1e-14


@lru_cache(maxsize=16)
def inverse_grid(nu):
    limit,n,constant=resolution(nu,QUANTILE_CDF_ERROR)
    n+=n%2
    # Student symmetry needs only the lower half of the certified CDF grid.
    theta=np.linspace(-limit,0.,n//2+1)
    return limit,2*limit/n,stdtr(nu,np.sqrt(nu)*np.tan(theta)),constant*np.exp((nu-1)*np.log(np.cos(theta)))


@njit(cache=True)
def inverse_interpolate(u,nu,limit,h,values,derivatives):
    result=np.empty_like(u);root=np.sqrt(nu)
    for j in range(len(u)):
        p=min(u[j],1-u[j])
        if p==.5:result[j]=0.;continue
        i=np.searchsorted(values,p)-1
        if i<0 or i>=len(values)-1:result[j]=np.nan;continue
        delta=values[i+1]-values[i];m=h*derivatives[i];n=h*derivatives[i+1]
        c=3*delta-2*m-n;d=-2*delta+m+n;t=(p-values[i])/delta
        lo=0.;hi=1.;passed=False
        # Safeguarded Newton solves the Hermite polynomial. Failure to reach
        # floating-point residual precision uses the direct inverse below.
        for iteration in range(16):
            residual=values[i]+t*(m+t*(c+t*d))-p
            if abs(residual)<=np.finfo(np.float64).eps*max(p,1e-8)*4:
                passed=True;break
            if residual>0:hi=t
            else:lo=t
            slope=m+t*(2*c+3*t*d)
            trial=t-residual/slope if slope>0 else (lo+hi)/2
            t=trial if lo<trial<hi else (lo+hi)/2
        theta=-limit+(i+t)*h
        result[j]=(1 if u[j]<.5 else -1)*root*np.tan(theta) if passed else np.nan
    return result


def quantiles(u,nu):
    # Small grids of observations and the Gaussian-limit numerical branch
    # remain direct: constructing the interpolation grid would cost more.
    if nu>1e5:return stdtrit(nu,u)
    _,n,_=resolution(nu,QUANTILE_CDF_ERROR)
    if n>=u.size:return stdtrit(nu,u)
    values=inverse_interpolate(np.ascontiguousarray(u).reshape(-1),nu,*inverse_grid(nu))
    unresolved=~np.isfinite(values)
    if unresolved.any():values[unresolved]=stdtrit(nu,u.reshape(-1)[unresolved])
    return values.reshape(u.shape)
