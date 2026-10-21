"""Streaming proper-score coefficients for a requested-horizon forecast path."""
import numpy as np
from numba import njit

def lead_masses(elapsed_calendar_days):
    days=np.asarray(elapsed_calendar_days,dtype=float)
    if days.ndim!=1 or len(days)==0 or not np.isfinite(days).all() or days[0]<=0 or np.any(np.diff(days)<=0):
        raise ValueError('Strictly increasing positive known validation calendar leads required')
    edges=np.log1p(days)
    return np.diff(np.r_[0.,edges])/edges[-1]/np.sqrt(np.arange(1,len(days)+1,dtype=float))

@njit(cache=True,fastmath=False)
def cross_absolute(x,y):
    total=0.;prefix=0.;sy=0.;j=0
    for value in y:sy+=value
    for value in x:
        while j<len(y) and y[j]<=value:prefix+=y[j];j+=1
        total+=(2*j-len(y))*value+sy-2*prefix
    return total/(len(x)*len(y))

@njit(cache=True,fastmath=False)
def _accumulate(paths,observations,masses,A,B):
    options,members,leads=paths.shape;ordered=np.empty((options,members))
    for t in range(leads):
        for i in range(options):
            value=0.
            for p in range(members):value+=abs(paths[i,p,t]-observations[t])
            A[i]+=masses[t]*(value/members)
            ordered[i]=np.sort(paths[i,:,t])
        for i in range(options):
            for j in range(i+1):
                contribution=masses[t]*cross_absolute(ordered[i],ordered[j])
                B[i,j]+=contribution
                if i!=j:B[j,i]+=contribution

def accumulate(paths,observations,masses,A,B):
    x=np.asarray(paths,dtype=float);y=np.asarray(observations,dtype=float);m=np.asarray(masses,dtype=float)
    if(x.ndim!=3 or x.shape[1]<2 or y.shape!=(x.shape[2],) or m.shape!=y.shape or A.shape!=(x.shape[0],) or B.shape!=(x.shape[0],x.shape[0])
       or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(m).all() or np.any(m<=0)):
        raise ValueError('Finite compatible forecast cube and positive prescribed lead masses required')
    # Caller retains A/B across chunks; every lead is accumulated in the same
    # chronological order, so changing chunk width cannot regroup the sums.
    _accumulate(x,y,m,A,B)
