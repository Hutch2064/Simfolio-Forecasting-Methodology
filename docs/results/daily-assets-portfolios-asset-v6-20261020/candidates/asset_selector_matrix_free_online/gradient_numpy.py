"""Vectorized weighted-CDF gradient; same empirical mixture, changed summation."""
import numpy as np
from numba import njit

@njit(cache=True,fastmath=False)
def _observation(x,y,m,A):
    for t in range(x.shape[2]):
        for i in range(x.shape[0]):
            total=0.
            for p in range(x.shape[1]):total+=abs(x[i,p,t]-y[t])
            A[i]+=m[t]*(total/x.shape[1])

def accumulate(paths,observations,masses,weights,A,cross):
    x=np.asarray(paths,dtype=float);y=np.asarray(observations,dtype=float);m=np.asarray(masses,dtype=float);w=np.asarray(weights,dtype=float)
    if(x.ndim!=3 or x.shape[1]<2 or y.shape!=(x.shape[2],) or m.shape!=y.shape or w.shape!=(x.shape[0],)
       or A.shape!=w.shape or cross.shape!=w.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
       or not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14)):
        raise ValueError('Finite aligned ensembles, positive masses and simplex mixture required')
    _observation(x,y,m,A)
    options,members,leads=x.shape
    support=np.flatnonzero(w>0);total=options*members;selected=len(support)*members
    # Choose by estimated comparison counts, never by an accuracy/model cap.
    subset_sweep=selected*np.log2(max(2,selected))+total*np.log2(max(2,selected))<total*np.log2(max(2,total))+total
    for t in range(leads):
        flat=x[:,:,t].reshape(-1)
        if subset_sweep:
            selected_values=x[support,:,t].reshape(-1);order=np.argsort(selected_values);values=selected_values[order]
            mass=np.repeat(w[support]/members,members)[order]
            prefix=np.r_[0.,np.cumsum(mass)];prefix_value=np.r_[0.,np.cumsum(mass*values)]
            rank=np.searchsorted(values,flat,side='right')
            distance=(2*prefix[rank]-w.sum())*flat+prefix_value[-1]-2*prefix_value[rank]
            # Every option is queried, including every zero-weight option.
            cross+=m[t]*(distance.reshape(options,members).sum(axis=1)/members)
            continue
        order=np.argsort(flat);values=flat[order];labels=order//members
        mass=w[labels]/members;prefix=np.cumsum(mass);prefix_value=np.cumsum(mass*values)
        # Every point tied at x adds zero to E|x-Y|, so a stable tie convention
        # is unnecessary mathematically; summation differs at floating precision.
        distance=(2*prefix-w.sum())*values+prefix_value[-1]-2*prefix_value
        cross+=m[t]*(np.bincount(labels,weights=distance,minlength=options)/members)
