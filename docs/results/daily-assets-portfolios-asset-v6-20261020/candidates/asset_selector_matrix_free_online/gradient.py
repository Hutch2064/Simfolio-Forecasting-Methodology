"""Compute A and B@w by a weighted CDF sweep, never constructing B."""
import numpy as np
from numba import njit

@njit(cache=True,fastmath=False)
def _accumulate(paths,observations,masses,weights,A,cross):
    options,members,leads=paths.shape
    for t in range(leads):
        flat=np.empty(options*members)
        for i in range(options):
            total=0.
            for p in range(members):
                value=paths[i,p,t];flat[i*members+p]=value
                total+=abs(value-observations[t])
            A[i]+=masses[t]*(total/members)
        order=np.argsort(flat);sums=np.zeros(options)
        for position in order:sums[position//members]+=flat[position]
        total_mass=0.;weighted_total=0.
        for i in range(options):
            total_mass+=weights[i];weighted_total+=weights[i]*(sums[i]/members)
        prefix_mass=0.;prefix_value=0.;distances=np.zeros(options);start=0
        while start<len(order):
            stop=start+1;value=flat[order[start]]
            while stop<len(order) and flat[order[stop]]==value:stop+=1
            for k in range(start,stop):
                label=order[k]//members;mass=weights[label]/members
                prefix_mass+=mass;prefix_value+=mass*flat[order[k]]
            distance=(2*prefix_mass-total_mass)*value+weighted_total-2*prefix_value
            for k in range(start,stop):distances[order[k]//members]+=distance
            start=stop
        for i in range(options):cross[i]+=masses[t]*(distances[i]/members)

def accumulate(paths,observations,masses,weights,A,cross):
    x=np.asarray(paths,dtype=float);y=np.asarray(observations,dtype=float);m=np.asarray(masses,dtype=float);w=np.asarray(weights,dtype=float)
    if(x.ndim!=3 or x.shape[1]<2 or y.shape!=(x.shape[2],) or m.shape!=y.shape or w.shape!=(x.shape[0],)
       or A.shape!=w.shape or cross.shape!=w.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
       or not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14)):
        raise ValueError('Finite aligned ensembles, positive masses and simplex mixture required')
    _accumulate(x,y,m,w,A,cross)
