"""Shared sorted sweep preserving each pair's original summation order."""
import numpy as np
from numba import njit

@njit(cache=True,fastmath=False)
def _accumulate(paths,observations,masses,A,B):
    options,members,leads=paths.shape
    for t in range(leads):
        # Each option's observation term keeps its original member order.
        for i in range(options):
            total=0.
            for p in range(members):total+=abs(paths[i,p,t]-observations[t])
            A[i]+=masses[t]*(total/members)
        flat=np.empty(options*members)
        for i in range(options):
            for p in range(members):flat[i*members+p]=paths[i,p,t]
        order=np.argsort(flat)
        sums=np.zeros(options)
        # Interleaving options preserves the sorted order within each option.
        for position in order:
            sums[position//members]+=flat[position]
        counts=np.zeros(options,dtype=np.int64);prefix=np.zeros(options);totals=np.zeros((options,options));start=0
        while start<len(order):
            stop=start+1;value=flat[order[start]]
            while stop<len(order)and flat[order[stop]]==value:stop+=1
            # Original per-pair reducer includes all y<=x before evaluating x.
            for k in range(start,stop):
                label=order[k]//members;counts[label]+=1;prefix[label]+=flat[order[k]]
            for k in range(start,stop):
                label=order[k]//members;x=flat[order[k]]
                for j in range(label+1):
                    totals[label,j]+=(2*counts[j]-members)*x+sums[j]-2*prefix[j]
            start=stop
        for i in range(options):
            for j in range(i+1):
                contribution=masses[t]*(totals[i,j]/(members*members))
                B[i,j]+=contribution
                if i!=j:B[j,i]+=contribution

def accumulate(paths,observations,masses,A,B):
    x=np.asarray(paths,dtype=float);y=np.asarray(observations,dtype=float);m=np.asarray(masses,dtype=float)
    if(x.ndim!=3 or x.shape[1]<2 or y.shape!=(x.shape[2],)or m.shape!=y.shape or A.shape!=(x.shape[0],)or B.shape!=(x.shape[0],x.shape[0])
       or not np.isfinite(x).all()or not np.isfinite(y).all()or not np.isfinite(m).all()or np.any(m<=0)):
        raise ValueError('Finite compatible forecast cube and positive prescribed lead masses required')
    _accumulate(x,y,m,A,B)
