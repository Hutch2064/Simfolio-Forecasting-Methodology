"""Original summation order, but compute only requested pairwise columns."""
import numpy as np
from numba import njit
from candidates.asset_selector_path_objective.coefficients import cross_absolute

@njit(cache=True,fastmath=False)
def _terms(paths,observations,masses,selected,A,diagonal,columns,include_terms):
    options,members,leads=paths.shape
    ordered=np.empty((options,members))
    for t in range(leads):
        for i in range(options):
            ordered[i]=np.sort(paths[i,:,t])
            if include_terms:
                value=0.
                for p in range(members):value+=abs(paths[i,p,t]-observations[t])
                A[i]+=masses[t]*(value/members)
                diagonal[i]+=masses[t]*cross_absolute(ordered[i],ordered[i])
        for i in range(options):
            for c in range(len(selected)):
                j=selected[c]
                # The dense reducer evaluates the higher index first.
                if i>=j:value=cross_absolute(ordered[i],ordered[j])
                else:value=cross_absolute(ordered[j],ordered[i])
                columns[i,c]+=masses[t]*value

def accumulate(paths,observations,masses,selected,A,diagonal,columns,*,include_terms=True):
    x=np.asarray(paths,dtype=float);y=np.asarray(observations,dtype=float);m=np.asarray(masses,dtype=float);selected=np.asarray(selected,dtype=np.int64)
    if(x.ndim!=3 or x.shape[1]<2 or y.shape!=(x.shape[2],) or m.shape!=y.shape
       or A.shape!=(x.shape[0],) or diagonal.shape!=A.shape or columns.shape!=(len(A),len(selected))
       or selected.ndim!=1 or len(set(selected.tolist()))!=len(selected) or np.any(selected<0) or np.any(selected>=len(A))
       or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(m).all() or np.any(m<=0)):
        raise ValueError('Finite compatible ensembles, positive masses and unique valid columns required')
    _terms(x,y,m,selected,A,diagonal,columns,include_terms)
