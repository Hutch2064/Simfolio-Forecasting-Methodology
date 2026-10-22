"""Canonical empirical CRPS, with only the Python outer pair loop compiled."""
import numpy as np
from numba import njit
from candidates.asset_selector_stacking.energy import cross_absolute_sorted


@njit(cache=False,fastmath=False)
def pair_matrix(sorted_x):
    k=sorted_x.shape[0];B=np.empty((k,k),dtype=np.float64)
    for i in range(k):
        for j in range(i+1):
            B[i,j]=B[j,i]=cross_absolute_sorted(sorted_x[i],sorted_x[j])
    return B


def coefficients(samples,observed):
    x=np.asarray(samples,dtype=float)
    if x.ndim!=2 or x.shape[1]<2 or not np.isfinite(x).all()or not np.isfinite(observed):
        raise ValueError('Finite equal-member predictive ensembles required')
    # NumPy sort, absolute values and row mean match the canonical code exactly.
    sorted_x=np.sort(x,axis=1);A=np.mean(np.abs(x-observed),axis=1)
    return A,pair_matrix(sorted_x)
