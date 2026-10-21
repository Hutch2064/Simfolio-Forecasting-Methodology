"""Exact future Gaussian simulation of the fitted corrected-DCC recursion.

The fixed training target is Ledoit-Wolf shrinkage; we do not claim Aielli's
consistent target estimator. No conditional-mean substitution inside paths.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def advance(normals,qpaths,target,a,b):
    steps,paths,assets=normals.shape
    out=np.empty((paths,steps,assets))
    for p in range(paths):
        q=qpaths[p].copy()
        for t in range(steps):
            root=np.linalg.cholesky(q)
            v=root@normals[t,p]
            for j in range(assets):out[p,t,j]=v[j]/np.sqrt(q[j,j])
            q=(1.-a-b)*target+b*q+a*np.outer(v,v)
        qpaths[p]=q
    return out
