"""Elliptical copula scatter from the sign U-statistic, including tied ranks.

Demarta and McNeil (2005), equations (9), (19), and Section 4.2.
Only numerical positive-definiteness repair is applied, not statistical shrinkage.
"""
import numpy as np
from scipy.stats import kendalltau


def positive_correlation(matrix):
    original=np.asarray(matrix,dtype=np.float64)
    eigenvalues,eigenvectors=np.linalg.eigh(original)
    floor=np.finfo(float).eps*len(original)*max(1.,float(eigenvalues.max()))
    repaired=bool(eigenvalues.min()<floor)
    if repaired:
        covariance=(eigenvectors*np.maximum(eigenvalues,floor))@eigenvectors.T
        scale=np.sqrt(np.diag(covariance))
        correlation=covariance/scale[:,None]/scale[None,:]
        correlation=(correlation+correlation.T)/2
        np.fill_diagonal(correlation,1.)
        eigenvalues,eigenvectors=np.linalg.eigh(correlation)
    else:correlation=original.copy()
    np.linalg.cholesky(correlation)
    root=(eigenvectors*np.sqrt(eigenvalues))@eigenvectors.T
    if np.array_equal(correlation,np.eye(len(correlation))):root=correlation.copy()
    return correlation,root,{'eigenvalue_repair_applied':repaired,
        'numerical_eigenvalue_floor':float(floor),
        'minimum_eigenvalue':float(eigenvalues.min()),
        'maximum_absolute_repair':float(np.abs(correlation-original).max())}


def kendall_correlation(innovations):
    x=np.asarray(innovations,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('finite aligned historical innovations required')
    n,p=x.shape;total=n*(n-1)//2
    untied=[]
    for column in x.T:
        counts=np.unique(column,return_counts=True)[1]
        untied.append(total-int(np.sum(counts*(counts-1)//2)))
    c=np.eye(p)
    for a in range(p):
        for b in range(a):
            if not untied[a] or not untied[b]:continue
            # SciPy's O(n log n) tau-b divides by untied-pair counts. Undo
            # that normalization to obtain the paper's sign U-statistic tau-a.
            tau_b=kendalltau(x[:,a],x[:,b],method='asymptotic').statistic
            tau_a=tau_b*np.sqrt(float(untied[a])*untied[b])/total
            c[a,b]=c[b,a]=np.sin(np.pi*tau_a/2)
    correlation,root,repair=positive_correlation(c)
    repair['estimator']='sin(pi/2 * empirical_sign_U_statistic_tau_a)'
    repair['ties']='zero contribution to pairwise sign U-statistic; no jitter'
    return correlation,root,repair
