"""Ledoit-Wolf sample covariance shrinkage, normalized to Gaussian correlation."""
import numpy as np


def shrunk_correlation(innovations):
    x=np.asarray(innovations,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2 or not np.isfinite(x).all():
        raise ValueError('finite aligned historical innovations required')
    n,p=x.shape;correlation=np.eye(p)
    centered=x-x.mean(axis=0);variance=np.mean(centered*centered,axis=0)
    active=np.flatnonzero(variance>0.)
    shrinkage=1.
    if len(active)>1:
        z=centered[:,active]/np.sqrt(variance[active]);q=len(active)
        sample=z.T@z/n;target=float(np.trace(sample)/q)
        delta=float(np.sum((sample-target*np.eye(q))**2)/q)
        beta=float((np.mean(np.sum(z*z,axis=1)**2)-np.sum(sample*sample))/(q*n))
        shrinkage=float(np.clip(beta/delta,0.,1.)) if delta>0. else 1.
        covariance=(1.-shrinkage)*sample+shrinkage*target*np.eye(q)
        scale=np.sqrt(np.diag(covariance));block=covariance/scale[:,None]/scale[None,:]
        correlation[np.ix_(active,active)]=block
    np.fill_diagonal(correlation,1.)
    eigenvalues,eigenvectors=np.linalg.eigh(correlation)
    tolerance=np.finfo(float).eps*max(1,p)*max(1.,float(np.max(eigenvalues)))
    if np.min(eigenvalues)<-tolerance:raise ArithmeticError('non-PSD covariance shrinkage')
    root=(eigenvectors*np.sqrt(np.maximum(eigenvalues,0.)))@eigenvectors.T
    if np.array_equal(correlation,np.eye(p)):root=np.eye(p)
    return correlation,root,shrinkage
