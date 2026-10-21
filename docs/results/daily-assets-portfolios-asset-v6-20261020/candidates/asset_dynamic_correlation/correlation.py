"""Training-only corrected-DCC state and cheap conditional-mean Q projection."""
import numpy as np
from . import state,shrinkage

def normalized_correlation(q):
    q=np.asarray(q,dtype=float);scale=np.sqrt(np.diag(q))
    if np.any(scale<=0)or not np.isfinite(q).all():raise ArithmeticError('Invalid conditional covariance')
    r=q/scale[:,None]/scale[None,:]
    return (r+r.T)/2

def prepare(pool):
    z=np.asarray(pool,dtype=float)
    if z.ndim!=2 or len(z)<24 or not np.isfinite(z).all():raise ValueError('Finite training innovation matrix required')
    target,_,shrinkage_weight=shrinkage.shrunk_correlation(z)
    # Scalar correlation has no identifiable dynamic parameter and no dependence to change.
    if z.shape[1]==1:return {'target':target,'next_Q':target.copy(),'a':0.,'b':0.,'shrinkage':shrinkage_weight,'scalar_identity':True}
    np.linalg.cholesky(target)
    theta,q,diagnostics=state.fit(z,target)
    if min(theta)<0 or theta.sum()>=1 or not np.isfinite(q).all():raise ArithmeticError('Invalid fitted corrected-DCC state')
    np.linalg.cholesky(q)
    return {'target':target,'next_Q':q,'a':float(theta[0]),'b':float(theta[1]),'shrinkage':shrinkage_weight,'fit':diagnostics,'scalar_identity':False}

def mean_Q(fit,start,count):
    if start<0 or count<1:raise ValueError('Positive forecast lead range required')
    s,q=fit['target'],fit['next_Q'];rho=fit['a']+fit['b']
    weight=rho**np.arange(start,start+count,dtype=float)
    return s[None,:,:]+weight[:,None,None]*(q-s)[None,:,:]

def roots(fit,start,count,mode,scalar_root):
    if mode not in ('static_shrinkage','conditional_mean'):raise ValueError(mode)
    if len(fit['target'])==1:return np.repeat(np.asarray(scalar_root)[None,:,:],count,axis=0)
    covariances=mean_Q(fit,start,count)if mode=='conditional_mean'else np.repeat(fit['target'][None,:,:],count,axis=0)
    return np.stack([np.linalg.cholesky(normalized_correlation(q))for q in covariances])
