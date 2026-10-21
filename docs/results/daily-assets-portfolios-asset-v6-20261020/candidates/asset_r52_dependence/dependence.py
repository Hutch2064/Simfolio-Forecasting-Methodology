"""R52 winner dependence law with the current leader's innovation input."""
import numpy as np
from scipy.special import ndtri,ndtr
from . import ranks,shrinkage,student_state,student_cdf,path_native

def fit(pool):
    x=np.asarray(pool,dtype=np.float64)
    if x.ndim!=2 or len(x)<24 or not np.isfinite(x).all():raise ValueError('Finite chronological innovation history required')
    if x.shape[1]==1:return {'target':np.eye(1),'next_Q':np.eye(1),'a':0.,'b':0.,'nu':np.inf,'scalar_identity':True,'target_shrinkage':1.}
    u=ranks.pseudo_observations(x);target,_,shrink=shrinkage.shrunk_correlation(ndtri(u))
    theta,q,nu,diagnostics=student_state.fit(u,target)
    if not np.isfinite(q).all()or min(theta)<0 or theta.sum()>=1:raise ArithmeticError('Invalid fitted R52 dependence state')
    np.linalg.cholesky(q)
    return {'target':target,'next_Q':q,'a':float(theta[0]),'b':float(theta[1]),'nu':float(nu),'scalar_identity':False,'target_shrinkage':float(shrink),'fit':diagnostics}

def gaussianize(normals,states,fit,scale_rng):
    nu=fit['nu'];p=normals.shape[-1]
    if not np.isinf(nu):
        scales=scale_rng.gamma(nu/2,2.,size=normals.shape[:2]+(1,))
        if np.any(scales<=0)or not np.isfinite(scales).all():raise ArithmeticError('Unresolved Student copula mixing scale')
        normals*=np.sqrt((nu-2)/scales)
    path_native.load_paths().scores(normals,np.ascontiguousarray(fit['target']),states,fit['a'],fit['b'],1)
    if np.isinf(nu):ndtr(normals,out=normals)
    else:
        normals*=np.sqrt(nu/(nu-2));student_cdf.probabilities(normals,nu,1024*normals.shape[1]*p)
    # Inherited R52 probability guard, not a new fitted clipping choice.
    np.clip(normals,1e-8,1.-1e-8,out=normals)
    ndtri(normals,out=normals)
    return normals
