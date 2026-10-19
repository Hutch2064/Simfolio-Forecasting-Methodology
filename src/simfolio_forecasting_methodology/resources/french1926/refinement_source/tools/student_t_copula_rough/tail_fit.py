"""Conditional rank pseudo-likelihood for t-copula degrees of freedom.

The scatter correlation is held fixed at M248's estimate. This is conditional
pseudo-likelihood, not joint maximum likelihood or posterior inference.
Demarta and McNeil (2005), doi:10.1111/j.1751-5823.2005.tb00254.x.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, ndtri, stdtrit


class CopulaLikelihood:
    def __init__(self, uniforms, correlation):
        self.u=np.asarray(uniforms,dtype=float)
        self.c=np.asarray(correlation,dtype=float)
        if self.u.ndim!=2 or self.c.shape!=(self.u.shape[1],)*2:
            raise ValueError('aligned copula observations and correlation required')
        if not np.isfinite(self.u).all() or np.any((self.u<=0)|(self.u>=1)):
            raise ValueError('copula observations must lie strictly inside (0,1)')
        np.linalg.cholesky(self.c)
        self.inverse=np.linalg.inv(self.c)
        self.logdet=float(np.linalg.slogdet(self.c)[1])
        self.evaluations=0

    def __call__(self, nu):
        self.evaluations+=1
        n,d=self.u.shape
        if np.isinf(nu):
            z=ndtri(self.u)
            q=np.sum((z@self.inverse)*z,axis=1)
            return float(-.5*n*self.logdet-.5*np.sum(q-np.sum(z*z,axis=1)))
        if not np.isfinite(nu) or nu<=0:return -np.inf
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            z=stdtrit(nu,self.u)
            q=np.sum((z@self.inverse)*z,axis=1)
            constant=gammaln((nu+d)/2)+(d-1)*gammaln(nu/2)-d*gammaln((nu+1)/2)-.5*self.logdet
            value=n*constant-(nu+d)/2*np.sum(np.log1p(q/nu))+(nu+1)/2*np.sum(np.log1p(z*z/nu))
        return float(value) if np.isfinite(value) else -np.inf


def fit_degrees_of_freedom(uniforms, correlation):
    likelihood=CopulaLikelihood(uniforms,correlation)
    gaussian=likelihood(np.inf)
    if likelihood.u.shape[1]==1:
        return np.inf,{'gaussian_log_likelihood':gaussian,'log_likelihood':gaussian,'evaluations':1}
    # x=nu/(1+nu) covers the entire positive parameter space. x=1 is the
    # explicit Gaussian endpoint; there is no prescribed minimum/maximum df.
    def objective(x):
        return -likelihood(x/(1-x)) if 0<x<1 else -gaussian if x==1 else np.inf
    result=minimize_scalar(objective,bounds=(0.,1.),method='bounded',options={'xatol':1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise ArithmeticError('t-copula conditional likelihood optimization failed')
    fitted=float(result.x/(1-result.x))
    score=-float(result.fun)
    if score<=gaussian:fitted=np.inf;score=gaussian
    return fitted,{'gaussian_log_likelihood':gaussian,'log_likelihood':score,
        'evaluations':likelihood.evaluations,'optimizer_coordinate':float(result.x)}
