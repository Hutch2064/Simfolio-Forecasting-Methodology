"""Adaptive Metropolis for Frontier's declared Gaussian log-square SV posterior.
Adaptation is restricted to burn-in. Parameter uncertainty is propagated by
an sixteen-node posterior mixture of existing filtered multiscale moment laws.
This is SV QML posterior inference, not a raw-return likelihood or full INLA.
"""
import hashlib
import math
from functools import lru_cache

import numpy as np
from complex_sv import ComplexSV
from scipy.special import expit

from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
    moment_return_curves,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

RECORDS=[]
def chain(y,theta,burn,kept,seed):
    rng=np.random.default_rng(seed)
    lower,upper=np.quantile(y,[.01,.99]);bounds=((lower-4,upper+4),(-7,7),(math.log(.02),math.log(2.5)))
    def logp(t):
        if any(t[i]<lo or t[i]>hi for i,(lo,hi) in enumerate(bounds)):return -np.inf
        return bd._sv_transformed_log_posterior(y,*t)
    t=np.asarray(theta).copy();lp=logp(t)
    covariance=np.diag([.01,.05,.03])**2
    chol=np.linalg.cholesky(covariance)
    trace=np.empty((burn+kept,3));accept=np.zeros(burn+kept)
    for i in range(burn+kept):
        proposal=t+chol@rng.normal(size=3);value=logp(proposal)
        if math.log(rng.random())<value-lp:t=proposal;lp=value;accept[i]=1
        trace[i]=t
        if 64<=i<burn and (i+1)%32==0:
            covariance=(2.38**2/3)*(np.cov(trace[:i+1],rowvar=False)+np.eye(3)*1e-6)
            chol=np.linalg.cholesky(covariance)
    return trace[burn:],float(accept[burn:].mean())

@lru_cache(maxsize=4096)
def parameter_fit(data):
    x=np.frombuffer(data,np.float64)
    fit=bd.fit_bdes_fastmap(x,filtered_innovations=True,fixed_mean=True)
    y,eps=bd._sv_observed_log_variance(x,x.mean())
    center=fit['posterior_center'];theta=(center[0],bd._logit(center[1]),math.log(center[2]))
    seed=int.from_bytes(hashlib.sha256(data).digest()[:4],'little')
    first,accept=chain(y,theta,1024,2048,seed)
    second,accept_second=chain(y,theta,1024,2048,(seed+99173)%2**32)
    draws=np.vstack((first,second))
    RECORDS.append({'data_sha256': hashlib.sha256(data).hexdigest(),'n': x.size,'postburn_acceptance': [accept,accept_second],'burn': 1024,'kept': 2048,'chains': 2})
    children=[]
    for t in draws[np.linspace(0,len(draws)-1,16,dtype=int)]:
        _,path,_var=bd._sv_kalman_rts_smoother_mean(y,t[0],float(expit(t[1])),math.exp(t[2]))
        child=dict(fit);child['bdes_multiscale_vol']=bd._bdes_multiscale_components(path,4,bd.BDES_MULTISCALE_GRID_FIXED)
        child['innovation_pool']=bd._standardized_empirical_innovation_pool(eps*np.exp(-.5*path),clip=None,method='mean_std')
        children.append(child)
    fit=dict(fit);fit['parameter_posterior_children']=children
    return fit

# Use the existing asset-level adapter; replace its two narrow module callbacks
# only inside this serial research process. No canonical package edit.
def install():
    import complex_sv
    original_sample=complex_sv.sampled_fit;original_marginal=complex_sv.sorted_moment_marginals
    def sampled(data,method,*args):
        if method=='parameter_mcmc':return parameter_fit(data)
        return original_sample(data,method,*args)
    def marginals(fit,sims,horizon):
        if 'parameter_posterior_children' not in fit:return original_marginal(fit,sims,horizon)
        children=fit['parameter_posterior_children'];curves=[moment_return_curves(c,horizon) for c in children]
        mean=np.mean([c[0] for c in curves],axis=0)
        variance=np.mean([c[1]**2+c[0]**2 for c in curves],axis=0)-mean**2
        pool=np.concatenate([c['innovation_pool'] for c in children])
        nodes=np.quantile(pool,np.linspace(.5/sims,1-.5/sims,sims));nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
        return np.clip(mean[None,:]+np.sqrt(np.maximum(variance,0))[None,:]*nodes[:,None],-1,1)
    complex_sv.sampled_fit=sampled;complex_sv.sorted_moment_marginals=marginals

CANDIDATES=(ComplexSV('sv_parameter_mcmc_twochain_sixteen_node_moment_mixture',method='parameter_mcmc',particles=0,burn=1024,kept=2048),ComplexSV('sv_parameter_mcmc_twochain_half_blend',method='parameter_mcmc',particles=0,burn=1024,kept=2048,blend=.5))
