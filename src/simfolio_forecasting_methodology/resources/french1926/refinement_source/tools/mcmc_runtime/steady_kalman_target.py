"""Exact scalar SV likelihood: reuse a variance only after float equality."""
import math

import numpy as np
from numba import njit


@njit(cache=True,fastmath=False)
def likelihood(y,level,phi,eta):
    if y.size==0 or not (np.isfinite(level) and np.isfinite(phi) and np.isfinite(eta)):
        return -math.inf
    if phi<0 or phi>=.999 or eta<=1e-6:return -math.inf
    q=float(eta*eta);mean=float(level)
    variance=float(max(q/max(1.-phi*phi,1e-4),1e-6));loglik=0.
    index=0
    while index<y.size:
        forecast_var=float(max(variance+4.934802200544679,1e-8))
        logterm=math.log(2.*math.pi*forecast_var)
        innovation=float(y[index]-mean)
        loglik+=-.5*(logterm+innovation*innovation/forecast_var)
        gain=variance/forecast_var
        updated_mean=mean+gain*innovation
        updated_var=float(max((1.-gain)*variance,1e-8))
        mean=float(level+phi*(updated_mean-level))
        next_variance=float(phi*phi*updated_var+q)
        index+=1
        if next_variance==variance:
            # The deterministic variance recurrence has reached an exact float
            # fixed point. Its log term and gain are bit-identical thereafter.
            while index<y.size:
                innovation=float(y[index]-mean)
                loglik+=-.5*(logterm+innovation*innovation/forecast_var)
                updated_mean=mean+gain*innovation
                mean=float(level+phi*(updated_mean-level))
                index+=1
            break
        variance=next_variance
    return float(loglik)

@njit(cache=True,fastmath=False)
def target(y,level,phi_logit,log_eta,ymean):
    if not (np.isfinite(level) and np.isfinite(phi_logit) and np.isfinite(log_eta)):return -math.inf
    if phi_logit>=0:
        z=math.exp(-float(phi_logit));phi=float(1./(1.+z))
    else:
        z=math.exp(float(phi_logit));phi=float(z/(1.+z))
    eta=math.exp(log_eta)
    value=likelihood(y,level,phi,eta)
    if not np.isfinite(value):return -math.inf
    prior_level=-.5*((level-ymean)/4.)**2
    prior_phi=-.5*((phi-.94)/.20)**2
    prior_eta=-.5*((math.log(eta)-math.log(.35))/1.)**2
    return float(value+prior_level+prior_phi+prior_eta+math.log(max(phi*(1.-phi),1e-300))+log_eta)

def make_chain():
    import inspect

    import parameter_mcmc_tuned as pm
    namespace=dict(pm.__dict__);namespace['fast_target']=target
    source=inspect.getsource(pm.chain).replace('rng=np.random.default_rng(seed)','rng=np.random.default_rng(seed)\n    ymean=float(np.mean(y))').replace('bd._sv_transformed_log_posterior(y,*t)','fast_target(y,*t,ymean)')
    exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    return namespace['chain']
