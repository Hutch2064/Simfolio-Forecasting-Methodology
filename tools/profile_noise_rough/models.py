"""Estimate Gaussian quasi-likelihood measurement variance jointly with rough parameters.

Rough H/kappa/scale retain their original priors. Log measurement variance is
an unpenalized nuisance parameter: profile QML with rough MAP regularization,
not full Bayesian integration. Wide log-variance bounds are numerical guards.
"""
import hashlib
import importlib.util
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('profile_noise_private',ROOT/'tools/conditional_empirical_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
base,noise=parent.base,parent.noise
SOURCE=hashlib.sha256(parent.SOURCE.encode()+b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()
base.SOURCE=noise.SOURCE=SOURCE

@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    identity=hashlib.sha256(data).hexdigest()
    key=hashlib.sha256((SOURCE+identity+str(lag)).encode()).hexdigest()
    def build():
        y,_=base.observed(data);_,initial_variance=noise.measurement_scale(data)
        initial_log_variance=math.log(initial_variance)
        def target(theta):
            prior=base.overlay.log_prior(theta[:3])
            if not np.isfinite(prior):return -np.inf
            scale=math.sqrt(4.934802200544679/math.exp(theta[3]))
            phi,w,q=base.overlay.configuration(theta[:3],f'dynamic:{lag}')
            # Jacobian is essential because scale now changes during fitting.
            likelihood=base.overlay.filter_rough(y*scale,phi,w,q*scale**2,float(y.mean())*scale)[0]
            return prior+likelihood+len(y)*math.log(scale)
        result=base.fit_map(target,[.1,math.log(1/63),math.log(.7),initial_log_variance],
            [(.03,.49),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3)),
             (initial_log_variance-20,initial_log_variance+20)],seconds)
        result.update(data_sha256=identity,contract=f'profile_log_noise_QML:{lag}',
            measurement_noise_variance=math.exp(result['map'][3]),noise_estimator='joint_unpenalized_QML')
        return result
    started=time.perf_counter()
    result=base.shell.controls.cache('profile_noise_rough_fit',key,build)
    base.shell.controls.timed('rough_fit_worker',started)
    return result

@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    y,_=base.observed(data);theta=np.frombuffer(theta_bytes,np.float64)
    phi,w,q=base.overlay.configuration(theta[:3],f'dynamic:{lag}')
    scale=math.sqrt(4.934802200544679/math.exp(theta[3]))
    state,p=base.overlay.terminal_filter(y*scale,phi,w,q*scale**2,float(y.mean())*scale)
    state=state/scale;p=p/scale**2
    eigen,vectors=np.linalg.eigh((p+p.T)/2)
    initial_root=vectors*np.sqrt(np.maximum(eigen,0))
    root=np.diag(np.sqrt(np.diag(q)))
    stationary=q/(1-phi[:,None]*phi[None,:])
    return phi,w,root,state,initial_root,np.zeros(horizon),np.full(horizon,float(w@stationary@w))

base.rough_fit=rough_fit
base.prepared=prepared
initialize=parent.initialize
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_multiscale_profile_noise_conditional_residual_dynamic_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
