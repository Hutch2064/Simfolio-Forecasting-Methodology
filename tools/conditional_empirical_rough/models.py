"""Factorial test: causal residual rough + empirical noise + state forecasts.

Load private adapter namespaces so the already scored candidates and baseline
remain unchanged. Only volatility inference/normalization changes.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

base=load('conditional_empirical_private_base',ROOT/'tools/conditional_residual_rough/models.py')
noise=load('conditional_empirical_private_noise',ROOT/'tools/empirical_rough_noise/models.py')
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for directory in
    (HERE,ROOT/'tools/conditional_residual_rough',ROOT/'tools/empirical_rough_noise')
    for p in sorted(directory.glob('*.py')))).hexdigest()
base.SOURCE=noise.SOURCE=SOURCE
noise.observed=base.observed
noise.predecessor_fit=base.predecessor_fit
noise.shell=base.shell
base.rough_fit=noise.rough_fit

@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    y,_=base.observed(data);theta=np.frombuffer(theta_bytes,np.float64)
    phi,w,q=base.overlay.configuration(theta,f'dynamic:{lag}')
    scale,_=noise.measurement_scale(data)
    state,p=base.overlay.terminal_filter(y*scale,phi,w,q*scale**2,float(y.mean())*scale)
    state=state/scale;p=p/scale**2
    eigen,vectors=np.linalg.eigh((p+p.T)/2)
    initial_root=vectors*np.sqrt(np.maximum(eigen,0))
    root=np.diag(np.sqrt(np.diag(q)))
    stationary=q/(1-phi[:,None]*phi[None,:])
    return phi,w,root,state,initial_root,np.zeros(horizon),np.full(horizon,float(w@stationary@w))

base.prepared=prepared
initialize=base.initialize
clear_path_cache=base.clear_path_cache
TIMINGS=base.TIMINGS
FIT_DIAGNOSTICS=base.FIT_DIAGNOSTICS
shell=base.shell
overlay=base.overlay
Candidate=base.PredecessorRoughMAP
CANDIDATES=(Candidate(model_id='asset_map_multiscale_conditional_residual_empirical_noise_dynamic_rough_map'),)
MODEL_IDS=(base.MODEL_IDS[0],CANDIDATES[0].model_id)
