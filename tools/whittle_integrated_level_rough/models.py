"""M200 covariance inference with a flat-prior integrated observation level.

Only the rough Gaussian terminal-state posterior changes. Covariance parameters
retain debiased Whittle MAP points; the nuisance intercept is analytically
integrated, including its uncertainty. The return-mean model is unchanged.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools/reml_rough'))
from gaussian_level_filter import restricted_filter
spec=importlib.util.spec_from_file_location('integrated_level_private_streamed',ROOT/'tools/streamed_rough_paths/models.py')
streamed=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=streamed;spec.loader.exec_module(streamed)
# Preserve the best undifferenced M200 covariance target, rather than M201.
streamed.rough_fit=streamed.parent.parent.rough_fit


@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    y,_=streamed.base.observed(data)
    _,noise_variance=streamed.parent.parent.base.noise.measurement_scale(data)
    phi,w,q=streamed.overlay.configuration(np.frombuffer(theta_bytes,np.float64),f'dynamic:{lag}')
    _,level,level_variance,state,p=restricted_filter(y,phi,w,q,noise_variance)
    eigen,vectors=np.linalg.eigh((p+p.T)/2)
    initial_root=vectors*np.sqrt(np.maximum(eigen,0))
    root=np.diag(np.sqrt(np.diag(q)))
    stationary=q/(1-phi[:,None]*phi[None,:])
    return phi,w,root,state,initial_root,np.zeros(horizon),np.full(horizon,float(w@stationary@w))


streamed.prepared=prepared


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    streamed.initialize(cache_root,vine_threads,state_workers)
    restricted_filter(np.zeros(4),np.array([.9]),np.ones(1),np.array([[.1]]),4.)


Candidate=streamed.Candidate
CANDIDATES=(Candidate(model_id='asset_map_multiscale_conditional_empirical_whittle_integrated_level_rough_map'),)
MODEL_IDS=(streamed.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=streamed.clear_path_cache
TIMINGS=streamed.TIMINGS
FIT_DIAGNOSTICS=streamed.FIT_DIAGNOSTICS
shell=streamed.shell
overlay=streamed.overlay
load_paths=streamed.load_paths
