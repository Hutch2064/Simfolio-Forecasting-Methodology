"""M236 with rough Hurst support widened to the theoretical open (0, 1/2)."""
import hashlib
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
parent=load('full_H_private_predictive_loading',ROOT/'tools/predictive_loading_rates_rough/models.py')
map_impl=load('full_H_private_map',HERE/'rough_map.py')
rough=parent.parent.parent

def log_prior(theta):
    if not (0.<theta[0]<.5 and np.isfinite(theta).all()):return -np.inf
    return -.5*((theta[1]-math.log(1/63))/2)**2-.5*((theta[2]-math.log(.7))/1.5)**2

rough.log_prior=log_prior;rough.fit_map=map_impl.fit_map
digest=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()
rough.SOURCE=hashlib.sha256((rough.SOURCE+digest).encode()).hexdigest()
parent.parent.SOURCE=hashlib.sha256((parent.parent.SOURCE+digest).encode()).hexdigest()
original_rough_fit=rough.rough_fit

@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    result=dict(original_rough_fit(data,lag,seconds))
    result['H_support']='theoretical open (0, 1/2); unchanged relative optimizer interior guard 1e-10'
    return result

rough.rough_fit=rough_fit;rough.parent.rough_fit=rough_fit;rough.backend.streamed.rough_fit=rough_fit
initialize=parent.initialize;Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_adaptive_predictive_loading_full_hurst_untruncated_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
