"""M243 asset law with a full, shrinkage-estimated static Gaussian copula."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from dependence import shrunk_correlation

spec=importlib.util.spec_from_file_location('static_copula_private_M243',ROOT/'tools/stationary_ar1_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
body=parent.parent;original_uniforms=body.shell.controls.gaussian_uniforms
STATIC_ENABLED=True


@lru_cache(maxsize=64)
def configuration(shape,data):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    scores=ndtri(body.shell.controls.dg.pseudo_observations(assets))
    return shrunk_correlation(scores)


@lru_cache(maxsize=2)
def gaussian_uniforms(shape,data,sims,horizon,seed):
    _,root,_=configuration(shape,data);rng=np.random.default_rng(seed)
    result=np.empty((sims,horizon,shape[1]))
    # Numerical memory blocking only; no observations, paths or lags omitted.
    for start in range(0,horizon,1024):
        stop=min(start+1024,horizon)
        z=rng.normal(size=(stop-start,sims,shape[1]))@root.T
        ndtr(z,out=z)
        np.clip(z,1e-8,1.-1e-8,out=z)
        result[:,start:stop]=z.transpose(1,0,2)
    return result


def select_copula(enabled):
    global STATIC_ENABLED
    STATIC_ENABLED=enabled
    body.shell.controls.gaussian_uniforms=gaussian_uniforms if enabled else original_uniforms


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        correlation,_,shrinkage=configuration(assets.shape,assets.tobytes())
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['static_shrinkage_copula_enabled']=STATIC_ENABLED
            record['return_copula_correlation']=correlation.tolist()
            record['return_copula_shrinkage']=shrinkage
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_static_shrinkage_copula_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():
    parent.clear_path_cache();gaussian_uniforms.cache_clear()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(STATIC_ENABLED)
