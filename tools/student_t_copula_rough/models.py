"""M248 asset forecasts with training-fitted Student-t copula dependence."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import stdtr

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from tail_fit import fit_degrees_of_freedom

spec=importlib.util.spec_from_file_location('student_copula_private_M248',ROOT/'tools/static_shrinkage_copula_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
ENABLED=True


@lru_cache(maxsize=64)
def configuration(shape,data):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    correlation,root,shrinkage=parent.configuration(shape,data)
    uniforms=parent.body.shell.controls.dg.pseudo_observations(assets)
    nu,diagnostics=fit_degrees_of_freedom(uniforms,correlation)
    return correlation,root,shrinkage,nu,diagnostics


@lru_cache(maxsize=2)
def student_uniforms(shape,data,sims,horizon,seed):
    _,root,_,nu,_=configuration(shape,data)
    if np.isinf(nu):return parent.gaussian_uniforms(shape,data,sims,horizon,seed)
    rng=np.random.default_rng(seed)
    # Separate stream leaves M248's Gaussian shocks and asset mapping intact.
    scale_rng=np.random.default_rng(np.random.SeedSequence([seed,0x54434f50]))
    result=np.empty((sims,horizon,shape[1]))
    for start in range(0,horizon,1024):
        stop=min(start+1024,horizon)
        z=rng.normal(size=(stop-start,sims,shape[1]))@root.T
        scale=scale_rng.gamma(nu/2,2/nu,size=(stop-start,sims,1))
        if np.any(scale<=0) or not np.isfinite(scale).all():
            raise ArithmeticError('unresolved t-copula mixing scale')
        z/=np.sqrt(scale)
        stdtr(nu,z,out=z)
        np.clip(z,1e-8,1.-1e-8,out=z)
        result[:,start:stop]=z.transpose(1,0,2)
    return result


def select_copula(enabled):
    global ENABLED
    ENABLED=enabled
    parent.select_copula(True)
    parent.body.shell.controls.gaussian_uniforms=student_uniforms if enabled else parent.gaussian_uniforms


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        _,_,_,nu,diagnostics=configuration(assets.shape,assets.tobytes())
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['student_t_copula_enabled']=ENABLED
            record['return_copula_degrees_of_freedom']=None if np.isinf(nu) else nu
            record['return_copula_gaussian_limit']=bool(np.isinf(nu))
            record['return_copula_tail_fit']=diagnostics
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_shrinkage_student_copula_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():
    parent.clear_path_cache();student_uniforms.cache_clear()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
