"""M251 asset forecasts with jointly fitted Student-copula scatter and tails."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from joint_fit import fit_joint_copula

spec=importlib.util.spec_from_file_location('joint_private_M251',ROOT/'tools/innovation_student_copula_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
original_configuration=parent.aligned_configuration
ENABLED=True


@lru_cache(maxsize=64)
def configuration(shape,data):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    initial_c,_,_,initial_nu,_,digest=original_configuration(shape,data)
    pools=parent.aligned_pools(assets,parent.parent.parent.body.refit)
    uniforms=parent.parent.parent.body.shell.controls.dg.pseudo_observations(pools)
    c,root,nu,diagnostics=fit_joint_copula(uniforms,initial_c,initial_nu)
    return c,root,0.,nu,diagnostics,digest


def select_copula(enabled):
    global ENABLED
    if enabled!=ENABLED:
        parent.parent.student_uniforms.cache_clear();parent.parent.parent.gaussian_uniforms.cache_clear()
    ENABLED=enabled
    parent.aligned_configuration=configuration if enabled else original_configuration
    parent.select_copula(True)


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['joint_copula_fit_enabled']=ENABLED
            record['return_copula_scatter_estimator']='joint_rank_maximum_pseudo_likelihood' if ENABLED else 'M251_Gaussian_rank_Ledoit_Wolf'
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_innovation_joint_student_copula_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():parent.clear_path_cache()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
