"""M251 asset forecasts with a tuning-free empirical beta innovation copula."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from beta_copula import rank_intervals, sample_copula

spec=importlib.util.spec_from_file_location('empirical_beta_private_M251',ROOT/'tools/innovation_student_copula_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
body=parent.parent.parent.body
ENABLED=True


@lru_cache(maxsize=64)
def configuration(shape,data):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    pools=parent.aligned_pools(assets,body.refit)
    low,high=rank_intervals(pools)
    diagnostics={'method':'empirical_beta_copula','historical_rows':len(pools),'dimensions':shape[1],
        'smoothing_degree':len(pools),'bandwidth_parameter':None,'degrees_of_freedom_parameter':None,
        'tie_policy':'average_independent_uniform_integer_rank_assignments_in_each_tie_block',
        'tied_observations_by_asset':np.count_nonzero(low!=high,axis=0).tolist(),
        'rank_intervals_sha256':hashlib.sha256(low.tobytes()+high.tobytes()).hexdigest()}
    return low,high,hashlib.sha256(pools.tobytes()).hexdigest(),diagnostics


@lru_cache(maxsize=2)
def beta_uniforms(shape,data,sims,horizon,seed):
    low,high,_,_=configuration(shape,data)
    row_rng=np.random.default_rng(seed)
    beta_rng=np.random.default_rng(np.random.SeedSequence([seed,0x42455441]))
    tie_rng=np.random.default_rng(np.random.SeedSequence([seed,0x54494553]))
    result=np.empty((sims,horizon,shape[1]))
    for start in range(0,horizon,1024):
        stop=min(start+1024,horizon)
        u=sample_copula(low,high,(stop-start)*sims,row_rng,beta_rng,tie_rng)
        if not np.isfinite(u).all() or np.any((u<0)|(u>1)):
            raise ArithmeticError('invalid empirical beta copula output')
        np.clip(u,1e-8,1-1e-8,out=u)
        result[:,start:stop]=u.reshape(stop-start,sims,shape[1]).transpose(1,0,2)
    return result


def select_copula(enabled):
    global ENABLED
    if enabled!=ENABLED:clear_path_cache()
    ENABLED=enabled
    parent.select_copula(True)
    shell.controls.gaussian_uniforms=beta_uniforms if enabled else parent.parent.student_uniforms


class Candidate(parent.parent.parent.parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        _,_,digest,diagnostics=configuration(assets.shape,assets.tobytes()) if ENABLED else (None,None,None,None)
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['empirical_beta_copula_enabled']=ENABLED
            record['copula_training_input']='unchanged_chronological_Student_standardized_innovation_pool'
            record['copula_innovation_matrix_sha256']=digest
            record['return_copula_beta_fit']=diagnostics
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_innovation_empirical_beta_copula_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():
    parent.clear_path_cache();beta_uniforms.cache_clear()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
