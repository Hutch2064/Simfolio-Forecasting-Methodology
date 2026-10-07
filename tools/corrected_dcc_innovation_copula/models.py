"""Unchanged M251 asset law with conditional Gaussian corrected-DCC dependence."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import cdcc_native as native
import cdcc_state as state

spec=importlib.util.spec_from_file_location('corrected_dcc_private_M251',ROOT/'tools/innovation_student_copula_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
body=parent.parent.parent.body;ENABLED=True
SOURCE=hashlib.sha256((body.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(HERE.glob('*')) if p.suffix in ('.py','.cpp'))).encode()).hexdigest()


@lru_cache(maxsize=64)
def configuration(shape,data):
    def build():
        assets=np.frombuffer(data,np.float64).reshape(shape)
        pools=parent.aligned_pools(assets,body.refit)
        u=body.shell.controls.dg.pseudo_observations(pools);z=np.ascontiguousarray(ndtri(u))
        target,_,shrinkage=parent.parent.parent.shrunk_correlation(z)
        theta,q,diagnostics=state.fit(z,target)
        if not np.isfinite(q).all() or np.linalg.eigvalsh(q).min()<=0.:
            raise ArithmeticError('invalid fitted cDCC terminal covariance')
        diagnostics.update({'method':'conditional_Gaussian_copula_corrected_DCC_QMLE',
            'target_estimator':'fixed_full_training_Gaussian_rank_Ledoit_Wolf',
            'consistent_cDCC_target_estimator_claim':False,'target_shrinkage':float(shrinkage),
            'a':float(theta[0]),'b':float(theta[1]),'target':target.tolist(),
            'terminal_Q':q.tolist(),'training_rows':len(z),'conditional_marginal_variance':1.})
        return target,q,*theta,hashlib.sha256(pools.tobytes()).hexdigest(),diagnostics
    return shell.controls.cache('corrected_DCC_Gaussian_copula',hashlib.sha256(data+SOURCE.encode()).hexdigest(),build)


@lru_cache(maxsize=2)
def dcc_uniforms(shape,data,sims,horizon,seed):
    target,q,a,b,_,_=configuration(shape,data)
    states=np.repeat(q[None,:,:],sims,axis=0);rng=np.random.default_rng(seed)
    result=np.empty((sims,horizon,shape[1]));backend=native.load_paths()
    for start in range(0,horizon,1024):
        stop=min(start+1024,horizon);z=rng.normal(size=(stop-start,sims,shape[1]))
        from numba import get_num_threads
        backend.scores(z,np.ascontiguousarray(target),states,a,b,get_num_threads() if shape[1]>=16 else 1)
        ndtr(z,out=z);np.clip(z,1e-8,1.-1e-8,out=z)
        result[:,start:stop]=z.transpose(1,0,2)
    return result


def select_copula(enabled):
    global ENABLED
    if enabled!=ENABLED:clear_path_cache()
    ENABLED=enabled;parent.select_copula(True)
    shell.controls.gaussian_uniforms=dcc_uniforms if enabled else parent.parent.student_uniforms


class Candidate(parent.parent.parent.parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        *_,digest,diagnostics=configuration(assets.shape,assets.tobytes()) if ENABLED else (None,None)
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['corrected_dcc_enabled']=ENABLED
            record['copula_training_input']='unchanged_chronological_Student_standardized_innovation_pool'
            record['copula_innovation_matrix_sha256']=digest
            record['return_copula_corrected_dcc_fit']=diagnostics
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_innovation_corrected_dcc_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():
    parent.clear_path_cache();dcc_uniforms.cache_clear()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
