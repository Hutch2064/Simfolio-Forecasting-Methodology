"""M255 asset law with finite-variance Student corrected-DCC dependence."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtri, ndtr

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import student_cdf
import student_state as state

spec=importlib.util.spec_from_file_location('student_cdcc_private_M255',ROOT/'tools/corrected_dcc_innovation_copula/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
body=parent.body;native=parent.native;ENABLED=True
_gaussian_fit=parent.state.fit


def gaussian_fit(z,s,**kwargs):
    if len(s)>=16 and "evaluator" not in kwargs:
        kwargs["evaluator"]=state.likelihood_kernel.legacy_gaussian_likelihood
    return _gaussian_fit(z,s,**kwargs)


parent.state.fit=gaussian_fit
SOURCE=hashlib.sha256((parent.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(HERE.glob('*')) if p.suffix in ('.py','.cpp'))).encode()).hexdigest()


@lru_cache(maxsize=64)
def configuration(shape,data):
    def build():
        assets=np.frombuffer(data,np.float64).reshape(shape)
        pools=parent.parent.aligned_pools(assets,body.refit)
        u=body.shell.controls.dg.pseudo_observations(pools)
        target,_,shrinkage=parent.parent.parent.parent.shrunk_correlation(ndtri(u))
        theta,q,nu,diagnostics=state.fit(u,target)
        if not np.isfinite(q).all() or np.linalg.eigvalsh(q).min()<=0.:
            raise ArithmeticError('invalid fitted Student cDCC terminal covariance')
        diagnostics.update({'method':'conditional_Student_copula_corrected_DCC_joint_a_b_nu_likelihood',
            'target_estimator':'fixed_full_training_Gaussian_rank_Ledoit_Wolf',
            'consistent_cDCC_target_estimator_claim':False,'target_shrinkage':float(shrinkage),
            'a':float(theta[0]),'b':float(theta[1]),'nu':None if np.isinf(nu) else float(nu),
            'gaussian_endpoint':bool(np.isinf(nu)),'target':target.tolist(),
            'terminal_Q':q.tolist(),'training_rows':len(u),'conditional_marginal_variance':1.,
            'shock_standardization':'z=sqrt((nu-2)/nu)*t_nu_inverse_CDF(u); w=sqrt(diag(Q))*z'})
        return target,q,*theta,nu,hashlib.sha256(pools.tobytes()).hexdigest(),diagnostics
    return shell.controls.cache('corrected_DCC_Student_copula',hashlib.sha256(data+SOURCE.encode()).hexdigest(),build)


@lru_cache(maxsize=2)
def dcc_uniforms(shape,data,sims,horizon,seed):
    target,q,a,b,nu,_,_=configuration(shape,data)
    gaussian_endpoint=np.isinf(nu)
    states=np.repeat(q[None,:,:],sims,axis=0);rng=np.random.default_rng(seed)
    scale_rng=np.random.default_rng(np.random.SeedSequence([seed,0x54434f50]))
    result=np.empty((sims,horizon,shape[1]));backend=native.load_paths()
    for start in range(0,horizon,1024):
        stop=min(start+1024,horizon);z=rng.normal(size=(stop-start,sims,shape[1]))
        if not gaussian_endpoint:
            scales=scale_rng.gamma(nu/2,2.,size=(stop-start,sims,1))
            if np.any(scales<=0) or not np.isfinite(scales).all():
                raise ArithmeticError('unresolved Student mixing scale')
            z*=np.sqrt((nu-2)/scales)
        from numba import get_num_threads
        backend.scores(z,np.ascontiguousarray(target),states,a,b,get_num_threads() if shape[1]>=16 else 1)
        if gaussian_endpoint:ndtr(z,out=z)
        else:
            z*=np.sqrt(nu/(nu-2));student_cdf.probabilities(z,nu,1024*sims*shape[1])
        np.clip(z,1e-8,1.-1e-8,out=z)
        result[:,start:stop]=z.transpose(1,0,2)
    return result


def select_copula(enabled):
    global ENABLED
    if enabled!=ENABLED:clear_path_cache()
    ENABLED=enabled;parent.select_copula(True)
    shell.controls.gaussian_uniforms=dcc_uniforms if enabled else parent.dcc_uniforms


class Candidate(parent.Candidate.__bases__[0]):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        *_,digest,diagnostics=configuration(assets.shape,assets.tobytes()) if ENABLED else (None,None)
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['student_corrected_dcc_enabled']=ENABLED
            record['copula_training_input']='unchanged_chronological_Student_standardized_innovation_pool'
            record['copula_innovation_matrix_sha256']=digest
            record['return_copula_student_corrected_dcc_fit']=diagnostics
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_innovation_student_corrected_dcc_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():
    parent.clear_path_cache();dcc_uniforms.cache_clear()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
