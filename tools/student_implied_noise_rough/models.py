"""M219 with Student-implied log-square centering and observation variance.

Differentiating the Student squared-return Mellin transform gives the exact
log-square mean/variance (Euler beta integral: https://dlmf.nist.gov/5.12.E3).
This replaces empirical proxy-noise moments, retaining empirical forecast nodes.
"""
import hashlib
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from student_log_moments import log_square_moments

spec=importlib.util.spec_from_file_location('student_moment_private_m219',ROOT/'tools/student_return_laplace_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()


@lru_cache(maxsize=64)
def measurement_scale(data):
    u=parent.refit(data)['student_return_laplace_fit']['inverse_df']
    _,R=log_square_moments(u)
    return math.sqrt(4.934802200544679/R),R


@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    y,eps=parent.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    fitted=parent.refit(data);level,phi,eta=fitted['posterior_center'][:3]
    bias,_=log_square_moments(fitted['student_return_laplace_fit']['inverse_df'])
    gaussian_bias,_=log_square_moments(0.)
    y=y+gaussian_bias-bias
    _,R=measurement_scale(data)
    return y-parent.parent.causal_predictor(y,level,phi,eta,R),eps


parent.parent.parent.base.base.observed=observed
parent.parent.parent.base.noise.observed=observed
parent.parent.parent.base.noise.measurement_scale=measurement_scale
parent.parent.parent.SOURCE=hashlib.sha256((parent.parent.parent.SOURCE+SOURCE).encode()).hexdigest()


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        for x in np.asarray(training.asset_log_returns,float).T:
            d=np.ascontiguousarray(x).tobytes()
            record=FIT_DIAGNOSTICS[self.model_id,hashlib.sha256(d).hexdigest()]
            u=parent.refit(d)['student_return_laplace_fit']['inverse_df']
            mean,variance=log_square_moments(u)
            record['rough_observation_noise']='analytic uncensored Student log-square moments from fitted inverse_df'
            record['conventional_offset_noise']='Gaussian causal proxy with Student log-square centering/variance; original floor retained'
            record['student_log_square_moments']={'mean':mean,'variance':variance,'inverse_df':u}
        return result


initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_student_return_laplace_implied_noise_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
