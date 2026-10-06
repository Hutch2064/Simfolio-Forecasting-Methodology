"""M237 without percentile clipping of rough log-square observations.

The existing near-zero floor and Student measurement moments are retained.
Only observation preprocessing changes; conventional fitting is unchanged.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('unclipped_private_learned_loading',ROOT/'tools/learned_loading_full_hurst_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
rough=parent.rough
from causal_noise import causal_predictor


@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    eps=(x-float(x.mean()))*100.0
    squared=eps*eps
    positive=squared[np.isfinite(squared)&(squared>0.)]
    floor=float(max(np.quantile(positive,.001)*.1,1e-10)) if positive.size else 1e-10
    fit=rough.parent.predecessor_fit(data)
    bias,noise=rough.parent.parent.log_square_moments(fit['student_return_laplace_fit']['inverse_df'])
    gaussian_bias,_=rough.parent.parent.log_square_moments(0.)
    y=np.log(np.maximum(squared,floor))-parent.shell.bd.SV_LOG_CHI_SQUARE_MEAN
    y=y+gaussian_bias-bias
    level,phi,eta=fit['posterior_center'][:3]
    return y-causal_predictor(y,level,phi,eta,noise),eps


original_target=rough.whittle_target


def whittle_target(theta,periodogram,n,noise_variance):
    try:
        return original_target(theta,periodogram,n,noise_variance)
    except ArithmeticError as error:
        if str(error)!='invalid analytic-covariance expected periodogram':
            raise
        return -np.inf


rough.whittle_target=whittle_target
rough.backend.base.base.observed=observed
rough.backend.base.noise.observed=observed
digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
rough.SOURCE=hashlib.sha256((rough.SOURCE+digest).encode()).hexdigest()
parent.parent.parent.SOURCE=hashlib.sha256((parent.parent.parent.SOURCE+digest).encode()).hexdigest()


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        for x in np.asarray(training.asset_log_returns,float).T:
            data=np.ascontiguousarray(x).tobytes()
            FIT_DIAGNOSTICS[self.model_id,hashlib.sha256(data).hexdigest()]['rough_observation_preprocessing']='percentile clipping removed; original near-zero floor retained'
        return result


initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_adaptive_predictive_loading_full_hurst_unclipped_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
