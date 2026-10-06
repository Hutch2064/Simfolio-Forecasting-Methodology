"""M243 future law retained; historical rough offset uses the same scalar AR1."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from matched_causal import predict

spec=importlib.util.spec_from_file_location('matched_offset_private_M243',ROOT/'tools/stationary_ar1_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
body=parent.parent
original_observed=body.rough.backend.base.base.observed
MATCHED_OFFSET_ENABLED=True


@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    y,eps=body.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    original,fit=body.refit(data)
    bias,variance=body.pathwise.parent.log_square_moments(original['student_return_laplace_fit']['inverse_df'])
    gaussian_bias,_=body.pathwise.parent.log_square_moments(0.)
    y=y+gaussian_bias-bias
    offset=predict(y,fit['level'],fit['coefficients'][0],fit['innovation_sd'],variance)
    return y-offset,eps


def select_offset(enabled):
    global MATCHED_OFFSET_ENABLED
    MATCHED_OFFSET_ENABLED=enabled
    chosen=observed if enabled else original_observed
    body.rough.backend.base.base.observed=chosen
    body.rough.backend.base.noise.observed=chosen


select_offset(True)
body.rough.SOURCE=hashlib.sha256((body.rough.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        for x in np.asarray(training.asset_log_returns,float).T:
            d=np.ascontiguousarray(x).tobytes()
            record=FIT_DIAGNOSTICS[self.model_id,hashlib.sha256(d).hexdigest()]
            record['matched_scalar_offset_enabled']=MATCHED_OFFSET_ENABLED
            record['rough_observation_offset']='stationary scalar AR1 one-step Gaussian predictor with unchanged M243 fitted proxy parameters and Student-implied measurement variance' if MATCHED_OFFSET_ENABLED else 'unchanged original M243 offset'
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_matched_offset_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
clear_path_cache=parent.clear_path_cache


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    predict(np.zeros(3),0.,.8,.1,5.)
