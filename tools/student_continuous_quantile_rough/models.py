"""M227 with exact first/second moments of the interpolated innovation quantile.

Normalization is in uniform probability space before the existing return clip;
it does not claim conditional dynamic-copula uniforms are independently uniform.
"""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('continuous_norm_private_m227',ROOT/'tools/untruncated_rough_priors/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.parent.streamed
student=parent.parent.parent.parent
ROOT=HERE.parents[1]


def normalize(nodes):
    x=np.asarray(nodes,float)
    left,right=x[:-1],x[1:]
    mean=float(np.mean((left+right)*.5))
    second=float(np.mean((left*left+left*right+right*right)/3))
    variance=second-mean*mean
    if not np.isfinite(variance) or variance<=0:raise ArithmeticError('degenerate continuous innovation variance')
    return (x-mean)/np.sqrt(variance)

@lru_cache(maxsize=128)
def nodes(data,n):
    pool=student.refit(data)['innovation_pool']
    return normalize(np.quantile(pool,np.linspace(.5/n,1-.5/n,n)))


def asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace

    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    mean,_=moment_return_curves(student.original_fit(data),uniforms.shape[1]);_,sd=moment_return_curves(student.refit(data),uniforms.shape[1])
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes(data,len(paths)));return mean,paths
parent.parent.predecessor_asset_paths=asset_paths

class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        for key,r in FIT_DIAGNOSTICS.items():
            if key[0]==self.model_id:r['innovation_normalization']='analytic uniform-probability moments of piecewise-linear quantile function; pre-clipping'
        return result
initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_student_continuous_quantile_untruncated_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
