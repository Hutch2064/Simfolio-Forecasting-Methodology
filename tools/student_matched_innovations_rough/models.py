"""M221 forecasting nodes from the same Student tail shape used in fitting.

The original finite-grid normalization/mapper and all volatility fits remain.
This is a discretized Student-shape forecast, not exact continuous Student draws.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtri, stdtrit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('student_matched_nodes_private_m221',ROOT/'tools/pathwise_multiscale_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)


def normalized_student_nodes(inverse_df,simulations):
    probability=np.linspace(.5/simulations,1.-.5/simulations,simulations)
    # Continuous unit-variance scaling cancels under the retained finite-grid
    # normalization. Avoid multiplying by a vanishing scale as nu approaches 2.
    result=ndtri(probability) if inverse_df==0 else stdtrit(1/inverse_df,probability)
    result-=result.mean();result/=np.sqrt(np.mean(result*result))
    return result


@lru_cache(maxsize=128)
def nodes(data,simulations):
    inverse_df=parent.predecessor_fit(data)['student_return_laplace_fit']['inverse_df']
    return normalized_student_nodes(inverse_df,simulations)


def asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace

    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    mean,_=moment_return_curves(parent.parent.parent.original_fit(data),uniforms.shape[1])
    _,sd=moment_return_curves(parent.predecessor_fit(data),uniforms.shape[1])
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes(data,len(uniforms)))
    return mean,paths


parent.predecessor_asset_paths=asset_paths


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        for x in np.asarray(training.asset_log_returns,float).T:
            data=np.ascontiguousarray(x).tobytes()
            record=FIT_DIAGNOSTICS[self.model_id,hashlib.sha256(data).hexdigest()]
            record['forecast_innovation_family']='Student quantile nodes with existing finite-grid normalization and interpolation'
            record['forecast_inverse_df']=record['student_return_laplace_fit']['inverse_df']
            record['additional_fitted_innovation_parameters']=0
        return result


initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_matched_student_innovations_pathwise_multiscale_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
