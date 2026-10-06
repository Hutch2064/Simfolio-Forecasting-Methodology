"""M207 volatility with a constant historical daily log-return mean.

Remove the volatility-scaled Sharpe drift in the forecasting marginal. The
conventional volatility fit, rough fit and empirical standardized innovations
are unchanged. Median long-horizon log growth remains the sample log mean,
without shrinkage toward zero or another prior mean.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
    moment_return_curves,
)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('historical_log_mean_private_exact',ROOT/'tools/exact_covariance_whittle_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.streamed


def predecessor_asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace
    fit=streamed.predecessor_fit(data)
    _,sd=moment_return_curves(fit,uniforms.shape[1])
    mean=np.full(uniforms.shape[1],float(np.frombuffer(data,np.float64).mean()))
    simulations=uniforms.shape[0]
    probability=np.linspace(.5/simulations,1-.5/simulations,simulations)
    nodes=np.quantile(fit['innovation_pool'],probability)
    nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes)
    return mean,paths


streamed.predecessor_asset_paths=predecessor_asset_paths
initialize=parent.initialize
Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_multiscale_historical_log_mean_exact_covariance_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
