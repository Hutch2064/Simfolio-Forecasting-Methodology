"""M207 with asset-level unconditional return variance targeting.

This adapts variance targeting to the existing Gaussian multiscale moment law.
The conventional fit, rough fit, empirical innovation nodes and mean curves are
unchanged. Only the return standard-deviation curves receive a training-only
constant that equates their stationary variance to the sample return variance.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import math

import numpy as np
from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('variance_targeted_private_exact',ROOT/'tools/exact_covariance_whittle_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.streamed


def stationary_return_variance(fit):
    q=fit['bdes_multiscale_vol'];base=fit['base_fit']
    phi=np.r_[q['phis'],q['residual_phi']]
    loading=np.r_[q['b'],1.]
    sd=np.r_[np.sqrt(np.maximum(q['q_var'],1e-10)),q['residual_innovation_sd']]
    common=np.r_[sd[:-1],sd[-1]*q['residual_common_loading']]
    noise=np.outer(common,common);noise[-1,-1]=sd[-1]**2
    variance=float(loading@(noise/(1-phi[:,None]*phi[None,:]))@loading)
    sigma_mean=math.exp(.5*q['ell']+.125*variance)/100
    sigma_second=math.exp(q['ell']+.5*variance)/10000
    drift=base['dlm_state_noise_var']/(1-base['dlm_state_transition_phi']**2)
    anchor=base['dlm_long_run_anchor_mean'];sample_sigma=base['sigma']
    return sigma_second*(1+(drift+anchor**2)/sample_sigma**2)-(anchor*sigma_mean/sample_sigma)**2


@lru_cache(maxsize=64)
def variance_scale(data):
    fit=streamed.predecessor_fit(data)
    target=float(np.var(np.frombuffer(data,np.float64),ddof=1))
    stationary=stationary_return_variance(fit)
    if not np.isfinite(stationary) or stationary<=0 or target<=0:
        raise ArithmeticError('variance targeting requires positive finite variances')
    return math.sqrt(target/stationary)


def predecessor_asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace
    fit=streamed.predecessor_fit(data)
    mean,sd=moment_return_curves(fit,uniforms.shape[1])
    sd=sd*variance_scale(data)
    simulations=uniforms.shape[0]
    probability=np.linspace(.5/simulations,1-.5/simulations,simulations)
    nodes=np.quantile(fit['innovation_pool'],probability)
    nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes)
    return mean,paths


streamed.predecessor_asset_paths=predecessor_asset_paths
initialize=parent.initialize
Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_multiscale_variance_targeted_exact_covariance_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
