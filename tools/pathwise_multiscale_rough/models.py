"""M220 with pathwise multiscale states, preserving pre-clipping return moments.

No new fitted parameters; original multiscale Gaussian transition/covariance.
The volatility multiplier has second moment one at each future date. Its RNG
is separate, retaining the original rough innovation stream and dependence.
"""
import hashlib
import importlib.util
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from pathwise_native import load_paths

spec=importlib.util.spec_from_file_location('pathwise_private_m220',ROOT/'tools/student_implied_noise_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.parent.streamed
predecessor_fit=streamed.predecessor_fit;rough_fit=streamed.rough_fit
predecessor_asset_paths=streamed.predecessor_asset_paths;prepared=streamed.prepared
shell=parent.shell;overlay=parent.overlay
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS


@lru_cache(maxsize=64)
def multiscale_configuration(data,horizon):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        predictive_state_moments,
    )

    fitted=predecessor_fit(data);q=fitted['bdes_multiscale_vol']
    phi=np.r_[q['phis'],q['residual_phi']];loading=np.r_[q['b'],1.]
    initial=np.r_[q['q_last'],q['residual_last']]
    sd=np.r_[np.sqrt(np.maximum(q['q_var'],1e-10)),q['residual_innovation_sd']]
    common=np.r_[sd[:-1],sd[-1]*q['residual_common_loading']]
    independent=np.zeros(len(phi));independent[-1]=sd[-1]*math.sqrt(max(0.,1-q['residual_common_loading']**2))
    mean,variance,_,_=predictive_state_moments(fitted,horizon)
    return phi,loading,common,independent,initial,float(q['ell']),mean,variance


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        training.validate()
        past,future=shell._validate_calendar(training,context)
        assets=np.asarray(training.asset_log_returns,np.float64)
        sims,horizon=context.simulations,context.horizon_days
        data=[assets[:,a].tobytes() for a in range(assets.shape[1])]
        lag=len(assets)+horizon-1
        started=time.perf_counter()
        def fit_asset(d):
            base=predecessor_fit(d)
            rough=rough_fit(d,lag,self.fit_seconds)
            return base,rough
        workers=min(len(data),max(1,shell.controls.STATE_WORKERS))
        if workers==1:posteriors=list(map(fit_asset,data))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:posteriors=list(pool.map(fit_asset,data))
        shell.controls.timed('fit_phase_wall',started)
        started=time.perf_counter()
        if assets.shape[1]>1:
            seed=shell.bd.deterministic_seed('copula_alternatives',shell.bd.FRONTIER_DEPENDENCE_ID,
                str(context.origin_date),horizon,sims)
            uniforms=shell.controls.gaussian_uniforms(assets.shape,assets.tobytes(),sims,horizon,seed)
        else:
            seed=shell.bd.deterministic_seed('moment_sv_single_asset',str(context.origin_date),horizon,sims)
            uniforms=np.random.default_rng(seed).random((sims,horizon,1))
        shell.controls.timed('dependence_paths',started)
        paths=np.empty_like(uniforms)
        started=time.perf_counter()
        from dynamic import selected_kernel
        for a,(d,(_,rough)) in enumerate(zip(data,posteriors)):
            identity=hashlib.sha256(d).hexdigest()
            theta=rough['map']
            kernel=selected_kernel(float(theta[0]),math.exp(theta[1]),lag)[2]
            FIT_DIAGNOSTICS[self.model_id,identity]={'model_id': self.model_id,'data_sha256': identity,
                'conventional_estimator': 'original_predecessor_MAP','rough_estimator': 'MAP_point_conditional_on_causal_SV_prediction','normalization': 'stationary_second_moment_one',
                'posterior_parameter_uncertainty': False,'rough_parameters': theta.tolist(),'kernel': kernel,
                'rough_fit': {k:v for k,v in rough.items() if k not in ('points','weights','map')}}
            fitted=predecessor_fit(d)
            FIT_DIAGNOSTICS[self.model_id,identity].update(student_return_laplace_fit=fitted['student_return_laplace_fit'],
                conventional_estimator='raw_Student_Laplace_MAP',return_mean_curve='unchanged_original_predecessor',
                multiscale_forecast='pathwise_existing_Gaussian_states_second_moment_normalized',
                student_log_square_moments=dict(zip(('mean','variance'),parent.log_square_moments(fitted['student_return_laplace_fit']['inverse_df']))))
            mean,base_paths=predecessor_asset_paths(d,uniforms[:,:,a])
            ms=multiscale_configuration(d,horizon)
            ms_rng=np.random.default_rng(shell.bd.deterministic_seed('pathwise_multiscale_prediction',identity,
                str(context.origin_date),horizon,sims))
            phi,w,root,state,initial_root,means,variances=prepared(d,theta.tobytes(),lag,horizon)
            rng=np.random.default_rng(shell.bd.deterministic_seed('rough_overlay_prediction',identity,
                str(context.origin_date),horizon,sims))
            for s in range(sims):
                initial=state+initial_root@rng.normal(size=phi.size)
                initial=phi*initial+root@rng.normal(size=phi.size)
                module,dot=load_paths()
                module.map_path(phi,w,root,initial,means,variances,rng.bit_generator.capsule,
                                dot,mean,base_paths[s],paths[s,:,a],*ms,ms_rng.bit_generator.capsule,True)
        shell.controls.timed('asset_predictive_paths',started)
        dates=shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        mask=np.asarray([date in dates for date in future])
        return shell.rejoin(paths,training.policy.weights,mask)



def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    load_paths()


clear_path_cache=parent.clear_path_cache
CANDIDATES=(Candidate(model_id='asset_map_student_implied_noise_pathwise_multiscale_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
