"""M221 future-state law with jointly fitted Student/copula leverage.

The Gaussian return rank at t drives the multiscale state innovation at t+1.
The original mean, marginal multiscale covariance and rough methodology remain.
Leverage uses sparse Laplace marginal MAP, not full posterior integration.
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
from leverage_native import load_paths
from scipy.special import ndtri

spec=importlib.util.spec_from_file_location('leverage_private_m220',ROOT/'tools/student_implied_noise_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.parent.streamed
predecessor_fit=streamed.predecessor_fit;rough_fit=streamed.rough_fit
prepared=streamed.prepared
shell=parent.shell;overlay=parent.overlay
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
import leverage_laplace

original_student_refit=parent.parent.refit
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()


@lru_cache(maxsize=64)
def joint_refit(data):
    def compute():
        original=original_student_refit(data)
        x=np.frombuffer(data,np.float64)
        y,eps=shell.bd._sv_observed_log_variance(x,float(x.mean()),winsorize=True)
        theta,h,variance,diagnostics=leverage_laplace.fit(eps,y,original['posterior_center'][:3],original['student_return_laplace_fit']['inverse_df'])
        level,phi,eta,_u,rho=theta
        pool=shell.bd._standardized_empirical_innovation_pool(eps*np.exp(-.5*h),clip=None,method='mean_std')
        if pool is None:raise ArithmeticError('invalid joint leverage innovation pool')
        fitted=original.copy()
        fitted.update(posterior_center=(level,phi,eta,float(h[-1]),rho),innovation_pool=pool,
                      bdes_multiscale_vol=shell.bd._bdes_multiscale_components(h,4,shell.bd.BDES_MULTISCALE_GRID_FIXED),
                      state_path_variance_last=float(variance[-1]),state_loglikelihood=diagnostics['loglikelihood'],
                      student_return_laplace_fit=diagnostics)
        return fitted
    return shell.controls.cache('joint_student_copula_leverage_MAP',hashlib.sha256(data+SOURCE.encode()).hexdigest(),compute)


parent.parent.refit=joint_refit
streamed.predecessor_fit=joint_refit
predecessor_fit=joint_refit
parent.parent.parent.parent.SOURCE=hashlib.sha256((parent.parent.parent.parent.SOURCE+SOURCE).encode()).hexdigest()



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
    rho=float(fitted['posterior_center'][4])
    mean,variance=leverage_laplace.conditioned_multiscale_moments(phi,loading,common,independent,mean,variance,rho,last_observed_score(data))
    return phi,loading,common,independent,initial,float(q['ell']),mean,variance


@lru_cache(maxsize=64)
def last_observed_score(data):
    fitted=predecessor_fit(data)
    x=np.frombuffer(data,np.float64)
    _,eps=shell.bd._sv_observed_log_variance(x,float(x.mean()),winsorize=True)
    h_last=fitted['posterior_center'][3]
    u=fitted['student_return_laplace_fit']['inverse_df']
    return float(leverage_laplace.scores(np.array([h_last]),np.array([eps[-1]]),u)[0][0])



def predecessor_asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace

    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
        predictive_state_moments,
    )
    fitted=predecessor_fit(data)
    horizon=uniforms.shape[1]
    mean,_=moment_return_curves(parent.parent.original_fit(data),horizon)
    mh,vh=multiscale_configuration(data,horizon)[-2:]
    _,_,drift_mean,drift_var=predictive_state_moments(fitted,horizon)
    sigma=fitted['base_fit']['sigma']
    expected_sigma=np.exp(.5*mh+.125*vh)/100.
    expected_sigma_squared=np.exp(mh+.5*vh)/10000.
    internal_mean=drift_mean*expected_sigma/sigma
    variance=expected_sigma_squared*(1.+(drift_var+drift_mean*drift_mean)/sigma**2)-internal_mean*internal_mean
    n=len(uniforms)
    nodes=np.quantile(fitted['innovation_pool'],np.linspace(.5/n,1.-.5/n,n))
    nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    paths=uniforms.copy()
    map_asset_inplace(paths,mean,np.sqrt(np.maximum(variance,0.)),nodes)
    return mean,paths


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
                conventional_estimator='joint_Student_Gaussian_copula_leverage_Laplace_MAP',return_mean_curve='unchanged_original_predecessor',
                multiscale_forecast='pathwise_lagged_leverage_conditioned_on_last_observation_second_moment_normalized',
                student_log_square_moments=dict(zip(('mean','variance'),parent.log_square_moments(fitted['student_return_laplace_fit']['inverse_df']))))
            mean,base_paths=predecessor_asset_paths(d,uniforms[:,:,a])
            ms=multiscale_configuration(d,horizon)
            rho=float(fitted['posterior_center'][4])
            gaussian_scores=ndtri(uniforms[:,:,a])
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
                                dot,mean,base_paths[s],paths[s,:,a],*ms,ms_rng.bit_generator.capsule,True,rho,gaussian_scores[s],last_observed_score(d))
        shell.controls.timed('asset_predictive_paths',started)
        dates=shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        mask=np.asarray([date in dates for date in future])
        return shell.rejoin(paths,training.policy.weights,mask)



def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    load_paths()
    leverage_laplace.likelihood_gradient(np.ones(3),0.,.9,.2,.1,0.)


clear_path_cache=parent.clear_path_cache
CANDIDATES=(Candidate(model_id='asset_map_joint_student_copula_leverage_pathwise_multiscale_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
