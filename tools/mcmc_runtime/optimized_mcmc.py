"""Output-identical execution of the scored two-chain, 16-node SV model."""
import copy
import hashlib
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import complex_sv
import numpy as np
import parameter_mcmc_tuned as pm
from cached_moments import make as make_moments
from cached_moments import power
from compiled_chain import chain as compiled_chain
from exact_path_kernels import fast_rejoin, parallel_rejoin_growth, sorted_map
from numba import get_num_threads, set_num_threads
from streamed_paths import map_asset_inplace
from streamed_paths import rejoin as streamed_rejoin
from streamed_paths import uniforms as streamed_uniforms

from simfolio_forecasting_methodology.models.asset_level.frontier import (
    _historical_rebalance_dates,
    _validate_calendar,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg

VERIFY=False
CHAIN_CHECKS=[]
PATH_CHECKS=[]
REFERENCE_CHAIN=pm.chain
FAST_MOMENTS=make_moments()
REFERENCE_MARGINALS=None

@lru_cache(maxsize=512)
def nodes(data,sims):
    fit=pm.parameter_fit(data)
    pool=np.concatenate([c['innovation_pool'] for c in fit['parameter_posterior_children']])
    values=np.quantile(pool,np.linspace(.5/sims,1-.5/sims,sims));values-=values.mean();values/=np.sqrt(np.mean(values*values))
    return values

def mixture_curves(fit,horizon):
    curves=[FAST_MOMENTS(c,horizon) for c in fit['parameter_posterior_children']]
    mean=np.mean([c[0] for c in curves],axis=0)
    variance=np.mean([c[1]**2+c[0]**2 for c in curves],axis=0)-mean**2
    return mean,np.sqrt(np.maximum(variance,0))

def marginals(data,fit,sims,horizon):
    mean,sd=mixture_curves(fit,horizon)
    return np.clip(mean[None,:]+sd[None,:]*nodes(data,sims)[:,None],-1,1)

def install(verify=False):
    global VERIFY,REFERENCE_MARGINALS
    VERIFY=verify
    set_num_threads(min(4,get_num_threads(),os.cpu_count() or 1))
    pm.install();REFERENCE_MARGINALS=complex_sv.sorted_moment_marginals
    fast=compiled_chain
    def checked(y,theta,burn,kept,seed):
        result=fast(y,theta,burn,kept,seed)
        if VERIFY:
            reference=REFERENCE_CHAIN(y,theta,burn,kept,seed)
            assert result[0].tobytes()==reference[0].tobytes() and result[1]==reference[1],('chain',len(y),seed)
            CHAIN_CHECKS.append({'n': len(y),'seed': seed,'observations_sha256': hashlib.sha256(y.tobytes()).hexdigest(),'burn': burn,'kept': kept,'draw_sha256': hashlib.sha256(result[0].tobytes()).hexdigest(),'acceptance': result[1]})
        return result
    pm.chain=checked
    from compiled_map import make as make_map
    original_map=bd._fit_sv_map_state_space_params
    fast_map=make_map()
    def map_fit(*args,**kwargs):
        result=fast_map(*args,**kwargs)
        if VERIFY:assert result==original_map(*args,**kwargs),'MAP parameter fit'
        return result
    bd._fit_sv_map_state_space_params=map_fit
    from parallel_chains import make as make_parallel_fit
    pm.parameter_fit=make_parallel_fit()
    from cached_dependence import make as make_dependence
    from finite_multiscale import make as make_multiscale
    original_multiscale=bd._bdes_multiscale_components
    finite_multiscale=make_multiscale()
    def multiscale(*args,**kwargs):
        result=finite_multiscale(*args,**kwargs)
        if VERIFY:
            assert pickle.dumps(result,protocol=5)==pickle.dumps(original_multiscale(*args,**kwargs),protocol=5),'complete multiscale fit'
        return result
    bd._bdes_multiscale_components=multiscale
    original_terminal=dg.kalman_terminal_posterior
    cached_terminal=make_dependence()
    def terminal(model):
        result=cached_terminal(model)
        if VERIFY:
            assert all(x.tobytes()==y.tobytes() for x,y in zip(result,original_terminal(model))),'dependence posterior'
        return result
    dg.kalman_terminal_posterior=terminal
    original_uniforms=dg.simulate_future_gaussian_uniforms
    def uniforms(model,simulations,horizon,rng):
        if VERIFY:
            clone=np.random.Generator(type(rng.bit_generator)())
            clone.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
        result=streamed_uniforms(model,simulations,horizon,rng)
        if VERIFY:
            assert result.tobytes()==original_uniforms(model,simulations,horizon,clone).tobytes(),'complete copula uniforms'
            assert rng.bit_generator.state==clone.bit_generator.state,'copula RNG consumption'
        return result
    dg.simulate_future_gaussian_uniforms=uniforms
    # Compile before measured work; keep the numerical workload unchanged.
    checked(np.array([-7.,-6.,-9.]),(-8.,bd._logit(.94),np.log(.35)),1024,2048,1)
    dummy=np.zeros((2,2,2));sorted_map(dummy,np.full_like(dummy,.5));fast_rejoin(dummy,(.5,.5),np.array([False,True]))
    map_asset_inplace(dummy[:,:,0],np.zeros(2),np.ones(2),np.array([-1.,1.]));streamed_rejoin(dummy,(.5,.5),np.array([False,True]))
    parallel_rejoin_growth(np.ones((2,2,2)),np.array([.5,.5]),np.array([False,True]),.0015)
    CHAIN_CHECKS.clear();PATH_CHECKS.clear();pm.parameter_fit.cache_clear();nodes.cache_clear();power.cache_clear()

@dataclass(frozen=True)
class OptimizedMCMC:
    model_id:str='sv_parameter_mcmc_twochain_sixteen_node_moment_mixture'
    fit_workers:int=0
    def simulate_daily_log_returns(self,training,context):
        training.validate();past,future=_validate_calendar(training,context)
        assets=np.asarray(training.asset_log_returns,dtype=np.float64);sims,horizon=context.simulations,context.horizon_days
        data=[assets[:,a].tobytes() for a in range(assets.shape[1])]
        cpu_count=os.cpu_count() or 1
        chain_lanes=2 if len(assets)>=8192 else 1
        automatic=min(2 if len(assets)<2048 else 6,max(1,cpu_count//chain_lanes))
        workers=self.fit_workers or automatic
        workers=min(workers,assets.shape[1])
        if workers==1:fits=[pm.parameter_fit(d) for d in data]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:fits=list(pool.map(pm.parameter_fit,data))
        if assets.shape[1]>1:
            dep=dg.fit_dynamic_gaussian_factor_model(assets)
            seed=bd.deterministic_seed('copula_alternatives',bd.FRONTIER_DEPENDENCE_ID,str(context.origin_date),horizon,sims)
            uniforms=dg.simulate_future_gaussian_uniforms(dep,sims,horizon,np.random.default_rng(seed))
        else:
            seed=bd.deterministic_seed('moment_sv_single_asset',str(context.origin_date),horizon,sims)
            uniforms=np.random.default_rng(seed).random((sims,horizon,1))
        if VERIFY:
            source=np.empty_like(uniforms);original_uniforms=uniforms.copy()
        for a,(d,fit) in enumerate(zip(data,fits)):
            mean,sd=mixture_curves(fit,horizon);standardized_nodes=nodes(d,sims)
            if VERIFY:
                values=np.clip(mean[None,:]+sd[None,:]*standardized_nodes[:,None],-1,1)
                assert values.tobytes()==REFERENCE_MARGINALS(fit,sims,horizon).tobytes(),('marginals',a)
                source[:,:,a]=values
            map_asset_inplace(uniforms[:,:,a],mean,sd,standardized_nodes)
        paths=uniforms
        if VERIFY:assert paths.tobytes()==dg.map_uniforms_to_marginal_paths(source,original_uniforms).tobytes(),'mapped asset paths'
        dates=_historical_rebalance_dates(past.append(future),training.policy.rebalance);mask=np.asarray([d in dates for d in future])
        result=streamed_rejoin(paths,training.policy.weights,mask)
        if VERIFY:
            assert result.tobytes()==dg.rebalanced_portfolio_log_paths(paths,training.policy.weights,mask).tobytes(),'daily portfolio paths'
            PATH_CHECKS.append({'asset_values': int(paths.size),'daily_values': int(result.size),'asset_sha256': hashlib.sha256(paths.tobytes()).hexdigest(),'daily_sha256': hashlib.sha256(result.tobytes()).hexdigest()})
        return result
