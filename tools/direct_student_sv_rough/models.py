"""Direct scalar Student/Laplace volatility dynamics without a second proxy fit."""
import hashlib
import importlib.util
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools/coupled_multiscale_rough'))
spec=importlib.util.spec_from_file_location('direct_student_private_M237',ROOT/'tools/learned_loading_full_hurst_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
sys.path.insert(0,str(HERE))
from coupled_native import load_paths

spec=importlib.util.spec_from_file_location('coupled_private_state',ROOT/'tools/coupled_multiscale_rough/state.py')
state=importlib.util.module_from_spec(spec);sys.modules[spec.name]=state;spec.loader.exec_module(state)
spec=importlib.util.spec_from_file_location('direct_student_terminal_helpers',ROOT/'tools/terminal_laplace_state_rough/terminal.py')
terminal_state=importlib.util.module_from_spec(spec);sys.modules[spec.name]=terminal_state;spec.loader.exec_module(terminal_state)
rough=parent.rough;learned=parent.parent.parent;pathwise=rough.parent
shell=parent.shell;overlay=parent.overlay;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
SOURCE=hashlib.sha256((rough.SOURCE+learned.SOURCE+hashlib.sha256((ROOT/'tools/coupled_multiscale_rough/state.py').read_bytes()).hexdigest()+hashlib.sha256((ROOT/'tools/terminal_laplace_state_rough/terminal.py').read_bytes()).hexdigest()+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*')) if p.suffix in ('.py','.cpp'))).encode()).hexdigest()


@lru_cache(maxsize=64)
def refit(data):
    def build():
        original=learned.original_student_fit(data)
        level,phi,eta,last=map(float,original['posterior_center'][:4])
        variance=float(original['state_path_variance_last'])
        if not np.isfinite(variance) or variance<0.:raise ArithmeticError('invalid Student terminal variance')
        fit={'success':True,'phis':[],'coefficients':[phi],'innovation_sd':eta,'level':level,
             'initial':[last-level],'component_count':0,'terminal_laplace_variance':variance,
             'estimator':'direct_raw_Student_return_Laplace_MAP_parameters',
             'historical_state':'raw-return conditional latent mode; no secondary clipping or proxy refit',
             'count_selection':'standard_scalar_logvol_AR1_state_model',
             'parameter_uncertainty':False}
        return original,fit
    return shell.controls.cache('direct_raw_Student_scalar_SV',hashlib.sha256(data+SOURCE.encode()).hexdigest(),build)


@lru_cache(maxsize=64)
def configuration(data,horizon,include_terminal=True):
    original,fit=refit(data);phis=np.array(fit['phis']);coefficients=np.array(fit['coefficients'])
    initial=np.array(fit['initial']);sd=fit['innovation_sd'];f=state.transition(phis,coefficients)
    if np.max(np.abs(np.linalg.eigvals(f)))>=1.:raise ArithmeticError('coupled transition lost numerical stationarity')
    mean,variance=state.moments(f,initial,sd*np.r_[1.,1.-phis],horizon)
    variance=terminal_state.variance_curve(variance,float(coefficients[0]),fit['terminal_laplace_variance'] if include_terminal else 0.)
    mean+=fit['level']
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        predictive_state_moments,
    )
    _,_,drift,drift_variance=predictive_state_moments(original,horizon)
    sigma=original['base_fit']['sigma']
    expected_sigma=np.exp(.5*mean+.125*variance)/100.
    expected_sigma_squared=np.exp(mean+.5*variance)/10000.
    implied_mean=drift*expected_sigma/sigma
    return_sd=np.sqrt(np.maximum(expected_sigma_squared*(1.+(drift_variance+drift*drift)/sigma**2)-implied_mean*implied_mean,0.))
    if not np.isfinite(return_sd).all():raise ArithmeticError('nonfinite coupled return variance curve')
    return phis,coefficients,sd,initial,fit['level'],mean,variance,return_sd


@dataclass(frozen=True)
class Candidate:
    model_id:str
    fit_seconds:float=10.
    terminal_uncertainty:bool=True

    def simulate_daily_log_returns(self,training,context):
        training.validate();past,future=shell._validate_calendar(training,context)
        assets=np.asarray(training.asset_log_returns,np.float64);sims=context.simulations;horizon=context.horizon_days
        data=[assets[:,a].tobytes() for a in range(assets.shape[1])];lag=len(assets)+horizon-1
        started=time.perf_counter()
        def fit_asset(d):return refit(d),rough.rough_fit(d,lag,self.fit_seconds)
        workers=min(len(data),max(1,shell.controls.STATE_WORKERS))
        if workers==1:posteriors=list(map(fit_asset,data))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:posteriors=list(pool.map(fit_asset,data))
        shell.controls.timed('fit_phase_wall',started);started=time.perf_counter()
        if assets.shape[1]>1:
            seed=shell.bd.deterministic_seed('copula_alternatives',shell.bd.FRONTIER_DEPENDENCE_ID,str(context.origin_date),horizon,sims)
            uniforms=shell.controls.gaussian_uniforms(assets.shape,assets.tobytes(),sims,horizon,seed)
        else:
            seed=shell.bd.deterministic_seed('moment_sv_single_asset',str(context.origin_date),horizon,sims)
            uniforms=np.random.default_rng(seed).random((sims,horizon,1))
        shell.controls.timed('dependence_paths',started);paths=np.empty_like(uniforms);started=time.perf_counter()
        from dynamic import selected_kernel
        from streamed_paths import map_asset_inplace

        from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
            moment_return_curves,
        )
        module,dot=load_paths()
        for a,(d,((original,fit),rough_fit)) in enumerate(zip(data,posteriors)):
            identity=hashlib.sha256(d).hexdigest();theta=rough_fit['map']
            phi,w,root,terminal,initial_root,rm,rv=pathwise.prepared(d,theta.tobytes(),lag,horizon)
            mean,_=moment_return_curves(learned.student.original_fit(d),horizon)
            mp,c,sd,initial,level,mm,mv,return_sd=configuration(d,horizon,self.terminal_uncertainty)
            base_paths=np.ascontiguousarray(uniforms[:,:,a]);map_asset_inplace(base_paths,mean,return_sd,learned.nodes(d,sims))
            rng=np.random.default_rng(shell.bd.deterministic_seed('rough_overlay_prediction',identity,str(context.origin_date),horizon,sims))
            memory_rng=np.random.default_rng(shell.bd.deterministic_seed('coupled_multiscale_prediction',identity,str(context.origin_date),horizon,sims))
            initial_rng=np.random.default_rng(shell.bd.deterministic_seed('conventional_terminal_laplace_state_prediction',identity,str(context.origin_date),horizon,sims))
            v0=fit['terminal_laplace_variance'] if self.terminal_uncertainty else 0.
            for s in range(sims):
                rinitial=terminal+initial_root@rng.normal(size=len(phi))
                rinitial=phi*rinitial+root@rng.normal(size=len(phi))
                memory_initial=terminal_state.initial_state(initial,v0,initial_rng)
                module.map_path(phi,w,root,rinitial,rm,rv,rng.bit_generator.capsule,dot,mean,base_paths[s],paths[s,:,a],mp,c,sd,memory_initial,level,mm,mv,memory_rng.bit_generator.capsule,True)
            FIT_DIAGNOSTICS[self.model_id,identity]={'model_id':self.model_id,'data_sha256':identity,
                'coupled_multiscale_fit':fit,'student_return_laplace_fit':original['student_return_laplace_fit'],
                'student_log_square_moments':dict(zip(('mean','variance'),pathwise.parent.log_square_moments(original['student_return_laplace_fit']['inverse_df']))),
                'rough_parameters':theta.tolist(),'rough_fit':{k:v for k,v in rough_fit.items() if k not in ('map','points','weights')},
                'kernel':selected_kernel(float(theta[0]),math.exp(theta[1]),lag)[2],
                'return_mean_curve':'unchanged_original_predecessor','parameter_uncertainty':False,
                'terminal_uncertainty_enabled':self.terminal_uncertainty,'terminal_initial_variance_used':v0,
                'conventional_state_uncertainty':'Student/Laplace terminal Gaussian variance under the same fitted AR1 state parameters; independent future Gaussian AR1 recursion',
                'conventional_normalization':'conditional Gaussian second moment one; return variance curve from same coupled state law',
                'rough_normalization':'unchanged M237 stationary Gaussian normalization'}
        shell.controls.timed('asset_predictive_paths',started)
        dates=shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        return shell.rejoin(paths,training.policy.weights,np.asarray([date in dates for date in future]))


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers);load_paths()
    state.innovations(np.arange(20.,dtype=float),np.empty(0),np.array([.5]))
    state.moments(np.array([[.5]]),np.zeros(1),np.ones(1),2)


CANDIDATES=(Candidate(model_id='asset_map_direct_student_sv_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
