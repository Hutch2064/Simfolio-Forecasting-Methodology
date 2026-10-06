"""M243 shell with exact endpoint residual volatility covariance ablations."""
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
spec=importlib.util.spec_from_file_location('coupled_private_M237',ROOT/'tools/learned_loading_full_hurst_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'tools/coupled_multiscale_rough'))
import kernel
from coupled_native import load_paths

spec=importlib.util.spec_from_file_location('coupled_private_state',ROOT/'tools/coupled_multiscale_rough/state.py')
state=importlib.util.module_from_spec(spec);sys.modules[spec.name]=state;spec.loader.exec_module(state)
rough=parent.rough;learned=parent.parent.parent;pathwise=rough.parent
spec=importlib.util.spec_from_file_location('endpoint_scalar_reference',ROOT/'tools/stationary_ar1_rough/state.py')
scalar=importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar)
state.fit=scalar.fit
shell=parent.shell;overlay=parent.overlay;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
SOURCE=hashlib.sha256((hashlib.sha256((ROOT/'tools/stationary_ar1_rough/state.py').read_bytes()).hexdigest()+rough.SOURCE+learned.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*')) if p.suffix in ('.py','.cpp'))).encode()).hexdigest()


@lru_cache(maxsize=64)
def refit(data):
    def build():
        import student_laplace_sv
        original=learned.original_student_fit(data);x=np.frombuffer(data,np.float64)
        _,eps=shell.bd._sv_observed_log_variance(x,float(x.mean()),winsorize=True)
        level,phi,eta=original['posterior_center'][:3]
        u=original['student_return_laplace_fit']['inverse_df']
        h,_,_,_,_,success,_,decrement=student_laplace_sv.latent_mode(eps*eps,level,phi,eta,u)
        if not success:raise ArithmeticError(f'Student latent reconstruction failed: {decrement}')
        return original,state.fit(np.clip(h,-18.,18.))
    return shell.controls.cache('coupled_multiscale_conditional_QMLE',hashlib.sha256(data+SOURCE.encode()).hexdigest(),build)


@lru_cache(maxsize=64)
def configuration(data,horizon):
    original,fit=refit(data);phis=np.array(fit['phis']);coefficients=np.array(fit['coefficients'])
    initial=np.array(fit['initial']);sd=fit['innovation_sd'];f=state.transition(phis,coefficients)
    if np.max(np.abs(np.linalg.eigvals(f)))>=1.:raise ArithmeticError('coupled transition lost numerical stationarity')
    mean,variance=state.moments(f,initial,sd*np.r_[1.,1.-phis],horizon)
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


@lru_cache(maxsize=128)
def endpoint_fit(data,lag,seconds,kind):
    if kind=='fractional':return rough.rough_fit(data,lag,seconds)
    identity=hashlib.sha256(data).hexdigest()
    def build():
        if kind=='none':return {'map':np.array([-1.,0.,0.]),'covariance_family':kind,'estimator':'no_residual_overlay','posterior_uncertainty':False,'fitted_parameter_count':0}
        y,_=rough.backend.base.base.observed(data)
        _,noise=rough.backend.base.noise.measurement_scale(data)
        differences=np.diff(y);n=len(differences);periodogram=np.abs(np.fft.rfft(differences))**2/n
        result=kernel.fit(periodogram,n,noise,kind,seconds)
        result.update(data_sha256=identity,measurement_noise_variance=noise,frequency_count=len(periodogram),zero_frequency='included_known_zero_difference_mean')
        return result
    started=time.perf_counter()
    result=shell.controls.cache('exact_endpoint_residual_MAP',hashlib.sha256(data+kind.encode()+SOURCE.encode()).hexdigest(),build)
    shell.controls.timed('rough_fit_worker',started)
    return result


@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    theta=np.frombuffer(theta_bytes,np.float64)
    if 0.<theta[0]<.5:return pathwise.prepared(data,theta_bytes,lag,horizon)
    phi,w,q,variance=kernel.configuration(theta)
    if theta[0]==-1.:return phi,w,q,np.zeros(1),q,np.zeros(horizon),np.zeros(horizon)
    y,_=rough.backend.base.base.observed(data)
    _,noise=rough.backend.base.noise.measurement_scale(data)
    scale=math.sqrt(4.934802200544679/noise)
    terminal,p=overlay.terminal_filter(y*scale,phi,w,q*scale**2,float(y.mean())*scale)
    terminal=terminal/scale;p=p/scale**2
    eigen,vectors=np.linalg.eigh((p+p.T)/2)
    initial_root=vectors*np.sqrt(np.maximum(eigen,0))
    return phi,w,np.diag(np.sqrt(np.diag(q))),terminal,initial_root,np.zeros(horizon),np.full(horizon,variance)


def kernel_diagnostics(theta,lag):
    if 0.<theta[0]<.5:
        from dynamic import selected_kernel
        return selected_kernel(float(theta[0]),math.exp(theta[1]),lag)[2]
    return {'selection':'exact_covariance_family_no_fractional_approximation','covariance_family':{-1.:'none',0.:'white',.5:'ou'}[float(theta[0])],
            'factors':0 if theta[0]==-1. else 1,'autocorrelation_error_upper_bound':0.}



@dataclass(frozen=True)
class Candidate:
    model_id:str
    fit_seconds:float=10.
    covariance_family:str='ou'

    def simulate_daily_log_returns(self,training,context):
        training.validate();past,future=shell._validate_calendar(training,context)
        assets=np.asarray(training.asset_log_returns,np.float64);sims=context.simulations;horizon=context.horizon_days
        data=[assets[:,a].tobytes() for a in range(assets.shape[1])];lag=len(assets)+horizon-1
        started=time.perf_counter()
        def fit_asset(d):return refit(d),endpoint_fit(d,lag,self.fit_seconds,self.covariance_family)
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
        from streamed_paths import map_asset_inplace

        from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
            moment_return_curves,
        )
        module,dot=load_paths()
        for a,(d,((original,fit),rough_fit)) in enumerate(zip(data,posteriors)):
            identity=hashlib.sha256(d).hexdigest();theta=rough_fit['map']
            phi,w,root,terminal,initial_root,rm,rv=prepared(d,theta.tobytes(),lag,horizon)
            mean,_=moment_return_curves(learned.student.original_fit(d),horizon)
            mp,c,sd,initial,level,mm,mv,return_sd=configuration(d,horizon)
            base_paths=np.ascontiguousarray(uniforms[:,:,a]);map_asset_inplace(base_paths,mean,return_sd,learned.nodes(d,sims))
            rng=np.random.default_rng(shell.bd.deterministic_seed('rough_overlay_prediction',identity,str(context.origin_date),horizon,sims))
            memory_rng=np.random.default_rng(shell.bd.deterministic_seed('coupled_multiscale_prediction',identity,str(context.origin_date),horizon,sims))
            for s in range(sims):
                rinitial=terminal+initial_root@rng.normal(size=len(phi))
                rinitial=phi*rinitial+root@rng.normal(size=len(phi))
                module.map_path(phi,w,root,rinitial,rm,rv,rng.bit_generator.capsule,dot,mean,base_paths[s],paths[s,:,a],mp,c,sd,initial,level,mm,mv,memory_rng.bit_generator.capsule,True)
            FIT_DIAGNOSTICS[self.model_id,identity]={'model_id':self.model_id,'data_sha256':identity,
                'coupled_multiscale_fit':fit,'student_return_laplace_fit':original['student_return_laplace_fit'],
                'student_log_square_moments':dict(zip(('mean','variance'),pathwise.parent.log_square_moments(original['student_return_laplace_fit']['inverse_df']))),
                'rough_parameters':theta.tolist(),'rough_fit':{k:v for k,v in rough_fit.items() if k not in ('map','points','weights')},
                'kernel':kernel_diagnostics(theta,lag),
                'return_mean_curve':'unchanged_original_predecessor','parameter_uncertainty':False,
                'conventional_state_uncertainty':'deterministic last fitted proxy state; stochastic coupled future recursion',
                'conventional_normalization':'conditional Gaussian second moment one; return variance curve from same coupled state law',
                'rough_normalization':'unchanged M237 stationary Gaussian normalization','residual_covariance_family':self.covariance_family}
        shell.controls.timed('asset_predictive_paths',started)
        dates=shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        return shell.rejoin(paths,training.policy.weights,np.asarray([date in dates for date in future]))


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers);load_paths()
    state.innovations(np.arange(20.,dtype=float),np.empty(0),np.array([.5]))
    state.moments(np.array([[.5]]),np.zeros(1),np.ones(1),2)


CANDIDATES=tuple(Candidate(model_id=model,covariance_family=kind) for kind,model in (
 ('none','asset_map_stationary_ar1_without_residual_overlay'),
 ('white','asset_map_stationary_ar1_white_residual_volatility'),
 ('ou','asset_map_stationary_ar1_ou_residual_volatility'),
 ('fractional','asset_map_stationary_ar1_fractional_control')))
MODEL_IDS=tuple(c.model_id for c in CANDIDATES)
clear_path_cache=parent.clear_path_cache
