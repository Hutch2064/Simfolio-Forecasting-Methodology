"""M237 conventional fits with one joint Gaussian rough/multiscale volatility law.

Conditional MAP calibrates rough covariance plus retained conventional and
Student measurement covariances on the raw log-square proxy. All states are
conditioned jointly. Future sigma comes directly from the combined log state;
direct mode uses no deterministic volatility curve; two normalization controls
retain M237 predictive variance with the same joint state dynamics.
This is modular Gaussian-proxy inference, not joint raw-return Bayesian inference.
"""
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
sys.path.insert(0,str(HERE))
from covariance import expected_periodogram
from joint_native import load_paths

spec=importlib.util.spec_from_file_location('joint_gaussian_private_m237',ROOT/'tools/learned_loading_full_hurst_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
rough=parent.rough;learned=parent.parent.parent;pathwise=rough.parent
shell=parent.shell;overlay=parent.overlay
SOURCE=hashlib.sha256((rough.SOURCE+learned.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*')) if p.suffix in ('.py','.cpp'))).encode()).hexdigest()
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS


@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    y,_=shell.bd._sv_observed_log_variance(x,float(x.mean()))
    fit=learned.refit(data)
    bias,noise=pathwise.parent.log_square_moments(fit['student_return_laplace_fit']['inverse_df'])
    gaussian_bias,_=pathwise.parent.log_square_moments(0.)
    return y+gaussian_bias-bias,noise


@lru_cache(maxsize=64)
def conventional(data):
    phi,w,common,independent,_,level,_,_=pathwise.multiscale_configuration(data,1)
    return phi,w,common,independent,np.outer(common,common)+np.outer(independent,independent),level


def target(theta,periodogram,n,fixed_spectrum):
    prior=rough.log_prior(theta)
    if not np.isfinite(prior):return -np.inf
    if theta[1]<math.log(np.nextafter(0.,1.)) or theta[1]>math.log(np.finfo(float).max)-math.log(n) or 2*theta[2]>math.log(np.finfo(float).max):return -np.inf
    with np.errstate(over='ignore',invalid='ignore'):
        expected=math.exp(2*theta[2])*rough.backend.unit_expected_periodogram(float(theta[0]),float(theta[1]),n)+fixed_spectrum
    if not np.isfinite(expected).all() or (expected<=0).any():return -np.inf
    terms=np.log(expected)+periodogram/expected;terms[0]*=.5
    if n%2==0:terms[-1]*=.5
    return prior-float(terms.sum())


@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    key=hashlib.sha256((SOURCE+hashlib.sha256(data).hexdigest()+str(lag)).encode()).hexdigest()
    def build():
        y,noise=observed(data);n=len(y)-1
        differences=np.diff(y);periodogram=np.abs(np.fft.rfft(differences))**2/n
        phi,w,_,_,q,_=conventional(data)
        conventional_spectrum=expected_periodogram(phi,w,q,n)
        fixed=conventional_spectrum+noise*rough.backend.noise_expected_periodogram(n)
        result=rough.fit_map(lambda theta:target(theta,periodogram,n,fixed),seconds)
        result.update(estimator='conditional_MAP_joint_conventional_rough_covariance_differenced_debiased_Whittle_proxy',
                      data_sha256=hashlib.sha256(data).hexdigest(),measurement_noise_variance=noise,
                      H_support='theoretical open (0, 1/2)',rough_prior_support='untruncated proper normals on log kappa and log scale; original centers and widths',
                      conventional_components=len(phi)-1,frequency_count=len(periodogram),
                      observed='raw Student-debiased clipped log-square proxy; no conventional predictor offset',
                      conventional_parameters='unchanged plug-in M237 fits; no joint parameter posterior')
        return result
    started=time.perf_counter();result=shell.controls.cache('joint_gaussian_rough_MAP',key,build)
    shell.controls.timed('rough_fit_worker',started);return result


@lru_cache(maxsize=64)
def prepared(data,theta_bytes,lag):
    y,noise=observed(data);theta=np.frombuffer(theta_bytes,np.float64)
    cp,cw,common,independent,cq,level=conventional(data)
    rp,rw,rq=overlay.configuration(theta,f'dynamic:{lag}')
    if np.count_nonzero(rq-np.diag(np.diag(rq))):raise ArithmeticError('joint forecast requires independent lifted rough innovations')
    phi=np.r_[cp,rp];w=np.r_[cw,rw]
    q=np.zeros((len(phi),len(phi)));q[:len(cp),:len(cp)]=cq;q[len(cp):,len(cp):]=rq
    scale=math.sqrt(4.934802200544679/noise)
    state,p=overlay.terminal_filter(y*scale,phi,w,q*scale**2,level*scale)
    state=state/scale;p=p/scale**2
    eigen,vectors=np.linalg.eigh((p+p.T)*.5)
    tolerance=256*np.finfo(float).eps*len(phi)*max(float(np.max(np.abs(eigen))),1.)
    if eigen.min() < -tolerance:raise ArithmeticError('materially indefinite joint terminal covariance')
    return phi,w,np.sqrt(np.diag(rq)),state,vectors*np.sqrt(np.maximum(eigen,0.)),common,independent,level,p


@lru_cache(maxsize=64)
def normalizers(data,theta_bytes,lag,horizon,mode):
    phi,w,sd,state,_root,common,independent,level,p=prepared(data,theta_bytes,lag)
    if mode=='stationary':
        _,_,_,_,_,_,means,variances=pathwise.multiscale_configuration(data,horizon)
        theta=np.frombuffer(theta_bytes,np.float64);rp,rw,rq=overlay.configuration(theta,f'dynamic:{lag}')
        stationary=rq/(1-rp[:,None]*rp[None,:])
        return means,variances+float(rw@stationary@rw)
    qroot=np.zeros((len(phi),len(phi)));m=len(common)
    qroot[:m,0]=common;qroot[:m,1]=independent
    qroot[m:,m:]=np.diag(sd)
    q=qroot@qroot.T
    means,variances=overlay.path_normalizers(phi,w,qroot,p*phi[:,None]*phi[None,:]+q,phi*state,horizon)
    return means+level,variances


@dataclass(frozen=True)
class Candidate:
    model_id: str
    fit_seconds: float = 10.
    normalization: str = 'direct'

    def simulate_daily_log_returns(self,training,context):
        training.validate();past,future=shell._validate_calendar(training,context)
        assets=np.asarray(training.asset_log_returns,np.float64);sims=context.simulations;horizon=context.horizon_days
        data=[assets[:,a].tobytes() for a in range(assets.shape[1])];lag=len(assets)+horizon-1
        started=time.perf_counter()
        def fit_asset(d):return learned.refit(d),rough_fit(d,lag,self.fit_seconds)
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

        from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
            moment_return_curves,
        )
        module,_=load_paths()
        for a,(d,(fitted,fit)) in enumerate(zip(data,posteriors)):
            identity=hashlib.sha256(d).hexdigest();theta=fit['map']
            phi,w,sd,state,root,common,independent,level,_=prepared(d,theta.tobytes(),lag)
            mean,_=moment_return_curves(learned.student.original_fit(d),horizon)
            asset_uniforms=np.ascontiguousarray(uniforms[:,:,a]);nodes=learned.nodes(d,sims)
            if self.normalization=='direct':
                base_sd=norm_mean=norm_variance=np.empty(0)
            else:
                _,base_sd=moment_return_curves(fitted,horizon)
                norm_mean,norm_variance=normalizers(d,theta.tobytes(),lag,horizon,self.normalization)
            rng=np.random.default_rng(shell.bd.deterministic_seed('joint_gaussian_volatility_prediction',identity,str(context.origin_date),horizon,sims))
            for s in range(sims):
                initial=state+root@rng.normal(size=len(phi))
                module.map_path(phi,w,sd,initial,common,independent,level,mean,asset_uniforms[s],nodes,rng.bit_generator.capsule,paths[s,:,a],base_sd,norm_mean,norm_variance,self.normalization!='direct')
            FIT_DIAGNOSTICS[self.model_id,identity]={'model_id':self.model_id,'data_sha256':identity,'rough_parameters':theta.tolist(),
                'rough_fit':{k:v for k,v in fit.items() if k not in ('map','points','weights')},'kernel':selected_kernel(float(theta[0]),math.exp(theta[1]),lag)[2],
                'student_return_laplace_fit':fitted['student_return_laplace_fit'],
                'student_log_square_moments':dict(zip(('mean','variance'),pathwise.parent.log_square_moments(fitted['student_return_laplace_fit']['inverse_df']))),
                'learned_multiscale_decay_fit':fitted['learned_decay_fit'],'return_mean_curve':'unchanged_original_predecessor',
                'volatility_forecast':'single_joint_Gaussian_logvol_state_law_'+self.normalization,
                'parameter_uncertainty':False,'state_uncertainty':'joint terminal Gaussian covariance, including cross-block covariance',
                'normalization':self.normalization,
                'normalization_contract':{'direct':'direct exp(logvol/2)/100; no separate moment curve',
                   'conditional':'retained M237 variance curve; joint Gaussian multiplier has conditional second moment one',
                   'stationary':'retained M237 variance curve; conventional predictive and rough stationary Gaussian normalization'}[self.normalization]}
        shell.controls.timed('asset_predictive_paths',started)
        dates=shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        return shell.rejoin(paths,training.policy.weights,np.asarray([date in dates for date in future]))


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers);load_paths()


CANDIDATES=(Candidate(model_id='asset_map_joint_gaussian_multiscale_rough_direct_volatility'),
            Candidate(model_id='asset_map_joint_gaussian_multiscale_rough_conditional_normalized',normalization='conditional'),
            Candidate(model_id='asset_map_joint_gaussian_multiscale_rough_stationary_normalized',normalization='stationary'))
MODEL_IDS=(parent.MODEL_IDS[0],*(c.model_id for c in CANDIDATES))
clear_path_cache=parent.clear_path_cache
