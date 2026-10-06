"""Volatility covariance experiment: causal residual Gaussian supOU log volatility.

Load private adapter namespaces so the already scored candidates and baseline
remain unchanged. Only volatility inference/normalization changes.
"""
import hashlib
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
ROOT=HERE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

base=load('gamma_supou_private_base',ROOT/'tools/conditional_residual_rough/models.py')
noise=load('gamma_supou_private_noise',ROOT/'tools/empirical_rough_noise/models.py')
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for directory in
    (HERE,ROOT/'tools/conditional_residual_rough',ROOT/'tools/empirical_rough_noise')
    for p in sorted(directory.glob('*.py')))).hexdigest()
base.SOURCE=noise.SOURCE=SOURCE
# Clone the Gaussian filtering helpers; replacing the covariance must not
# mutate the already scored rough candidates in other namespaces.
private_overlay=load('gamma_supou_private_overlay',ROOT/'tools/rough_bayesian/overlay.py')
from gamma_kernel import configuration, log_prior

private_overlay.configuration=configuration
private_overlay.log_prior=log_prior
base.overlay=noise.overlay=private_overlay
noise.observed=base.observed
noise.predecessor_fit=base.predecessor_fit
noise.shell=base.shell
import time

from posterior import fit_map


@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    identity=hashlib.sha256(data).hexdigest()
    contract=f'Gamma_mixed_OU_residual_empirical_noise_MAP_v1:{lag}'
    key=hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y,_=base.observed(data);scale,R=noise.measurement_scale(data);scaled=y*scale
        def target(theta):
            prior=private_overlay.log_prior(theta)
            if not np.isfinite(prior):return -np.inf
            phi,w,q=private_overlay.configuration(theta,f'dynamic:{lag}')
            return prior+private_overlay.filter_rough(scaled,phi,w,q*scale**2,float(scaled.mean()))[0]
        result=fit_map(target,[0.,math.log(1/63),math.log(.7)],
            [(math.log(.03),math.log(10)),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3))],seconds)
        result.update(contract=contract,data_sha256=identity,measurement_noise_variance=R,estimator='MAP_Gaussian_Gamma_mixed_OU')
        return result
    started=time.perf_counter();result=base.shell.controls.cache('predecessor_MAP_rough_fit',key,build)
    base.shell.controls.timed('rough_fit_worker',started);return result

base.rough_fit=rough_fit

@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    y,_=base.observed(data);theta=np.frombuffer(theta_bytes,np.float64)
    phi,w,q=base.overlay.configuration(theta,f'dynamic:{lag}')
    scale,_=noise.measurement_scale(data)
    state,p=base.overlay.terminal_filter(y*scale,phi,w,q*scale**2,float(y.mean())*scale)
    state=state/scale;p=p/scale**2
    eigen,vectors=np.linalg.eigh((p+p.T)/2)
    initial_root=vectors*np.sqrt(np.maximum(eigen,0))
    root=np.diag(np.sqrt(np.diag(q)))
    stationary=q/(1-phi[:,None]*phi[None,:])
    return phi,w,root,state,initial_root,np.zeros(horizon),np.full(horizon,float(w@stationary@w))

base.prepared=prepared
initialize=base.initialize
clear_path_cache=base.clear_path_cache
TIMINGS=base.TIMINGS
FIT_DIAGNOSTICS=base.FIT_DIAGNOSTICS
shell=base.shell
overlay=base.overlay
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

predecessor_fit=base.predecessor_fit
predecessor_asset_paths=base.predecessor_asset_paths
@dataclass(frozen=True)
class PredecessorRoughMAP:
    """MAP predecessor plus a stationary-normalized Gaussian supOU layer.

    Both component estimates are MAP points, not parameter-posterior draws.
    Gaussian Gamma-mixed OU states describe residual log-volatility after causal conventional predictions;
    their conditional terminal uncertainty and future innovations are simulated.
    The existing Gaussian log-square proxy is a quasi likelihood for returns.
    """
    model_id:str='asset_map_multiscale_conditional_empirical_gamma_supou_map'
    fit_seconds:float=10.

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
        from gamma_kernel import selected_kernel
        for a,(d,(_,rough)) in enumerate(zip(data,posteriors)):
            identity=hashlib.sha256(d).hexdigest()
            theta=rough['map']
            kernel=selected_kernel(float(theta[0]),math.exp(theta[1]),lag)[2]
            FIT_DIAGNOSTICS[self.model_id,identity]={'model_id': self.model_id,'data_sha256': identity,
                'conventional_estimator': 'original_predecessor_MAP','volatility_estimator': 'MAP_Gaussian_Gamma_mixed_OU','normalization': 'stationary_second_moment_one',
                'posterior_parameter_uncertainty': False,'volatility_parameters': theta.tolist(),'kernel': kernel,
                'rough_fit': {k:v for k,v in rough.items() if k not in ('points','weights','map')}}
            mean,base_paths=predecessor_asset_paths(d,uniforms[:,:,a])
            phi,w,root,state,initial_root,means,variances=prepared(d,theta.tobytes(),lag,horizon)
            rng=np.random.default_rng(shell.bd.deterministic_seed('rough_overlay_prediction',identity,
                str(context.origin_date),horizon,sims))
            for s in range(sims):
                initial=state+initial_root@rng.normal(size=phi.size)
                initial=phi*initial+root@rng.normal(size=phi.size)
                normals=np.empty((horizon+1,phi.size));normals[0]=phi*state
                normals[1:]=rng.normal(size=(horizon,phi.size))
                multiplier=overlay.multiplier_independent_prepared(phi,w,root,initial,normals,means,variances)
                paths[s,:,a]=np.clip(mean+(base_paths[s]-mean)*multiplier,-1,1)
        shell.controls.timed('asset_predictive_paths',started)
        dates=shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        mask=np.asarray([date in dates for date in future])
        return shell.rejoin(paths,training.policy.weights,mask)

Candidate=PredecessorRoughMAP
CANDIDATES=(Candidate(model_id='asset_map_multiscale_conditional_empirical_gamma_supou_map'),)
MODEL_IDS=(base.MODEL_IDS[0],CANDIDATES[0].model_id)
