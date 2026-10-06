"""Original MAP multiscale predecessor plus dynamic-resolution MAP rough volatility.

Research only: the mean, empirical innovations, dependence and portfolio policy
remain the predecessor's. The rough fit uses the winner's Gaussian log-square
quasi likelihood, priors and accuracy-controlled covariance approximation.
Parameter uncertainty is not integrated in this MAP candidate.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'tools/rough_bayesian'),str(ROOT/'tools/mcmc_runtime'),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('rough_inference_shell',ROOT/'tools/rough_bayesian/models.py')
shell=importlib.util.module_from_spec(spec);sys.modules[spec.name]=shell;spec.loader.exec_module(shell)
import overlay
from streamed_paths import map_asset_inplace
from posterior import fit_map,InferenceLimit

TIMINGS=shell.TIMINGS
FIT_DIAGNOSTICS={}
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    shell.controls.initialize(cache_root,vine_threads,state_workers)
    overlay.native_filter()
    map_asset_inplace(np.zeros((2,2)),np.zeros(2),np.ones(2),np.array([-1.,1.]))
    shell.rejoin(np.zeros((8,2,2)),[.5,.5],[False,True])
    shell.rejoin(np.zeros((8,1025,2)),[.5,.5],np.zeros(1025,dtype=bool))


def clear_path_cache():shell.clear_path_cache()


@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    return shell.bd._sv_observed_log_variance(x,float(x.mean()))


@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    identity=hashlib.sha256(data).hexdigest()
    contract=f'predecessor_dynamic_rough_MAP_v1:{lag}'
    key=hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y,_=observed(data)
        def target(theta):
            prior=overlay.log_prior(theta)
            if not np.isfinite(prior):return -np.inf
            phi,w,q=overlay.configuration(theta,f'dynamic:{lag}')
            return prior+overlay.filter_rough(y,phi,w,q,float(y.mean()))[0]
        result=fit_map(target,[.1,math.log(1/63),math.log(.7)],
            [(.03,.49),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3))],seconds)
        result.update(contract=contract,data_sha256=identity)
        return result
    started=time.perf_counter()
    result=shell.controls.cache('predecessor_MAP_rough_fit',key,build)
    shell.controls.timed('rough_fit_worker',started)
    return result


@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    y,_=observed(data);t=np.frombuffer(theta_bytes,np.float64)
    phi,w,q=overlay.configuration(t,f'dynamic:{lag}')
    state,p=overlay.terminal_filter(y,phi,w,q,float(y.mean()))
    eigen,vectors=np.linalg.eigh((p+p.T)/2)
    initial_root=vectors*np.sqrt(np.maximum(eigen,0))
    root=np.diag(np.sqrt(np.diag(q)))
    next_p=p*phi[:,None]*phi[None,:]+q
    means,variances=overlay.path_normalizers(phi,w,root,next_p,phi*state,horizon)
    return phi,w,root,state,initial_root,means,variances


@lru_cache(maxsize=64)
def predecessor_fit(data):
    """Use the actual promoted predecessor; do not substitute another SV fit."""
    source=hashlib.sha256(Path(shell.bd.__file__).read_bytes()).hexdigest()
    key=hashlib.sha256(data+(SOURCE+source).encode()).hexdigest()
    return shell.controls.cache('map_predecessor_asset',key,lambda:
        shell.bd.fit_bdes_fastmap(np.frombuffer(data,np.float64),
            filtered_innovations=True,fixed_mean=True))


def predecessor_asset_paths(data,uniforms):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    fit=predecessor_fit(data)
    # Reuse the same curves instead of computing all multiscale moments twice.
    mean,sd=moment_return_curves(fit,uniforms.shape[1])
    simulations=uniforms.shape[0]
    probability=np.linspace(.5/simulations,1.-.5/simulations,simulations)
    nodes=np.quantile(fit['innovation_pool'],probability)
    nodes-=nodes.mean()
    nodes/=np.sqrt(np.mean(nodes*nodes))
    # The harness's exact compiled mapper avoids dense quantile/interpolation
    # temporaries and a redundant sort. It clips each endpoint before mixing,
    # preserving the predecessor's marginal law even at the clipping limits.
    paths=uniforms.copy()
    map_asset_inplace(paths,mean,sd,nodes)
    return mean,paths


@dataclass(frozen=True)
class PredecessorRoughMAP:
    """Exact MAP predecessor plus a normalized, dynamically resolved rough layer.

    Both component estimates are MAP points, not parameter-posterior draws.
    The rough historical Gaussian states are integrated by the existing filter;
    their conditional terminal uncertainty and future innovations are simulated.
    The existing Gaussian log-square proxy is a quasi likelihood for returns.
    """
    model_id:str='asset_map_predecessor_dynamic_rough_map'
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
        from dynamic import selected_kernel
        for a,(d,(_,rough)) in enumerate(zip(data,posteriors)):
            identity=hashlib.sha256(d).hexdigest()
            theta=rough['map']
            kernel=selected_kernel(float(theta[0]),math.exp(theta[1]),lag)[2]
            FIT_DIAGNOSTICS[self.model_id,identity]=dict(model_id=self.model_id,data_sha256=identity,
                conventional_estimator='original_predecessor_MAP',rough_estimator='MAP_point',
                posterior_parameter_uncertainty=False,rough_parameters=theta.tolist(),kernel=kernel,
                rough_fit={k:v for k,v in rough.items() if k not in ('points','weights','map')})
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
CANDIDATES=(PredecessorRoughMAP(),)
MODEL_IDS=(shell.MODEL_IDS[0],CANDIDATES[0].model_id)
