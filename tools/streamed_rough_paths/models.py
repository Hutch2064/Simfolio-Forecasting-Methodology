"""Runtime-only M201 optimization: unchanged model, fitting, seeds and draw order."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import math,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from stream_native import load_paths
spec=importlib.util.spec_from_file_location('streamed_private_differenced',ROOT/'tools/differenced_whittle_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
base=parent.parent.base.base
shell=parent.shell
overlay=parent.overlay
predecessor_fit=base.predecessor_fit
rough_fit=base.rough_fit
predecessor_asset_paths=base.predecessor_asset_paths
prepared=base.prepared
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
TIMINGS=parent.TIMINGS


def stream_multiplier(phi,w,root,initial,means,variances,capsule):
    module,dot=load_paths()
    return module.multiplier(phi,w,root,initial,means,variances,capsule,dot)


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
            FIT_DIAGNOSTICS[self.model_id,identity]=dict(model_id=self.model_id,data_sha256=identity,
                conventional_estimator='original_predecessor_MAP',rough_estimator='MAP_point_conditional_on_causal_SV_prediction',normalization='stationary_second_moment_one',
                posterior_parameter_uncertainty=False,rough_parameters=theta.tolist(),kernel=kernel,
                rough_fit={k:v for k,v in rough.items() if k not in ('points','weights','map')})
            mean,base_paths=predecessor_asset_paths(d,uniforms[:,:,a])
            phi,w,root,state,initial_root,means,variances=prepared(d,theta.tobytes(),lag,horizon)
            rng=np.random.default_rng(shell.bd.deterministic_seed('rough_overlay_prediction',identity,
                str(context.origin_date),horizon,sims))
            for s in range(sims):
                initial=state+initial_root@rng.normal(size=phi.size)
                initial=phi*initial+root@rng.normal(size=phi.size)
                module,dot=load_paths()
                module.map_path(phi,w,root,initial,means,variances,rng.bit_generator.capsule,
                                dot,mean,base_paths[s],paths[s,:,a])
        shell.controls.timed('asset_predictive_paths',started)
        dates=shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        mask=np.asarray([date in dates for date in future])
        return shell.rejoin(paths,training.policy.weights,mask)


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    load_paths()


clear_path_cache=parent.clear_path_cache
CANDIDATES=(Candidate(model_id=parent.CANDIDATES[0].model_id),)
MODEL_IDS=parent.MODEL_IDS
