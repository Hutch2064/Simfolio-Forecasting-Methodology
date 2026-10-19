"""Prepared research adapters only. No import-time initialization or queue.

Selected RMS/variance-only-corrected-node base; conditional mean and/or latent
covariance emission change, with exact existing runtime/calendar/accounting.
"""
from contextlib import nullcontext
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
KINDS=('covariance_only','mean_only','combined','selected_control')
_M=None;_A=None


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


NATIVE=load('conditional_candidate_native',HERE/'native.py')
MARGINAL=load('conditional_candidate_marginal',HERE/'marginal.py')
COPULA=load('conditional_candidate_copula',HERE/'copula.py')
MATH=load('conditional_candidate_math',HERE.parent/'conditional_integration_design/math_helpers.py')


def initialize():
    global _M,_A
    if _M is not None:return
    model=load('conditional_candidate_selected',HERE.parents[1]/'innovation_mean_variance/models.py')
    adapter=load('conditional_candidate_runtime',HERE.parents[1]/'runtime/adapter.py')
    model.initialize();NATIVE.load_paths()
    adapter.single_growth(np.ones((1,1,1)),np.zeros(1,dtype=np.bool_),.0015)
    _M=model;_A=adapter


def _sha(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def forecast(task,kind,runtime=True):
    if _M is None:raise RuntimeError('explicit dedicated-worker initialize required; no queue is installed')
    if kind not in KINDS:raise ValueError(kind)
    owner=_M.REF.MODELS
    with _A.hooks(owner) if runtime else nullcontext() as shell:
        paths,stages,diagnostics=_forecast(task,kind)
        if runtime:stages.update(shell.timings)
    return paths,stages,diagnostics


def _forecast(task,kind):
    task.training.validate();c=_M.BASE.context(task)
    owner=_M.REF.MODELS;body=owner.parent.body;shell=owner.shell
    past,future=shell._validate_calendar(task.training,c)
    assets=np.asarray(task.training.asset_log_returns,np.float64)
    data=[assets[:,a].tobytes() for a in range(assets.shape[1])]
    groups,hashes=MATH.history_groups(data);groups=np.ascontiguousarray(groups,dtype=np.int64)
    stages={};tick=time.perf_counter()
    for d in data:body.refit(d)
    stages['conventional_fit_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
    configurations=[body.configuration(d,c.horizon_days) for d in data]
    stages['retained_configuration_curve_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
    variances=np.column_stack([x[6] for x in configurations])
    if not np.isfinite(variances).all() or np.any(variances<0):raise ArithmeticError('invalid fitted Gaussian state variance')
    attenuations=np.exp(-variances/8.)
    stages['analytic_attenuation_curve_seconds']=time.perf_counter()-tick
    covariance=kind in ('covariance_only','combined');mean_adjustment=kind in ('mean_only','combined')
    control=shell.controls;original_uniforms=control.gaussian_uniforms;copula_diagnostics={}

    def integrated(shape,raw,sims,horizon,seed):
        if shape!=assets.shape or raw!=assets.tobytes() or sims!=c.simulations or horizon!=c.horizon_days:
            raise AssertionError('conditional copula input context changed')
        u,d=COPULA.uniforms(owner.parent,NATIVE.load_paths(),shape,raw,sims,horizon,seed,attenuations,groups,stages)
        copula_diagnostics.update(d);return u

    tick=time.perf_counter()
    if covariance:control.gaussian_uniforms=integrated
    try:
        # Preserve the original _return_uniforms dimension-one identity handler.
        # It never calls gaussian_uniforms for one holding; no extra cDCC fit.
        uniforms=_M.REF.REFERENCE._return_uniforms(assets,c)
    finally:control.gaussian_uniforms=original_uniforms
    stages['dependence_seconds']=time.perf_counter()-tick
    from streamed_paths import map_asset_inplace
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    paths=np.empty_like(uniforms);records=[];tick=time.perf_counter();compensation_seconds=0.
    for a,d in enumerate(data):
        mean,_=moment_return_curves(body.learned.student.original_fit(d),c.horizon_days)
        return_sd=configurations[a][7]
        original_nodes=body.learned.nodes(d,c.simulations)
        nodes=_M.transform(original_nodes,'variance_only_preserve_mean')
        base=np.ascontiguousarray(uniforms[:,:,a]);map_asset_inplace(base,mean,return_sd,nodes)
        delta=None
        if mean_adjustment:
            start=time.perf_counter();delta=MARGINAL.compensation(mean,return_sd,nodes,variances[:,a])
            compensation_seconds+=time.perf_counter()-start
        paths[:,:,a]=MARGINAL.finish(base,mean,delta)
        record={'training_sha256':hashes[a],'history_group':int(groups[a]),
            'original_mean_sha256':_sha(mean),'original_return_sd_sha256':_sha(return_sd),
            'original_nodes_sha256':_sha(original_nodes),'selected_corrected_nodes_sha256':_sha(nodes),
            'state_variance_sha256':_sha(variances[:,a]),'d_sha256':_sha(attenuations[:,a]),
            'state_variance_first_terminal':variances[[0,-1],a].tolist(),
            'expected_multiplier_first_terminal':attenuations[[0,-1],a].tolist()}
        if delta is not None:record.update(mean_compensation_sha256=_sha(delta),
            mean_compensation_first_terminal=delta[[0,-1]].tolist(),
            mean_compensation_max_abs=float(np.max(np.abs(delta))),
            mean_compensation_horizon_average=float(delta.mean()))
        records.append(record)
    stages['asset_paths_seconds']=time.perf_counter()-tick
    stages['mean_compensation_curve_seconds']=compensation_seconds;tick=time.perf_counter()
    dates=shell._historical_rebalance_dates(past.append(future),task.training.policy.rebalance)
    result=shell.rejoin(paths,task.training.policy.weights,np.asarray([d in dates for d in future]))
    stages['accounting_seconds']=time.perf_counter()-tick
    return result,stages,{'kind':kind,'assets':records,'copula':copula_diagnostics,
        'copula_hook_restored':control.gaussian_uniforms is original_uniforms,
        'mean_scope':'clipped-knot continuous input deviation; add after mapper, final clip unchanged',
        'no_exact_final_clip_mean_or_mapped_covariance_integration_claim':True}
