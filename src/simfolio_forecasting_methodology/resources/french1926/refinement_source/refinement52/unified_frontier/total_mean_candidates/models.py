"""One total-K plus full conditional total-mean law. No import-time learner."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
KIND='total_kernel_full_conditional_total_mean'
G=None
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
MEAN=load('full_total_mean_pure',HERE/'mean.py')
def sha(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def initialize():
    global G
    if G is not None:return
    module=load('full_total_mean_sole_rough_stack',HERE.parent/'rough_moment_candidates/models.py')
    module.initialize();G=module

def prepare(task,stack):
    """One retained rough preparation per exact history, even when K is one."""
    assets=np.asarray(task.training.asset_log_returns,np.float64);h=task.horizon_days
    histories=[assets[:,j].tobytes() for j in range(assets.shape[1])]
    groups,hashes=stack._C.MATH.history_groups(histories)
    variance=np.empty((h,len(histories)));drough=np.empty_like(variance)
    records=[];stages={};lag=len(assets)+h-1
    for group in np.unique(groups):
        columns=np.flatnonzero(groups==group);j=int(columns[0]);captured=[]
        original=stack._REFS['normalizer']
        def normalizer(*args):
            mean,var=original(*args);captured.append(np.maximum(np.asarray(var),0.));return mean,var
        refs={**stack._REFS,'normalizer':normalizer}
        d,record,timing=stack.MOMENTS.prepare(histories[j],lag,h,refs)
        if len(captured)!=1 or captured[0].shape!=(h,):raise AssertionError('rough normalizer must provide exactly one complete curve')
        v=captured[0]
        if sha(v)!=record['conditional_variance']['sha256'] or not np.array_equal(d,np.exp(-v/8.)):
            raise AssertionError('shared rough mean and covariance curves diverged')
        variance[:,columns]=v[:,None];drough[:,columns]=d[:,None]
        record['history_group']=int(group);record['asset_columns']=columns.tolist();records.append(record)
        for name,value in timing.items():stages[name]=stages.get(name,0.)+value
    return {'assets':assets,'histories':histories,'groups':np.ascontiguousarray(groups,dtype=np.int64),
        'training_sha256':hashes,'rough_variance':variance,'rough_attenuations':drough,
        'rough_moments':records,'stages':stages,'rough_fits_requested':len(records)}

@contextmanager
def shared_law(task,stack,prepared):
    C=stack._C;copula=C.COPULA.uniforms;compensation=C.MARGINAL.compensation
    assets=prepared['assets'];groups=prepared['groups'];h=task.horizon_days;n=assets.shape[1]
    state={'copula_calls':0,'mean_calls':0,'assets':[],'total_attenuation_sha256':None,
        '_omitted_ratio_values':[],'omitted_variance_diagnostic_seconds':0.}
    def emit(dependence,backend,shape,raw,sims,horizon,seed,dc,input_groups,stages):
        if shape!=assets.shape or raw!=assets.tobytes() or horizon!=h or not np.array_equal(input_groups,groups):
            raise AssertionError('total mean copula context changed')
        state['copula_calls']+=1
        if state['copula_calls']!=1:raise AssertionError('total covariance must be applied exactly once')
        dtotal=dc*prepared['rough_attenuations'];state['total_attenuation_sha256']=sha(dtotal)
        return copula(dependence,backend,shape,raw,sims,horizon,seed,dtotal,input_groups,stages)
    def location(mean,sd,nodes,vc):
        j=state['mean_calls'];state['mean_calls']+=1
        if j>=n:raise AssertionError('one original marginal compensation call per asset required')
        delta,total=MEAN.compensation(mean,sd,nodes,vc,prepared['rough_variance'][:,j])
        tick=time.perf_counter();ratio=MEAN.omitted_variance_ratio(mean,sd,nodes,total)
        state['omitted_variance_diagnostic_seconds']+=time.perf_counter()-tick
        state['_omitted_ratio_values'].append(ratio)
        state['assets'].append({'asset_column':j,'training_sha256':prepared['training_sha256'][j],
            'conventional_variance_sha256':sha(vc),'rough_variance_sha256':sha(prepared['rough_variance'][:,j]),
            'total_variance_sha256':sha(total),'total_expected_multiplier_first_terminal':np.exp(-total[[0,-1]]/8.).tolist(),
            'full_total_compensation_sha256':sha(delta),'full_total_compensation_first_terminal':delta[[0,-1]].tolist(),
            'full_total_compensation_horizon_average':float(delta.mean()),
            'omitted_variance_ratio':MEAN.ratio_summary(ratio)})
        return delta
    C.COPULA.uniforms=emit;C.MARGINAL.compensation=location
    try:yield state
    finally:
        C.COPULA.uniforms=copula;C.MARGINAL.compensation=compensation
        state['copula_and_mean_hooks_restored']=C.COPULA.uniforms is copula and C.MARGINAL.compensation is compensation

def forecast(task,runtime=True):
    if G is None:raise RuntimeError('explicit sole-worker initialize required; no queue is installed')
    tick=time.perf_counter();prepared=prepare(task,G);preparation_seconds=time.perf_counter()-tick
    with shared_law(task,G,prepared) as state:
        paths,stages,diagnostics=G._C.forecast(task,'combined',runtime=runtime)
    n=prepared['assets'].shape[1]
    if state['mean_calls']!=n or state['copula_calls']!=(0 if n==1 else 1):
        raise AssertionError('frozen marginal/copula call topology changed')
    if len(diagnostics['assets'])!=n:raise AssertionError('frozen marginal diagnostic topology changed')
    for j,(record,original) in enumerate(zip(state['assets'],diagnostics['assets'])):
        if record['training_sha256']!=original['training_sha256'] or record['conventional_variance_sha256']!=original['state_variance_sha256']:
            raise AssertionError('mean adjustment used a different original marginal context')
        if record['full_total_compensation_sha256']!=original['mean_compensation_sha256']:
            raise AssertionError('mean adjustment was not emitted unchanged')
    stages.update(prepared['stages']);stages['total_rough_preparation_wall_seconds']=preparation_seconds
    stages['omitted_variance_diagnostic_seconds']=state['omitted_variance_diagnostic_seconds']
    ratios=state.pop('_omitted_ratio_values')
    state['omitted_variance_ratio']=MEAN.ratio_summary(np.concatenate(ratios))
    diagnostics['kind']=KIND;diagnostics['full_total_mean']={**state,
        'history_groups':prepared['groups'].tolist(),'rough_fits_requested':prepared['rough_fits_requested'],
        'rough_moments':prepared['rough_moments'],'rough_variance_matrix_sha256':sha(prepared['rough_variance']),
        'mean_law':'clipped-knot deviation b * expm1(-(vc+vr)/8), after mapper, existing final clip',
        'K_one_is_not_d_one':True,'future_SV_states_drawn':False,
        'scope':'unit-second-moment conditional Gaussian first-moment counterpart; no complete M261, postclip mean, variance or joint distribution parity'}
    return paths,stages,diagnostics
