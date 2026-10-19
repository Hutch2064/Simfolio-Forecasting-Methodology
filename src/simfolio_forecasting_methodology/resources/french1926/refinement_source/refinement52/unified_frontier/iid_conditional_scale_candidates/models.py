"""Prepared ONE IID-scale contrast; no import-time learner or queue.

Numerical eligibility is conditional on current total-mean missing M261 on a
complete panel. This module does not admit or launch any numerical execution.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
KIND='iid_conditional_unit_second_moment_scale'
_T=None
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
SCALE=load('iid_conditional_scale_pure',HERE/'scale.py')
def sha(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def initialize():
    global _T
    if _T is not None:return
    module=load('iid_scale_sole_inherited_stack',HERE.parent/'total_mean_candidates/models.py')
    module.initialize();_T=module

@contextmanager
def iid_law(task,prepared,stack):
    C=stack._C;body=C._M.REF.MODELS.parent.body
    configuration=body.configuration;finish=C.MARGINAL.finish
    configuration_curves={};group_draws={};records=[];terminal_base=[];terminal_scaled=[]
    state={'assets':records,'group_draws':[],'finish_calls':0,'configuration_calls':0,'stages':{}}
    def observe(data,horizon):
        value=configuration(data,horizon);state['configuration_calls']+=1
        if horizon!=task.horizon_days or data not in prepared['histories']:raise AssertionError('different conventional context')
        if data in configuration_curves and not np.array_equal(configuration_curves[data],value[6]):raise AssertionError('repeated conventional variance changed')
        configuration_curves[data]=value[6];return value
    def scaled(mapped,mean,delta=None):
        j=state['finish_calls'];state['finish_calls']+=1
        if j>=len(prepared['histories']) or delta is not None:raise AssertionError('selected control only, one original finish per column')
        data=prepared['histories'][j];vc=np.asarray(configuration_curves[data],np.float64);vr=prepared['rough_variance'][:,j]
        if vc.shape!=vr.shape or not np.isfinite(vc).all() or np.any(vc<0):raise ArithmeticError('invalid original conventional curve')
        total=vc+vr;group=int(prepared['groups'][j]);base=finish(mapped,mean,None)
        tick=time.perf_counter()
        if group not in group_draws:
            s,logs,draw=SCALE.draw_group(total,base.shape[0],task.seed,task.origin_date,data)
            group_draws[group]=(s,logs,sha(total));state['group_draws'].append({'history_group':group,**draw})
        s,logs,total_sha=group_draws[group]
        if total_sha!=sha(total):raise AssertionError('exact duplicate total conditional variances differ')
        result,diag=SCALE.apply(base,mean,s,logs)
        state['stages']['iid_scale_draw_and_apply_seconds']=state['stages'].get('iid_scale_draw_and_apply_seconds',0.)+time.perf_counter()-tick
        terminal_base.append(base.sum(axis=1));terminal_scaled.append(result.sum(axis=1))
        records.append({'asset_column':j,'history_group':group,'training_sha256':prepared['training_sha256'][j],
            'point_mean_sha256':sha(mean),'conventional_variance_sha256':sha(vc),'rough_variance_sha256':sha(vr),'total_variance_sha256':total_sha,
            'mapped_paths_sha256':sha(base),'scaled_paths_sha256':sha(result),**diag})
        return result
    body.configuration=observe;C.MARGINAL.finish=scaled
    try:yield state
    finally:
        body.configuration=configuration;C.MARGINAL.finish=finish
        state['hooks_restored']=body.configuration is configuration and C.MARGINAL.finish is finish
        if terminal_base:
            state['mapped_asset_terminal_log_covariance']=np.atleast_2d(np.cov(np.column_stack(terminal_base),rowvar=False)).tolist()
            state['scaled_asset_terminal_log_covariance']=np.atleast_2d(np.cov(np.column_stack(terminal_scaled),rowvar=False)).tolist()

def forecast(task,runtime=True):
    if _T is None:raise RuntimeError('explicit sole-worker initialization requires later parent admission')
    tick=time.perf_counter();prepared=_T.prepare(task,_T.G);preparation_seconds=time.perf_counter()-tick
    C=_T.G._C
    # selected_control leaves original Student/Q/CDF emission unchanged and
    # supplies no deterministic location compensation or total-K hook.
    with iid_law(task,prepared,_T.G) as state:paths,stages,diagnostics=C.forecast(task,'selected_control',runtime=runtime)
    n=len(prepared['histories'])
    if state['finish_calls']!=n or len(state['group_draws'])!=len(set(prepared['groups'])):raise AssertionError('IID history/asset topology changed')
    if len(diagnostics['assets'])!=n:raise AssertionError('original marginal diagnostics changed')
    for new,old in zip(state['assets'],diagnostics['assets']):
        if new['training_sha256']!=old['training_sha256'] or new['point_mean_sha256']!=old['original_mean_sha256'] or new['conventional_variance_sha256']!=old['state_variance_sha256']:
            raise AssertionError('IID scale used different original training/point/variance curves')
    stages.update(prepared['stages']);stages.update(state.pop('stages'));stages['rough_preparation_wall_seconds']=preparation_seconds
    diagnostics['kind']=KIND;diagnostics['iid_conditional_scale']={**state,'rough_moments':prepared['rough_moments'],
        'rough_fits_requested':prepared['rough_fits_requested'],'future_SV_states_drawn':False,
        'law':'clip(m+(g_selected_RMS-m)*exp(.5*sqrt(vc+vr)*Z-(vc+vr)/4),-1,1)',
        'total_K_applied':False,'deterministic_mean_compensation_applied':False,
        'scope':'IID group scale changes marginal tails and cross-asset dependence; no original M261, serial-state, postclip mean/variance or CAGR parity'}
    return paths,stages,diagnostics
