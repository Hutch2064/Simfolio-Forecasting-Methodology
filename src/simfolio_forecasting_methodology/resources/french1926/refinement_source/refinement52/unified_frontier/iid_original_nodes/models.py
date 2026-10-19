"""One node-only contrast on the frozen all-volatility IID post-finish law."""
from contextlib import contextmanager,nullcontext
import importlib.util
import inspect
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
KIND='iid_conditional_unit_second_moment_scale_original_nodes'
IID=None
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def initialize():
    global IID
    if IID is not None:return
    module=load('original_nodes_sole_existing_IID',HERE.parent/'iid_conditional_scale_candidates/models.py')
    module.initialize();IID=module

@contextmanager
def original_nodes():
    C=IID._T.G._C;model=C._M;original=model.transform;calls=[]
    def retained(nodes,mode):
        frame=inspect.currentframe().f_back
        try:assert frame.f_code is C._forecast.__code__ and mode=='variance_only_preserve_mean','unexpected node-transform boundary'
        finally:del frame
        calls.append({'nodes_sha256':IID.sha(nodes),'same_array_object_returned':True});return nodes
    model.transform=retained
    try:yield calls
    finally:model.transform=original

def forecast(task,runtime=True):
    if IID is None:raise RuntimeError('explicit sole-worker initialization required; no admission or queue')
    with original_nodes() as calls:paths,stages,diagnostics=IID.forecast(task,runtime=runtime)
    n=task.training.asset_log_returns.shape[1]
    assert len(calls)==len(diagnostics['assets'])==n
    for call,asset in zip(calls,diagnostics['assets']):
        assert call['nodes_sha256']==asset['original_nodes_sha256']==asset['selected_corrected_nodes_sha256'],'mapped original nodes changed'
    diagnostics['kind']=KIND;diagnostics['original_nodes_selector']={'selection':'literal original RMS return nodes; identity at sole selected-control transform','calls':calls,'transform_restored':True,'point_SD_dependence_IID_seed_and_clip_order_unchanged':True}
    diagnostics['iid_conditional_scale']['law']='clip(m+(g_original_RMS-m)*exp(.5*sqrt(vc+vr)*Z-(vc+vr)/4),-1,1); g includes original RMS finish/clip'
    return paths,stages,diagnostics

def original_RMS_control(task,runtime=True):
    """Same private stack and exact U/models.py RMS_original_nodes branch."""
    C=IID._T.G._C
    with C._A.hooks(C._M.REF.MODELS) if runtime else nullcontext():
        paths,stages,_=C._M.BASE.forecast(task,'deterministic_RMS')
    return paths,stages

def zero_variance_path_gate(task,runtime=True):
    """Future correctness gate only; no zero-variance candidate or selector."""
    owner=IID._T.G._C._M.REF.MODELS;draw=IID.SCALE.draw_group
    def unit(variance,*args,**kwargs):return draw(np.zeros_like(variance),*args,**kwargs)
    owner.clear_path_cache();IID.SCALE.draw_group=unit
    try:paths,_,_=forecast(task,runtime=runtime)
    finally:IID.SCALE.draw_group=draw
    owner.clear_path_cache();control,_=original_RMS_control(task,runtime=runtime)
    assert paths.shape==control.shape==(240,task.horizon_days) and paths.tobytes()==control.tobytes(),'unit-scale original RMS fullpath mismatch'
    return {'passed':True,'complete_paths_byte_equal':True,'path_values':int(paths.size),'correctness_control_only':True}
