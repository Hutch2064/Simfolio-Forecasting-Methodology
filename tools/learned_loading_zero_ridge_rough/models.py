"""Matched M237 control removing only the fixed loading ridge penalty."""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
parent=load('zero_ridge_private_full_H',ROOT/'tools/learned_loading_full_hurst_rough/models.py')
rate_fit=load('zero_ridge_private_fit',HERE/'rate_fit_zero.py')
construction=load('zero_ridge_private_components',HERE/'components_zero.py')
learned=parent.parent.parent

def fit_rates(h,rates):
    fitted,receipt=rate_fit.fit_adaptive_loading(h,rates)
    receipt['ridge_scale']=0.
    return fitted,receipt

learned.fit_rates=fit_rates;learned.components=construction.components
learned.SOURCE=hashlib.sha256((learned.SOURCE+''.join(p.read_text() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()
initialize=parent.initialize;Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_adaptive_predictive_loading_full_hurst_zero_ridge_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
