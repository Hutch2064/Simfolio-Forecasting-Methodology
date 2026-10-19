"""M242 forecasting shell with stationary one-state proxy likelihood."""
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('stationary_ar1_private_M242',ROOT/'tools/coupled_multiscale_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
spec=importlib.util.spec_from_file_location('stationary_ar1_private_estimator',HERE/'state.py')
estimator=importlib.util.module_from_spec(spec);sys.modules[spec.name]=estimator;spec.loader.exec_module(estimator)
# Replace only this private instance's estimator, preserving shared state helpers.
parent.state=SimpleNamespace(fit=estimator.fit,transition=parent.state.transition,
                             moments=parent.state.moments,innovations=parent.state.innovations)
parent.SOURCE=hashlib.sha256((parent.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()
Candidate=parent.Candidate;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;initialize=parent.initialize;load_paths=parent.load_paths
clear_path_cache=parent.clear_path_cache
CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_proxy_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
