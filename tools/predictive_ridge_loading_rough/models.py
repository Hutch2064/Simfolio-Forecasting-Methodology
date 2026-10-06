"""M237 with identifiable data-selected loading ridge; forecasting shell retained."""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('predictive_ridge_private_full_H',ROOT/'tools/learned_loading_full_hurst_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
spec=importlib.util.spec_from_file_location('predictive_ridge_private_fit',HERE/'ridge_fit.py')
ridge_fit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ridge_fit;spec.loader.exec_module(ridge_fit)
learned=parent.parent.parent
learned.fit_rates=ridge_fit.fit_adaptive
learned.SOURCE=hashlib.sha256((learned.SOURCE+''.join(p.read_text() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()
initialize=parent.initialize;Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_adaptive_predictive_ridge_loading_full_hurst_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
