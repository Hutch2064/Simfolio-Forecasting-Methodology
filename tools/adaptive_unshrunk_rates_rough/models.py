"""Matched learned-rate control without signal-median loading shrinkage."""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('adaptive_unshrunk_rates_rough_private_learned_rates',ROOT/'tools/learned_multiscale_rates_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
parent.fit_rates=lambda h, rates: parent.rate_fit.fit_adaptive(h, rates, shrink_loadings=False)
original_components=parent.components
parent.components=lambda h, rates, **kwargs: original_components(h, rates, shrink_loadings=False, **kwargs)
parent.SOURCE=hashlib.sha256((parent.SOURCE+Path(__file__).read_text()).encode()).hexdigest()
initialize=parent.initialize;Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_adaptive_unshrunk_multiscale_rates_untruncated_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
