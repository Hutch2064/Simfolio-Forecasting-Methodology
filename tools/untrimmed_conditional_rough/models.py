"""Untrimmed log-square QL ablation of conditional empirical-noise rough SV."""
from functools import lru_cache
import hashlib,importlib.util,math
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('untrimmed_rough_private',ROOT/'tools/conditional_empirical_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
base,noise=parent.base,parent.noise
SOURCE=hashlib.sha256(parent.SOURCE.encode()+b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()
base.SOURCE=noise.SOURCE=SOURCE

def log_square(values):
    x=np.asarray(values,np.float64)
    squared=((x-float(x.mean()))*100)**2
    # Only protect log(0); no empirical quantile flooring or tail trimming.
    return np.log(np.maximum(squared,np.finfo(np.float64).tiny))-base.shell.bd.SV_LOG_CHI_SQUARE_MEAN

@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    original,eps=base.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    level,phi,eta,_,_=base.predecessor_fit(data)['posterior_center']
    return log_square(x)-base.causal_log_variance_predictor(original,level,phi,eta),eps

@lru_cache(maxsize=64)
def measurement_scale(data):
    pool=base.predecessor_fit(data)['innovation_pool']
    variance=float(np.var(log_square(pool*.01),ddof=1))
    if not np.isfinite(variance) or variance<=0:raise ValueError('nonpositive log-square noise variance')
    return math.sqrt(4.934802200544679/variance),variance

base.observed=noise.observed=observed
noise.measurement_scale=measurement_scale
prepared=parent.prepared
initialize=parent.initialize
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_multiscale_untrimmed_conditional_residual_empirical_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
