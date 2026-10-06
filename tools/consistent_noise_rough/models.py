"""M210 with empirical observation noise in its causal conventional offset.

The original conventional parameter fit and return generator remain unchanged.
This is a two-stage plug-in Gaussian quasi-likelihood experiment, not a joint
raw-return likelihood. No additional parameter is introduced.
"""
from functools import lru_cache
import hashlib,importlib.util
from pathlib import Path
import sys
import numpy as np
from numba import njit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('consistent_noise_private_m210',ROOT/'tools/relaxed_hurst_differenced_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
parent.SOURCE=hashlib.sha256((parent.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()

@njit(cache=True,nogil=True)
def causal_predictor(y,level,phi,eta,noise):
    result=np.empty(y.size);mean=level
    variance=max(eta*eta/max(1-phi*phi,1e-4),1e-6)
    for t in range(y.size):
        result[t]=mean
        gain=variance/(variance+noise)
        mean=level+phi*(mean+gain*(y[t]-mean)-level)
        variance=phi*phi*max((1-gain)*variance,1e-8)+eta*eta
    return result

@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    y,eps=parent.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    level,phi,eta,_,_=parent.streamed.predecessor_fit(data)['posterior_center']
    _,noise=parent.base.noise.measurement_scale(data)
    return y-causal_predictor(y,level,phi,eta,noise),eps

parent.base.base.observed=observed
parent.base.noise.observed=observed

class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        for key,record in FIT_DIAGNOSTICS.items():
            if key[0]==self.model_id:
                record['conventional_offset_noise']='same empirical log-square innovation variance as rough fit'
                record['conventional_parameter_fit']='unchanged original Gaussian-noise MAP; two-stage plug-in offset only'
        return result

initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_multiscale_consistent_empirical_noise_differenced_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
