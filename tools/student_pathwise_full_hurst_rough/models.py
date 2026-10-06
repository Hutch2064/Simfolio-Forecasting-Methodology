"""M221 with uniform H support over the full rough domain (0, 1/2).

Only H support changes. The likelihood, other priors, numerical accuracy,
Student fit, empirical nodes, multiscale future states and mean are retained.
"""
import hashlib
import importlib.util
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('student_full_hurst_private_m221',ROOT/'tools/pathwise_multiscale_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
backend=parent.parent.parent.parent.parent
SOURCE=hashlib.sha256((backend.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


def log_prior(theta):
    if not (0.<theta[0]<.5 and math.log(1/2520)<theta[1]<math.log(.5)
            and math.log(.05)<theta[2]<math.log(3)):return -np.inf
    return (-.5*((theta[1]-math.log(1/63))/2)**2
            -.5*((theta[2]-math.log(.7))/1.5)**2)


def whittle_target(theta,periodogram,n,noise_variance):
    prior=log_prior(theta)
    if not np.isfinite(prior):return -np.inf
    expected=math.exp(2*theta[2])*backend.unit_expected_periodogram(float(theta[0]),float(theta[1]),n)+noise_variance*backend.noise_expected_periodogram(n)
    if not np.isfinite(expected).all() or (expected<=0).any():
        raise ArithmeticError('invalid analytic-covariance expected periodogram')
    terms=np.log(expected)+periodogram/expected
    terms[0]*=.5
    if n%2==0:terms[-1]*=.5
    return prior-float(terms.sum())


@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    identity=hashlib.sha256(data).hexdigest()
    contract=f'M221_full_theoretical_H_analytic_differenced_debiased_Whittle_MAP_v1:{lag}'
    key=hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y,_=backend.base.base.observed(data)
        _,noise_variance=backend.base.noise.measurement_scale(data)
        differences=np.diff(y);periodogram=np.abs(np.fft.rfft(differences))**2/len(differences)
        result=backend.base.noise.fit_map(lambda theta:whittle_target(theta,periodogram,len(differences),noise_variance),
            [.1,math.log(1/63),math.log(.7)],
            [(0.,.5),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3))],seconds)
        result.update(contract=contract,data_sha256=identity,measurement_noise_variance=noise_variance,
            estimator='MAP_analytic_covariance_differenced_debiased_Whittle_quasi_likelihood',
            H_support='(0, 1/2); same optimizer interior numerical guards',
            frequency_count=len(periodogram),zero_frequency='included_known_zero_difference_mean')
        return result
    started=time.perf_counter();result=parent.shell.controls.cache('predecessor_MAP_rough_fit',key,build)
    parent.shell.controls.timed('rough_fit_worker',started)
    return result


parent.rough_fit=rough_fit
backend.streamed.rough_fit=rough_fit
initialize=parent.initialize
Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_student_implied_noise_pathwise_multiscale_full_hurst_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
