"""M209 with the Hurst support lower bound relaxed from 0.03 to 0.01.

Debiased Whittle remains a Gaussian quasi likelihood. Only its numerical
covariance representation changes: the dynamic OU lift is resolved after
fitting, for the unchanged conditional state filter and future paths.
"""
import hashlib
import importlib.util
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import gamma, kv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('relaxed_hurst_differenced_private_streamed',ROOT/'tools/streamed_rough_paths/models.py')
streamed=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=streamed;spec.loader.exec_module(streamed)
parent=streamed.parent.parent
base=parent.base
SOURCE=hashlib.sha256((parent.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


@lru_cache(maxsize=64)
def unit_expected_periodogram(hurst,log_kappa,n):
    x=math.exp(log_kappa)*np.arange(1,n+1)
    # DLMF 13.6.10: exact Bessel-K form of the same covariance.
    covariance=np.r_[1.,2**(1-hurst)/gamma(hurst)*x**hurst*kv(hurst,x)]
    differences=np.empty(n)
    differences[0]=2*(covariance[0]-covariance[1])
    differences[1:]=2*covariance[1:n]-covariance[:n-1]-covariance[2:n+1]
    # The differenced finite-sample spectrum includes zero and Nyquist.
    return 2*np.fft.rfft(differences*(1-np.arange(n)/n)).real-differences[0]


@lru_cache(maxsize=64)
def noise_expected_periodogram(n):
    return 2*(1-(1-1/n)*np.cos(2*np.pi*np.arange(n//2+1)/n))


def log_prior(theta):
    if not (.01<theta[0]<.49 and math.log(1/2520)<theta[1]<math.log(.5)
            and math.log(.05)<theta[2]<math.log(3)):return -np.inf
    return (-.5*((theta[1]-math.log(1/63))/2)**2
            -.5*((theta[2]-math.log(.7))/1.5)**2)


def whittle_target(theta,periodogram,n,noise_variance):
    prior=log_prior(theta)
    if not np.isfinite(prior):return -np.inf
    expected=math.exp(2*theta[2])*unit_expected_periodogram(float(theta[0]),float(theta[1]),n)+noise_variance*noise_expected_periodogram(n)
    if not np.isfinite(expected).all() or (expected<=0).any():
        raise ArithmeticError('invalid analytic-covariance expected periodogram')
    terms=np.log(expected)+periodogram/expected
    terms[0]*=.5
    if n%2==0:terms[-1]*=.5
    return prior-float(terms.sum())


@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    identity=hashlib.sha256(data).hexdigest()
    contract=f'relaxed_H_floor_analytic_differenced_debiased_Whittle_MAP_v1:{lag}'
    key=hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y,_=base.base.observed(data)
        _,noise_variance=base.noise.measurement_scale(data)
        differences=np.diff(y)
        periodogram=np.abs(np.fft.rfft(differences))**2/len(differences)
        result=base.noise.fit_map(lambda theta:whittle_target(theta,periodogram,len(differences),noise_variance),
            [.1,math.log(1/63),math.log(.7)],
            [(.01,.49),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3))],seconds)
        result.update(contract=contract,data_sha256=identity,measurement_noise_variance=noise_variance,
            estimator='MAP_analytic_covariance_differenced_debiased_Whittle_quasi_likelihood',
            frequency_count=len(periodogram),zero_frequency='included_known_zero_difference_mean')
        return result
    started=time.perf_counter()
    result=base.shell.controls.cache('predecessor_MAP_rough_fit',key,build)
    base.shell.controls.timed('rough_fit_worker',started)
    return result


streamed.rough_fit=rough_fit
initialize=streamed.initialize
Candidate=streamed.Candidate
CANDIDATES=(Candidate(model_id='asset_map_multiscale_conditional_empirical_relaxed_hurst_differenced_rough_map'),)
MODEL_IDS=(streamed.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=streamed.clear_path_cache
TIMINGS=streamed.TIMINGS
FIT_DIAGNOSTICS=streamed.FIT_DIAGNOSTICS
shell=streamed.shell
overlay=streamed.overlay
load_paths=streamed.load_paths
