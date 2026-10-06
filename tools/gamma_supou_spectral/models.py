"""Gamma-mixed Gaussian OU volatility with direct debiased Whittle inference.

Fit the exact analytic covariance rather than rebuilding a lift in each trial.
Only predictive state filtering/simulation uses the accuracy-controlled lift.
This is a new pseudolikelihood estimator, not parity with Gaussian inference.
"""
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('gamma_spectral_private', ROOT/'tools/gamma_supou_volatility/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
SOURCE = hashlib.sha256((parent.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


@lru_cache(maxsize=128)
def expected_unit_periodogram(log_shape, log_kappa, n):
    lag = np.arange(n)
    covariance = np.exp(-math.exp(log_shape)*np.log1p(math.exp(log_kappa)*lag))
    return 2*np.fft.rfft(covariance*(1-lag/n)).real-1.


def whittle_target(theta, periodogram, n, noise_variance):
    prior = parent.overlay.log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    expected = (math.exp(2*theta[2])*expected_unit_periodogram(float(theta[0]), float(theta[1]), n)[1:]
                + noise_variance)
    if not np.isfinite(expected).all() or (expected <= 0).any():
        raise ArithmeticError('invalid Gamma covariance periodogram')
    terms = np.log(expected)+periodogram/expected
    if n % 2 == 0:
        terms[-1] *= .5
    return prior-float(terms.sum())


@lru_cache(maxsize=64)
def rough_fit(data, lag, seconds):
    identity = hashlib.sha256(data).hexdigest()
    contract = f'Gamma_mixed_OU_direct_debiased_Whittle_MAP_v1:{lag}'
    key = hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y, _ = parent.base.observed(data)
        _, R = parent.noise.measurement_scale(data)
        periodogram = np.abs(np.fft.rfft(y-y.mean())[1:])**2/len(y)
        result = parent.fit_map(lambda theta: whittle_target(theta, periodogram, len(y), R),
            [0.,math.log(1/63),math.log(.7)],
            [(math.log(.03),math.log(10)),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3))],seconds)
        result.update(contract=contract,data_sha256=identity,measurement_noise_variance=R,
            estimator='MAP_exact_Gamma_covariance_debiased_Whittle',
            parameter_likelihood_lift=False,frequency_count=len(periodogram))
        return result
    started = time.perf_counter()
    result = parent.shell.controls.cache('predecessor_MAP_rough_fit',key,build)
    parent.shell.controls.timed('rough_fit_worker',started)
    return result


parent.rough_fit = rough_fit
initialize = parent.initialize
clear_path_cache = parent.clear_path_cache
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
shell = parent.shell
overlay = parent.overlay
Candidate = parent.Candidate
CANDIDATES = (Candidate(model_id='asset_map_multiscale_conditional_empirical_gamma_supou_whittle_map'),)
MODEL_IDS = (parent.MODEL_IDS[0], CANDIDATES[0].model_id)
