"""Debiased Whittle on first differences: Sykulski et al. (2019), section 4.2.

Only rough-parameter inference changes from M197. The undifferenced conditional
Gaussian state filter and future return generator are retained. Differencing
removes a constant observation level; it is an explicit estimator experiment.
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
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('differenced_whittle_private', ROOT/'tools/debiased_whittle_rough/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
base = parent.base
SOURCE = hashlib.sha256((parent.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


from differenced_whittle_kernel import differenced_expected_periodogram


def whittle_target(theta, periodogram, n, lag, noise_variance):
    prior = base.overlay.log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    phi, w, q = base.overlay.configuration(theta, f'dynamic:{lag}')
    mass = np.diag(q)*w*w/(1-phi*phi)
    # Original independent measurement errors become an MA(1) after
    # differencing. Treating them as independent with variance 2R is wrong.
    expected = differenced_expected_periodogram(
        np.append(phi, 0.), np.append(mass, noise_variance), n)
    if not np.isfinite(expected).all() or (expected <= 0).any():
        raise ArithmeticError('invalid differenced expected periodogram')
    terms = np.log(expected)+periodogram/expected
    terms[0] *= .5
    if n % 2 == 0:
        terms[-1] *= .5
    return prior-float(terms.sum())


@lru_cache(maxsize=64)
def rough_fit(data, lag, seconds):
    identity = hashlib.sha256(data).hexdigest()
    contract = f'differenced_debiased_Whittle_residual_empirical_noise_MAP_v1:{lag}'
    key = hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y, _ = base.base.observed(data)
        _, noise_variance = base.noise.measurement_scale(data)
        differences = np.diff(y)
        periodogram = np.abs(np.fft.rfft(differences))**2/len(differences)
        result = base.noise.fit_map(
            lambda theta: whittle_target(theta, periodogram, len(differences), lag, noise_variance),
            [.1, math.log(1/63), math.log(.7)],
            [(.03,.49), (math.log(1/2520),math.log(.5)), (math.log(.05),math.log(3))], seconds)
        result.update(contract=contract, data_sha256=identity,
            measurement_noise_variance=noise_variance,
            estimator='MAP_differenced_debiased_Whittle_quasi_likelihood',
            frequency_count=len(periodogram), zero_frequency='included_known_zero_difference_mean')
        return result
    started = time.perf_counter()
    result = base.shell.controls.cache('predecessor_MAP_rough_fit', key, build)
    base.shell.controls.timed('rough_fit_worker', started)
    return result


base.base.rough_fit = rough_fit


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    parent.initialize(cache_root, vine_threads, state_workers)
    differenced_expected_periodogram(np.array([.9]), np.ones(1), 8)


clear_path_cache = base.clear_path_cache
TIMINGS = base.TIMINGS
FIT_DIAGNOSTICS = base.FIT_DIAGNOSTICS
shell = base.shell
overlay = base.overlay
Candidate = base.Candidate
CANDIDATES = (Candidate(model_id='asset_map_multiscale_conditional_empirical_differenced_whittle_rough_map'),)
MODEL_IDS = (base.MODEL_IDS[0], CANDIDATES[0].model_id)
