"""M197 volatility structure with debiased Whittle rough-parameter inference.

Sykulski et al. (2019), Biometrika 106(2), equations (7) and (9).
The finite-sample expected periodogram is exact for the selected OU mixture.
The frequency-domain likelihood remains an approximation to Gaussian likelihood;
this is a new estimator experiment, not a parity optimization or raw-return Bayes.
"""
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('whittle_private_conditional_empirical', ROOT / 'tools/conditional_empirical_rough/models.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
SOURCE = hashlib.sha256(b''.join(p.read_bytes() for directory in
    (HERE, ROOT/'tools/conditional_empirical_rough', ROOT/'tools/conditional_residual_rough', ROOT/'tools/empirical_rough_noise')
    for p in sorted(directory.glob('*.py')))).hexdigest()


@njit(cache=True, nogil=True)
def expected_periodogram(phi, stationary_mass, n):
    """Exact triangular-window covariance transform at positive DFT frequencies.

    Sum (1-l/n)*z**l analytically for each OU factor, z=phi*exp(-iw).
    At DFT frequencies z**n=phi**n. No frequency subsampling or factor cap.
    """
    result = np.zeros(n//2)
    for j in range(1, n//2+1):
        omega = 2*math.pi*j/n
        for k in range(phi.size):
            z = phi[k]*complex(math.cos(omega), -math.sin(omega))
            triangle = z/(1-z) - z*(1-phi[k]**n)/(n*(1-z)**2)
            result[j-1] += stationary_mass[k]*(1+2*triangle.real)
    return result


def whittle_target(theta, periodogram, n, lag, noise_variance):
    prior = base.overlay.log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    phi, w, q = base.overlay.configuration(theta, f'dynamic:{lag}')
    mass = np.diag(q)*w*w/(1-phi*phi)
    expected = expected_periodogram(phi, mass, n) + noise_variance
    if not np.isfinite(expected).all() or (expected <= 0).any():
        raise ArithmeticError('invalid finite-sample expected periodogram')
    terms = np.log(expected) + periodogram/expected
    # Restore Gaussian 1/2 omitted in the paper's MLE objective because we add
    # priors. Positive frequencies account for conjugate pairs; Nyquist is real.
    if n % 2 == 0:
        terms[-1] *= .5
    return prior-float(terms.sum())


@lru_cache(maxsize=64)
def rough_fit(data, lag, seconds):
    identity = hashlib.sha256(data).hexdigest()
    contract = f'debiased_Whittle_residual_empirical_noise_MAP_v1:{lag}'
    key = hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y, _ = base.base.observed(data)
        _, noise_variance = base.noise.measurement_scale(data)
        # The sample mean is a plug-in nuisance estimate, as in M197. Omit
        # frequency zero; demeaning leaves all other DFT ordinates unchanged.
        periodogram = np.abs(np.fft.rfft(y-y.mean())[1:])**2/len(y)
        result = base.noise.fit_map(
            lambda theta: whittle_target(theta, periodogram, len(y), lag, noise_variance),
            [.1, math.log(1/63), math.log(.7)],
            [(.03,.49), (math.log(1/2520),math.log(.5)), (math.log(.05),math.log(3))], seconds)
        result.update(contract=contract, data_sha256=identity,
            measurement_noise_variance=noise_variance,
            estimator='MAP_debiased_Whittle_quasi_likelihood',
            frequency_count=len(periodogram), zero_frequency='omitted_sample_mean_nuisance')
        return result
    started = time.perf_counter()
    result = base.shell.controls.cache('predecessor_MAP_rough_fit', key, build)
    base.shell.controls.timed('rough_fit_worker', started)
    return result


base.base.rough_fit = rough_fit


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    base.initialize(cache_root, vine_threads, state_workers)
    expected_periodogram(np.array([.9]), np.ones(1), 8)


clear_path_cache = base.clear_path_cache
TIMINGS = base.TIMINGS
FIT_DIAGNOSTICS = base.FIT_DIAGNOSTICS
shell = base.shell
overlay = base.overlay
Candidate = base.Candidate
CANDIDATES = (Candidate(model_id='asset_map_multiscale_conditional_empirical_debiased_whittle_rough_map'),)
MODEL_IDS = (base.MODEL_IDS[0], CANDIDATES[0].model_id)
