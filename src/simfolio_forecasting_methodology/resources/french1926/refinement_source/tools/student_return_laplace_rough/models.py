"""Student-return Laplace conventional fit plus unchanged M212 rough methodology.

Conventional state/parameter inference uses Gaussian return observations directly;
rough retains the Gaussian log-square quasi likelihood and empirical-pool noise.
Mean forecast is preserved from the original predecessor, byte for byte.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import student_laplace_sv

spec = importlib.util.spec_from_file_location('student_laplace_private_m212', ROOT / 'tools/consistent_noise_rough/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
streamed = parent.parent.streamed
original_fit = streamed.predecessor_fit
SOURCE = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()
shell = parent.shell


@lru_cache(maxsize=64)
def refit(data):
    def compute():
        original = original_fit(data)
        x = np.frombuffer(data, np.float64)
        y, eps = shell.bd._sv_observed_log_variance(x, float(x.mean()), winsorize=True)
        theta, h, variance, diagnostics = student_laplace_sv.fit(eps * eps, y, original['posterior_center'][:3])
        level, phi, eta = theta[:3]
        pool = shell.bd._standardized_empirical_innovation_pool(eps * np.exp(-.5 * h), clip=None, method='mean_std')
        if pool is None:
            raise ArithmeticError('invalid raw-return Laplace innovation pool')
        innov = (h[1:] - level - phi * (h[:-1] - level)) / eta
        rho = shell.bd._finite_correlation(pool[:len(innov)], innov)
        fitted = original.copy()
        fitted.update(posterior_center=(level, phi, eta, float(h[-1]), float(rho)),
                      innovation_pool=pool,
                      bdes_multiscale_vol=shell.bd._bdes_multiscale_components(h, 4, shell.bd.BDES_MULTISCALE_GRID_FIXED, fitted_phi=phi),
                      state_path_variance_last=float(variance[-1]),
                      state_loglikelihood=diagnostics["loglikelihood"], student_return_laplace_fit=diagnostics)
        return fitted
    return shell.controls.cache('student_return_laplace_conventional_MAP', hashlib.sha256(data + SOURCE.encode()).hexdigest(), compute)


streamed.predecessor_fit = refit
parent.parent.base.noise.predecessor_fit = refit


@lru_cache(maxsize=64)
def observed(data):
    x = np.frombuffer(data, np.float64)
    y, eps = shell.bd._sv_observed_log_variance(x, float(x.mean()))
    level, phi, eta = refit(data)['posterior_center'][:3]
    _, noise = parent.parent.base.noise.measurement_scale(data)
    return y - parent.causal_predictor(y, level, phi, eta, noise), eps


parent.parent.base.base.observed = observed
parent.parent.base.noise.observed = observed
parent.parent.SOURCE = hashlib.sha256((parent.parent.SOURCE + SOURCE).encode()).hexdigest()


def asset_paths(data, uniforms):
    from streamed_paths import map_asset_inplace

    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )

    mean, _ = moment_return_curves(original_fit(data), uniforms.shape[1])
    _, sd = moment_return_curves(refit(data), uniforms.shape[1])
    n = len(uniforms)
    nodes = np.quantile(refit(data)['innovation_pool'], np.linspace(.5 / n, 1 - .5 / n, n))
    nodes -= nodes.mean()
    nodes /= np.sqrt(np.mean(nodes * nodes))
    paths = uniforms.copy()
    map_asset_inplace(paths, mean, sd, nodes)
    return mean, paths


streamed.predecessor_asset_paths = asset_paths


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self, training, context):
        result = super().simulate_daily_log_returns(training, context)
        data = np.asarray(training.asset_log_returns, float)
        for a in range(data.shape[1]):
            d = data[:, a].tobytes()
            record = FIT_DIAGNOSTICS[self.model_id, hashlib.sha256(d).hexdigest()]
            record['student_return_laplace_fit'] = refit(d)['student_return_laplace_fit']
            record['conventional_parameter_fit'] = 'raw unit-variance Student returns with sparse Laplace latent integration; MAP hyperparameters'
            record['conventional_offset_noise'] = 'Gaussian causal proxy with empirical innovation-pool variance; rough quasi likelihood unchanged'
            record['return_mean_curve'] = 'unchanged_original_predecessor'
        return result


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    parent.initialize(cache_root, vine_threads, state_workers)
    student_laplace_sv.likelihood_gradient(np.ones(3), 0., .9, .2, 0.)
CANDIDATES = (Candidate(model_id='asset_map_student_return_laplace_multiscale_dynamic_rough'),)
MODEL_IDS = (parent.MODEL_IDS[0], CANDIDATES[0].model_id)
clear_path_cache = parent.clear_path_cache
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
shell = parent.shell
overlay = parent.overlay
load_paths = parent.load_paths
