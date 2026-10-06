"""M227 with data-learned conventional multiscale decay rates; mean unchanged."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import rate_fit
from components import components

spec = importlib.util.spec_from_file_location('learned_rates_private_m227', ROOT / 'tools/untruncated_rough_priors/models.py')
parent = importlib.util.module_from_spec(spec); sys.modules[spec.name] = parent; spec.loader.exec_module(parent)
pathwise = parent.parent
student = pathwise.parent.parent
bd = parent.shell.bd
SOURCE = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()
original_student_fit = pathwise.predecessor_fit
fit_rates = rate_fit.fit


@lru_cache(maxsize=64)
def refit(data):
    def build():
        import student_laplace_sv
        original = original_student_fit(data)
        x = np.frombuffer(data, np.float64)
        _, eps = bd._sv_observed_log_variance(x, float(x.mean()), winsorize=True)
        level, phi, eta = original['posterior_center'][:3]
        u = original['student_return_laplace_fit']['inverse_df']
        h, _, _, _, _, success, _, decrement = student_laplace_sv.latent_mode(eps * eps, level, phi, eta, u)
        if not success: raise ArithmeticError(f'Student latent reconstruction failed: {decrement}')
        # Preserve the existing finite history preprocessing in the construction.
        h = np.clip(h, -18., 18.)
        phis, diagnostics = fit_rates(h, original['bdes_multiscale_vol']['phis'])
        fitted = original.copy()
        fitted['bdes_multiscale_vol'] = components(h, phis, loading_scale=diagnostics.get("loading_scale"))
        fitted['learned_decay_fit'] = diagnostics
        return fitted
    return parent.shell.controls.cache('M227_learned_multiscale_rates', hashlib.sha256(data + SOURCE.encode()).hexdigest(), build)


@lru_cache(maxsize=64)
def nodes(data, n):
    pool = original_student_fit(data)['innovation_pool']
    x = np.quantile(pool, np.linspace(.5 / n, 1 - .5 / n, n))
    x -= x.mean(); x /= np.sqrt(np.mean(x * x))
    return x


def asset_paths(data, uniforms):
    from streamed_paths import map_asset_inplace

    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    mean, _ = moment_return_curves(student.original_fit(data), uniforms.shape[1])
    _, sd = moment_return_curves(refit(data), uniforms.shape[1])
    paths = uniforms.copy(); map_asset_inplace(paths, mean, sd, nodes(data, len(paths)))
    return mean, paths


pathwise.predecessor_fit = refit
pathwise.predecessor_asset_paths = asset_paths


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self, training, context):
        result = super().simulate_daily_log_returns(training, context)
        for x in np.asarray(training.asset_log_returns, float).T:
            data = np.ascontiguousarray(x).tobytes()
            record = FIT_DIAGNOSTICS[self.model_id, hashlib.sha256(data).hexdigest()]
            record['learned_multiscale_decay_fit'] = refit(data)['learned_decay_fit']
        return result


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    parent.initialize(cache_root, vine_threads, state_workers)
    rate_fit.residual_innovations(np.arange(20., dtype=float), np.array([.9, .95]))


CANDIDATES = (Candidate(model_id='asset_map_learned_multiscale_rates_untruncated_dynamic_rough'),)
MODEL_IDS = (parent.MODEL_IDS[0], CANDIDATES[0].model_id)
clear_path_cache = parent.clear_path_cache
TIMINGS = parent.TIMINGS; FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
shell = parent.shell; overlay = parent.overlay; load_paths = parent.load_paths
