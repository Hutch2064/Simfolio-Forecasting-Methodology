"""Exact M256 with only the conventional EWMA timescale selector replaced.

The same component construction receives data-selected rates. No changes to
the fitted historical path, mean formula, simulated SV, rough fit or copula.
Select one variant per process so inference caches cannot mix methodologies.
"""
import hashlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('timescales_private_M256',
    ROOT/'tools/student_corrected_dcc_innovation_copula/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
rates = parent.body.learned.rate_fit
components = parent.body.learned.components
bd = parent.shell.bd
ORIGINAL_COMPONENTS = bd._bdes_multiscale_components
RAW_FIT = bd.fit_bdes_fastmap
VARIANTS = ('autocorrelation', 'predictive_one', 'predictive_adaptive', 'stationary_one', 'existing_sv_rate')
VARIANT = os.environ.get('SIMFOLIO_TIMESCALE_VARIANT', VARIANTS[0])
if VARIANT not in VARIANTS:
    raise ValueError(f'Unknown timescale variant: {VARIANT}')
MODEL_ID = 'asset_m256_data_timescales_' + VARIANT


def cleaned_history(h):
    # Identical preprocessing to the original component constructor.
    h = np.asarray(h, dtype=float)
    finite = h[np.isfinite(h)]
    fill = float(np.nanmedian(finite)) if finite.size else 0.
    return np.clip(np.nan_to_num(h, nan=fill, posinf=fill, neginf=fill), -18., 18.)


@lru_cache(maxsize=128)
def selected_components(data, fitted_phi=None):
    h = np.frombuffer(data, np.float64)
    if VARIANT == 'existing_sv_rate':
        if fitted_phi is None:
            raise ValueError('Existing SV timescale requires its fitted persistence')
        phis = np.array([np.clip(fitted_phi, np.nextafter(0., 1.), np.nextafter(1., 0.))])
        receipt = {'estimator': 'reuse_unchanged_existing_SV_fitted_persistence',
            'phis': phis.tolist(), 'component_count': 1,
            'count_selection': 'single_existing_volatility_persistence_control',
            'additional_parameter_fits': 0}
    elif VARIANT == 'autocorrelation':
        x = h - h.mean()
        denominator = float(x[:-1] @ x[:-1])
        rho = float(x[:-1] @ x[1:]) / denominator if denominator > 0. else 0.
        phis = np.array([np.clip(rho, np.nextafter(0., 1.), np.nextafter(1., 0.))])
        receipt = {'estimator': 'centered_log_volatility_conditional_AR1_rate',
            'phis': phis.tolist(), 'component_count': 1,
            'count_selection': 'single_data_fitted_rate_control'}
    elif VARIANT == 'stationary_one':
        fitted = parent.body.state.fit(h)
        phis = np.array([np.clip(fitted['coefficients'][0],
            np.nextafter(0., 1.), np.nextafter(1., 0.))])
        receipt = dict(fitted)
        receipt.update(estimator='stationary_Gaussian_AR1_timescale_on_same_original_latent_history',
            phis=phis.tolist(), component_count=1,
            count_selection='single_stationary_likelihood_fitted_rate_control')
    elif VARIANT == 'predictive_one':
        phis, receipt = rates.fit_one(h, None)
    else:
        phis, receipt = rates.fit_adaptive(h, None)
    result = components(h, phis)
    result['timescale_selection'] = receipt
    return result


def data_components(h_path, k_star, scale_grid=bd.BDES_MULTISCALE_GRID_FIXED, *, fitted_phi=None):
    # k_star/grid are legacy arguments; neither chooses a candidate rate/count.
    return selected_components(cleaned_history(h_path).tobytes(),
        float(fitted_phi) if VARIANT == 'existing_sv_rate' and fitted_phi is not None else None)


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self, training, context):
        result = super().simulate_daily_log_returns(training, context)
        for x in np.asarray(training.asset_log_returns, float).T:
            data = np.ascontiguousarray(x).tobytes()
            key = self.model_id, hashlib.sha256(data).hexdigest()
            q = parent.body.learned.student.original_fit(data)['bdes_multiscale_vol']
            FIT_DIAGNOSTICS[key]['mean_scaling_timescales'] = q['timescale_selection']
            FIT_DIAGNOSTICS[key]['return_mean_curve'] = 'original_formula_with_data_selected_timescales_only'
        return result


CANDIDATES = (Candidate(model_id=MODEL_ID),)
# The runner treats index zero as its historical production reference. No
# baseline candidate is scheduled or evaluated in this experiment.
MODEL_IDS = (parent.MODEL_IDS[0], MODEL_ID)
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
TIMINGS = parent.TIMINGS
shell = parent.shell
overlay = parent.overlay
load_paths = parent.load_paths
state = parent.state
student_cdf = parent.student_cdf
native = parent.native
select_copula = parent.select_copula
clear_path_cache = parent.clear_path_cache


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    if cache_root is not None:
        cache_root = Path(cache_root)/VARIANT
    parent.initialize(cache_root, vine_threads, state_workers)
    # The retained compiler setup inspects the original constructor's source.
    # Install the selection adapter only after that unchanged setup finishes.
    bd._bdes_multiscale_components = data_components
    rates.residual_innovations(np.arange(20., dtype=float), np.array([.9]))


def source_hashes():
    # Complete source closure, not only the adapter's source files.
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'src', ROOT/'tools') for p in sorted(folder.rglob('*'))
        if p.suffix in ('.py', '.cpp')}
