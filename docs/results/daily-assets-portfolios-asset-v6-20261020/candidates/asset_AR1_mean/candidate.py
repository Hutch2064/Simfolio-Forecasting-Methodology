"""Same GARCH-fade raw half; conditional AR1-GARCH innovations in the filtered half."""
from pathlib import Path
import hashlib
import json
import numpy as np
from asset_paths import WITH_REPLACEMENT, WITHOUT_REPLACEMENT
from blocks import BlockLimits
from candidates.asset_circular_blocks.circular import CircularBlockStream
from candidates.asset_fading_vol import fade
from candidates.asset_fading_vol.kernels import advance_raw
from . import fit_ar
from .kernels import advance as advance_AR
from candidates.asset_prior_best import mean, fit, kernels

OWN = Path(__file__).resolve().parent
MODEL_ID='model_testing_asset_circular_GARCH_fade_AR1_FHS_mean_v1'


def seed(namespace, value):
    return int.from_bytes(hashlib.sha256(f'{namespace}:{value}'.encode()).digest()[:8], 'little')


class ARPool:
    model_id=MODEL_ID
    def __init__(self):
        self.replacement,self.limits=True,BlockLimits()

    @property
    def dependency_identity(self):
        return {'method': self.model_id, 'forecast_level': 'joint_assets',
            'paths_per_forecast': 1000, 'component_weights': [.5, .5],
            'raw_parent': 'Uniform circular blocks with replacement; all training rows eligible starts; modulo historical wrap',
            'block_limits': {'minimum': self.limits.minimum, 'maximum': self.limits.maximum},
            'asset_fits': 'Same raw GARCH fit plus separately variance-targeted GARCH on training AR1 innovations, per constituent',
            'joint_shocks': 'Same historical residual row across all fitted constituents',
            'mean_uncertainty': 'Joint weighted-contiguous-batch covariance, matched Gaussian variance and smooth per-asset mean regularization',
            'raw_volatility_conditioning': 'Existing GARCH conditional expected-variance fade',
            'raw_overlay_long_run_variance_ratio': 1.,
            'filtered_conditional_mean':'Stationary AR1, training OLS with HAC-uncertainty shrinkage; innovation GARCH fit; same long-run mean treatment',
            'portfolio_accounting': 'asset paths and target weights inside each scenario at every native endpoint',
            'numeric_change_from_old_scalar_model': 'Log-domain asset/portfolio wealth, no legacy scalar simple-return clipping; finite forecast required for every path',
            'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OWN.glob('*.py')},
            'outcome_fitted_weights_or_routes': False}

    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        x = np.asarray(history, dtype=float); w = np.asarray(weights, dtype=float)
        if (x.ndim != 2 or len(x) < 16 or not np.isfinite(x).all()
                or x.shape[1] != len(w) or np.any(w < 0) or not np.isfinite(w).all()
                or not np.isclose(w.sum(), 1., rtol=0, atol=1e-14) or paths % 4 or paths < 4):
            raise ValueError('Finite asset matrix, valid portfolio and equal even component ensembles required')
        half, assets = paths//2, x.shape[1]
        batch = mean.estimate(x)
        fitted = [fit.fit_variance_target(x[:, a]) for a in range(assets)]
        ARfits=[fit_ar.innovation_fit(x[:,a],fitted[a]) for a in range(assets)]
        phi=np.array([f[0] for f in ARfits])
        conditional=[f[1] for f in ARfits]
        pool = np.stack([f['pool'] for f in conditional], axis=1)
        innovation_params={key:np.array([f[key] for f in conditional]) for key in ['target_variance','alpha','beta','next_variance_normalized']}
        params = {key: np.asarray([f[key] for f in fitted]) for key in
                  ['mean', 'target_variance', 'alpha', 'beta', 'next_variance_normalized']}
        raw = CircularBlockStream(len(x), horizon, half, context_seed,
                               replacement=self.replacement, limits=self.limits)
        shocks = np.random.default_rng(seed('joint_FHS_asset_rows_v1', context_seed))
        delta = [mean.draws(batch, half, seed(name, context_seed)) for name in
                 ['raw_joint_mean_uncertainty_v1', 'FHS_joint_mean_uncertainty_v1']]
        target = np.full(assets, -np.inf); target[w > 0] = np.log(w[w > 0])
        states = [(np.tile(target, (half, 1)), np.zeros((half, assets)), np.zeros(half)) for _ in range(2)]
        variance = np.tile(innovation_params['next_variance_normalized'], (half, 1))
        previous=np.tile(x[-1],(half,1))
        initial=params['next_variance_normalized']
        decay=params['alpha']+params['beta']
        self.last_overlay={'initial':initial.copy(),'decay':decay.copy()}
        for start in range(0, horizon, width):
            count = min(width, horizon-start)
            indices = [raw.take(count), shocks.integers(0, len(x), size=(count, half)).T.copy()]
            scales=fade.scale_table(initial,decay,start,count)
            outputs=[advance_raw(x,indices[0],delta[0],params['mean'],scales,target,*states[0]),
                     advance_AR(pool,indices[1],delta[1],params['mean'],
                         innovation_params['target_variance'],innovation_params['alpha'],innovation_params['beta'],variance,
                         phi,previous,target,*states[1])]
            result = np.concatenate(outputs, axis=0)
            if not np.isfinite(result).all() or not np.isfinite(variance).all():
                raise ValueError('Nonfinite asset FHS or portfolio forecast; no path exclusions')
            self.last_asset_log_paths = np.concatenate([s[1] for s in states])
            self.last_asset_weights = np.exp(np.concatenate([s[0] for s in states]))
            self.last_batch = batch
            self.last_AR_coefficients=phi.copy()
            self.last_fit_parameters = [{k: v for k, v in f.items() if k != 'pool'} for f in fitted]
            yield start, result



def build():
    return ARPool()
