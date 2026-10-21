"""Same joint-asset hybrid with a training-selected stationary raw-block component."""
from pathlib import Path
import hashlib
import json
import numpy as np
from asset_paths import WITH_REPLACEMENT, WITHOUT_REPLACEMENT
from blocks import BlockLimits, JointBlockStream
from candidates.asset_prior_best import mean, fit, kernels
from .stationary import block_length, StationaryStream, selector_identity

OWN = Path(__file__).resolve().parent
MODEL_ID = 'model_testing_asset_equal_PPW_stationary_FHS_smooth_joint_mean_uncertainty_v1'


def seed(namespace, value):
    return int.from_bytes(hashlib.sha256(f'{namespace}:{value}'.encode()).digest()[:8], 'little')


class AdaptivePool:
    model_id = MODEL_ID

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'forecast_level': 'joint_assets',
            'paths_per_forecast': 1000, 'component_weights': [.5, .5],
            'raw_parent': 'Training-selected stationary bootstrap, shared asset rows and circular continuation',
            'block_rule': 'Maximum corrected PPW stationary mean over active constituent returns and centered squares, constrained to1..training rows; geometric actual lengths',
            'selector_dependency': selector_identity(),
            'asset_fits': 'Same two-start variance-targeted GARCH(1,1) estimator as prior best, per constituent',
            'joint_shocks': 'Same historical residual row across all fitted constituents',
            'mean_uncertainty': 'Joint weighted-contiguous-batch covariance, matched Gaussian variance and smooth per-asset mean regularization',
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
        pool = np.stack([f['pool'] for f in fitted], axis=1)
        params = {key: np.asarray([f[key] for f in fitted]) for key in
                  ['mean', 'target_variance', 'alpha', 'beta', 'next_variance_normalized']}
        length = block_length(x[:, w > 0])
        self.last_raw_block_length = length
        raw = StationaryStream(len(x), horizon, half, context_seed, length)
        shocks = np.random.default_rng(seed('joint_FHS_asset_rows_v1', context_seed))
        delta = [mean.draws(batch, half, seed(name, context_seed)) for name in
                 ['raw_joint_mean_uncertainty_v1', 'FHS_joint_mean_uncertainty_v1']]
        target = np.full(assets, -np.inf); target[w > 0] = np.log(w[w > 0])
        states = [(np.tile(target, (half, 1)), np.zeros((half, assets)), np.zeros(half)) for _ in range(2)]
        variance = np.tile(params['next_variance_normalized'], (half, 1))
        for start in range(0, horizon, width):
            count = min(width, horizon-start)
            indices = [raw.take(count), shocks.integers(0, len(x), size=(count, half)).T.copy()]
            outputs = [kernels.advance(x, pool, indices[i], delta[i], params['mean'],
                params['target_variance'], params['alpha'], params['beta'], variance,
                bool(i), target, *states[i]) for i in range(2)]
            result = np.concatenate(outputs, axis=0)
            if not np.isfinite(result).all() or not np.isfinite(variance).all():
                raise ValueError('Nonfinite asset FHS or portfolio forecast; no path exclusions')
            self.last_asset_log_paths = np.concatenate([s[1] for s in states])
            self.last_asset_weights = np.exp(np.concatenate([s[0] for s in states]))
            self.last_batch = batch
            self.last_fit_parameters = [{k: v for k, v in f.items() if k != 'pool'} for f in fitted]
            yield start, result



def build():
    return AdaptivePool()
