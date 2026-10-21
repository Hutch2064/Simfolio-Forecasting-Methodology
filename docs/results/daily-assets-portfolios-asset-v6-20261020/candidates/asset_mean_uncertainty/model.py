"""Simple joint bootstrap plus training-estimated correlated mean uncertainty."""
import numpy as np
from blocks import BlockLimits, JointBlockStream
from candidates.asset_prior_best import mean, kernels
from candidates.asset_prior_best.candidate import seed


class JointMeanBootstrap:
    def __init__(self, replacement, shrink):
        self.replacement, self.shrink = replacement, shrink

    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        x, w = np.asarray(history, dtype=float), np.asarray(weights, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(w) or not np.isfinite(x).all():
            raise ValueError('Finite aligned constituent matrix required')
        if np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-14):
            raise ValueError('Long-only unit portfolio required')
        batch = mean.estimate(x)
        if not self.shrink:
            batch['center'] = batch['sample_mean'].copy()
        delta = mean.draws(batch, paths, seed('raw_joint_mean_uncertainty_v1', context_seed))
        stream = JointBlockStream(len(x), horizon, paths, context_seed,
                                  replacement=self.replacement, limits=BlockLimits())
        target = np.full(len(w), -np.inf)
        target[w > 0] = np.log(w[w > 0])
        weights_state = np.tile(target, (paths, 1))
        assets, portfolio = np.zeros((paths, len(w))), np.zeros(paths)
        zero, variance = np.zeros(len(w)), np.ones((paths, len(w)))
        for start in range(0, horizon, width):
            count = min(width, horizon-start)
            result = kernels.advance(x, x, stream.take(count), delta, zero, zero,
                                     zero, zero, variance, False, target,
                                     weights_state, assets, portfolio)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite portfolio forecast; no exclusions')
            self.last_asset_log_paths = assets.copy()
            self.last_asset_weights = np.exp(weights_state)
            self.last_mean_delta = delta.copy()
            yield start, result
