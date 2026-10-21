"""Natural endpoint contrast: the hybrid's filtered component, with all paths used."""
import numpy as np
from candidates.asset_prior_best import fit, mean, kernels
from candidates.asset_prior_best.candidate import seed


class JointFilteredSimulation:
    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        x, w = np.asarray(history, dtype=float), np.asarray(weights, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(w) or not np.isfinite(x).all():
            raise ValueError('Finite joint history required')
        if np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-14):
            raise ValueError('Long-only unit weights required')
        fitted = [fit.fit_variance_target(x[:, a]) for a in range(x.shape[1])]
        pool = np.stack([f['pool'] for f in fitted], axis=1)
        parameters = {name: np.asarray([f[name] for f in fitted]) for name in
                      ['mean', 'target_variance', 'alpha', 'beta', 'next_variance_normalized']}
        delta = mean.draws(mean.estimate(x), paths,
                           seed('FHS_joint_mean_uncertainty_v1', context_seed))
        shocks = np.random.default_rng(seed('joint_FHS_asset_rows_v1', context_seed))
        target = np.full(len(w), -np.inf)
        target[w > 0] = np.log(w[w > 0])
        weights_state = np.tile(target, (paths, 1))
        assets, portfolio = np.zeros((paths, len(w))), np.zeros(paths)
        variance = np.tile(parameters['next_variance_normalized'], (paths, 1))
        for start in range(0, horizon, width):
            count = min(width, horizon-start)
            indices = shocks.integers(0, len(x), size=(count, paths)).T.copy()
            result = kernels.advance(x, pool, indices, delta, parameters['mean'],
                parameters['target_variance'], parameters['alpha'], parameters['beta'],
                variance, True, target, weights_state, assets, portfolio)
            if not np.isfinite(result).all() or not np.isfinite(variance).all():
                raise ValueError('Nonfinite filtered forecast; no path exclusions')
            self.last_asset_log_paths = assets.copy()
            self.last_asset_weights = np.exp(weights_state)
            yield start, result
