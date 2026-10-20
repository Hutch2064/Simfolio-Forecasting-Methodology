"""Simulate constituent log paths and portfolio holdings inside every scenario."""
import numpy as np
from numba import njit
from blocks import BlockLimits, JointBlockStream

WITH_REPLACEMENT = 'model_testing_asset_joint_blocks_replacement_v6'
WITHOUT_REPLACEMENT = 'model_testing_asset_joint_blocks_history_partition_v6'


@njit(cache=True, fastmath=False)
def advance(asset_history, indices, target_log_weights, log_weights,
            cumulative_asset, cumulative_portfolio, position, rebalance_every):
    paths, width = indices.shape
    assets = asset_history.shape[1]
    out = np.empty((paths, width))
    for p in range(paths):
        for t in range(width):
            if rebalance_every and (position+t) % rebalance_every == 0:
                for a in range(assets):
                    log_weights[p, a] = target_log_weights[a]
            maximum = -np.inf
            for a in range(assets):
                value = asset_history[indices[p, t], a]
                cumulative_asset[p, a] += value
                log_weights[p, a] += value
                maximum = max(maximum, log_weights[p, a])
            total = 0.
            for a in range(assets):
                total += np.exp(log_weights[p, a]-maximum)
            step = maximum+np.log(total)
            cumulative_portfolio[p] += step
            for a in range(assets):
                log_weights[p, a] -= step
            out[p, t] = cumulative_portfolio[p]
    return out


class JointAssetBootstrap:
    def __init__(self, *, replacement, limits=BlockLimits()):
        self.replacement, self.limits = replacement, limits
        self.model_id = WITH_REPLACEMENT if replacement else WITHOUT_REPLACEMENT

    def terminal_chunks(self, history, weights, horizon, paths, seed, *, width=64,
                        rebalance_every=1):
        x = np.asarray(history, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if (x.ndim != 2 or len(x) < 2 or x.shape[1] != len(w)
                or not np.isfinite(x).all() or not np.isfinite(w).all()
                or np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-14)
                or rebalance_every < 0 or width < 1):
            raise ValueError('Finite aligned assets, long-only unit weights and valid policy required')
        stream = JointBlockStream(len(x), horizon, paths, seed,
                                  replacement=self.replacement, limits=self.limits)
        target = np.full(len(w), -np.inf)
        target[w > 0] = np.log(w[w > 0])
        log_weights = np.tile(target, (paths, 1))
        cumulative_asset = np.zeros((paths, len(w)))
        cumulative_portfolio = np.zeros(paths)
        for start in range(0, horizon, width):
            count = min(width, horizon-start)
            terminal = advance(x, stream.take(count), target, log_weights,
                               cumulative_asset, cumulative_portfolio, start, rebalance_every)
            if not np.isfinite(terminal).all():
                raise ValueError('Nonfinite asset-level portfolio forecast')
            # Expose constituent state for audit, never reconstruct it from portfolio paths.
            self.last_asset_log_paths = cumulative_asset.copy()
            self.last_asset_weights = np.exp(log_weights)
            yield start, terminal
