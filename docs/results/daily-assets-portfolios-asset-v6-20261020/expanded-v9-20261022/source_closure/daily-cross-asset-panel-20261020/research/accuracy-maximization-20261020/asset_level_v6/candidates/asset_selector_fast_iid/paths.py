"""Exact IID row draws and existing batched filtered paths; no menu reduction."""
import numpy as np
from asset_paths import JointAssetBootstrap, advance
from candidates.asset_selector_universe.bank import BANK, model as original_model
from candidates.asset_selector_batched_paths.paths import group_chunks


class FastBootstrap(JointAssetBootstrap):
    def terminal_chunks(self, history, weights, horizon, paths, seed, *, width=64, rebalance_every=1):
        x = np.asarray(history, dtype=np.float64); w = np.asarray(weights, dtype=np.float64)
        if (x.ndim != 2 or len(x) < 2 or x.shape[1] != len(w) or not np.isfinite(x).all()
                or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-14)
                or rebalance_every < 0 or width < 1):
            raise ValueError('Finite aligned assets, long-only unit weights and valid policy required')
        if not self.replacement or self.limits.resolve(len(x), horizon) != (1, 1):
            yield from super().terminal_chunks(x, w, horizon, paths, seed, width=width, rebalance_every=rebalance_every)
            return
        if paths < 1:
            raise ValueError('Positive simulation count required')
        # integers(1,2) in the original constant-length stream consumes no
        # random bits. Vector bounded draws retain the exact child-RNG sequence.
        indices = np.empty((paths, horizon), dtype=np.int64)
        for p, child in enumerate(np.random.SeedSequence(int(seed)).spawn(paths)):
            indices[p] = np.random.default_rng(child).integers(0, len(x), size=horizon)
        target = np.full(len(w), -np.inf); target[w > 0] = np.log(w[w > 0])
        log_weights = np.tile(target, (paths, 1)); asset = np.zeros((paths, len(w))); portfolio = np.zeros(paths)
        for start in range(0, horizon, width):
            count = min(width, horizon-start)
            values = advance(x, np.ascontiguousarray(indices[:, start:start+count]), target,
                             log_weights, asset, portfolio, start, rebalance_every)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite asset-level portfolio forecast')
            self.last_asset_log_paths = asset.copy(); self.last_asset_weights = np.exp(log_weights)
            yield start, values


def model(index, prepared):
    spec = BANK[index]
    if spec['kind'] == 'blocks':
        original = original_model(index, prepared)
        return FastBootstrap(replacement=original.replacement, limits=original.limits)
    return original_model(index, prepared)


def all_chunks(history, weights, horizon, paths, context_seed, current, prepared, *, width=64):
    current = list(current)
    if not current or len(set(current)) != len(current) or any(i < 0 or i >= len(BANK) for i in current) or width < 1:
        raise ValueError('Unique valid options and positive chunk width required')
    positions = {index: offset for offset, index in enumerate(current)}; groups = {}; generators = []
    for index in current:
        spec = BANK[index]
        if spec['kind'] == 'filtered':
            groups.setdefault((spec['family'], spec['innovations']), []).append(index)
        else:
            generators.append(([index], model(index, prepared).terminal_chunks(history, weights, horizon, paths, context_seed, width=width), False))
    for (family, law), ids in groups.items():
        generators.append((ids, group_chunks(history, weights, horizon, paths, context_seed, ids, prepared['families'][family], width=width), True))
    for start in range(0, horizon, width):
        count = min(width, horizon-start); cube = np.empty((len(current), paths, count))
        for ids, generator, grouped in generators:
            position, values = next(generator)
            if not grouped:
                values = values[None, :, :]
            if position != start or values.shape != (len(ids), paths, count) or not np.isfinite(values).all():
                raise ArithmeticError('Incomplete grouped predictive path')
            for offset, index in enumerate(ids):
                cube[positions[index]] = values[offset]
        yield start, cube
