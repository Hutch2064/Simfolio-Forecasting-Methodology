"""Matched menu ablations; retain the original financial laws and draw streams."""
import numpy as np
from candidates.asset_selector_universe import bank as base
from candidates.asset_selector_fast_iid import paths as fast
from candidates.asset_fast_selector.candidate import BASE_CONFIG
from candidates.asset_trained_pool.candidate import TrainedPool
from candidates.asset_selector_ppw.blocks import fit_lengths, PPWJointBootstrap

BANK = [*base.BANK, {'kind': 'parent_leg', 'leg': 'raw'},
        {'kind': 'parent_leg', 'leg': 'filtered'},
        {'kind': 'PPW_blocks', 'sampling': 'stationary'},
        {'kind': 'PPW_blocks', 'sampling': 'circular'}]
MENUS = {'original': list(range(166)), 'no_parent': list(range(1, 166)),
         'flat': list(range(1, 168)), 'original_PPW': [*range(166), 168, 169],
         'flat_PPW': list(range(1, 170))}


class ParentLeg(TrainedPool):
    def __init__(self, leg, prepared):
        if leg not in ('raw', 'filtered'):
            raise ValueError('Known exact parent leg required')
        bundle = {**prepared, 'pool_fit': {**prepared['pool_fit'], 'raw_fraction': float(leg == 'raw')}}
        super().__init__(BASE_CONFIG, bundle)
        self.model_id = 'selector_exact_parent_' + leg + '_leg_v1'


def prefix_fits(history, eligible, weights):
    original = [i for i in eligible if i < 166]
    if any(i in (166, 167) for i in eligible) and 0 not in original:
        original.append(0)
    prepared, invalid = base.prefix_fits(history, original)
    if 0 in invalid:
        for i in (166, 167):
            if i in eligible:
                invalid[i] = dict(invalid[0])
    added = [i for i in eligible if i in (168, 169)]
    if added:
        try:
            prepared['PPW'] = fit_lengths(history, weights)
        except Exception as e:
            for i in added:
                invalid[i] = {'type': type(e).__name__, 'reason': str(e)}
    return prepared, {i: error for i, error in invalid.items() if i in eligible}


def model(index, prepared):
    if index < 166:
        return fast.model(index, prepared)
    if index < 168:
        return ParentLeg(BANK[index]['leg'], prepared['parent'])
    return PPWJointBootstrap(BANK[index]['sampling'], prepared['PPW'])


def all_chunks(history, weights, horizon, paths, seed, valid, prepared, *, width=64):
    original = [i for i in valid if i < 166]
    positions = {i: j for j, i in enumerate(valid)}
    generators = [(i, model(i, prepared).terminal_chunks(history, weights, horizon, paths, seed, width=width))
                  for i in valid if i >= 166]
    original_chunks = fast.all_chunks(history, weights, horizon, paths, seed, original, prepared, width=width) if original else None
    for start in range(0, horizon, width):
        count = min(width, horizon-start); cube = np.empty((len(valid), paths, count))
        if original_chunks is not None:
            position, values = next(original_chunks)
            if position != start:
                raise ArithmeticError('Original menu path cursor differs')
            for j, i in enumerate(original):
                cube[positions[i]] = values[j]
        for i, generator in generators:
            position, values = next(generator)
            if position != start or values.shape != (paths, count) or not np.isfinite(values).all():
                raise ArithmeticError('New component path is incomplete')
            cube[positions[i]] = values
        yield start, cube
