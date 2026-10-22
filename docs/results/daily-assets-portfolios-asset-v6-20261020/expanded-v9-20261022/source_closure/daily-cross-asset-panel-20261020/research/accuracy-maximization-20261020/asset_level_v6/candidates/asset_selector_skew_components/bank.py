"""Composable skew innovations; no parameter learned from outer panel scores."""
import hashlib, itertools, json
import numpy as np
from candidates.asset_selector_menu_components import bank as base
from candidates.asset_selector_universe import bank as original
from candidates.asset_feature_batch import fits
from .shape import fit
from .paths import group_chunks

SKEW = [{'kind': 'SAS_filtered', 'family': family, 'innovations': 'SAS_BIC',
         'mean_mode': mean, 'mean_uncertainty': uncertainty}
        for family, mean, uncertainty in itertools.product(fits.FAMILIES, ('sample', 'shrink', 'zero'), (False, True))]
BANK = [*base.BANK, *SKEW]
MENUS = {'original': base.MENUS['original'], 'flat': base.MENUS['flat'],
         'flat_PPW': base.MENUS['flat_PPW'],
         'flat_SAS': [*base.MENUS['flat'], *range(170, len(BANK))],
         'flat_PPW_SAS': [*base.MENUS['flat_PPW'], *range(170, len(BANK))]}


def prefix_fits(history, eligible, weights):
    needed = [i for i in eligible if i < 170]
    families = {BANK[i]['family'] for i in eligible if i >= 170}
    for family in families:
        # The existing model fitter is shared, rather than refitting every mean
        # and uncertainty combination or changing its financial parameters.
        index = next(i for i, spec in enumerate(original.BANK) if spec.get('family') == family)
        if index not in needed:
            needed.append(index)
    prepared, invalid = base.prefix_fits(history, needed, weights)
    for family in families:
        ids = [i for i in eligible if i >= 170 and BANK[i]['family'] == family]
        try:
            if family not in prepared['families']:
                raise ArithmeticError('Underlying volatility fit is unavailable')
            bundle = prepared['families'][family]
            prepared['families'][family] = {**bundle, 'SAS_fit': [fit(column) for column in bundle['filter']['pool'].T]}
        except Exception as error:
            for i in ids:
                invalid[i] = {'type': type(error).__name__, 'reason': str(error)}
    return prepared, {i: error for i, error in invalid.items() if i in eligible}


class SkewOption:
    def __init__(self, spec, bundle):
        if spec not in SKEW:
            raise ValueError('Known standardized skew component required')
        self.spec = dict(spec); self.bundle = bundle
        self.model_id = 'selector_joint_SAS_component_v1_' + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    @property
    def dependency_identity(self):
        return {'method': self.model_id, 'spec': self.spec, 'forecast_level': 'joint_assets',
                'shape': 'Training likelihood with one-parameter BIC comparison against Gaussian; delta=1 skew-only subfamily',
                'dependence': 'Existing Gaussian copula, monotone standardized marginal transforms; Pearson correlations can change',
                'source': 'Qualified prior SAS density/likelihood plus existing grouped asset/holdings kernel'}
    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        if paths < 2 or paths % 2 or width < 1:
            raise ValueError('Even positive ensemble and chunk width required')
        for start, cube in group_chunks(history, weights, horizon, paths, context_seed, [self.spec], self.bundle, width=width):
            yield start, cube[0]


def model(index, prepared):
    return base.model(index, prepared) if index < 170 else SkewOption(BANK[index], prepared['families'][BANK[index]['family']])


def all_chunks(history, weights, horizon, paths, seed, valid, prepared, *, width=64):
    original_ids = [i for i in valid if i < 170]; positions = {i: j for j, i in enumerate(valid)}
    groups = {}
    for i in valid:
        if i >= 170:
            groups.setdefault(BANK[i]['family'], []).append(i)
    existing = base.all_chunks(history, weights, horizon, paths, seed, original_ids, prepared, width=width) if original_ids else None
    generators = [(ids, group_chunks(history, weights, horizon, paths, seed, [BANK[i] for i in ids], prepared['families'][family], width=width)) for family, ids in groups.items()]
    for start in range(0, horizon, width):
        count = min(width, horizon-start); cube = np.empty((len(valid), paths, count))
        if existing is not None:
            position, values = next(existing)
            if position != start:
                raise ArithmeticError('Base menu path cursor differs')
            for j, i in enumerate(original_ids):
                cube[positions[i]] = values[j]
        for ids, generator in generators:
            position, values = next(generator)
            if position != start or values.shape != (len(ids), paths, count) or not np.isfinite(values).all():
                raise ArithmeticError('Skew component path is incomplete')
            for j, i in enumerate(ids):
                cube[positions[i]] = values[j]
        yield start, cube
