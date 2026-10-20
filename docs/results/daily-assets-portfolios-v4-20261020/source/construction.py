"""Observed common valuations, interval compounding, constant target weights."""
from fractions import Fraction
import hashlib
import itertools
import json
import numpy as np


def canonical_weights(weights):
    combined = {}
    for sid, value in weights.items():
        value = Fraction(value)
        if value < 0:
            raise ValueError('Long-only unlevered portfolio weights required')
        if value:
            combined[str(sid)] = value
    if sum(combined.values()) != 1 or len(combined) < 2:
        raise ValueError('At least two assets and exact weights summing to one required')
    return tuple((sid, w.numerator, w.denominator) for sid, w in sorted(combined.items()))


def mix(sleeves, shares):
    weights = {}
    for sleeve, share in zip(sleeves, shares, strict=True):
        for sid, weight in sleeve.items():
            weights[sid] = weights.get(sid, Fraction(0))+Fraction(share)*Fraction(weight)
    return {sid: w for sid, w in weights.items() if w}


def portfolio_returns(assets, weights):
    """Use only common observed endpoints; retain every intervening return.

    The first common observed valuation is an anchor, not an invented return.
    Each later interval compounds ALL component observations since that anchor.
    Constant target weights are restored at every common endpoint. No filling,
    dropping intervening crypto weekend returns or averaging logarithmic returns.
    """
    canonical_weights(weights)
    common = None
    for sid in weights:
        dates, logs = assets[sid]
        dates, logs = np.asarray(dates), np.asarray(logs)
        if dates.ndim != 1 or dates.shape != logs.shape or len(dates) < 2:
            raise ValueError('Aligned native observation arrays required')
        if np.any(dates[1:] <= dates[:-1]) or not np.isfinite(logs).all():
            raise ValueError('Unique increasing dates and finite log returns required')
        common = dates if common is None else np.intersect1d(common, dates, assume_unique=True)
    if len(common) < 2:
        raise ValueError('No common observed valuation intervals')
    simple = np.zeros(len(common)-1)
    for sid, weight in weights.items():
        dates, logs = assets[sid]
        cumulative = np.cumsum(logs, dtype=np.float64)
        positions = np.searchsorted(dates, common)
        interval_log = np.diff(cumulative[positions])
        simple += float(weight)*np.expm1(interval_log)
    if not np.isfinite(simple).all() or np.any(simple <= -1):
        raise ValueError('Invalid constructed unlevered portfolio returns')
    return common[1:].astype('datetime64[D]'), np.log1p(simple).astype('<f8'), str(common[0])


def catalogue(rows):
    """Coverage rules, not a count cap or a grid of minor weight/rebalance tweaks."""
    by_id = {r['series_id']: r for r in rows}
    ids = {'FR0551', 'EQX0112', 'FR0525'}
    ids.update(r['series_id'] for r in rows if r['asset_class'] in
               {'fixed_income', 'commodities', 'foreign_exchange', 'cryptocurrencies'})
    sleeves = {sid: {sid: Fraction(1)} for sid in sorted(ids)}
    classes = {sid: by_id[sid]['asset_class'] for sid in ids}
    definitions = {}
    excluded = []

    def add(name, weights, family, purpose):
        signature = canonical_weights(weights)
        if signature in definitions:
            excluded.append({'name': name, 'same_weights_as': definitions[signature]['display_name'],
                             'reason': 'identical_exact_asset_weights_and_construction_rule'})
            return
        payload = {'weights': signature, 'construction': 'constant_weights_common_observed_endpoint'}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        definitions[signature] = {'series_id': 'P_'+digest[:16], 'display_name': name,
            'family': family, 'purpose': purpose, 'weights': {sid: float(Fraction(n, d)) for sid, n, d in signature},
            'rational_weights': [list(x) for x in signature], 'construction': payload['construction'],
            'source_ids': [x[0] for x in signature], 'definition_sha256': digest}

    # Every unordered cross-class pair of admitted broad source exposures.
    # All durations/credit families and all distinct commodities/major currencies
    # remain represented. Numerous existing equity tilts do not create a huge
    # second catalogue of near-equivalent mixes.
    for left, right in itertools.combinations(sorted(sleeves), 2):
        if classes[left] == classes[right]:
            continue
        pair = '__'.join(sorted((classes[left], classes[right])))
        add(f'50 {left} / 50 {right}', mix([sleeves[left], sleeves[right]], [Fraction(1, 2)]*2),
            'cross_class_pair__'+pair, 'neutral_cross_class_pair')

    fx = ['FX_AUD_USD', 'FX_CAD_USD', 'FX_GBP_USD', 'FX_JPY_USD', 'FX_CHF_USD']
    anchors = {'equities': {'FR0551': Fraction(1)}, 'fixed_income': {'IEFSIM': Fraction(1)},
        'commodities': {'GSGSIM': Fraction(1)}, 'real_estate': {'FR0525': Fraction(1)},
        'foreign_exchange': {sid: Fraction(1, len(fx)) for sid in fx},
        'cryptocurrencies': {'BTC': Fraction(1)}}
    for subset_size in range(2, len(anchors)+1):
        for subset in itertools.combinations(sorted(anchors), subset_size):
            family = ('cross_class_pair__'+'__'.join(subset)) if subset_size == 2 else f'balanced_{subset_size}_classes'
            add('Balanced '+' / '.join(subset),
                mix([anchors[c] for c in subset], [Fraction(1, subset_size)]*subset_size),
                family, 'complete_major_asset_class_subset_coverage')
    for left, right in itertools.combinations(sorted(anchors), 2):
        family = 'cross_class_pair__'+'__'.join((left, right))
        for high, low in [(left, right), (right, left)]:
            add(f'Concentrated90 {high} / 10 {low}',
                mix([anchors[high], anchors[low]], [Fraction(9, 10), Fraction(1, 10)]),
                family, 'material_directional_concentration_stress')

    strategic = [
        ('US balanced', {'FR0551': 60, 'IEFSIM': 40}),
        ('Global balanced', {'FR0551': 30, 'EQX0112': 30, 'IEFSIM': 40}),
        ('Defensive', {'FR0551': 20, 'IEFSIM': 80}),
        ('Growth', {'FR0551': 80, 'IEFSIM': 20}),
        ('Real assets', {'FR0551': 50, 'IEFSIM': 30, 'GLDSIM': 10, 'FR0525': 10}),
        ('Commodity diversified', {'FR0551': 50, 'IEFSIM': 30, 'GSGSIM': 20}),
        ('Inflation diversified', {'FR0551': 20, 'IEFSIM': 40, 'TIPSIM': 20, 'GSGSIM': 20}),
        ('Broad cross asset', {'FR0551': 40, 'IEFSIM': 25, 'TIPSIM': 10, 'GSGSIM': 10, 'FR0525': 10, 'GLDSIM': 5}),
        ('Small digital allocation', {'FR0551': 60, 'IEFSIM': 35, 'BTC': 5}),
        ('Short-duration reserve', {'FR0551': 30, 'IEFSIM': 40, 'SHYSIM': 30}),
        ('Treasury duration barbell', {'SHYSIM': 50, 'TLTSIM': 50}),
        ('Nominal/inflation protection', {'IEFSIM': 50, 'TIPSIM': 50}),
        ('Credit diversified', {'IEFSIM': 50, 'VWESX_YAHOO_DAILY': 25, 'VWEHX_YAHOO_DAILY': 25}),
        ('International diversified', {'EQX0112': 50, 'TPINX_YAHOO_DAILY': 30, 'GLDSIM': 20}),
        ('US/international equity', {'FR0551': 50, 'EQX0112': 50}),
    ]
    for name, weights in strategic:
        add(name, {sid: Fraction(weight, 100) for sid, weight in weights.items()},
            'strategic_multiasset', 'distinct_conventional_or_risk_exposure_template')
    # Every broad five-industry sector, once, against the same defensive sleeve.
    sectors = [r for r in rows if r['series_id'].startswith('FR')
               and r.get('provenance', {}).get('original_series', {}).get('dataset_family') == 'industry_5'
               and r.get('provenance', {}).get('original_series', {}).get('weighting') == 'value']
    assert len(sectors) == 5
    for row in sectors:
        add('Sector70 '+row['display_name']+' / Treasury30',
            {row['series_id']: Fraction(7, 10), 'IEFSIM': Fraction(3, 10)},
            'sector_concentration', 'complete_broad_sector_concentration_coverage')
    return list(definitions.values()), excluded
