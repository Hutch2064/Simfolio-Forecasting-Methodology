"""Descriptive finite-ensemble diagnostics; no contribution to model scores."""
import numpy as np

QUANTILES = (.05, .10, .25, .50, .75, .90, .95)
BANDS = (('all', 1, 10**9), ('1_30_days', 1, 30), ('31_365_days', 31, 365),
         ('366_1826_days', 366, 1826), ('1827_7305_days', 1827, 7305),
         ('7306_plus_days', 7306, 10**9))


def diagnostics(terminal, observed):
    terminal = np.asarray(terminal, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if terminal.ndim != 2 or observed.shape != (terminal.shape[1],):
        raise ValueError('Paths by horizon predictions and one outcome per horizon required')
    if terminal.shape[0] < 2 or not np.isfinite(terminal).all() or not np.isfinite(observed).all():
        raise ValueError('Finite ensemble and observations required')
    n = len(terminal)
    less = np.sum(terminal < observed[None, :], axis=0)
    equal = np.sum(terminal == observed[None, :], axis=0)
    midpoint = (less+.5*equal)/n
    bins = np.minimum((midpoint*10).astype(int), 9)
    result = {f'CDF_bin_{i}': (bins == i).astype(float) for i in range(10)}
    result['CDF_midpoint'] = midpoint
    result['forecast_tie_fraction'] = equal/n
    ordered = np.sort(terminal, axis=0)
    quantiles = {}
    for probability in QUANTILES:
        # Left-continuous generalized inverse of the finite empirical CDF.
        q = ordered[int(np.ceil(probability*n))-1]
        key = f'{probability:.2f}'
        quantiles[probability] = q
        result['below_quantile_'+key] = (observed < q).astype(float)
        result['equal_quantile_'+key] = (observed == q).astype(float)
    for lower, upper, label in ((.05, .95, '90'), (.10, .90, '80'), (.25, .75, '50')):
        result['central_interval_coverage_'+label] = ((observed >= quantiles[lower]) & (observed <= quantiles[upper])).astype(float)
    assert np.all((midpoint >= 0) & (midpoint <= 1))
    return result


def weighted_summary(values, elapsed_days, weights):
    elapsed_days = np.asarray(elapsed_days)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0) or abs(weights.sum()-1) > 1e-12:
        raise ValueError('Canonical positive normalized calendar weights required')
    out = {}
    for name, low, high in BANDS:
        mask = (elapsed_days >= low) & (elapsed_days <= high)
        mass = float(weights[mask].sum())
        if not mass:
            continue
        w = weights[mask]/mass
        out[name] = {metric: float(np.dot(value[mask], w)) for metric, value in values.items()}
        assert abs(sum(out[name][f'CDF_bin_{i}'] for i in range(10))-1) < 1e-12
    return out
