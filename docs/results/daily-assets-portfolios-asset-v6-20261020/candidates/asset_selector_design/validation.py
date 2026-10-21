"""Uncapped historical validation designs; all positions are past observations."""
from candidates.asset_selector_universe_precision.candidate import schedule as established
from candidates.asset_selector_universe.bank import BANK

RULES = ('established', 'support_balanced', 'target_first', 'overlap_midpoints')


def schedule(rows, requested_horizon, rule):
    if rule not in RULES:
        raise ValueError('Unknown historical validation design')
    eligible, original = established(rows, requested_horizon)
    minimum = max(2 if BANK[i]['kind'] == 'blocks' else 24 for i in eligible)
    if rule == 'established':
        return eligible, original
    if rule == 'overlap_midpoints':
        horizon = original[0][1]
        positions = {start for start, _ in original}
        positions.update((a[0]+b[0])//2 for a, b in zip(original, original[1:]))
        return eligible, [(start, horizon) for start in sorted(positions)]
    if rule == 'support_balanced':
        horizon = min(original[0][1], max(1, rows//3))
        reserve = max(minimum, horizon)
    else:
        horizon = min(requested_horizon, rows-minimum)
        reserve = minimum
    count = (rows-reserve)//horizon
    if count < 1:
        raise ArithmeticError('No completed historical fold; no fixed-model fallback')
    return eligible, [(rows-j*horizon, horizon) for j in range(count, 0, -1)]
