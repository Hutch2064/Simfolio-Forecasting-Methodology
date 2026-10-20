"""Preserve finite native paths; continue positive-overflow paths in log wealth.

The simple-return limits and reported1e-300 wealth floor are unchanged. Native
underflow-to-zero remains absorbing. There is no new return or variance cap.
"""
import math
import numpy as np
from numba import njit

LOG_FLOOR = math.log(1e-300)
LOG_UNDERFLOW = math.log(np.nextafter(0., 1.))-math.log(2.)


@njit(cache=True, fastmath=False, inline='always')
def wealth_step(wealth, log_wealth, logarithmic, previous, cumulative, simple):
    if logarithmic:
        log_wealth += math.log1p(simple)
        if log_wealth < LOG_UNDERFLOW:
            # Preserve the existing numerical absorbing-zero rule.
            wealth, logarithmic, current = 0., False, LOG_FLOOR
        else:
            current = max(log_wealth, LOG_FLOOR)
    else:
        product = wealth*(1.+simple)
        if math.isfinite(product):
            wealth = product
            current = math.log(max(wealth, 1e-300))
        elif math.isinf(product) and wealth > 0 and math.isfinite(simple):
            log_wealth = math.log(wealth)+math.log1p(simple)
            logarithmic = True
            current = max(log_wealth, LOG_FLOOR)
        else:
            raise ValueError('Invalid wealth product outside finite clipped return law')
    cumulative += current-previous
    return wealth, log_wealth, logarithmic, current, cumulative


@njit(cache=True, fastmath=False)
def raw_block(history, source, indices, delta, wealth, logs, modes, previous, cumulative):
    out = np.empty(indices.shape)
    upper = np.finfo(np.float64).max/10.
    max_log = math.log1p(upper)
    overflows = 0
    for p in range(len(wealth)):
        for h in range(indices.shape[1]):
            i = indices[p, h]
            if delta[p] == 0.:
                simple = source[i]
            else:
                raw = history[i]+delta[p]
                simple = upper if raw > max_log else min(max(math.expm1(raw), -.999999), upper)
            was = modes[p]
            wealth[p], logs[p], modes[p], previous[p], cumulative[p] = wealth_step(
                wealth[p], logs[p], modes[p], previous[p], cumulative[p], simple)
            overflows += int(modes[p] and not was)
            out[p, h] = cumulative[p]
    return out, overflows


@njit(cache=True, fastmath=False)
def fhs_block(pool, indices, mean, target, alpha, beta, variance, wealth, logs, modes, previous, cumulative):
    out = np.empty(indices.shape)
    upper = np.finfo(np.float64).max/10.
    max_log = np.log1p(upper)
    omega = 1.-alpha-beta
    overflows = 0
    for p in range(len(wealth)):
        for h in range(indices.shape[1]):
            normalized = np.sqrt(variance[p])*pool[indices[p, h]]
            raw = mean[p]+np.sqrt(target)*normalized
            if raw > max_log:
                simple = upper
            else:
                simple = np.expm1(raw)
                if simple < -.999999:
                    simple = -.999999
            was = modes[p]
            wealth[p], logs[p], modes[p], previous[p], cumulative[p] = wealth_step(
                wealth[p], logs[p], modes[p], previous[p], cumulative[p], simple)
            overflows += int(modes[p] and not was)
            out[p, h] = cumulative[p]
            variance[p] = omega+alpha*normalized*normalized+beta*variance[p]
    return out, overflows
