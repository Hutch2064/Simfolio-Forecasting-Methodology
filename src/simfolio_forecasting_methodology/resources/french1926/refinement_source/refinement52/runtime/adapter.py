"""Research-only exact calendar and single-holding shell specialization.

No fit, draw, path, horizon, scoring or policy changes. ``call`` swaps only its
owner module's shell reference during one synchronous call; each worker must
own its model modules. Original generic rejoin remains the multiasset fallback.
"""
from contextlib import contextmanager
import math
import time

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=False, fastmath=False)
def single_growth(growth, mask, cost):
    """Same scalar operation order as exact_path_kernels.rejoin_growth.

    Keep cumulative holdings and exp/log round trips: log returns are not a
    byte-identical shortcut even for a single unit-weight holding.
    """
    sims, horizon, _ = growth.shape
    out = np.empty((sims, horizon))
    for sim in range(sims):
        holding = 1.
        for day in range(horizon):
            previous = -0. + holding
            holding *= growth[sim, day, 0]
            ending = -0. + holding
            if mask[day]:
                denominator = max(ending, 1e-300)
                turnover = -0. + abs(1. - holding / denominator)
                ending = max(ending - ending * .5 * turnover * cost, 0.)
                holding = ending * 1.
            out[sim, day] = math.log(max(ending, 1e-300) / max(previous, 1e-300))
    return out


class Shell:
    def __init__(self, original):
        self.original = original
        self.timings = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _validate_calendar(self, training, context):
        # Delegate every original check; replace only expected-calendar creation.
        # Rather than monkeypatching pandas globally, retain the original source
        # checks verbatim below and use its fallback for timezone-bearing inputs.
        tick = time.perf_counter()
        try:
            return self._calendar(training, context)
        finally:
            self.timings['calendar_seconds'] = time.perf_counter() - tick

    def _calendar(self, training, context):
        if training.training_dates is None:
            raise ValueError('frontier_requires_explicit_training_dates')
        if context.origin_date is None or not str(context.origin_date).strip():
            raise ValueError('frontier_requires_explicit_origin_date')
        if context.future_dates is None:
            raise ValueError('frontier_requires_explicit_future_dates')
        train_dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(training.training_dates), errors='raise'))
        future_dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(context.future_dates), errors='raise'))
        if train_dates.tz is not None or future_dates.tz is not None:
            return self.original._validate_calendar(training, context)
        if len(train_dates) != len(training.portfolio_log_returns):
            raise ValueError('frontier_training_dates_do_not_align')
        if len(future_dates) != int(context.horizon_days):
            raise ValueError('frontier_future_dates_do_not_align')
        if len(train_dates) == 0 or len(future_dates) == 0 or train_dates.hasnans or future_dates.hasnans:
            raise ValueError('frontier_calendar_contains_missing_dates')
        if not train_dates.is_monotonic_increasing or not train_dates.is_unique:
            raise ValueError('frontier_training_dates_must_be_strictly_increasing')
        if not future_dates.is_monotonic_increasing or not future_dates.is_unique:
            raise ValueError('frontier_future_dates_must_be_strictly_increasing')
        if future_dates[0] <= train_dates[-1]:
            raise ValueError('frontier_future_dates_must_follow_training_dates')
        # BDay(1) and bdate_range use a Monday-Friday, holiday-free calendar,
        # normalized to midnight. busday_offset implements that same calendar.
        next_day = train_dates[-1].to_datetime64().astype('datetime64[D]') + np.timedelta64(1, 'D')
        expected_future = pd.DatetimeIndex(np.busday_offset(next_day,
            np.arange(int(context.horizon_days)), roll='forward').astype('datetime64[ns]'))
        if not future_dates.equals(expected_future):
            raise ValueError('frontier_future_dates_must_follow_source_business_day_policy')
        try:
            origin = pd.Timestamp(str(context.origin_date)).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError('frontier_invalid_origin_date') from exc
        if origin != train_dates[-1].normalize():
            raise ValueError('frontier_origin_date_must_equal_training_end')
        return train_dates, future_dates

    def rejoin(self, paths, weights, mask, *, cost_per_turnover_bps=15.):
        tick = time.perf_counter()
        try:
            target = np.maximum(np.asarray(weights, np.float64), 0.)
            target /= target.sum()
            if paths.ndim != 3 or paths.shape[2] != 1 or target.shape != (1,) or target[0] != 1.:
                return self.original.rejoin(paths, weights, mask,
                    cost_per_turnover_bps=cost_per_turnover_bps)
            mask = np.asarray(mask, np.bool_)
            cost = max(float(cost_per_turnover_bps), 0.) / 10000.
            out = np.empty(paths.shape[:2])
            for start in range(0, len(paths), 8):
                stop = min(start + 8, len(paths))
                # Preserve original NumPy ufuncs and original block geometry.
                growth = np.exp(np.clip(paths[start:stop], -745., 50.))
                out[start:stop] = single_growth(growth, mask, cost)
            return out
        finally:
            self.timings['rejoin_seconds'] = time.perf_counter() - tick


@contextmanager
def hooks(owner):
    original = owner.shell
    proxy = Shell(original)
    owner.shell = proxy
    try:
        yield proxy
    finally:
        owner.shell = original


def call(owner, candidate, training, context):
    """Return (full daily paths, shell stage timings) for an existing model."""
    with hooks(owner) as shell:
        result = candidate.simulate_daily_log_returns(training, context)
    return result, dict(shell.timings)
