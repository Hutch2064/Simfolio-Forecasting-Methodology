"""Equal-day, training-SD/sqrt(native-lead) marginal CRPS protocol.

This version preserves the asset-v6 aggregation hierarchy. It does not replace
the frozen calendar-log scorer used by earlier experiments.
"""
import common
import numpy as np
from score_main import aggregate_records

SCORE_PROTOCOL_ID = 'daily_portfolio_panel_crps_v7_train_sd_sqrt_native_lead_equal_days_two_branch'


def origin_score(losses, history, origin_date, future_dates):
    losses = np.asarray(losses, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    dates = np.asarray(future_dates).astype('datetime64[D]')
    if losses.ndim != 1 or not losses.size or dates.shape != losses.shape:
        raise ValueError('CRPS and native dates must have matching nonempty horizons')
    if history.ndim != 1 or history.size < 2 or not np.all(np.isfinite(history)):
        raise ValueError('training history needs at least two finite observations')
    if not np.all(np.isfinite(losses)) or np.any(losses < 0):
        raise ValueError('CRPS losses must be finite and nonnegative')
    origin = np.datetime64(str(origin_date)[:10], 'D')
    if np.isnat(origin) or np.any(np.isnat(dates)):
        raise ValueError('origin and native dates must be valid')
    if np.any(dates <= origin) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError('native dates must follow the origin and increase strictly')
    sd = float(np.std(history, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('training-only sample SD must be positive and finite')
    leads = np.arange(1, losses.size + 1, dtype=np.float64)
    score = float(np.mean(losses / (sd * np.sqrt(leads))))
    if not np.isfinite(score):
        raise ValueError('normalized origin score is nonfinite')
    return score
