"""External v8 headline: three equal requested-calendar-horizon buckets.

The request score and within-branch hierarchy are unchanged v7. Branch scores
are combined before buckets, retaining portfolio-only quarters. No learner,
forecast, control, or retained loss is modified. February 29 anniversaries
clamp to February 28 in non-leap years.
"""
from datetime import date
from pathlib import Path
import sys

_V = str(Path(__file__).resolve().parent / 'asset_level_v6')
if _V not in sys.path:
    sys.path.insert(0, _V)
from score_aligned_v7 import origin_score
from score_main import legacy as _legacy

SCORE_PROTOCOL_ID = 'daily_portfolio_panel_crps_v8_train_sd_sqrt_native_lead_equal_calendar_horizon_buckets'
BUCKETS = ('short', 'medium', 'long')


def _date(value):
    return date.fromisoformat(str(value)[:10])


def _anniversary(origin, years):
    try:
        return origin.replace(year=origin.year + years)
    except ValueError:
        return origin.replace(year=origin.year + years, day=28)


def horizon_bucket(origin_date, requested_end_date):
    """Short <=3 calendar years; medium >3 and <20; long >=20."""
    origin, end = _date(origin_date), _date(requested_end_date)
    if end <= origin:
        raise ValueError('Requested end date must follow the origin')
    if end <= _anniversary(origin, 3):
        return 'short'
    return 'long' if end >= _anniversary(origin, 20) else 'medium'


def _bucket(record):
    annotation = record.get('horizon_bucket')
    if annotation is not None and annotation not in BUCKETS:
        raise ValueError('Unknown horizon_bucket: ' + str(annotation))
    end = record.get('requested_end_date', record.get('end_date'))
    if end is not None:
        derived = horizon_bucket(record['origin_date'], end)
        if annotation is not None and annotation != derived:
            raise ValueError('horizon_bucket disagrees with requested calendar dates')
        return derived
    if annotation is None:
        raise ValueError('Explicit horizon_bucket or requested end date required; native H is insufficient')
    return annotation


def _aggregate_bucket(records, metadata, score_key):
    branches = {}
    coverage = {}
    for branch in ('assets', 'portfolios'):
        rows = [r for r in records if metadata[str(r['series_id'])]['panel_branch'] == branch]
        coverage[branch] = len(rows)
        branches[branch] = _legacy.aggregate_records(rows, metadata, score_key=score_key)
    assets, portfolios = (branches[b]['score'] for b in ('assets', 'portfolios'))
    # No branch or quarter is dropped to manufacture common support.
    complete = assets is not None
    weight = .5 if complete and portfolios is not None else 0.
    value = (assets + portfolios) / 2 if weight else assets
    return {'score': value, 'complete': complete, 'branches': branches,
            'branch_coverage': coverage, 'portfolio_weight': weight,
            'maximum_portfolio_weight': weight,
            'incomplete_reason': None if complete else 'No individual-asset support in bucket'}


def aggregate_records(records, metadata, *, score_key='score'):
    """Equal-third Q requires all three buckets with asset support.

    Within each bucket, independently aggregate each branch using the existing
    series-quarter/family/class/quarter hierarchy, then average the two branch
    scores equally when both exist. Asset-only buckets retain 100% assets.
    Unlike v7's joint-quarter branch blend, this retains portfolio-only quarters
    while capping the complete bucket's portfolio weight at 50%. A whole bucket
    lacking assets is incomplete. Never average per-bucket percentage gains.
    """
    grouped = {bucket: [] for bucket in BUCKETS}
    for record in records:
        grouped[_bucket(record)].append(record)
    results = {bucket: _aggregate_bucket(rows, metadata, score_key)
               for bucket, rows in grouped.items()}
    coverage = {bucket: len(rows) for bucket, rows in grouped.items()}
    complete = all(result['complete'] for result in results.values())
    scores = {bucket: results[bucket]['score'] for bucket in BUCKETS}
    return {'score_protocol_id': SCORE_PROTOCOL_ID,
            'score': sum(scores.values()) / 3 if complete else None,
            'complete': complete,
            'missing_buckets': [bucket for bucket in BUCKETS if not coverage[bucket]],
            'incomplete_buckets': [bucket for bucket in BUCKETS if not results[bucket]['complete']],
            'bucket_weights': {bucket: 1 / 3 for bucket in BUCKETS},
            'bucket_coverage': coverage, 'bucket_scores': scores,
            'buckets': results, 'diagnostics_in_score': False}
