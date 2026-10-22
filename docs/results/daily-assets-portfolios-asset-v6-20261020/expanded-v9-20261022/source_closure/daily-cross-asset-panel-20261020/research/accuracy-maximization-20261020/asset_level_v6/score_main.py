"""One canonical score: normalized CRPS, calendar leads, capped portfolio branch."""
import common
import numpy as np
import scoring as legacy

SCORE_PROTOCOL_ID = 'daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch'


def origin_score(losses, history, origin_date, future_dates):
    # Exact existing sqrt(h) calculation, now the sole selected objective.
    return legacy.normalized_origin_scores(losses, history, origin_date, future_dates)[1]


def aggregate_records(records, metadata, *, score_key='score'):
    records = list(records)
    branches = {}
    for branch in ['assets', 'portfolios']:
        rows = [r for r in records if metadata[str(r['series_id'])]['panel_branch'] == branch]
        branches[branch] = legacy.aggregate_records(rows, metadata, score_key=score_key)
    asset_quarters = branches['assets']['quarter_scores']
    portfolio_quarters = branches['portfolios']['quarter_scores']
    unsupported = sorted(set(portfolio_quarters)-set(asset_quarters))
    if unsupported:
        raise ValueError('Portfolio quarters lack matched individual-asset support: '+str(unsupported))
    quarters = {q: (value+portfolio_quarters[q])/2 if q in portfolio_quarters else value
                for q, value in asset_quarters.items()}
    return {'score': float(np.mean(list(quarters.values()))) if quarters else None,
            'quarter_count': len(quarters), 'quarter_scores': quarters,
            'quarter_branch_scores': {q: {branch: data['quarter_scores'][q]
                for branch, data in branches.items() if q in data['quarter_scores']} for q in quarters},
            'portfolio_weight_by_quarter': {q: .5 if q in portfolio_quarters else 0. for q in quarters},
            'maximum_portfolio_weight': .5 if portfolio_quarters else 0.,
            'diagnostics_in_score': False}
