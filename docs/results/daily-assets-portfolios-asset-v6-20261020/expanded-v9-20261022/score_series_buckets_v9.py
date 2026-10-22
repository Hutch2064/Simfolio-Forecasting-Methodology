"""Posthoc external V9 prototype; no quarters and no forecast/learner changes.

Freeze weights from the entire planned schedule before reading losses. Missing
requests contribute no invented loss: observed numerator/support are reported,
with a conditional partial mean; final headline requires complete coverage.
"""
import math
import hashlib
import json
from collections import defaultdict
from score_horizon_buckets_v8 import BUCKETS, _bucket
import numpy as np

SCORE_PROTOCOL_ID = 'daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters'


def origin_score(losses, history, origin_date, future_dates):
    """Raw daily mean CRPS; history is accepted for API compatibility, not scaling."""
    loss=np.asarray(losses,dtype=np.float64)
    dates=np.asarray(future_dates).astype('datetime64[D]')
    origin=np.datetime64(str(origin_date)[:10],'D')
    if loss.ndim!=1 or not loss.size or dates.shape!=loss.shape:
        raise ValueError('Matching nonempty daily losses and dates required')
    if not np.isfinite(loss).all() or np.any(loss<0):
        raise ValueError('Finite nonnegative raw CRPS required')
    if np.isnat(origin) or np.isnat(dates).any() or np.any(dates<=origin) or np.any(dates[1:]<=dates[:-1]):
        raise ValueError('Valid strictly increasing dates after origin required')
    return float(np.mean(loss))


def _key(row):
    if 'request_id' in row:
        return str(row['request_id'])
    if 'task_key' in row:
        return str(row['task_key'])
    return repr((str(row['series_id']), str(row['origin_date']),
                 str(row.get('requested_end_date',row.get('end_date'))), _bucket(row)))


def build_weights(planned_records, metadata):
    """Freeze equal hierarchical series weights and within-series request means."""
    grouped = {b: defaultdict(list) for b in BUCKETS}
    identities = set()
    for row in planned_records:
        key = _key(row)
        if key in identities:
            raise ValueError('Duplicate planned request identity: '+key)
        identities.add(key)
        grouped[_bucket(row)][str(row['series_id'])].append(key)
    requests = {}
    for bucket, series in grouped.items():
        tree = {}
        for sid in series:
            meta = metadata[sid]
            branch = meta['panel_branch']
            if branch not in ('assets','portfolios'):
                raise ValueError('Unknown panel branch')
            tree.setdefault(branch,{}).setdefault(meta['asset_class'],{}).setdefault(meta['family'],[]).append(sid)
        if 'assets' not in tree:
            raise ValueError('Planned bucket requires individual-asset support: '+bucket)
        for branch, classes in tree.items():
            for families in classes.values():
                for members in families.values():
                    series_weight = 1/len(tree)/len(classes)/len(families)/len(members)
                    for sid in members:
                        for key in series[sid]:
                            requests[key]={'bucket':bucket,'series_id':sid,
                                           'weight':series_weight/len(series[sid])}
    fingerprint=hashlib.sha256(json.dumps(requests,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'score_protocol_id':SCORE_PROTOCOL_ID,'weight_plan_sha256':fingerprint,'requests':requests,
            'planned_request_count':len(requests)}


def aggregate_records(records, metadata, *, weights, score_key='score'):
    """Use fixed full-schedule coefficients; never fill unseen requests with zero."""
    if weights['score_protocol_id'] != SCORE_PROTOCOL_ID:
        raise ValueError('Wrong frozen weight protocol')
    seen=set(); totals={b:[] for b in BUCKETS};support={b:[] for b in BUCKETS}
    counts={b:0 for b in BUCKETS}
    for row in records:
        key=_key(row)
        if key in seen:raise ValueError('Duplicate observed request identity: '+key)
        seen.add(key)
        entry=weights['requests'][key]
        if entry['bucket'] != _bucket(row) or entry['series_id'] != str(row['series_id']):
            raise ValueError('Observed request identity disagrees with plan')
        value=float(row[score_key])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Request score must be finite and nonnegative')
        b=entry['bucket'];w=entry['weight']
        totals[b].append(w*value);support[b].append(w);counts[b]+=1
    buckets={}
    for b in BUCKETS:
        mass=math.fsum(support[b]); numerator=math.fsum(totals[b])
        expected=sum(e['bucket']==b for e in weights['requests'].values())
        buckets[b]={'observed_weight_support':mass,'observed_weighted_numerator':numerator,
                    'partial_score':numerator/mass if mass else None,
                    'observed_requests':counts[b],'planned_requests':expected,
                    'complete':counts[b]==expected}
    complete=all(d['complete'] for d in buckets.values())
    partial=(math.fsum(d['partial_score'] for d in buckets.values())/3
             if all(d['partial_score'] is not None for d in buckets.values()) else None)
    return {'score_protocol_id':SCORE_PROTOCOL_ID,'score':partial if complete else None,
            'partial_score':partial,'complete':complete,'buckets':buckets,
            'weight_plan_sha256':weights['weight_plan_sha256'],
            'observed_request_ids':sorted(seen),
            'partial_score_warning':'Conditional observed-support means can move as missing requests arrive; not a final headline',
            'bucket_weights':{b:1/3 for b in BUCKETS}}


def compare(algorithm, baseline):
    """Two explicitly different estimands; only compare identical coverage."""
    if (algorithm['weight_plan_sha256'] != baseline['weight_plan_sha256'] or
            algorithm['observed_request_ids'] != baseline['observed_request_ids']):
        raise ValueError('Identical frozen weights and matched request identities required')
    for b in BUCKETS:
        for k in ('observed_requests','planned_requests','observed_weight_support'):
            if algorithm['buckets'][b][k] != baseline['buckets'][b][k]:
                raise ValueError('Matched request coverage required')
    if not algorithm['complete'] or not baseline['complete']:
        raise ValueError('Final comparison requires full planned coverage')
    a=[algorithm['buckets'][b]['partial_score'] for b in BUCKETS]
    ref=[baseline['buckets'][b]['partial_score'] for b in BUCKETS]
    if min(ref)<=0:raise ValueError('Positive baseline bucket losses required')
    return {'percent_gain_of_equal_bucket_loss':100*(1-math.fsum(a)/math.fsum(ref)),
            'mean_equal_bucket_percent_gain':100*math.fsum(1-x/y for x,y in zip(a,ref))/3,
            'posthoc_definition':True,
            'relative_metric_interpretation':'Equal relative influence per bucket; baseline-dependent loss weights differ from equal-bucket Q'}
