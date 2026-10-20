"""Training-only numerical qualification, exact native simple/log round trip."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import time
import numpy as np
import candidate
import hmm
import sweep as api
OWN = Path(__file__).resolve().parent
ROOT = candidate.ROOT
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
plan = json.loads(roster.read_text())
records = []
for case in plan['phases']['development']:
    row = case['series']
    _, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        prefix = np.log1p(np.clip(np.expm1(values[:origin['position']+1]), -.999999, np.finfo(float).max/10))
        started = time.perf_counter()
        fitted = hmm.fit(prefix)
        stationary = np.array([fitted['transition'][1, 0], fitted['transition'][0, 1]])
        stationary /= stationary.sum()
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'training_n': len(prefix), 'fit_seconds': time.perf_counter()-started,
            'fit': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items()},
            'stationary_distribution': stationary.tolist()})
        if len(records) % 25 == 0:
            print(json.dumps({'training_prefixes_qualified': len(records), 'expected': 245}), flush=True)
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'constant_histories': sum(r['fit']['constant_history'] for r in records),
    'starts_not_numerically_converged': sum(not s['success'] for r in records for s in r['fit']['starts']),
    'maximum_valid_start_gradient_per_observation': max(s['gradient_per_observation'] for r in records for s in r['fit']['starts'] if s['success']),
    'median_variance_ratio': float(np.median([r['fit']['variance'][1]/r['fit']['variance'][0] for r in records if not r['fit']['constant_history']])),
    'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
