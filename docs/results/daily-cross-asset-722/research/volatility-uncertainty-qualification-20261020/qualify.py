"""Twelve predeclared short/long training prefixes: importance precision only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import hashlib
import datetime
import numpy as np
import posterior
import sweep as api
OWN = Path(__file__).resolve().parent
ROOT = posterior.ROOT
roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
plan = json.loads(roster.read_text())
cases, seen = [], set()
for case in plan['phases']['development']:
    cls = case['series']['asset_class']
    if cls not in seen:
        seen.add(cls)
        ordered = sorted(case['origins'], key=lambda o: o['position'])
        cases.extend((case['series'], o) for o in (ordered[0], ordered[-1]))
assert len(cases) == 12
records = []
for row, origin in cases:
    _, returns, _ = api.load_series(ROOT/'inputs', row)
    history = returns[:origin['position']+1]
    y = history-history.mean()
    y /= np.sqrt(np.mean(y*y))
    mode, covariance = posterior.fit_proposal(y)
    replicas = [posterior.importance_nodes(y, mode, covariance, seed, 12) for seed in (3101, 3102)]
    error = float(np.max(np.abs(replicas[0]['mean']-replicas[1]['mean'])))
    records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
        'origin_label': origin['origin_label'], 'training_n': len(y),
        'prefix_sha256': hashlib.sha256(history.tobytes()).hexdigest(),
        'mode': mode.tolist(), 'proposal_base_covariance': covariance.tolist(),
        'replicas': [{'mean': r['mean'].tolist(), 'covariance': r['covariance'].tolist(),
                      'ESS': r['ESS'], 'largest_normalized_weight': r['largest_normalized_weight'],
                      'nodes': r['nodes']} for r in replicas],
        'replicated_coefficient_mean_max_abs_difference': error})
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'training_prefixes': len(records),
    'classes': len(seen), 'forecast_scores_or_targets_read': False,
    'minimum_replica_ESS': min(x['ESS'] for r in records for x in r['replicas']),
    'maximum_replicated_coefficient_mean_abs_difference': max(r['replicated_coefficient_mean_max_abs_difference'] for r in records),
    'nodes_per_scramble': 4096, 'scrambles_per_prefix': 2,
    'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OWN.glob('*.py')},
    'roster_sha256': hashlib.sha256(roster.read_bytes()).hexdigest(), 'records': records,
    'financial_model_not_yet_admitted': True,
    'limitation': 'Working Gaussian likelihood, fixed sample mean and unconditional variance. Stationarity-constrained flat coefficient prior; empirical-shock predictive integration and all245-prefix precision checks remain unimplemented.'}
(OWN/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('source_hashes', 'records')}), flush=True)
