"""All-prefix fit/gradient qualification; no field forecasts or evaluation scores."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import candidate
import crps_fit
import sweep as api
OWN = Path(__file__).resolve().parent
ROOT = candidate.ROOT
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
plan = json.loads(roster.read_text())
records = []
for case in plan['phases']['development']:
    row = case['series']
    dates, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        prefix = values[:origin['position']+1]
        fit = crps_fit.fit_crps(prefix)
        original = crps_fit.gaussian.fit_variance_target(prefix)
        y = (prefix-prefix.mean())/np.sqrt(np.mean((prefix-prefix.mean())**2))
        parent_crps = crps_fit.conditional_loss(y, original['alpha'], original['beta'])[0]
        assert np.isfinite(fit['pool']).all()
        assert abs(fit['pool'].mean()) < 1e-12 and abs(np.mean(fit['pool']**2)-1) < 1e-12
        assert fit['alpha'] >= 0 and fit['beta'] >= 0 and fit['alpha']+fit['beta'] < 1
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'alpha': fit['alpha'], 'beta': fit['beta'], 'parent_alpha': original['alpha'],
            'parent_beta': original['beta'], 'CRPS_objective': fit['objective'],
            'parent_CRPS_objective': parent_crps, 'fit_seconds': fit['fit_seconds'],
            'fixed_start_results': fit['fixed_start_results']})
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'median_fit_seconds': float(np.median([r['fit_seconds'] for r in records])),
    'CRPS_fit_lower_training_CRPS_count': sum(r['CRPS_objective'] < r['parent_CRPS_objective'] for r in records),
    'CRPS_fit_numerical_boundary_equivalent_count': sum(abs(r['CRPS_objective']-r['parent_CRPS_objective']) < 1e-10 for r in records),
    'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
