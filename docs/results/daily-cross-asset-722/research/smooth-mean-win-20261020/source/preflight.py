"""Training-only mean signal and regularization audit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import candidate
import sweep as api
OWN = Path(__file__).resolve().parent
ROOT = candidate.ROOT
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
plan = json.loads(roster.read_text())
records = []
law = candidate.retained.parent.incumbent.base
for case in plan['phases']['development']:
    row = case['series']
    dates, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        prefix = values[:origin['position']+1]
        scale, df, lrv, batches = law.batch_mean_scale(prefix)
        mu = float(prefix.mean())
        se = candidate.retained.gaussian_sd(df, scale)
        center = candidate.regularized_mean(mu, se)
        assert np.isfinite(center) and abs(center) <= abs(mu)
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'], 'family': row['family'],
            'origin_label': origin['origin_label'], 'training_rows': len(prefix),
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'sample_mean': mu, 'regularized_mean': center, 'Gaussian_mean_se': se,
            'shrink_weight': center/mu if mu else 0.0,
            'first_half_mean': float(prefix[:len(prefix)//2].mean()),
            'second_half_mean': float(prefix[len(prefix)//2:].mean())})
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'median_shrink_weight': float(np.median([r['shrink_weight'] for r in records])),
    'opposite_training_half_mean_sign_count': sum(r['first_half_mean']*r['second_half_mean'] < 0 for r in records),
    'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
