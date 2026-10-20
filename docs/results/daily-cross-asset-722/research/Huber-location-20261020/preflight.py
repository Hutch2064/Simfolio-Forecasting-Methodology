"""Training-only fixed robust-location assessment with the exact native return round trip."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import candidate
import location
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
    _, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        prefix = np.log1p(np.clip(np.expm1(values[:origin['position']+1]), -.999999, np.finfo(float).max/10))
        center = location.estimate(prefix)
        mean = float(prefix.mean())
        scale, df, lrv, batches = law.batch_mean_scale(prefix)
        se = candidate.retained.gaussian_sd(df, scale)
        new = candidate.parent.regularized_mean(center, se)
        old = candidate.parent.regularized_mean(mean, se)
        assert np.isfinite([center, new, old]).all()
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'sample_mean': mean, 'Huber_location': center, 'retained_mean_SE': se,
            'new_regularized_drift': new, 'prior_regularized_drift': old,
            'zero_MAD': bool(np.median(np.abs(prefix-np.median(prefix))) == 0)})
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'fixed_Huber_constant': location.C, 'zero_MAD_prefixes': sum(r['zero_MAD'] for r in records),
    'median_location_minus_sample_mean': float(np.median([r['Huber_location']-r['sample_mean'] for r in records])),
    'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
