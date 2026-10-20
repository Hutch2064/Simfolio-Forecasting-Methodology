"""Measure residual block lengths on training-only prefixes before forecasting."""
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
prior = candidate.retained.parent.incumbent.prior
for case in plan['phases']['development']:
    row = case['series']
    dates, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        prefix = values[:origin['position']+1]
        fit = candidate.retained.FHS.base.fit_variance_target(prefix)
        pool = fit['pool']
        signed = prior._politis_white_block_length(pool)
        squared = prior._politis_white_block_length((pool-pool.mean())**2)
        block = min(len(pool), max(signed, squared))
        assert np.isfinite(block) and 1 <= block <= len(pool)
        records.append({'series_id': row['series_id'], 'family': row['family'],
            'asset_class': row['asset_class'], 'origin_label': origin['origin_label'],
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'training_rows': len(prefix), 'signed_block': float(signed),
            'squared_block': float(squared), 'chosen_block': float(block),
            'filtered_lag1': float(np.corrcoef(pool[:-1], pool[1:])[0, 1]),
            'mean_log_GARCH_multiplier': float(np.mean(np.log(fit['beta']+fit['alpha']*pool**2)))
                if fit['alpha']+fit['beta'] > 0 else None})
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'median_fitted_residual_block': float(np.median([r['chosen_block'] for r in records])),
    'block_greater_than_one_count': sum(r['chosen_block'] > 1 for r in records), 'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
