"""Distinguish one-step moving-average dependence using training prefixes only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import sys
import datetime
import numpy as np
import ma1
OWN = Path(__file__).resolve().parent
ROOT = next(p for p in OWN.parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import sweep as api
roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
records = []
for case in json.loads(roster.read_text())['phases']['development']:
    row = case['series']
    _, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        history = values[:origin['position']+1]
        fitted = ma1.fit(history)
        halves = [ma1.fit(x)['theta'] for x in np.array_split(history, 2)]
        errors = fitted.pop('errors')
        factors = fitted.pop('variance_factors')
        assert np.isfinite(errors).all() and np.all(factors >= 1.-1e-12)
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'prefix_sha256': hashlib.sha256(history.tobytes()).hexdigest(), **fitted,
            'two_half_theta': halves, 'same_nonzero_sign_in_both_halves': bool(halves[0]*halves[1] > 0)})
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'training_prefixes': len(records), 'forecast_scores_or_targets_read': False,
    'roster_sha256': hashlib.sha256(roster.read_bytes()).hexdigest(),
    'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OWN.glob('*.py')},
    'classes': {cls: {'prefixes': len(rows), 'median_theta': float(np.median([r['theta'] for r in rows])),
        'same_sign_in_both_halves_fraction': float(np.mean([r['same_nonzero_sign_in_both_halves'] for r in rows]))}
        for cls in sorted({r['asset_class'] for r in records})
        for rows in [[r for r in records if r['asset_class'] == cls]]},
    'records': records, 'overlapping_prefixes_not_independent': True,
    'not_a_forecast_candidate_admission': True,
    'unresolved_before_field': 'Coherent conditional-volatility innovation deconvolution, latent boundary uncertainty, parameter uncertainty and native accounting integration still require qualification; no field outcomes exist.'}
(OWN/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
