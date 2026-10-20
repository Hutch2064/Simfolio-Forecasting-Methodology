"""Training-only HARCH numerical qualification and descriptive GARCH residual checks."""
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
import harch
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
        fitted = harch.fit(prefix)
        seconds = time.perf_counter()-started
        old = candidate.retained.FHS.base.fit_variance_target(prefix)
        y = (prefix-prefix.mean())/np.std(prefix)
        features = harch.lag_features(y)
        residual_squared = old['pool']**2
        correlations = {str(lag): float(np.corrcoef(features[22:, j], residual_squared[22:])[0, 1])
                        for j, lag in enumerate(harch.LAGS)}
        assert np.isfinite(list(correlations.values())).all()
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'fit_seconds': seconds,
            'fit': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items()
                    if k not in ['pool', 'last_squared_normalized']},
            'GARCH_training_QMLE': old['objective'],
            'descriptive_GARCH_squared_filtered_shock_correlation_with_lag_squared_returns': correlations})
        if len(records) % 25 == 0:
            print(json.dumps({'training_prefixes_qualified': len(records), 'expected': 245}), flush=True)
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'fixed_lags': list(harch.LAGS), 'constant_variance_selected': sum(r['fit']['constant_variance_selected'] for r in records),
    'starts_not_numerically_converged': sum(not s['eligible'] for r in records for s in r['fit']['fixed_start_results']),
    'median_training_alpha': np.median([r['fit']['alpha'] for r in records], axis=0).tolist(),
    'median_training_GARCH_residual_correlations': {str(lag): float(np.median([
        r['descriptive_GARCH_squared_filtered_shock_correlation_with_lag_squared_returns'][str(lag)] for r in records])) for lag in harch.LAGS},
    'correlations_descriptive_overlapping_training_prefixes_not_independent_significance': True,
    'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
