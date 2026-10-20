"""Qualify the frozen likelihood on training prefixes only, before field scores."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
from time import perf_counter
import numpy as np
import candidate
import sweep as api
import t_fit

OWN = Path(__file__).resolve().parent
ROOT = candidate.ROOT
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
plan = read(source)
assert api.load_panel(ROOT/'inputs')['panel_fingerprint'] == plan['panel_fingerprint']
assert all(sha(ROOT/k) == v for k, v in plan['shared_hashes'].items())
rows = []
started = perf_counter()
for case in plan['phases']['development']:
    row = case['series']
    dates, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        prefix = values[:origin['position']+1].copy()
        prefix.flags.writeable = False
        gaussian = t_fit.gaussian.fit_variance_target(prefix)
        fit = t_fit.fit_student(prefix)
        # Conditional Gaussian residual kurtosis measures the motivation without held-out returns.
        pool = gaussian['pool']
        rows.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'training_rows': len(prefix), 'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'Gaussian_filtered_excess_kurtosis': float(np.mean(pool**4)-3),
            **{k: v for k, v in fit.items() if k not in ('pool', 'fixed_start_results')}})
        if len(rows) % 25 == 0:
            print(json.dumps({'qualified_prefixes': len(rows), 'expected': 245, 'elapsed_seconds': perf_counter()-started}), flush=True)
assert len(rows) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
          'training_prefixes': len(rows), 'families': len({r['family'] for r in rows}),
          'classes': len({r['asset_class'] for r in rows}), 'field_scores_read': False,
          'metadata_roster_sha256': sha(source), 'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')},
          'elapsed_seconds': perf_counter()-started,
          'Student_selected': sum(r['likelihood_selected'] == 'Student_t' for r in rows),
          'median_Gaussian_filtered_excess_kurtosis': float(np.median([r['Gaussian_filtered_excess_kurtosis'] for r in rows])),
          'records': rows}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
