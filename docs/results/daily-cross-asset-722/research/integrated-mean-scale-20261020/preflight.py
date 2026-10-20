"""Training-only local-scale qualification, before reading any candidate field score."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
from scipy.integrate import quad
import candidate
import sweep as api
OWN = Path(__file__).resolve().parent
ROOT = candidate.ROOT
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
plan = json.loads(roster.read_text())
records = []
law = candidate.retained.parent.incumbent.base
p = (np.arange(120, dtype=float)+.5)/120
for case in plan['phases']['development']:
    row = case['series']
    dates, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        prefix = values[:origin['position']+1]
        scale, df, lrv, batches = law.batch_mean_scale(prefix)
        mu = float(prefix.mean())
        se = candidate.retained.gaussian_sd(df, scale)
        z = mu/se if se else 0.
        u = candidate.local_scale_quantile(p, z)
        error = float(np.max(np.abs(candidate.local_scale_cdf(u, z)-p)))
        assert error < 1e-10
        a = z*z/2
        ii = [quad(lambda t: t**k*np.exp(a*(t*t-1)), 0, 1, epsabs=1e-13)[0] for k in (0, 2, 4)]
        ew, ew2 = ii[1]/ii[0], ii[2]/ii[0]
        offsets = candidate.mean_offsets(mu, df, scale, 120, 317)
        assert np.isfinite(offsets).all()
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'sample_mean': mu, 'mean_SE': se, 'z': z, 'mean_retention': ew,
            'posterior_mean': mu*ew, 'posterior_variance': se*se*ew+mu*mu*(ew2-ew*ew),
            'CDF_inverse_max_error': error})
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'maximum_abs_training_mean_z': max(abs(r['z']) for r in records),
    'maximum_CDF_inverse_error': max(r['CDF_inverse_max_error'] for r in records),
    'median_mean_retention': float(np.median([r['mean_retention'] for r in records])), 'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
