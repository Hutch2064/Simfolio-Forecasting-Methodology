"""Frozen all245 training-prefix qualification of posterior precision and native pools."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from time import perf_counter
import datetime
import hashlib
import json
import numpy as np
import candidate
import posterior
import sweep as api
OWN = Path(__file__).resolve().parent
ROOT = candidate.ROOT
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
plan = json.loads(roster.read_text())
qualification = json.loads((OWN/'qualification_plan.json').read_text())
assert qualification['roster_sha256'] == sha(roster)
assert all(sha(OWN/k) == v for k, v in qualification['source_hashes'].items())
records = []
timings = []
for case in plan['phases']['development']:
    row = case['series']
    _, values, _ = api.load_series(ROOT/'inputs', row)
    for origin in case['origins']:
        started = perf_counter()
        prefix = values[:origin['position']+1]
        draws, diagnostic = posterior.parameter_draws(prefix, 120, 317)
        assert diagnostic['qualified'] and np.all(draws >= 0) and np.all(draws.sum(axis=1) < 1.)
        prepared = candidate.basis(prefix)
        maximum_mean = maximum_variance_error = 0.
        for a, b in draws:
            fit = candidate.conditional_fit(prepared, a, b)
            maximum_mean = max(maximum_mean, abs(float(fit['pool'].mean())))
            maximum_variance_error = max(maximum_variance_error, abs(float(np.mean(fit['pool']**2))-1.))
            assert np.isfinite(fit['pool']).all() and fit['next_variance_normalized'] > 0
        assert maximum_mean < 1e-12 and maximum_variance_error < 1e-12
        elapsed = perf_counter()-started
        timings.append(elapsed)
        records.append({'series_id': row['series_id'], 'asset_class': row['asset_class'],
            'family': row['family'], 'origin_label': origin['origin_label'],
            'training_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'training_n': len(prefix), 'quadrature_seed': 317, 'seconds': elapsed,
            'maximum_absolute_innovation_pool_mean': maximum_mean,
            'maximum_innovation_pool_variance_error': maximum_variance_error,
            'posterior_diagnostic': diagnostic})
        if len(records) % 10 == 0:
            assert all(sha(OWN/k) == v for k, v in qualification['source_hashes'].items())
            status = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'stage': 'training_numerical_qualification', 'completed_prefixes': len(records),
                'expected_prefixes': 245, 'mean_remaining_compute_seconds_lower_bound': float(np.mean(timings)*(245-len(records))),
                'median_remaining_compute_seconds_lower_bound': float(np.median(timings)*(245-len(records))),
                'scope': 'Measured training qualification compute only; excludes setup/IO/contention and any future field test.'}
            (OWN/'preflight_status.json').write_text(json.dumps(status, indent=2)+'\n')
            print(json.dumps(status), flush=True)
assert len(records) == 245
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True,
    'training_prefixes': len(records), 'families': len({r['family'] for r in records}),
    'classes': len({r['asset_class'] for r in records}), 'field_scores_read': False,
    'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')}, 'metadata_roster_sha256': sha(roster),
    'precision_rules': posterior.PRECISION, 'bounded_node_powers': list(posterior.NODE_POWERS),
    'median_prefix_seconds': float(np.median(timings)),
    'largest_nodes_per_scramble': max(r['posterior_diagnostic']['precision_attempts'][-1]['nodes_per_scramble'] for r in records),
    'minimum_qualified_ESS': min(r['posterior_diagnostic']['precision_attempts'][-1]['minimum_ESS'] for r in records),
    'records': records}
(OWN/'preflight_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_hashes')}), flush=True)
