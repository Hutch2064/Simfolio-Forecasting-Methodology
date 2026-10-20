"""Freeze the reused broad roster and fit training prefixes, never field scores."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
from . import gjr_fit, adapter
import sweep_v5 as api
import score_main

OWN = Path(__file__).resolve().parent
V5 = OWN.parent
RESEARCH = V5.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p, d: api.original._atomic_json(p, d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def qualify():
    assert not (OWN/'plan_receipt.json').exists()
    panel = api.load_panel(V5/'inputs', verify_files=False)
    old = read(RESEARCH/'persistent_mean_v4/plan.json')
    phases = old['phases']
    rows = {r['series_id']: r for r in panel['series']}
    for phase, cases in phases.items():
        assert len(cases) == 355
        for case in cases:
            assert case['origin'] in api.schedule(V5/'inputs', rows[case['series_id']])
    pairs = [{(c['series_id'], c['origin']['origin_label']) for c in phases[p]}
             for p in ['development', 'reserved']]
    assert not pairs[0] & pairs[1]
    sources = {str(p.relative_to(V5)): sha(p) for p in OWN.glob('*.py')}
    plan = {'utc': now(), 'candidate_model_id': adapter.MODEL_ID,
        'parent_model_id': adapter.s.BEST, 'paths': 1000, 'workers_when_available': 12,
        'panel_fingerprint': panel['panel_fingerprint'], 'input_root': str(V5/'inputs'),
        'score_protocol_id': score_main.SCORE_PROTOCOL_ID,
        'source_hashes': sources, 'design_sha256': sha(OWN/'DESIGN.md'),
        'source_roster_sha256': sha(RESEARCH/'persistent_mean_v4/plan.json'),
        'phases': phases, 'economic_families': 70,
        'native_source_closure_digest': api.original.stable_digest(api.original._source_closure()),
        'field_forecasts_started': False, 'field_scores_read': False,
        'independent_validation_claim': False, 'diagnostics_have_zero_promotion_influence': True,
        'gate': 'Complete lower sole mainQ than incumbent on355 development then355 reserved; no failures/integrity errors. Broad/full required before promotion.',
        'refitting_prior_cases': 'Training-only source cohort; no original240-path loss reuse as1,000-path scores.'}
    write(OWN/'plan.json', plan)
    write(OWN/'qualification_active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'running', 'field_scores_read': False})
    records, errors = [], []
    for phase, cases in phases.items():
        cache = {}
        for case in cases:
            sid, origin = case['series_id'], case['origin']
            if sid not in cache:
                cache[sid] = api.load_series(V5/'inputs', rows[sid])
            _, values, _ = cache[sid]
            x = np.log1p(np.clip(np.expm1(values[:origin['position']+1]), -.999999, np.finfo(float).max/10))
            try:
                fitted = gjr_fit.fit(x)
                scalar = {k:v for k,v in fitted.items() if k != 'pool'}
                assert fitted['omega_forecast'] > 0 and fitted['innovation_persistence'] < 1
                assert fitted['objective'] <= fitted['nested_parent_objective']+1e-13
                record = {'phase': phase, 'series_id': sid, 'origin_label': origin['origin_label'],
                    'training_sha256': hashlib.sha256(x.tobytes()).hexdigest(), **scalar}
                records.append(record)
            except Exception as exc:
                errors.append({'phase': phase, 'series_id': sid, 'origin_label': origin['origin_label'],
                               'type': type(exc).__name__, 'message': str(exc)})
            if (len(records)+len(errors)) % 50 == 0:
                print(json.dumps({'utc': now(), 'training_prefixes': len(records), 'failures': len(errors),
                                  'field_scores_read': False}), flush=True)
        cache.clear()
    assert all(sha(V5/name) == digest for name, digest in sources.items())
    receipt = {'utc': now(), 'complete': not errors and len(records) == 710,
        'training_prefixes': len(records), 'fit_failures': len(errors), 'errors': errors,
        'negative_shocks_have_larger_fit_response': sum(x['alpha_negative'] > x['alpha_positive'] for x in records),
        'nested_symmetric_parent_selected': sum(x['nested_parent_selected'] for x in records),
        'field_scores_read': False, 'field_forecasts_started': False,
        'records': records, 'diagnostics_are_not_score_or_model_selection': True}
    write(OWN/'preflight.json', receipt)
    write(OWN/'plan_receipt.json', {'utc': now(), 'plan_sha256': sha(OWN/'plan.json'),
        'preflight_sha256': sha(OWN/'preflight.json'), 'complete': receipt['complete'],
        'before_field_forecasts': True})
    write(OWN/'qualification_active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'terminal'})
    print(json.dumps({k:v for k,v in receipt.items() if k != 'records'}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    qualify()
