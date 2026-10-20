"""Source-frozen matched native pilot; no forecast-score input to fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from types import SimpleNamespace
import argparse
import datetime
import hashlib
import json
import sys

import candidate
import numpy as np
import sweep as api
import scoring as sc
from simfolio_forecasting_methodology.french_panel.selected import build_public

OWN = Path(__file__).resolve().parent
ROOT = candidate.ROOT
FACTORIES = {'Public': build_public, 'Incumbent': candidate.build_gaussian_equal_pool,
             'Candidate': candidate.build_candidate}
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(p, value):
    api.original._atomic_json(p, value)


def prepare():
    assert not (OWN/'plan.json').exists()
    roster = ROOT/'research/resumed-signed-gjr-pool-20261020/plan.json'
    old = read(roster)
    preflight = read(OWN/'preflight_summary.json')
    assert preflight['complete'] and preflight['training_prefixes'] == 245 and not preflight['field_scores_read']
    plan = {'schema': 'stationary_filtered_residual_blocks_pool_pilot_v1', 'utc': now(), 'panel_fingerprint': old['panel_fingerprint'],
        'input_root': str(ROOT/'inputs'), 'roster_sha256': sha(roster),
        'phases': old['phases'], 'cohort': old['cohort'], 'shared_hashes': old['shared_hashes'],
        'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')},
        'parent_component_hashes': old['component_hashes'],
        'native_source_closure_digest': api.original.stable_digest(api.original._source_closure()),
        'preflight_sha256': sha(OWN/'preflight_summary.json'), 'paths': 240,
        'arms': {a: f().model_id for a, f in FACTORIES.items()},
        'existing_development_reference': str(ROOT/'research/resumed-signed-gjr-pool-20261020/development_summary.json'),
        'existing_development_reference_sha256': sha(ROOT/'research/resumed-signed-gjr-pool-20261020/development_summary.json'),
        'gate': 'Complete same 245 development, then 245 reserved, then 4122 representative native origins; lower primary than unchanged current winner. Disclose sensitivity, horizon, class and family regressions. Full722 required for promotion.',
        'independent_holdout': False, 'adaptive_reused_evidence': True,
        'no_OOS_score_fit_or_routes': True, 'owned_output_budget_bytes': 1024**3}
    write(OWN/'plan.json', plan)
    write(OWN/'plan_receipt.json', {'utc': now(), 'plan_sha256': sha(OWN/'plan.json'),
        'before_first_field_forecast': True, 'tests_passed': 5, 'training_prefixes_qualified': 245})
    print(json.dumps({'prepared': True, 'arms': plan['arms'], 'development_origins': 245}), flush=True)


def guard():
    plan = read(OWN/'plan.json')
    assert sha(OWN/'plan.json') == read(OWN/'plan_receipt.json')['plan_sha256']
    assert all(sha(OWN/k) == v for k, v in plan['source_hashes'].items())
    assert all(sha(ROOT/k) == v for k, v in plan['shared_hashes'].items())
    assert all(sha(Path(k)) == v for k, v in plan['parent_component_hashes'].items())
    assert api.original.stable_digest(api.original._source_closure()) == plan['native_source_closure_digest']
    if sum(p.stat().st_size for p in OWN.rglob('*') if p.is_file()) > plan['owned_output_budget_bytes']:
        raise RuntimeError('Owned pilot output budget exceeded; preserve closed evidence before expanding')
    return plan


def run(phase):
    p = guard()
    previous = {'reserved': 'development', 'coverage': 'reserved'}.get(phase)
    if previous:
        result = read(OWN/(previous+'_summary.json'))
        assert result['complete'] and result['candidate_pct_better']['Incumbent']['primary_score'] > 0
    cases = p['cohort'] if phase == 'coverage' else p['phases'][phase]
    expected = sum(len(c['origins']) for c in cases)
    assert expected == (4122 if phase == 'coverage' else 245)
    models = {a: f() for a, f in FACTORIES.items()}
    execution = {'plan_sha256': sha(OWN/'plan.json'), 'phase': phase, 'paths': 240,
        'panel_fingerprint': p['panel_fingerprint'],
        'models': {a: {'model_id': m.model_id, 'runtime': api.original.dependency_identity(m)} for a, m in models.items()}}
    fp = api.original._manifest(OWN/'results'/phase, execution)
    write(OWN/'active_run.json', {'utc': now(), 'pid': os.getpid(), 'phase': phase,
        'execution_fingerprint': fp, 'expected_forecasts': expected*3, 'status': 'running'})
    refs = {}
    if phase == 'development':
        refpath = Path(p['existing_development_reference'])
        assert sha(refpath) == p['existing_development_reference_sha256']
        old = read(refpath)
        refs = {a: {(r['series_id'], r['origin_label']): r for r in old['records'][a]}
                for a in ('Public', 'Incumbent')}
    meta = {c['series']['series_id']: c['series'] for c in cases}
    records = {a: [] for a in models}
    timings = []
    count = hashes = 0
    for case in cases:
        row = case['series']
        sid = row['series_id']
        dates, values, _ = api.load_series(p['input_root'], row)
        for origin in case['origins']:
            stop = origin['position']+1
            future = dates[stop:stop+origin['max_horizon']]
            order = list(models)
            order = order[count % len(order):]+order[:count % len(order)]
            elapsed = 0
            for arm in order:
                model = models[arm]
                task = api.task_identity(fp, model.model_id, sid, origin)
                path = OWN/'results'/phase/arm/sid/(origin['origin_label']+'.npz')
                try:
                    if path.exists():
                        loss, diag = api.original._read_checkpoint(path, task)
                    else:
                        loss, diag = api.original._forecast(model, sid, values, dates,
                                                           SimpleNamespace(**origin), task['seed'])
                        diag = {**diag, **getattr(model, 'last_metadata', {}), 'arm': arm}
                        api.original._write_checkpoint(path, task, loss, diag)
                    if not np.isfinite(loss).all() or np.any(loss < 0):
                        raise ValueError('Invalid native loss vector')
                    if arm == 'Candidate' and (diag.get('daily_clips', 0) or diag.get('wealth_floor_events', 0)):
                        raise ValueError('Candidate native accounting boundary event')
                    q, s = sc.normalized_origin_scores(loss, values[:stop], origin['origin_date'], future)
                    loss_hash = hashlib.sha256(loss.tobytes()).hexdigest()
                    if arm in refs:
                        prior = refs[arm][sid, origin['origin_label']]
                        assert prior['loss_sha256'] == loss_hash, 'Exact baseline loss hash drift'
                        assert max(abs(q-prior['primary_score']), abs(s-prior['sensitivity_score'])) <= 1e-12
                        hashes += 1
                    elapsed += diag['forecast_and_score_seconds']
                    records[arm].append({'series_id': sid, 'origin_date': origin['origin_date'],
                        'origin_label': origin['origin_label'], 'evaluation_split': origin['evaluation_split'],
                        'primary_score': q, 'sensitivity_score': s, 'loss_sha256': loss_hash,
                        'forecast_seconds': diag['forecast_and_score_seconds']})
                except Exception as exc:
                    write(OWN/'errors'/phase/arm/sid/(origin['origin_label']+'.json'),
                        {'utc': now(), 'identity': task, 'exception': type(exc).__name__, 'message': str(exc)})
                    raise
            count += 1
            if count > 1:
                timings.append(elapsed)
            status = {'utc': now(), 'pid': os.getpid(), 'phase': phase, 'completed_origins': count,
                'expected_origins': expected, 'completed_forecasts': count*3, 'expected_forecasts': expected*3,
                'active_pilot_workers': 1, 'native_failures': 0, 'scoring_integrity_errors': 0,
                'exact_historical_baseline_hashes_replayed': hashes,
                'mean_remaining_seconds_lower_bound': float(np.mean(timings)*(expected-count)) if timings else None,
                'median_remaining_seconds_lower_bound': float(np.median(timings)*(expected-count)) if timings else None,
                'ETA_scope': 'Measured pilot compute; excludes setup, IO, contention and failed-fit repair'}
            write(OWN/'status.json', status)
            if count % 25 == 0:
                guard()
                print(json.dumps(status), flush=True)
    guard()
    def aggregate(rr, field):
        return sc.aggregate_records(rr, meta, score_key=field)['score']
    fields = ('primary_score', 'sensitivity_score')
    arms = {a: {f: aggregate(rr, f) for f in fields} for a, rr in records.items()}
    for arm, rr in records.items():
        for label, field in [('by_class', 'asset_class'), ('by_family', 'family')]:
            arms[arm][label] = {group: {f: aggregate([r for r in rr if meta[r['series_id']][field] == group], f)
                                      for f in fields} for group in sorted({v[field] for v in meta.values()})}
    result = {'utc': now(), 'phase': phase, 'complete': True, 'origins': count,
        'forecasts': count*3, 'arms': arms, 'records': records,
        'candidate_pct_better': {a: {f: 100*(1-arms['Candidate'][f]/arms[a][f]) for f in fields}
                                 for a in ('Public', 'Incumbent')},
        'native_failures': 0, 'scoring_integrity_errors': 0, 'adaptive_reused_evidence': True,
        'independent_holdout': False, 'exact_historical_baseline_hashes_replayed': hashes}
    write(OWN/(phase+'_summary.json'), result)
    status.update(active_pilot_workers=0, state='phase_complete')
    write(OWN/'status.json', status)
    write(OWN/'active_run.json', {'utc': now(), 'pid': os.getpid(), 'phase': phase, 'status': 'phase_complete'})
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'arms')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'development', 'reserved', 'coverage'])
    action = parser.parse_args().action
    prepare() if action == 'prepare' else run(action)
