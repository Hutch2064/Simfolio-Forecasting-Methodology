"""Matched main-score pilot using12 slots only after the baseline queue closes."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'asset_level_v6'))
import common
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'asset_level_v6'))
from candidates.asset_equilibrium_blocks import candidate as adapters
import sweep_v6 as api
import score_main
import reporter
import fcntl

OWN = Path(__file__).resolve().parent/'asset_level_v6/candidates/asset_equilibrium_blocks'
V5 = OWN.parents[1]
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: api.original._atomic_json(p, d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def guard():
    plan = read(OWN/'plan.json')
    receipt = read(OWN/'plan_receipt.json')
    assert receipt['complete'] and receipt['plan_sha256'] == sha(OWN/'plan.json')
    assert plan['runner_sha256'] == sha(Path(__file__))
    assert all(sha(V5/name) == value for name, value in plan['source_hashes'].items())
    assert sha(OWN/'DESIGN.md') == plan['design_sha256']
    return plan


def wait_for_baseline_completion():
    full = V5/'candidates/asset_fading_vol/full'
    result = read(full/'verification_receipt.json')
    assert result['complete'] and result['successful_tasks'] == 64049
    assert result['native_failures'] == result['scoring_integrity_errors'] == 0
    assert read(full/'active.json')['status'] == 'terminal'
    assert read(full/'reporter_active.json')['status'] == 'terminal'
    official = read(V5/'official_scoreboard.json')
    assert official['current_best_model_id'] == result['model_id']
    report = read(full/'scoreboard.json')
    assert report['scoring_integrity_errors'] == report['native_failures'] == 0
    return report


def run(phase):
    slots = (V5.parent/'asset_v6_forecast_slots.lock').open('a')
    fcntl.flock(slots, fcntl.LOCK_EX)
    plan = guard()
    baseline_report = wait_for_baseline_completion()
    if phase == 'reserved':
        assert read(OWN/'development/summary.json')['advance_to_next_stage']
    if phase == 'broad':
        assert read(OWN/'reserved/summary.json')['advance_to_next_stage']
    destination = OWN/phase
    destination.mkdir(exist_ok=True)
    cases = plan['phases'][phase]
    tasks = [{'series_id': c['series_id'], 'origin_label': c['origin']['origin_label']} for c in cases]
    baseline_plan = read(V5/'launch_plan.json')
    files = dict(plan['source_hashes'])
    definitions = {'equilibrium': (adapters.MODEL_ID, adapters.build)}
    admitted = {adapters.MODEL_ID} if phase == 'development' else set(read(OWN/('development' if phase == 'reserved' else 'reserved')/'summary.json')['advancing_model_ids'])
    assert admitted, 'No model earned this next phase'
    control = {'workers': 12, 'stop': False, 'models': {}}
    registry = {'schema': 'french-external-model-registry-v1', 'source_root': str(V5), 'models': {}}
    for name, (mid, factory) in definitions.items():
        if mid not in admitted:
            continue
        model = factory()
        factory_name = 'candidates.asset_equilibrium_blocks.candidate:build'
        identity = {'model_id': mid, 'factory': factory_name, 'source_files': files,
                    'numerical_contract': api.original.dependency_identity(model)}
        digest = api.original.stable_digest(identity)
        control['models'][mid] = {'enabled': True, 'factory': factory_name, 'implementation_digest': digest}
        registry['models'][mid] = {'identity': identity, 'factory': factory_name, 'implementation_digest': digest}
    write(destination/'control.json', control)
    write(destination/'registry.json', registry)
    write(destination/'selected_tasks.json', tasks)
    write(destination/'runner_receipt.json', {'utc': now(), 'before_field_forecasts': True,
        'source_hashes': files, 'plan_sha256': sha(OWN/'plan.json'),
        'selected_tasks_sha256': sha(destination/'selected_tasks.json'),
        'baseline_scoreboard_sha256': sha(V5/'official_scoreboard.json'),
        'baseline_complete_before_use': True, 'new_full_panel_admission': False})
    peak = 0
    pids = set()
    def observe(status):
        nonlocal peak
        peak = max(peak, status['active_workers'])
        pids.update(x['pid'] for x in status['worker_assignments'] if x.get('pid'))
    write(destination/'active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'running'})
    final = api.run_controller(V5/'inputs', destination/'results', destination/'control.json',
        registry_path=destination/'registry.json', selected_tasks=tasks,
        poll_seconds=.05, progress_callback=observe)
    guard()
    assert all(sha(V5/name) == value for name, value in files.items())
    panel = api.load_panel(V5/'inputs', verify_files=False)
    by_id = {r['series_id']: (i, r) for i, r in enumerate(panel['series'])}
    metadata = {sid: r for sid, (_, r) in by_id.items()}
    reference = baseline_report['reference_model_id']
    current_best = baseline_report['complete_coverage_leaderboard'][0]['model_id']
    candidate_ids = list(control['models'])
    mids = list(dict.fromkeys([*candidate_ids, current_best, reference]))
    records = {mid: [] for mid in mids}
    roots = {mid: api.model_directory(destination/'results' if mid in candidate_ids else V5/'candidates/asset_fading_vol/full/results' if mid == current_best else V5/'results', mid) for mid in mids}
    fingerprints = {mid: read(root/'execution.json')['fingerprint'] for mid, root in roots.items()}
    missing = []
    verified = 0
    for case in cases:
        sid, origin = case['series_id'], case['origin']
        i, row = by_id[sid]
        dates, values, origins = api.load_series(V5/'inputs', row, verify_files=False)
        j = origins.index(origin)
        for mid in mids:
            cp, _ = api.task_paths(roots[mid], (i, j))
            if not cp.exists():
                missing.append({'model_id': mid, 'series_id': sid, 'origin_label': origin['origin_label']})
                continue
            loss, _ = api.original._read_checkpoint(cp, api.task_identity(fingerprints[mid], mid, sid, origin))
            score = score_main.origin_score(loss, values[:origin['position']+1], origin['origin_date'],
                dates[origin['position']+1:origin['position']+1+origin['max_horizon']])
            records[mid].append({'series_id': sid, 'origin_date': origin['origin_date'], 'score': score,
                'loss_sha256': hashlib.sha256(np.asarray(loss, dtype=np.float64).tobytes()).hexdigest()})
            verified += 1
    complete = not missing and final['failed_active_tasks'] == 0 and all(len(x) == len(cases) for x in records.values())
    scores = {mid: reporter.scalar(rows, metadata) if complete else None for mid, rows in records.items()}
    advancing = [mid for mid in candidate_ids if complete and scores[mid] < scores[current_best]]
    advance = bool(advancing)
    summary = {'utc': now(), 'phase': phase, 'complete': complete, 'cases_per_model': len(cases),
        'workers_requested': 12, 'peak_workers': peak, 'worker_pids': sorted(pids),
        'forecast_successes': final['completed_active_tasks'], 'native_failures': final['failed_active_tasks'],
        'scoring_integrity_errors': 0, 'missing': missing, 'checkpoint_identities_verified': verified,
        'sole_main_scores': scores, 'worse_asset_baseline_reference_model_id': reference,
        'current_best_reference_model_id': current_best,
        'percent_better_vs_worse_asset_baseline': {mid: 100*(1-value/scores[reference]) if complete else None for mid, value in scores.items()},
        'advance_to_next_stage': advance, 'advancing_model_ids': advancing,
        'candidate_model_ids': candidate_ids, 'total_expected_selected_forecasts': len(cases)*len(candidate_ids),
        'independent_validation_claim': False,
        'diagnostics_have_zero_promotion_influence': True, 'new_full_panel_admission': False,
        'records': records}
    write(destination/'summary.json', summary)
    write(destination/'active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'terminal'})
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development', 'reserved', 'broad'])
    run(parser.parse_args().phase)
