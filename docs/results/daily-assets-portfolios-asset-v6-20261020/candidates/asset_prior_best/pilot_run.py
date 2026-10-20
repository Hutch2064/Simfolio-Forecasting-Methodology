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
import common
from . import candidate as adapter
import sweep_v6 as api
import score_main
import reporter

OWN = Path(__file__).resolve().parent
V5 = OWN.parents[1]
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: api.original._atomic_json(p, d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def guard():
    plan = read(OWN/'plan.json')
    receipt = read(OWN/'plan_receipt.json')
    assert receipt['complete'] and receipt['plan_sha256'] == sha(OWN/'plan.json')
    assert receipt['preflight_sha256'] == sha(OWN/'broad_training_preflight.json')
    assert all(sha(V5/name) == value for name, value in plan['source_hashes'].items())
    assert sha(OWN/'DESIGN.md') == plan['design_sha256']
    return plan


def wait_for_baseline_completion():
    # A stage marker alone never authorizes a second12-worker forecast queue.
    active = read(V5/'controller_active.json')
    if active['status'] != 'terminal':
        raise RuntimeError('Baseline scheduler owns the12 forecast slots; do not oversubscribe')
    import subprocess
    process = subprocess.run(['ps', '-p', str(active['pid']), '-o', 'command='], capture_output=True, text=True)
    if 'asset_level_v6/run.py' in process.stdout:
        raise RuntimeError('Baseline controller process is still live')
    terminal = read(V5/'controller_terminal_receipt.json')
    state = terminal['final_status']
    assert state['active_workers'] == state['ready_queue'] == state['failed_active_tasks'] == 0
    assert state['completed_active_tasks'] == state['expected_active_tasks'] == 128098
    report = read(V5/'results/scoreboard.json')
    assert report['reference_selection_preliminary'] is False
    assert report['scoring_integrity_errors'] == report['native_failures'] == 0
    assert len(report['complete_coverage_leaderboard']) == 2
    assert read(V5/'full_pair_independent_audit.json')['complete_pair_audit']
    return report


def run(phase):
    plan = guard()
    baseline_report = wait_for_baseline_completion()
    if phase == 'reserved':
        assert read(OWN/'development/summary.json')['advance_to_next_stage']
    destination = OWN/phase
    destination.mkdir(exist_ok=True)
    cases = plan['phases'][phase]
    tasks = [{'series_id': c['series_id'], 'origin_label': c['origin']['origin_label']} for c in cases]
    model = adapter.build()
    baseline_plan = read(V5/'launch_plan.json')
    files = dict(baseline_plan['source_hashes'])
    files.update({str(p.relative_to(V5)): sha(p) for p in OWN.glob('*.py')})
    files[str((OWN/'parent_selection.json').relative_to(V5))] = sha(OWN/'parent_selection.json')
    identity = {'model_id': model.model_id, 'factory': 'candidates.asset_prior_best.candidate:build',
                'source_files': files, 'numerical_contract': api.original.dependency_identity(model)}
    digest = api.original.stable_digest(identity)
    control = {'workers': 12, 'stop': False, 'models': {model.model_id: {'enabled': True,
        'factory': identity['factory'], 'implementation_digest': digest}}}
    registry = {'schema': 'french-external-model-registry-v1', 'source_root': str(V5),
        'models': {model.model_id: {'identity': identity, 'factory': identity['factory'], 'implementation_digest': digest}}}
    write(destination/'control.json', control)
    write(destination/'registry.json', registry)
    write(destination/'selected_tasks.json', tasks)
    write(destination/'runner_receipt.json', {'utc': now(), 'before_field_forecasts': True,
        'source_hashes': files, 'plan_sha256': sha(OWN/'plan.json'),
        'selected_tasks_sha256': sha(destination/'selected_tasks.json'),
        'baseline_scoreboard_sha256': sha(V5/'results/scoreboard.json'),
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
    mids = list(dict.fromkeys([adapter.MODEL_ID, current_best, reference]))
    records = {mid: [] for mid in mids}
    roots = {mid: api.model_directory(destination/'results' if mid == adapter.MODEL_ID else V5/'results', mid) for mid in mids}
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
    complete = not missing and final['failed_active_tasks'] == 0 and all(len(x) == 355 for x in records.values())
    scores = {mid: reporter.scalar(rows, metadata) if complete else None for mid, rows in records.items()}
    advance = bool(complete and scores[adapter.MODEL_ID] is not None and scores[adapter.MODEL_ID] < scores[current_best])
    summary = {'utc': now(), 'phase': phase, 'complete': complete, 'cases_per_model': 355,
        'workers_requested': 12, 'peak_workers': peak, 'worker_pids': sorted(pids),
        'forecast_successes': final['completed_active_tasks'], 'native_failures': final['failed_active_tasks'],
        'scoring_integrity_errors': 0, 'missing': missing, 'checkpoint_identities_verified': verified,
        'sole_main_scores': scores, 'worse_asset_baseline_reference_model_id': reference,
        'current_best_reference_model_id': current_best,
        'percent_better_vs_worse_asset_baseline': {mid: 100*(1-value/scores[reference]) if complete else None for mid, value in scores.items()},
        'advance_to_next_stage': advance, 'independent_validation_claim': False,
        'diagnostics_have_zero_promotion_influence': True, 'new_full_panel_admission': False,
        'records': records}
    write(destination/'summary.json', summary)
    write(destination/'active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'terminal'})
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development', 'reserved'])
    run(parser.parse_args().phase)
