"""Complete frozen asset candidate after broad gate, reusing identical pilot losses."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import shutil
import sys
import subprocess
O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
C = V/'candidates/asset_prior_best'
F = C/'full'
sys.path.insert(0, str(V))
import common
import sweep_v6 as api
from candidates.asset_prior_best import candidate, pilot_run
from audit_asset_v6_full_pair import independent_origin, independent_reduce
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    slots = (O/'asset_v6_forecast_slots.lock').open('a')
    fcntl.flock(slots, fcntl.LOCK_EX)
    plan = pilot_run.guard()
    baseline_report = pilot_run.wait_for_baseline_completion()
    broad = read(C/'broad/summary.json')
    assert broad['complete'] and broad['advance_to_next_stage']
    assert broad['cases_per_model'] == 7418 and broad['native_failures'] == broad['scoring_integrity_errors'] == 0
    assert not (F/'active.json').exists(), 'Inspect any previous full evaluation before retrying'
    subprocess.run([sys.executable, '/Users/aidanhutchison/simfolio/storage_preflight.py',
                    '--expected-gib', '1.5', '--artifact-root', str(O), '--budget-gib', '12'], check=True)
    F.mkdir(exist_ok=True)
    for name in ('control.json', 'registry.json'):
        shutil.copy2(C/'broad'/name, F/name)
    control = api.read_control(F/'control.json')
    assert control['workers'] == 12
    specification = control['models'][candidate.MODEL_ID]
    api.admitted_contract(candidate.MODEL_ID, specification, F/'registry.json')
    panel = api.load_panel(V/'inputs', verify_files=True)
    target = api.model_directory(F/'results', candidate.MODEL_ID)
    target.mkdir(parents=True, exist_ok=True)
    manifests = []
    reused = []
    for phase in ('development', 'reserved', 'broad'):
        source = api.model_directory(C/phase/'results', candidate.MODEL_ID)
        manifest = read(source/'execution.json')
        manifests.append(manifest)
        if not (target/'execution.json').exists():
            shutil.copy2(source/'execution.json', target/'execution.json')
        assert manifest['fingerprint'] == read(target/'execution.json')['fingerprint']
        for i, row in enumerate(panel['series']):
            for j, origin in enumerate(api.schedule(V/'inputs', row)):
                cp, _ = api.task_paths(source, (i, j))
                if not cp.exists():
                    continue
                _, diagnostics = api.original._read_checkpoint(cp, api.task_identity(manifest['fingerprint'], candidate.MODEL_ID, row['series_id'], origin))
                assert diagnostics['simulations'] == 1000 and diagnostics['joint_asset_paths_simulated']
                destination, _ = api.task_paths(target, (i, j))
                assert not destination.exists(), 'Predeclared phases must be disjoint'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cp, destination)
                assert sha(cp) == sha(destination)
                reused.append({'phase': phase, 'task': [i, j], 'checkpoint_sha256': sha(destination)})
    assert len(reused) == 8128
    api.original._atomic_json(F/'admission.json', {'utc': now(), 'expected_tasks': 64049,
        'paths': 1000, 'workers': 12, 'same_source_bound_pilot_checkpoints_reused': len(reused),
        'reused_checkpoint_records': reused, 'panel_fingerprint': panel['panel_fingerprint'],
        'not_promoted_before_complete_full_score': True, 'baseline_results_unchanged': True,
        'runner_sha256': sha(Path(__file__)), 'broad_summary_sha256': sha(C/'broad/summary.json')})
    api.original._atomic_json(F/'active.json', {'utc': now(), 'pid': os.getpid(), 'status': 'running'})
    final = api.run_controller(V/'inputs', F/'results', F/'control.json', registry_path=F/'registry.json', poll_seconds=.05)
    pilot_run.guard()
    assert final['completed_active_tasks'] == final['expected_active_tasks'] == 64049
    assert final['failed_active_tasks'] == final['active_workers'] == final['ready_queue'] == 0
    fingerprint = read(target/'execution.json')['fingerprint']
    records, compute, commitment = [], 0., hashlib.sha256()
    for i, row in enumerate(panel['series']):
        dates, values, origins = api.load_series(V/'inputs', row, verify_files=True)
        ids = [sid for sid, _, _ in row['rational_weights']] if row['panel_branch'] == 'portfolios' else [row['series_id']]
        for j, origin in enumerate(origins):
            cp, _ = api.task_paths(target, (i, j))
            loss, d = api.original._read_checkpoint(cp, api.task_identity(fingerprint, candidate.MODEL_ID, row['series_id'], origin))
            assert d['simulations'] == 1000 and d['all_paths_scored_at_every_horizon']
            assert d['forecast_level'] == 'joint_assets' and d['constituent_ids'] == ids
            n = origin['position']+1
            value = independent_origin(loss, values[:n], origin['origin_date'], dates[n:n+origin['max_horizon']])
            records.append((row['series_id'], origin['origin_date'], value))
            compute += d['forecast_and_score_seconds']
            commitment.update(f'{i}/{j}:{sha(cp)}\n'.encode())
    main_score = independent_reduce(records, {r['series_id']: r for r in panel['series']})['main_score']
    audit = read(V/'full_pair_independent_audit.json')
    reference, parent = audit['worse_reference_model_id'], audit['better_starting_model_id']
    ref_score = audit['models'][reference]['main_score']
    promote = main_score < audit['models'][parent]['main_score']
    summary = {'utc': now(), 'model_id': candidate.MODEL_ID, 'complete': True, 'series': 1264,
               'successful_tasks': 64049, 'native_failures': 0, 'scoring_integrity_errors': 0,
               'simulations': 1000, 'main_score': main_score, 'sole_reference_model_id': reference,
               'reference_main_score': ref_score, 'percent_better_vs_worse_reference': 100*(1-main_score/ref_score),
               'beats_better_parent_on_full_identical_coverage': promote,
               'eligible_for_current_best_promotion': promote, 'checkpoint_commitment': commitment.hexdigest(),
               'worker_compute_seconds': compute, 'runtime_scope': 'Recorded successful input preparation, forecasts and CRPS; no controlled speedup claim',
               'independent_validation_or_significance_claim': False, 'goal_complete': False}
    api.original._atomic_json(F/'verification_receipt.json', summary)
    api.original._atomic_json(F/'active.json', {'utc': now(), 'pid': os.getpid(), 'status': 'terminal'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
