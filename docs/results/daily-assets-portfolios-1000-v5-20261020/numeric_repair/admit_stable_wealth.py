"""Checkpoint-boundary numeric repair qualification; never exceed12 forecasts."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import sys
import time
OWN = Path(__file__).resolve().parent
V5 = OWN/'simulation_1000_v5'
sys.path.insert(0, str(V5))
import sweep_v5 as api
from research_stable_wealth import adapter

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p,d: api.original._atomic_json(p,d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def run():
    control_path = V5/'control.json'
    old_control = read(control_path)
    assert old_control['workers'] == 12 and not old_control['stop']
    assert adapter.MODEL_ID not in old_control['models']
    current = read(V5/'results/status.json')
    prior_id = adapter.s.BEST
    old = current['models'][prior_id]
    assert old['completed_tasks']+old['failed_tasks'] == old['expected_tasks'] == 64049
    assert old['failed_tasks'] == 1
    active = read(V5/'controller_active.json')
    os.kill(active['pid'], 0)
    errors = list((api.model_directory(V5/'results', prior_id)/'errors').rglob('*.json'))
    assert len(errors) == 1
    failure = read(errors[0])
    replay = read(OWN/'v5_stable_failed_origin_replay.json')
    assert replay['all_1000_paths_and_2762_horizons_finite_and_scored']
    assert replay['source_error_sha256'] == sha(errors[0])
    destination = OWN/'stable_wealth_qualification_v5'
    destination.mkdir(exist_ok=True)
    model = adapter.build()
    initial = read(V5/'launch_plan.json')
    sources = dict(initial['source_hashes'])
    sources.update({str(p.relative_to(V5)): sha(p) for p in (V5/'research_stable_wealth').glob('*.py')})
    identity = {'model_id': model.model_id, 'factory': 'research_stable_wealth.adapter:build',
        'source_files': sources, 'numerical_contract': api.original.dependency_identity(model)}
    digest = api.original.stable_digest(identity)
    spec = {'enabled': True, 'factory': identity['factory'], 'implementation_digest': digest}
    registry_record = {'identity': identity, 'factory': identity['factory'], 'implementation_digest': digest}
    pilot_control = {'workers': 12, 'stop': False, 'models': {model.model_id: spec}}
    pilot_registry = {'schema': 'french-external-model-registry-v1', 'source_root': str(V5),
                      'models': {model.model_id: registry_record}}
    write(destination/'control.json', pilot_control)
    write(destination/'registry.json', pilot_registry)
    tasks = read(V5/'smoke_tasks.json')
    failed_task = {'series_id': failure['identity']['series_id'],
                   'origin_label': failure['identity']['origin']['origin_label']}
    assert failed_task not in tasks
    tasks.append(failed_task)
    write(destination/'selected_tasks.json', tasks)
    write(destination/'qualification_plan.json', {'utc': now(), 'before_qualification_forecasts': True,
        'source_hashes': sources, 'runner_sha256': sha(Path(__file__)), 'selected_tasks': tasks,
        'paths': 1000, 'same_native_seeds_and_horizons': True, 'expected_old_success_loss_hash_matches': 54,
        'old_error_sha256': sha(errors[0]), 'scientific_score_promotion': False,
        'reason': 'Numerical overflow repair only; no new clipping or scientific fit/sampling change'})
    admitted = False
    try:
        paused = {**old_control, 'stop': True}
        write(control_path, paused)
        write(destination/'active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'draining_main_at_boundaries'})
        deadline = time.monotonic()+120
        while True:
            current = read(V5/'results/status.json')
            os.kill(active['pid'], 0)
            if current['active_workers'] == 0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Main workers did not reach checkpoint boundary in120seconds')
            time.sleep(.1)
        peak = 0
        pids = set()
        def observe(status):
            nonlocal peak
            peak = max(peak, status['active_workers'])
            pids.update(x['pid'] for x in status['worker_assignments'] if x.get('pid'))
        write(destination/'active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'qualification_running',
                                       'main_workers_verified_zero': True})
        final = api.run_controller(V5/'inputs', destination/'results', destination/'control.json',
            registry_path=destination/'registry.json', selected_tasks=tasks, poll_seconds=.05,
            progress_callback=observe)
        assert final['completed_active_tasks'] == 55 and final['failed_active_tasks'] == 0
        assert peak == 12
        panel = api.load_panel(V5/'inputs', verify_files=False)
        by_id = {r['series_id']: (i,r) for i,r in enumerate(panel['series'])}
        new_root = api.model_directory(destination/'results', model.model_id)
        old_root = api.model_directory(V5/'results', prior_id)
        new_fp = read(new_root/'execution.json')['fingerprint']
        old_fp = read(old_root/'execution.json')['fingerprint']
        matched = 0
        for task in tasks:
            i, row = by_id[task['series_id']]
            origins = api.schedule(V5/'inputs', row)
            j = next(j for j,o in enumerate(origins) if o['origin_label'] == task['origin_label'])
            origin = origins[j]
            cp,_ = api.task_paths(new_root, (i,j))
            losses, diag = api.original._read_checkpoint(cp, api.task_identity(new_fp, model.model_id, row['series_id'], origin))
            assert diag['simulations'] == 1000
            old_cp,_ = api.task_paths(old_root, (i,j))
            if task != failed_task:
                prior_losses,_ = api.original._read_checkpoint(old_cp, api.task_identity(old_fp, prior_id, row['series_id'], origin))
                assert hashlib.sha256(losses.tobytes()).hexdigest() == hashlib.sha256(prior_losses.tobytes()).hexdigest()
                matched += 1
            else:
                assert hashlib.sha256(losses.tobytes()).hexdigest() == replay['loss_sha256']
        assert matched == 54
        assert all(sha(V5/name) == value for name,value in sources.items())
        assert read(control_path) == paused
        receipt = {'utc': now(), 'successes': 55, 'forecast_failures': 0, 'scoring_integrity_errors': 0,
            'peak_workers': peak, 'worker_pids': sorted(pids), 'main_was_drained_before_pilot': True,
            'regular_native_full_loss_vectors_exact_matches': matched, 'reproduced_failure_now_finite': True,
            'paths_per_forecast': 1000, 'arbitrary_caps_or_score_changes': False,
            'old_failed_version_preserved': True, 'new_model_id': model.model_id,
            'new_full_queue_from_first_origin': True, 'full_result_claim': False}
        write(destination/'verification_receipt.json', receipt)
        registry = read(V5/'registry.json')
        registry['models'][model.model_id] = registry_record
        write(V5/'registry.json', registry)
        fresh = read(control_path)
        fresh['models'][prior_id]['enabled'] = False
        fresh['models'][model.model_id] = spec
        fresh['stop'] = False
        assert sum(x['enabled'] for x in fresh['models'].values()) == 4
        write(V5/'control.json', fresh)
        admitted = True
        write(destination/'admission_receipt.json', {**receipt, 'utc': now(),
            'control_sha256': sha(V5/'control.json'), 'registry_sha256': sha(V5/'registry.json'),
            'old_fully_attempted_version_successes': 64048, 'old_fully_attempted_version_generation_failures': 1,
            'reason': 'Qualified arithmetic repair, not a score-based new winner promotion',
            'live_existing_controller_pid': active['pid'], 'no_main_controller_restart': True,
            'no_pilot_checkpoint_imports_to_full_run': True})
        write(destination/'active.json', {'pid': os.getpid(), 'utc': now(), 'status': 'terminal_admitted_and_main_resumed'})
        print(json.dumps(receipt), flush=True)
    finally:
        if not admitted:
            write(control_path, old_control)


if __name__ == '__main__':
    run()
