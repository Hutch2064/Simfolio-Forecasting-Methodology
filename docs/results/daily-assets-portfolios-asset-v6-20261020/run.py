"""Source-bound shared12-slot scheduler for actual joint asset forecasts."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import hashlib
import datetime
import common
import sweep_v6 as api

OWN = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: api.original._atomic_json(p, d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def guard():
    plan = read(OWN/'launch_plan.json')
    assert sha(OWN/'launch_plan.json') == read(OWN/'launch_receipt.json')['plan_sha256']
    assert all(sha(OWN/name) == digest for name, digest in plan['source_hashes'].items())
    assert sha(OWN/'smoke_tasks.json') == plan['smoke_tasks_sha256']
    assert api.original.stable_digest(api.original._source_closure()) == plan['native_source_closure_digest']
    return plan


def run(stage):
    plan = guard()
    if stage == 'full':
        receipt = read(OWN/'smoke_receipt.json')
        assert receipt['successes'] == receipt['expected_attempts'] == 154
        assert receipt['failures'] == receipt['scoring_integrity_errors'] == 0
        assert receipt['peak_workers'] == 12
        assert read(OWN/'full_admission.json')['admitted']
    active = {'utc': now(), 'pid': os.getpid(), 'stage': stage, 'status': 'running'}
    write(OWN/'controller_active.json', active)
    peak, pids = 0, set()
    def observe(status):
        nonlocal peak
        peak = max(peak, status['active_workers'])
        pids.update(w['pid'] for w in status.get('worker_assignments', []) if w.get('pid'))
    final = api.run_controller(OWN/'inputs', OWN/'results', OWN/'control.json',
        registry_path=OWN/'registry.json', poll_seconds=.05,
        selected_tasks=read(OWN/'smoke_tasks.json') if stage == 'smoke' else None,
        max_tasks=None if stage == 'smoke' else plan['expected_tasks_per_model'], progress_callback=observe)
    guard(); active.update(utc=now(), status='terminal')
    write(OWN/'controller_active.json', active)
    if stage == 'smoke':
        import reporter
        report = reporter.build()
        write(OWN/'smoke_receipt.json', {'utc': now(), 'paths_per_forecast': 1000,
            'expected_attempts': 154, 'successes': final['completed_active_tasks'],
            'failures': final['failed_active_tasks'], 'scoring_integrity_errors': report['scoring_integrity_errors'],
            'peak_workers': peak, 'worker_pids': sorted(pids), 'final_status': final,
            'score_protocol_id': plan['score_protocol_id'], 'full_suite_complete': False})
    else:
        write(OWN/'controller_terminal_receipt.json', {'utc': now(), 'final_status': final,
            'peak_workers': peak, 'worker_pids': sorted(pids), 'required_attempts': 128098})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['smoke', 'full'])
    run(parser.parse_args().stage)
