"""One-shot authorized baseline-to-pilot handoff; never restarts scientific jobs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import fcntl
import json
import subprocess
import time
import traceback

O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
C = V/'candidates/asset_prior_best'
PY = '/Users/aidanhutchison/simfolio/tmp/catalogue52-20261018/venv/bin/python'
read = lambda p: json.loads(p.read_text())
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(p, data):
    temp = p.with_suffix(p.suffix+'.new')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temp, p)


def update(stage, **details):
    write(O/'asset_v6_research_handoff_active.json',
          {'utc': now(), 'pid': os.getpid(), 'stage': stage, **details})


def command_live(pid, fragment):
    process = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                             capture_output=True, text=True)
    return fragment in process.stdout


def run_script(name, *, cwd=O):
    with (O/(Path(name).stem+'.log')).open('a') as log:
        subprocess.run([PY, str(O/name)], cwd=cwd, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def pilot(phase):
    destination = C/phase
    assert not (destination/'active.json').exists(), 'Existing pilot must be inspected, never blindly restarted'
    subprocess.run([PY, '/Users/aidanhutchison/simfolio/storage_preflight.py',
                    '--expected-gib', '.15', '--artifact-root', str(O),
                    '--budget-gib', '12'], check=True)
    with (O/f'asset_prior_best_{phase}.log').open('a') as log:
        child = subprocess.Popen([PY, '-m', 'candidates.asset_prior_best.pilot_run', phase],
                                 cwd=V, stdout=log, stderr=subprocess.STDOUT)
        update('pilot_running', phase=phase, pilot_pid=child.pid, forecast_slots=12)
        ledger = read(O/'goal_progress.json')
        ledger.update(stage=f'asset_prior_best_{phase}_running',
                      active_pilot={'phase': phase, 'pid': child.pid,
                                    'active_path': str(destination/'active.json'),
                                    'status_path': str(destination/'results/status.json'),
                                    'summary_path': str(destination/'summary.json'),
                                    'expected_tasks': 355, 'workers_requested': 12},
                      current_status_path=str(destination/'results/status.json'),
                      scientific_jobs_running=12, actual_goal_tool_status='active')
        write(O/'goal_progress.json', ledger)
        code = child.wait()
        assert code == 0, f'{phase} pilot exited {code}; inspect log before any retry'
    summary = read(destination/'summary.json')
    assert summary['complete'] and summary['forecast_successes'] == 355
    assert summary['native_failures'] == summary['scoring_integrity_errors'] == 0
    return summary


def main():
    lock = (O/'asset_v6_research_handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (C/'development/active.json').exists(), 'A prior handoff already launched this version'
    while True:
        active = read(V/'controller_active.json')
        live = command_live(active['pid'], 'asset_level_v6/run.py full')
        if active['status'] == 'terminal' and not live:
            break
        assert live, 'Baseline controller missing without terminal receipt; inspect, do not restart'
        update('waiting_for_live_baseline_controller', baseline_pid=active['pid'])
        time.sleep(15)
    while True:
        report = read(V/'results/scoreboard.json')
        reporter = read(V/'reporter_active.json')
        live = command_live(reporter['pid'], 'asset_level_v6/reporter.py')
        if reporter['status'] == 'terminal' and not live:
            break
        assert live, 'Reporter missing before final scan; inspect, do not restart'
        update('waiting_for_live_final_scoring', reporter_pid=reporter['pid'])
        time.sleep(15)
    assert len(report['complete_coverage_leaderboard']) == 2
    assert report['scoring_integrity_errors'] == report['native_failures'] == 0
    update('independent_full_pair_audit')
    if not (V/'full_pair_independent_audit.json').exists():
        run_script('audit_asset_v6_full_pair.py')
    audit = read(V/'full_pair_independent_audit.json')
    assert audit['complete_pair_audit'] and audit['total_checkpoints_verified'] == 128098
    if not (C/'parent_selection.json').exists():
        run_script('bind_asset_prior_parent.py')
    update('publishing_bound_candidate_before_field_tests')
    run_script('publish_asset_v6_prepared.py')
    ledger = read(O/'goal_progress.json')
    worse, better = audit['worse_reference_model_id'], audit['better_starting_model_id']
    ledger.update(baseline_selection_pending=False, baseline_model_id=worse,
                  baseline_display_name=read(V/'model_names.json').get(worse, worse),
                  baseline_main_score=audit['models'][worse]['main_score'],
                  baseline_main_score_scope='Complete paired64,049-task independently audited joint-asset reference at1,000paths',
                  current_best_model_id=better,
                  current_best_main_score=audit['models'][better]['main_score'],
                  baseline_pair_full_audit=str(V/'full_pair_independent_audit.json'),
                  prior_best_asset_candidate_parent=better,
                  previous_goal_turn_classification='progress')
    write(O/'goal_progress.json', ledger)
    development = pilot('development')
    reserved = pilot('reserved') if development['advance_to_next_stage'] else None
    decision = 'broader_review_required' if reserved and reserved['advance_to_next_stage'] else 'reject_frozen_contrast_no_full_escalation'
    ledger = read(O/'goal_progress.json')
    ledger.update(stage=f'asset_prior_best_{decision}', active_pilot=None,
                  scientific_jobs_running=0,
                  prior_best_asset_pilot={'development_summary': str(C/'development/summary.json'),
                      'reserved_summary': str(C/'reserved/summary.json') if reserved else None,
                      'decision': decision, 'goal_complete': False},
                  pending=['Continue evidence-led accuracy search; no stopping judgment yet'],
                  actual_goal_tool_status='active')
    write(O/'goal_progress.json', ledger)
    update('terminal', decision=decision, goal_complete=False)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        update('attention_required', traceback=traceback.format_exc(),
               scientific_jobs_not_restarted=True, goal_complete=False)
        raise
