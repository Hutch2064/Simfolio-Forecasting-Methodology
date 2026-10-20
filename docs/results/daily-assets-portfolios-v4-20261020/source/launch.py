"""Fresh five-model v4 admission, canonical smoke, shared-worker full queue."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib
import json
import common
import score_main
import reporter
import sweep as api

OWN = common.OWN
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: api.original._atomic_json(p, d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    assert not (OWN/'launch_plan.json').exists()
    panel = api.load_panel(OWN/'inputs')
    assert read(OWN/'input_verification.json')['complete']
    old = OWN.parent/'mean_shrinkage_promotion'
    control = read(old/'control.json')
    assert sum(v['enabled'] for v in control['models'].values()) == 5
    control['stop'] = False
    control['workers'] = 12
    registry = read(old/'external_model_registry.json')
    registry['models'] = {mid: value for mid, value in registry['models'].items() if mid in control['models']}
    write(OWN/'registry.json', registry)
    write(OWN/'control.json', control)
    for mid, spec in control['models'].items():
        api.admitted_contract(mid, spec, OWN/'registry.json')
        module, name = spec['factory'].split(':')
        model = getattr(importlib.import_module(module), name)()
        assert model.model_id == mid
    # Both short and long temporal forecasts in every original asset class and
    # every added portfolio economic family, under the full native identities.
    groups = {}
    for row in panel['series']:
        key = row['asset_class'] if row['panel_branch'] == 'assets' else row['family']
        groups.setdefault(key, row)
    smoke = []
    for row in groups.values():
        origins = api.schedule(OWN/'inputs', row)
        selected = [origins[0], next(o for o in origins if o['train_fraction'] is not None)]
        smoke.extend({'series_id': row['series_id'], 'origin_label': o['origin_label']} for o in selected)
    write(OWN/'smoke_tasks.json', smoke)
    plan = {'utc': now(), 'panel_fingerprint': panel['panel_fingerprint'],
        'score_protocol_id': score_main.SCORE_PROTOCOL_ID, 'initial_models': control['models'],
        'expected_tasks_per_model': panel['expected_tasks'], 'expected_initial_attempts': 5*panel['expected_tasks'],
        'smoke_tasks_per_model': len(smoke), 'smoke_economic_groups': list(groups),
        'paths': 240, 'workers': 12, 'source_hashes': {p.name: sha(p) for p in OWN.glob('*.py')},
        'scoring_document_sha256': sha(OWN/'SCORING.md'), 'initial_registry_sha256': sha(OWN/'registry.json'),
        'smoke_tasks_sha256': sha(OWN/'smoke_tasks.json'),
        'native_source_closure_digest': api.original.stable_digest(api.original._source_closure()),
        'source_root_budget_bytes': 8*1024**3,
        'future_admission': 'Only source-frozen serious candidates after matched representative main-score validation; start full queue from first origin, preserve old catalogue entries. Live control/registry admission is explicit and separately receipted.',
        'score_selected_or_diagnostic_routing': False, 'existing_models_recomputed_fresh': True}
    write(OWN/'launch_plan.json', plan)
    write(OWN/'launch_receipt.json', {'utc': now(), 'plan_sha256': sha(OWN/'launch_plan.json'), 'before_first_model_forecast': True})
    print(json.dumps({'prepared': True, 'tasks_per_model': panel['expected_tasks'],
        'initial_attempts': plan['expected_initial_attempts'], 'smoke_tasks_per_model': len(smoke)}), flush=True)


def guard():
    plan = read(OWN/'launch_plan.json')
    assert sha(OWN/'launch_plan.json') == read(OWN/'launch_receipt.json')['plan_sha256']
    assert all(sha(OWN/name) == value for name, value in plan['source_hashes'].items())
    assert sha(OWN/'SCORING.md') == plan['scoring_document_sha256']
    assert sha(OWN/'smoke_tasks.json') == plan['smoke_tasks_sha256']
    assert api.original.stable_digest(api.original._source_closure()) == plan['native_source_closure_digest']
    assert sum(p.stat().st_size for p in OWN.rglob('*') if p.is_file()) < plan['source_root_budget_bytes']
    return plan


def run(stage):
    plan = guard()
    if stage == 'full':
        smoke = read(OWN/'smoke_receipt.json')
        assert smoke['three_Public_and_current_winner_smoke_failures'] == 0
        assert smoke['scoring_integrity_errors'] == 0
    write(OWN/'controller_active.json', {'utc': now(), 'pid': os.getpid(), 'stage': stage, 'status': 'running'})
    api.run_controller(OWN/'inputs', OWN/'results', OWN/'control.json', registry_path=OWN/'registry.json',
        poll_seconds=.1, selected_tasks=read(OWN/'smoke_tasks.json') if stage == 'smoke' else None,
        max_tasks=None if stage == 'smoke' else plan['expected_tasks_per_model'])
    guard()
    if stage == 'smoke':
        report = reporter.build()
        fast = 'R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION'
        unexpected = sum(m['native_failures'] for mid, m in report['models'].items() if mid != fast)
        receipt = {'utc': now(), 'stage': 'smoke', 'three_Public_and_current_winner_smoke_failures': unexpected,
            'FAST_native_failures': report['models'][fast]['native_failures'],
            'scoring_integrity_errors': report['scoring_integrity_errors'],
            'completed_forecasts': sum(m['completed_tasks'] for m in report['models'].values()),
            'expected_attempts': 5*plan['smoke_tasks_per_model'], 'same_full_execution_identities': True,
            'sole_main_score': score_main.SCORE_PROTOCOL_ID}
        write(OWN/'smoke_receipt.json', receipt)
        assert unexpected == report['scoring_integrity_errors'] == 0
    else:
        write(OWN/'controller_terminal_receipt.json', {'utc': now(), 'pid': os.getpid(),
            'stage': 'full', 'status': 'terminal_after_required_attempts', 'final_status': read(OWN/'results/status.json')})
    write(OWN/'controller_active.json', {'utc': now(), 'pid': os.getpid(), 'stage': stage, 'status': 'terminal'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'smoke', 'full'])
    action = parser.parse_args().action
    prepare() if action == 'prepare' else run(action)
