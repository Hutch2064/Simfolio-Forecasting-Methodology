"""Freeze the new asset architecture without altering any panel data or score."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import hashlib
import datetime
import common
import sweep_v6 as api
import asset_models
import score_main

OWN = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: api.original._atomic_json(p, d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    assert not (OWN/'launch_plan.json').exists(), 'Existing immutable launch plan'
    qualification = read(OWN/'qualification/receipt.json')
    assert qualification['successes'] == qualification['expected_forecasts'] == 154
    assert not qualification['failures'] and len(qualification['worker_pids']) == 12
    for name in ('blocks.py', 'asset_paths.py', 'inputs.py', 'profile.json'):
        assert sha(OWN/name) == qualification['source_hashes'][name]
    old = OWN.parent/'simulation_1000_v5'
    panel = read(old/'inputs/panel.json'); inputs = OWN/'inputs'
    inputs.mkdir(exist_ok=True)
    for relative in [*panel['files'], 'panel.json']:
        source, destination = old/'inputs'/relative, inputs/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source, destination)
        assert sha(destination) == sha(source)
    assert api.load_panel(inputs)['panel_fingerprint'] == panel['panel_fingerprint']
    assert sha(OWN/'score_main.py') == sha(old/'score_main.py')
    factories = {'model_testing_asset_joint_blocks_replacement_v6': 'build_replacement',
                 'model_testing_asset_joint_blocks_history_partition_v6': 'build_without_replacement'}
    sources = {p.name: sha(p) for p in OWN.glob('*.py')}
    sources['profile.json'] = sha(OWN/'profile.json')
    registry = {'schema': 'french-external-model-registry-v1', 'source_root': str(OWN), 'models': {}}
    control = {'workers': 12, 'stop': False, 'models': {}}
    for mid, name in factories.items():
        factory = 'asset_models:'+name
        model = getattr(asset_models, name)()
        identity = {'model_id': mid, 'factory': factory, 'source_files': sources,
                    'numerical_contract': api.original.dependency_identity(model)}
        digest = api.original.stable_digest(identity)
        registry['models'][mid] = {'identity': identity, 'factory': factory, 'implementation_digest': digest}
        control['models'][mid] = {'enabled': True, 'factory': factory, 'implementation_digest': digest}
    write(OWN/'registry.json', registry); write(OWN/'control.json', control)
    for mid, spec in control['models'].items():
        api.admitted_contract(mid, spec, OWN/'registry.json')
    qualification_tasks = read(OWN/'qualification/plan.json')['tasks']
    chosen = sorted({(i, j) for i, j, _ in qualification_tasks})
    tasks = [{'series_id': panel['series'][i]['series_id'],
              'origin_label': api.schedule(inputs, panel['series'][i])[j]['origin_label']}
             for i, j in chosen]
    write(OWN/'smoke_tasks.json', tasks)
    plan = {'utc': now(), 'source_hashes': sources, 'models': control['models'],
        'workers': 12, 'paths_per_forecast': 1000, 'expected_tasks_per_model': 64049,
        'expected_attempts': 128098, 'panel_series': 1264, 'individual_series': 722,
        'portfolio_series': 542, 'panel_fingerprint': panel['panel_fingerprint'],
        'source_data_and_score_unchanged': True, 'forecast_level': 'joint_assets',
        'profile': read(OWN/'profile.json'), 'score_protocol_id': score_main.SCORE_PROTOCOL_ID,
        'native_source_closure_digest': api.original.stable_digest(api.original._source_closure()),
        'smoke_tasks_per_model': len(tasks), 'smoke_tasks_sha256': sha(OWN/'smoke_tasks.json'),
        'registry_sha256': sha(OWN/'registry.json'), 'qualification_receipt_sha256': sha(OWN/'qualification/receipt.json'),
        'full_run_admitted': False, 'old_portfolio_checkpoints_imported': False,
        'reference_selection': 'Worse main Q of the two configurations on identical full successful coverage',
        'research_start': 'New selected asset-level baseline, not old scalar portfolio winner'}
    write(OWN/'launch_plan.json', plan)
    write(OWN/'launch_receipt.json', {'utc': now(), 'plan_sha256': sha(OWN/'launch_plan.json')})
    print(json.dumps({'prepared': True, 'models': 2, 'smoke_tasks_per_model': len(tasks),
                      'full_tasks': 128098, 'score_source_unchanged': True}), flush=True)


if __name__ == '__main__':
    prepare()
