"""Fresh four-model admission; input arrays and single main score unchanged."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import common
import sweep_v5 as api
import retained_models
import streaming
import score_main

OWN = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: api.original._atomic_json(p, d)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    assert not (OWN/'launch_plan.json').exists()
    old = OWN.parent/'portfolio_expansion_v4'
    panel = read(old/'inputs/panel.json')
    assert panel['simulations_per_origin'] == 240
    assert len(panel['series']) == 1264 and panel['expected_tasks'] == 64049
    inputs = OWN/'inputs'
    inputs.mkdir(exist_ok=True)
    for relative, digest in panel['files'].items():
        source, destination = old/'inputs'/relative, inputs/relative
        assert sha(source) == digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            # Immutable inputs: hard links avoid a second437MB data copy.
            os.link(source, destination)
        assert sha(destination) == digest
    previous = panel.pop('panel_fingerprint')
    panel.update(simulations_per_origin=1000,
        previous_240_path_panel_fingerprint=previous,
        simulation_protocol_id=api.PROTOCOL,
        score_protocol_id=score_main.SCORE_PROTOCOL_ID)
    panel['panel_fingerprint'] = hashlib.sha256(json.dumps(panel, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    write(inputs/'panel.json', panel)
    assert api.load_panel(inputs)['panel_fingerprint'] == panel['panel_fingerprint']
    factories = {streaming.ADAPTIVE: 'build_adaptive', streaming.REPLACEMENT: 'build_replacement',
                 streaming.CYCLES: 'build_cycles', streaming.BEST: 'build_best'}
    control = {'workers': 12, 'stop': False, 'models': {}}
    registry = {'schema': 'french-external-model-registry-v1', 'source_root': str(OWN), 'models': {}}
    sources = {p.name: sha(p) for p in OWN.glob('*.py')}
    for mid, name in factories.items():
        factory = 'retained_models:'+name
        model = getattr(retained_models, name)()
        identity = {'model_id': mid, 'factory': factory, 'source_files': sources,
                    'numerical_contract': api.original.dependency_identity(model)}
        digest = api.original.stable_digest(identity)
        registry['models'][mid] = {'identity': identity, 'factory': factory, 'implementation_digest': digest}
        control['models'][mid] = {'enabled': True, 'factory': factory, 'implementation_digest': digest}
    write(OWN/'registry.json', registry)
    write(OWN/'control.json', control)
    for mid, spec in control['models'].items():
        api.admitted_contract(mid, spec, OWN/'registry.json')
    groups = {}
    for row in panel['series']:
        group = row['asset_class'] if row['panel_branch'] == 'assets' else row['family']
        if group not in groups or row['observations'] > groups[group]['observations']:
            groups[group] = row
    tasks = []
    for row in groups.values():
        origins = api.schedule(inputs, row)
        chosen = [origins[0], max(origins, key=lambda o: o['max_horizon'])]
        for o in chosen:
            item = {'series_id': row['series_id'], 'origin_label': o['origin_label']}
            if item not in tasks:
                tasks.append(item)
    write(OWN/'smoke_tasks.json', tasks)
    old_score = old/'score_main.py'
    assert sha(OWN/'score_main.py') == sha(old_score), 'Main score changed'
    plan = {'utc': now(), 'models': control['models'], 'workers': 12, 'paths_per_forecast': 1000,
        'expected_tasks_per_model': 64049, 'expected_attempts': 4*64049,
        'panel_series': 1264, 'individual_series': 722, 'portfolio_series': 542,
        'panel_fingerprint': panel['panel_fingerprint'], 'old_panel_fingerprint': previous,
        'all_input_data_and_origin_files_SHA256_unchanged': True,
        'score_protocol_id': score_main.SCORE_PROTOCOL_ID, 'sole_main_score_source_unchanged': True,
        'source_hashes': sources, 'native_source_closure_digest': api.original.stable_digest(api.original._source_closure()),
        'smoke_tasks_per_model': len(tasks), 'smoke_groups': list(groups),
        'smoke_tasks_sha256': sha(OWN/'smoke_tasks.json'), 'registry_sha256': sha(OWN/'registry.json'),
        'full_run_admitted': False, 'old_240_path_results_are_not_1000_path_baselines': True,
        'FAST_removed_from_new_roster': True, 'peak_workers_verified': False,
        'future_candidates': 'Source-bound training-only streamed ensemble adapters, admitted only after matched representative sole-main-score validation; start from first origin in shared12worker scheduler.'}
    write(OWN/'launch_plan.json', plan)
    write(OWN/'launch_receipt.json', {'utc': now(), 'plan_sha256': sha(OWN/'launch_plan.json'),
                                    'before_first_v5_forecast': True})
    print(json.dumps({'prepared': True, 'series': 1264, 'paths': 1000, 'models': 4,
                      'smoke_tasks_per_model': len(tasks)}), flush=True)


if __name__ == '__main__':
    prepare()
