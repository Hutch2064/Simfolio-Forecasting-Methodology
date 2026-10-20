"""Append both fully attempted asset baseline versions after remote recovery."""
from pathlib import Path
import datetime
import hashlib
import json
import sys

O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
R = O/'repository'
sys.path.insert(0, str(R/'tools/full_panel_catalogue'))
import catalogue
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive = read(O/'archives/asset-v6-baselines-1000-20261020/receipt.json')
    assert archive['complete'] and not archive['scientific_originals_removed']
    audit = read(V/'full_pair_independent_audit.json')
    assert audit['complete_pair_audit'] and audit['total_checkpoints_verified'] == 128098
    report = read(V/'results/scoreboard.json')
    assert report['native_failures'] == report['scoring_integrity_errors'] == 0
    control, panel, names = [read(V/name) for name in ('control.json', 'inputs/panel.json', 'model_names.json')]
    path = R/'docs/results/daily-cross-asset-722/catalogue.json'
    ledger = read(path)
    reference = audit['worse_reference_model_id']
    ref_score = audit['models'][reference]['main_score']
    for mid, specification in control['models'].items():
        assert specification['enabled']
        root = V/'results/models'/hashlib.sha256(mid.encode()).hexdigest()
        execution = read(root/'execution.json')
        model = audit['models'][mid]
        assert model['verified'] == 64049
        parts = [r for r in archive['archives'] if r['group'] in (root.name, 'shared', 'audit')]
        assert catalogue.retirement_allowed({'archives': parts}, archive)
        definition = ('Uniform bounded contiguous nonwrapping historical blocks, sampled with replacement.'
                      if mid.endswith('replacement_v6') else
                      'Random contiguous chronological partition, shuffled without reusing a day within each history cycle; repartition and refill after exhaustion.')
        record = {
            'identity': {'model_id': mid, 'implementation_digest': specification['implementation_digest']},
            'display_name': names[mid],
            'definition': definition+' Joint constituent rows, actual asset paths and portfolio holdings, fixed target rebalancing inside every path. Uniform block lengths1..min(training rows,forecast observations); no Politis–White.',
            'version': 'v6_joint_assets_1000_source_bound_execution',
            'forecast_level': 'joint_assets', 'retained_in_active_set': True,
            'score_protocol_id': report['score_protocol_id'],
            'execution_fingerprint': execution['fingerprint'],
            'coverage': {'panel_fingerprint': panel['panel_fingerprint'], 'series': 1264,
                         'expected_tasks': 64049, 'successful_tasks': 64049,
                         'native_failed_tasks': 0, 'native_fit_failures': 0,
                         'native_generation_failures': 0, 'missing_tasks': 0,
                         'scoring_integrity_errors': 0, 'status': 'complete'},
            'scores': {'main': model['main_score']},
            'comparison': {'reference_model_id': reference, 'matched_successful_tasks': 64049,
                           'public_scores': {'main': ref_score},
                           'percent_better': {'main': 100*(1-model['main_score']/ref_score)},
                           'preliminary': False,
                           'reference_rule': 'Worse of the two completed joint-asset TestFolio-style configurations on identical full coverage; one main score.'},
            'runtime': {'forecast_and_score_worker_seconds': model['compute_seconds'],
                        'scope': 'Recorded successful causal preparation, joint simulation and1000-path exact CRPS; excludes IO/setup/queue. No controlled speedup claim.',
                        'wall_elapsed_seconds': None},
            'reproducibility': {'actual_factory': specification['factory'],
                                'actual_implementation_digest': specification['implementation_digest'],
                                'full_execution_manifest_sha256': sha(root/'execution.json'),
                                'full_execution_archive_member': str((root/'execution.json').relative_to(O)),
                                'source_environment': execution['identity']['runtime'],
                                'input_manifest_sha256': sha(V/'inputs/panel.json'),
                                'main_score_source_sha256': report['score_source_sha256'],
                                'paths_per_forecast': 1000,
                                'seed_contract': 'coherent_daily_seed(origin_date,max_horizon,1000)',
                                'full_independent_audit_sha256': sha(V/'full_pair_independent_audit.json'),
                                'evidence_root': '../daily-assets-portfolios-asset-v6-20261020',
                                'preservation_receipt': '../daily-assets-portfolios-asset-v6-20261020/closed-baseline-archive-receipt.json'},
            'archives': parts}
        catalogue.append(ledger, record)
    ledger.update(current_score_protocol_id=report['score_protocol_id'],
                  current_panel_fingerprint=panel['panel_fingerprint'],
                  active_model_ids=list(control['models']),
                  current_best_model_id=audit['better_starting_model_id'],
                  current_best_report='../daily-assets-portfolios-asset-v6-20261020/README.md',
                  updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='Append-only full-panel implementation versions, including losers; identical executions count once. Current asset baseline pair complete; improvement research remains active.')
    path.write_text(json.dumps(ledger, indent=2)+'\n')
    (path.parent/'CATALOGUE.md').write_text(catalogue.render(ledger))
    print(json.dumps({'unique_version_count': ledger['unique_full_panel_models'],
                      'full_panel_executions': ledger['full_panel_executions'],
                      'goal_complete': False}), flush=True)


if __name__ == '__main__':
    main()
