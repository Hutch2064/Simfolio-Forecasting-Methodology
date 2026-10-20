"""One-score v4 reporter, using the verified immutable native-loss ingestion."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time
import common
import score_main as main
import native_index
import scoring as ingestion
import sweep as api

OWN = common.OWN
PUBLIC_IDS = ('model_testing_joint_stationary_bootstrap_v1',
              'model_testing_joint_uniform_bounded_block_bootstrap_v1',
              'model_testing_joint_uniform_bounded_cycle_bootstrap_v1')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def native_failures(model_root):
    root = Path(model_root)
    return sum(not (root/'tasks'/p.relative_to(root/'errors').with_suffix('.npz')).exists()
               for p in (root/'errors').rglob('*.json')) if (root/'errors').exists() else 0


def scalar(records, metadata):
    try:
        return main.aggregate_records(records, metadata)['score']
    except ValueError as exc:
        if 'lack matched individual-asset support' in str(exc):
            return None
        raise


def reduce_records(records_by_model, metadata, expected, complete):
    """One fixed reference after Public completes; exact intersections before it."""
    available = [mid for mid in PUBLIC_IDS if mid in records_by_model]
    reference = None
    reference_scope = set()
    reference_scores = {}
    if len(available) == len(PUBLIC_IDS):
        reference_scope = set.intersection(*(set(records_by_model[mid]) for mid in PUBLIC_IDS))
        for mid in PUBLIC_IDS:
            value = scalar([records_by_model[mid][k] for k in sorted(reference_scope)], metadata)
            if value is not None:
                reference_scores[mid] = value
        if len(reference_scores) == len(PUBLIC_IDS):
            reference = max(reference_scores, key=lambda mid: (reference_scores[mid], mid))
    pairs = {}
    if reference:
        for mid, records in records_by_model.items():
            keys = sorted(set(records) & set(records_by_model[reference]))
            candidate = scalar([records[k] for k in keys], metadata)
            public = scalar([records_by_model[reference][k] for k in keys], metadata)
            pairs[mid] = {'reference_model_id': reference, 'matched_successful_tasks': len(keys),
                'main_score': candidate, 'reference_main_score': public,
                'percent_better': 100*(1-candidate/public) if candidate is not None and public else None,
                'preliminary': not (complete.get(mid) and complete.get(reference) and set(keys) == expected),
                'score_pending_matched_branch_support': candidate is None}
    full = [mid for mid in records_by_model if complete.get(mid) and set(records_by_model[mid]) == expected]
    leaderboard = sorted([{'model_id': mid, 'main_score': scalar(records_by_model[mid].values(), metadata),
                           'successful_tasks': len(expected)} for mid in full], key=lambda r: r['main_score'])
    return {'reference_model_id': reference, 'reference_rule': 'worst_of_three_Public_on_identical_coverage',
        'reference_selection_matched_tasks': len(reference_scope),
        'reference_selection_preliminary': not (len(available) == 3 and all(complete.get(mid) for mid in PUBLIC_IDS)),
        'reference_comparator_main_scores': reference_scores, 'paired_comparisons': pairs,
        'complete_coverage_leaderboard': leaderboard,
        'incomplete_models_are_not_ranked_in_full_coverage_leaderboard': True}


def build():
    control_path = OWN/'control.json'
    control_hash = sha(control_path)
    control = api.read_control(control_path)
    panel = ingestion._load_panel(api, OWN/'inputs')
    if panel.get('score_protocol_id') != main.SCORE_PROTOCOL_ID:
        raise ValueError('Panel score protocol differs from sole main objective')
    metadata = {r['series_id']: r for r in panel['series']}
    expected = {ingestion._task_key(r['series_id'], o) for r in panel['series'] for o in api.schedule(OWN/'inputs', r)}
    records = {}
    models = {}
    errors = native = 0
    for mid, spec in control['models'].items():
        if not spec['enabled']:
            continue
        root = api.model_directory(OWN/'results', mid)
        if not (root/'execution.json').exists():
            records[mid] = {}
            models[mid] = {'completed_tasks': 0, 'expected_tasks': len(expected), 'complete': False,
                'main_score_on_own_coverage': None, 'native_failures': 0, 'scoring_integrity_errors': 0,
                'state': 'waiting_for_controller_admission'}
            continue
        failures = 0
        try:
            summary, raw = native_index.collect(OWN/'inputs', panel, root, mid,
                OWN/'results/native-loss-index.sqlite3', api, spec)
            records[mid] = {k: {'series_id': r['series_id'], 'origin_date': r['origin_date'],
                               'score': r['score']} for k, r in raw.items()}
            integrity = summary['scoring_integrity_errors']
            failures = summary['native_failures']
            models[mid] = {'completed_tasks': summary['completed_tasks'], 'expected_tasks': len(expected),
                'complete': bool(summary['complete']),
                'main_score_on_own_coverage': scalar(records[mid].values(), metadata),
                'native_failures': failures, 'scoring_integrity_errors': integrity,
                'missing_tasks': summary['missing_tasks'], 'state': 'complete' if summary['complete'] else 'partial',
                'errors': summary['errors']}
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            integrity = 1
            failures = native_failures(root)
            records[mid] = {}
            models[mid] = {'completed_tasks': 0, 'expected_tasks': len(expected), 'complete': False,
                'main_score_on_own_coverage': None, 'native_failures': failures,
                'scoring_integrity_errors': integrity, 'state': 'integrity_error',
                'errors': [{'kind': type(exc).__name__, 'message': str(exc)[:500]}]}
        native += failures
        errors += integrity
    if sha(control_path) != control_hash:
        raise ingestion.ControlChangedDuringScan('Admission changed during scan; retry without publishing stale control')
    report = {'schema': 'single-main-portfolio-panel-scoreboard-v4', 'generated_at_utc': now(),
        'score_protocol_id': main.SCORE_PROTOCOL_ID, 'panel_fingerprint': panel['panel_fingerprint'],
        'score_source_sha256': sha(main.__file__), 'reporter_source_sha256': sha(__file__),
        'native_loss_ingestion_source_sha256': native_index.source_digest(),
        'control_sha256': control_hash, 'models': models,
        'native_failures': native, 'scoring_integrity_errors': errors,
        'expected_tasks_per_model': len(expected), 'diagnostics_in_score': False,
        'one_main_score_column_only': True,
        **reduce_records(records, metadata, expected, {mid: r['complete'] for mid, r in models.items()})}
    api.original._atomic_json(OWN/'results/scoreboard.json', report)
    return report


def watch(seconds):
    api.original._atomic_json(OWN/'reporter_active.json', {'pid': os.getpid(), 'status': 'running', 'utc': now()})
    while True:
        try:
            report = build()
            print(json.dumps({'utc': now(), 'completed_tasks': sum(m['completed_tasks'] for m in report['models'].values()),
                'native_failures': report['native_failures'], 'scoring_integrity_errors': report['scoring_integrity_errors'],
                'reference_model_id': report['reference_model_id']}), flush=True)
        except ingestion.ControlChangedDuringScan:
            pass
        if (OWN/'controller_terminal_receipt.json').exists():
            # Controller exits at stage boundaries; do one final published scan.
            build()
            api.original._atomic_json(OWN/'reporter_active.json', {'pid': os.getpid(), 'status': 'terminal', 'utc': now()})
            break
        time.sleep(seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['once', 'watch'])
    parser.add_argument('--seconds', type=float, default=120.)
    args = parser.parse_args()
    build() if args.action == 'once' else watch(args.seconds)
