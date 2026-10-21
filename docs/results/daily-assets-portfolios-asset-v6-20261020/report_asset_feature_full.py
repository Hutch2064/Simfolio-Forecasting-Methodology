"""Exact matched progress for the frozen full candidate; official promotion is separate."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import datetime
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
C = V/'candidates/asset_feature_batch'
F = C/'full'
sys.path.insert(0, str(V))
import common
import sweep_v6 as api
import native_index
import reporter
MODEL_ID = json.loads((C/'full_selection.json').read_text())['model_id']
read = lambda p: json.loads(p.read_text())
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def build():
    panel = api.load_panel(V/'inputs', verify_files=False)
    metadata = {r['series_id']: r for r in panel['series']}
    audit = read(V/'full_pair_independent_audit.json')
    assert audit['complete_pair_audit']
    records, models = {}, {}
    connection = sqlite3.connect(f'file:{V/"results/native-loss-index.sqlite3"}?mode=ro', uri=True)
    try:
        for mid in reporter.PUBLIC_IDS:
            fingerprint = read(api.model_directory(V/'results', mid)/'execution.json')['fingerprint']
            rows = connection.execute('SELECT task_key,series_id,origin_date,main_score FROM task_scores_main WHERE model_id=? AND execution_fingerprint=? AND status=? AND scorer_hash=?',
                                      (mid, fingerprint, 'ok', native_index.source_digest())).fetchall()
            assert len(rows) == 64049
            records[mid] = {k: {'series_id': sid, 'origin_date': date, 'score': value} for k, sid, date, value in rows}
            value = reporter.scalar(records[mid].values(), metadata)
            assert abs(value-audit['models'][mid]['main_score']) < 2e-12
            models[mid] = {'complete': True, 'completed_tasks': 64049, 'expected_tasks': 64049,
                           'main_score_on_own_coverage': value, 'native_failures': 0, 'scoring_integrity_errors': 0}
    finally:
        connection.close()
    control = api.read_control(F/'control.json')
    root = api.model_directory(F/'results', MODEL_ID)
    summary, rows = native_index.collect(V/'inputs', panel, root, MODEL_ID,
                                         F/'results/native-loss-index.sqlite3', api, control['models'][MODEL_ID])
    records[MODEL_ID] = {k: {'series_id': r['series_id'], 'origin_date': r['origin_date'], 'score': r['score']} for k, r in rows.items()}
    models[MODEL_ID] = summary
    models[MODEL_ID]['main_score_on_own_coverage'] = reporter.scalar(records[MODEL_ID].values(), metadata)
    expected = set(records[reporter.PUBLIC_IDS[0]])
    assert expected == set(records[reporter.PUBLIC_IDS[1]])
    pairs = reporter.reduce_records(records, metadata, expected, {mid: row['complete'] for mid, row in models.items()})
    assert pairs['reference_model_id'] == audit['worse_reference_model_id']
    report = {'utc': now(), 'models': models, 'native_failures': summary['native_failures'],
              'scoring_integrity_errors': summary['scoring_integrity_errors'],
              'one_main_score': True, 'score_protocol_id': panel['score_protocol_id'],
              'experimental_full_candidate_evaluation_not_official_promotion': True, **pairs}
    api.original._atomic_json(F/'scoreboard.json', report)
    print(json.dumps({'utc': now(), 'candidate_completed': summary['completed_tasks'],
                      'native_failures': summary['native_failures'],
                      'scoring_integrity_errors': summary['scoring_integrity_errors'],
                      'paired_candidate': pairs['paired_comparisons'][MODEL_ID]}), flush=True)


def main():
    while not (F/'active.json').exists():
        process = subprocess.run(['ps', '-p', str(read(O/'asset_feature_full_launch.json')['pid']), '-o', 'command='], capture_output=True, text=True)
        assert 'run_asset_feature_full.py' in process.stdout, 'Owner missing before admission; inspect instead of restarting'
        time.sleep(5)
    api.original._atomic_json(F/'reporter_active.json', {'pid': os.getpid(), 'status': 'running', 'utc': now()})
    while True:
        build()
        if (F/'verification_receipt.json').exists():
            build()
            api.original._atomic_json(F/'reporter_active.json', {'pid': os.getpid(), 'status': 'terminal', 'utc': now()})
            break
        active = read(F/'active.json')
        process = subprocess.run(['ps', '-p', str(active['pid']), '-o', 'command='], capture_output=True, text=True)
        assert 'run_asset_feature_full.py' in process.stdout, 'Controller ended without final verification; inspect native errors and log'
        time.sleep(60)


if __name__ == '__main__':
    main()
