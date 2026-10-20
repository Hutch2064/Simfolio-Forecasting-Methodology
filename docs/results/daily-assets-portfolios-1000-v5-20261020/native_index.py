"""Single-score checkpoint index with separate native and integrity failures."""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import numpy as np
import common
import score_main
import scoring as verified

SCHEMA = '''CREATE TABLE IF NOT EXISTS task_scores_main(
 panel_fingerprint TEXT NOT NULL, scorer_hash TEXT NOT NULL, model_id TEXT NOT NULL,
 execution_fingerprint TEXT NOT NULL, task_key TEXT NOT NULL, series_id TEXT NOT NULL,
 origin_date TEXT NOT NULL, main_score REAL NOT NULL, loss_sha256 TEXT NOT NULL,
 checkpoint_path TEXT NOT NULL, checkpoint_sha256 TEXT NOT NULL, status TEXT NOT NULL,
 scan_id TEXT NOT NULL,
 PRIMARY KEY(panel_fingerprint,scorer_hash,model_id,execution_fingerprint,task_key));'''


def source_digest():
    paths = [Path(__file__), Path(score_main.__file__), Path(verified.__file__)]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def collect(panel_root, panel, model_root, mid, cache_path, api, spec):
    identity, fingerprint = verified._read_model_manifest(model_root, panel['panel_fingerprint'], mid, api)
    if identity.get('factory') != spec['factory'] or identity.get('implementation_digest') != spec['implementation_digest']:
        raise ValueError('Model execution factory/digest differs from enabled control')
    protocol_hash = source_digest()
    scan = hashlib.sha256(os.urandom(32)).hexdigest()
    native = integrity = missing = processed = 0
    examples = []
    expected = 0
    connection = sqlite3.connect(cache_path)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute(SCHEMA)
    try:
        with connection:
            for i, row in enumerate(panel['series']):
                dates, values, origins = verified._load_series(api, panel_root, row)
                if verified._canonical(origins) != verified._canonical(api.schedule(panel_root, row)):
                    raise ValueError('Origin calendar differs from frozen input')
                for j, origin in enumerate(origins):
                    expected += 1
                    key = verified._task_key(row['series_id'], origin)
                    args = (panel['panel_fingerprint'], protocol_hash, mid, fingerprint, key)
                    checkpoint, error = api.task_paths(model_root, (i, j))
                    checkpoint, error = Path(checkpoint), Path(error)
                    cached = connection.execute('SELECT * FROM task_scores_main WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?', args).fetchone()
                    if not checkpoint.is_file():
                        if cached is not None:
                            connection.execute('UPDATE task_scores_main SET status=?,scan_id=? WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?', ('missing', scan, *args))
                        if error.is_file():
                            native += 1
                        else:
                            missing += 1
                        continue
                    try:
                        digest = verified._file_sha256(checkpoint)
                        if cached is not None:
                            if cached['checkpoint_path'] != str(checkpoint) or cached['checkpoint_sha256'] != digest:
                                connection.execute('UPDATE task_scores_main SET status=?,scan_id=? WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?', ('drift', scan, *args))
                                raise ValueError('Immutable checkpoint byte/path drift')
                            connection.execute('UPDATE task_scores_main SET status=?,scan_id=? WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND task_key=?', ('ok', scan, *args))
                            continue
                        loss, _ = api.original._read_checkpoint(checkpoint, api.task_identity(fingerprint, mid, row['series_id'], origin))
                        horizon = origin['max_horizon']
                        position = origin['position']
                        if np.asarray(loss).shape != (horizon,):
                            raise ValueError('Native loss horizon differs')
                        value = score_main.origin_score(loss, values[:position+1], origin['origin_date'], dates[position+1:position+1+horizon])
                        if verified._file_sha256(checkpoint) != digest:
                            raise ValueError('Checkpoint changed during score calculation')
                        loss_digest = hashlib.sha256(np.asarray(loss, dtype=np.float64).tobytes()).hexdigest()
                        connection.execute('INSERT INTO task_scores_main VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (*args, row['series_id'], origin['origin_date'], value, loss_digest, str(checkpoint), digest, 'ok', scan))
                        processed += 1
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError, IndexError) as exc:
                        integrity += 1
                        if len(examples) < 20:
                            examples.append({'series_id': row['series_id'], 'origin_label': origin['origin_label'],
                                             'kind': type(exc).__name__, 'message': str(exc)[:300]})
            selected = connection.execute('SELECT task_key,series_id,origin_date,main_score,loss_sha256 FROM task_scores_main WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND scan_id=? AND status=?',
                (panel['panel_fingerprint'], protocol_hash, mid, fingerprint, scan, 'ok')).fetchall()
    finally:
        connection.close()
    records = {r['task_key']: {'series_id': r['series_id'], 'origin_date': r['origin_date'],
                              'score': r['main_score'], 'loss_sha256': r['loss_sha256']} for r in selected}
    return {'completed_tasks': len(records), 'expected_tasks': expected, 'native_failures': native,
            'scoring_integrity_errors': integrity, 'missing_tasks': missing, 'processed': processed,
            'complete': len(records) == expected and native == integrity == missing == 0, 'errors': examples}, records
