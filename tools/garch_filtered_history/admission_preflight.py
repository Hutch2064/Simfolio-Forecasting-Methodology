"""Fit-only admission of 52 assets x 51 canonical training prefixes.

Does not construct future return targets, forecasts, or OOS scores.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

for key in ('NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(HERE)]
import numpy as np
from simfolio_forecasting_methodology.data import CANONICAL_TICKERS, load_canonical_engine_inputs
from simfolio_forecasting_methodology.origins import origin_schedule
from simfolio_forecasting_methodology.runner import PortfolioPolicy, TrainingData, _training_digest


def fit_one(item):
    import models
    record, data = item
    started = time.perf_counter()
    try:
        receipt = models.fit_asset(data)[4]
        record.update(passed=True, fit=receipt, retry_count=receipt['retry_count'])
    except Exception as error:
        record.update(passed=False, error=type(error).__name__, message=str(error),
            attempts=getattr(error, 'attempts', []))
    record['seconds'] = time.perf_counter()-started
    return record


def source_hashes():
    paths = [p for folder in (ROOT/'src', ROOT/'tools') for p in folder.rglob('*')
             if p.suffix in ('.py', '.cpp')]
    paths.append(ROOT/'tests/test_garch_filtered_history.py')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def atomic(path, result):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(clean(result), indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--output', type=Path, default=HERE/'admission_preflight.json')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError('preflight requires one to twelve processes')
    source = source_hashes()
    manifest = args.data/'canonical_data_manifest.json'
    manifest_data = json.loads(manifest.read_text())
    assets, _ = load_canonical_engine_inputs(args.data)
    origins = origin_schedule(assets.index)
    if len(CANONICAL_TICKERS) != 52 or len(origins) != 51:
        raise ValueError('canonical training-prefix denominator mismatch')
    tasks = []
    for ticker in CANONICAL_TICKERS:
        column = assets[ticker].to_numpy(dtype=float)
        for origin in origins:
            x = np.ascontiguousarray(column[:origin.position+1])
            dates = assets.index[:origin.position+1].to_numpy()
            training = TrainingData(x, x[:, None], PortfolioPolicy((ticker,), (1.,), 'none'), dates)
            record = {'index': len(tasks), 'ticker': ticker, 'origin': origin.origin_date,
                'origin_label': origin.origin_label, 'training_rows': len(x),
                'training_digest': _training_digest(training),
                'asset_data_sha256': hashlib.sha256(x.tobytes()).hexdigest()}
            tasks.append((record, x.tobytes()))
    if len(tasks) != 2652:
        raise ValueError('missing training prefixes')
    # Verify exact prefix identity against both previously failed official tasks.
    for folder, index in (('garch-single52', 57), ('garch-repaired-single52', 67)):
        previous = ROOT.parent/'expanded'/folder/'manifest.json'
        if previous.exists():
            identity = json.loads(previous.read_text())['task_identities'][index]
            if tasks[index][0]['training_digest'] != identity['training_digest']:
                raise ValueError('preflight training prefix differs from official task')
    result = {'scope': 'fit_admission_only_no_forecast_no_OOS_scores',
        'candidate': 'asset_m261_mean_gjr_garch_student_filtered_history',
        'required_prefixes': 2652, 'workers': args.workers,
        'data_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'dataset_id': manifest_data['dataset_id'],
        'dataset_derived_fingerprints': manifest_data['derived_fingerprints'],
        'source_hashes': source, 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('numpy','scipy','numba','pandas','arch')},
        'thread_environment': {key: os.environ.get(key) for key in
            ('NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_CACHE_DIR','TMPDIR')},
        'records': []}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(fit_one, item) for item in tasks]
        for future in as_completed(futures):
            result['records'].append(future.result())
            if len(result['records']) % 100 == 0:
                result.update(completed=len(result['records']), elapsed_seconds=time.perf_counter()-started)
                atomic(args.output, result)
                print(json.dumps({'completed':result['completed'], 'failures':sum(not r['passed'] for r in result['records']),
                    'elapsed_seconds':result['elapsed_seconds']}), flush=True)
    result['records'].sort(key=lambda record: record['index'])
    changed = [path for path, digest in source.items()
               if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest]
    result.update(completed=len(result['records']), elapsed_seconds=time.perf_counter()-started,
        failures=sum(not r['passed'] for r in result['records']),
        retried_prefixes=sum(r.get('retry_count', 0)>0 for r in result['records']),
        summed_fit_seconds=sum(r['seconds'] for r in result['records']),
        source_changed_paths=changed)
    result['all_prefixes_admitted'] = not result['failures'] and not changed
    atomic(args.output, result)
    print(json.dumps({key:value for key,value in result.items() if key not in
        ('records','source_hashes','dataset_derived_fingerprints','thread_environment','packages')}), flush=True)


if __name__ == '__main__':
    main()
