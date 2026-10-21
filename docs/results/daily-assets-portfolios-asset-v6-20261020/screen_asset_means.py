"""Preliminary100-path multi-hypothesis screen; never an official panel score."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import multiprocessing
import sys
import time
import numpy as np
O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
D = V/'candidates/asset_mean_uncertainty/screen100'
sys.path.insert(0, str(V))
import common
import sweep_v6 as api
import reporter
import score_main
from asset_paths import JointAssetBootstrap
from inputs import AssetInputs
from candidates.asset_mean_uncertainty.model import JointMeanBootstrap
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def initialize():
    global PANEL, ROWS
    PANEL = api.load_panel(V/'inputs', verify_files=False)
    ROWS = {r['series_id']: r for r in PANEL['series']}


def execute(job):
    number, case, replacement, variant = job
    start_time = time.perf_counter()
    row, origin = ROWS[case['series_id']], case['origin']
    dates, values, origins = api.load_series(V/'inputs', row, verify_files=False)
    assert origin in origins
    stop, horizon = origin['position']+1, origin['max_horizon']
    history, weights, ids = AssetInputs(V/'inputs').training(row, stop)
    model = JointAssetBootstrap(replacement=replacement) if variant == 'raw' else JointMeanBootstrap(replacement, variant == 'shrunk_mean')
    seed = api.original.coherent_daily_seed(origin['origin_date'], horizon, 100)
    realized, losses = np.cumsum(values[stop:stop+horizon]), np.empty(horizon)
    cursor = 0
    for first, paths in model.terminal_chunks(history, weights, horizon, 100, seed):
        assert first == cursor and paths.shape[0] == 100
        cursor += paths.shape[1]
        losses[first:cursor] = empirical_crps_by_horizon(paths, realized[first:cursor])
    assert cursor == horizon and np.isfinite(losses).all()
    arm = ('replacement_' if replacement else 'history_partition_')+variant
    destination = D/arm
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/f'{number:04}.npz', losses=losses)
    return {'arm': arm, 'series_id': row['series_id'], 'origin_date': origin['origin_date'],
            'origin_label': origin['origin_label'], 'score': score_main.origin_score(losses, values[:stop], origin['origin_date'], dates[stop:stop+horizon]),
            'paths': 100, 'constituents': ids, 'seed': seed, 'pid': os.getpid(),
            'worker_compute_seconds': time.perf_counter()-start_time,
            'loss_sha256': hashlib.sha256(losses.tobytes()).hexdigest()}


def main():
    lock = (O/'asset_v6_forecast_slots.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert read(V/'controller_active.json')['status'] == 'terminal'
    assert not (D/'active.json').exists(), 'Do not silently rerun a completed or running screen'
    D.mkdir(parents=True, exist_ok=True)
    cases = read(V/'candidates/asset_prior_best/plan.json')['phases']['development']
    assert len(cases) == 355
    api.original._atomic_json(D/'active.json', {'pid': os.getpid(), 'status': 'running', 'workers': 12, 'paths': 100})
    sources = [Path(__file__), V/'candidates/asset_mean_uncertainty/model.py', V/'candidates/asset_prior_best/mean.py', V/'candidates/asset_prior_best/kernels.py', V/'candidates/asset_prior_best/candidate.py']
    hashes = {str((V/name).relative_to(O)): digest for name, digest in read(V/'launch_plan.json')['source_hashes'].items()}
    hashes.update({str(p.relative_to(O)): sha(p) for p in sources})
    jobs = [(n, c, replacement, variant) for n, c in enumerate(cases) for replacement in (False, True) for variant in ('raw', 'unshrunk_mean', 'shrunk_mean')]
    api.original._atomic_json(D/'plan.json', {'cases': cases, 'paths': 100, 'workers': 12, 'source_hashes': hashes,
        'panel_fingerprint': read(V/'inputs/panel.json')['panel_fingerprint'],
        'mechanism': 'History-cycle ordering cannot represent uncertainty about the finite-history mean. Test correlated batch mean uncertainty, with and without the previous best smooth shrinkage, while preserving each raw block sampler.',
        'is_official_panel_score': False, 'eligible_for_promotion_without_1000_path_confirmation': False})
    rows, errors = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                job = futures[future]
                errors.append({'case': job[1], 'replacement': job[2], 'variant': job[3], 'exception_type': type(exc).__name__, 'message': str(exc)})
                api.original._atomic_json(D/'native_errors.json', errors)
            if (len(rows)+len(errors)) % 100 == 0:
                api.original._atomic_json(D/'progress.json', {'completed': len(rows), 'native_failures': len(errors), 'expected': len(jobs), 'paths': 100})
    assert all(sha(O/name) == value for name, value in hashes.items())
    metadata = {r['series_id']: r for r in api.load_panel(V/'inputs', verify_files=False)['series']}
    scores = {arm: reporter.scalar([r for r in rows if r['arm'] == arm], metadata) if not errors else None for arm in sorted({r['arm'] for r in rows})}
    assert errors or all(v is not None for v in scores.values())
    summary = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': not errors,
               'cases_per_arm': 355, 'forecast_successes': len(rows), 'native_failures': len(errors),
               'paths': 100, 'workers_requested': 12, 'worker_pids': sorted({r['pid'] for r in rows}),
               'exploratory_main_Q': scores, 'is_official_panel_score': False,
               'eligible_for_promotion_without_1000_path_confirmation': False,
               'independent_validation_claim': False, 'records': rows}
    api.original._atomic_json(D/'summary.json', summary)
    api.original._atomic_json(D/'active.json', {'pid': os.getpid(), 'status': 'terminal', 'paths': 100})
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
