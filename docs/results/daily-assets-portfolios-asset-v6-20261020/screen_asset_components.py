"""Frozen100-path component contrast; reuses verified same-count raw/mean arms."""
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
import subprocess
import sys
import time
import numpy as np
O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
D = V/'candidates/asset_mean_uncertainty/components_screen100'
OLD = V/'candidates/asset_mean_uncertainty/screen100'
sys.path.insert(0, str(V))
import common
import sweep_v6 as api
import reporter
import score_main
from inputs import AssetInputs
from candidates.asset_prior_best.candidate import AssetPool
from candidates.asset_mean_uncertainty.fhs_only import JointFilteredSimulation
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def initialize():
    global ROWS, REPLACEMENT
    ROWS = {r['series_id']: r for r in api.load_panel(V/'inputs', verify_files=False)['series']}
    REPLACEMENT = read(V/'candidates/asset_prior_best/parent_selection.json')['better_model_id'].endswith('replacement_v6')


def execute(job):
    number, case, arm = job
    started = time.perf_counter()
    row, origin = ROWS[case['series_id']], case['origin']
    dates, values, origins = api.load_series(V/'inputs', row, verify_files=False)
    assert origin in origins
    stop, horizon = origin['position']+1, origin['max_horizon']
    history, weights, ids = AssetInputs(V/'inputs').training(row, stop)
    model = AssetPool(REPLACEMENT) if arm == 'hybrid' else JointFilteredSimulation()
    seed = api.original.coherent_daily_seed(origin['origin_date'], horizon, 100)
    losses, realized, cursor = np.empty(horizon), np.cumsum(values[stop:stop+horizon]), 0
    for first, paths in model.terminal_chunks(history, weights, horizon, 100, seed):
        assert first == cursor and paths.shape[0] == 100
        cursor += paths.shape[1]
        losses[first:cursor] = empirical_crps_by_horizon(paths, realized[first:cursor])
    assert cursor == horizon and np.isfinite(losses).all()
    destination = D/arm
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/f'{number:04}.npz', losses=losses)
    return {'arm': arm, 'series_id': row['series_id'], 'origin_date': origin['origin_date'],
            'origin_label': origin['origin_label'], 'score': score_main.origin_score(losses, values[:stop], origin['origin_date'], dates[stop:stop+horizon]),
            'paths': 100, 'constituents': ids, 'seed': seed, 'pid': os.getpid(),
            'worker_compute_seconds': time.perf_counter()-started,
            'loss_sha256': hashlib.sha256(losses.tobytes()).hexdigest(), 'reused': False}


def main():
    slots = (O/'asset_v6_forecast_slots.lock').open('a')
    fcntl.flock(slots, fcntl.LOCK_EX)
    assert not (D/'active.json').exists(), 'Inspect an existing run rather than blindly restarting'
    subprocess.run([sys.executable, '/Users/aidanhutchison/simfolio/storage_preflight.py',
                    '--expected-gib', '.2', '--artifact-root', str(O), '--budget-gib', '12'], check=True)
    old_plan, old = read(OLD/'plan.json'), read(OLD/'summary.json')
    assert old['complete'] and old['paths'] == 100 and old['native_failures'] == 0
    assert all(sha(O/name) == value for name, value in old_plan['source_hashes'].items())
    cases = old_plan['cases']
    assert len(cases) == 355
    by_key = {(c['series_id'], c['origin']['origin_label']): (i, c) for i, c in enumerate(cases)}
    selection = read(V/'candidates/asset_prior_best/parent_selection.json')
    assert selection['better_model_id'] == 'model_testing_asset_joint_blocks_history_partition_v6'
    rows = []
    for r in old['records']:
        if r['arm'] not in ('replacement_raw', 'history_partition_shrunk_mean'):
            continue
        index, case = by_key[r['series_id'], r['origin_label']]
        path = OLD/r['arm']/f'{index:04}.npz'
        with np.load(path, allow_pickle=False) as saved:
            loss = saved['losses']
        assert hashlib.sha256(loss.tobytes()).hexdigest() == r['loss_sha256']
        rows.append({**r, 'arm': 'fixed_reference' if r['arm'] == 'replacement_raw' else 'raw_shrunk',
                     'reused': True, 'verified_original_checkpoint': str(path), 'original_checkpoint_sha256': sha(path)})
    assert len(rows) == 710
    D.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), V/'candidates/asset_mean_uncertainty/fhs_only.py',
             V/'candidates/asset_prior_best/candidate.py', V/'candidates/asset_prior_best/fit.py',
             V/'candidates/asset_prior_best/mean.py', V/'candidates/asset_prior_best/kernels.py']
    hashes = {**old_plan['source_hashes'], **{str(p.relative_to(O)): sha(p) for p in files}}
    api.original._atomic_json(D/'plan.json', {'cases': cases, 'paths': 100, 'workers': 12,
        'source_hashes': hashes, 'reused100path_forecasts': 710, 'new_forecasts': 710,
        'mechanism': 'Compare the natural raw-only, equal-pool and filtered-only component endpoints; no mixture-weight grid or outcome-fitted allocation.',
        'fixed_reference': selection['worse_reference_model_id'],
        'is_official_panel_score': False, 'eligible_for_promotion': False})
    api.original._atomic_json(D/'active.json', {'pid': os.getpid(), 'status': 'running', 'paths': 100, 'workers': 12})
    errors = []
    jobs = [(i, c, arm) for i, c in enumerate(cases) for arm in ('hybrid', 'filtered_only')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({'job': futures[future], 'exception_type': type(exc).__name__, 'message': str(exc)})
                api.original._atomic_json(D/'native_errors.json', errors)
            if (len(rows)+len(errors)) % 100 == 0:
                api.original._atomic_json(D/'progress.json', {'new_completed': sum(not r['reused'] for r in rows), 'new_expected': 710, 'reused': 710, 'native_failures': len(errors)})
    assert all(sha(O/name) == value for name, value in hashes.items())
    metadata = {r['series_id']: r for r in api.load_panel(V/'inputs', verify_files=False)['series']}
    scores = {arm: reporter.scalar([r for r in rows if r['arm'] == arm], metadata) if not errors else None
              for arm in ('fixed_reference', 'raw_shrunk', 'hybrid', 'filtered_only')}
    assert errors or all(value is not None for value in scores.values())
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': not errors,
              'new_forecasts': sum(not r['reused'] for r in rows), 'reused_forecasts': 710,
              'cases_per_arm': 355, 'paths': 100, 'native_failures': len(errors),
              'worker_pids': sorted({r['pid'] for r in rows if not r['reused']}),
              'exploratory_main_Q': scores,
              'percent_better_vs_same100path_fixed_reference': {arm: 100*(1-value/scores['fixed_reference']) if not errors else None for arm, value in scores.items()},
              'filtered_only_warrants1000path_confirmation': bool(not errors and scores['filtered_only'] < scores['hybrid']),
              'is_official_panel_score': False, 'eligible_for_promotion': False,
              'independent_validation_claim': False, 'records': rows}
    api.original._atomic_json(D/'summary.json', result)
    api.original._atomic_json(D/'active.json', {'pid': os.getpid(), 'status': 'terminal', 'paths': 100})
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
