"""Twelve-worker real-data qualification, separate from full-panel scoring."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import hashlib
import datetime
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from inputs import AssetInputs
from asset_paths import JointAssetBootstrap
from blocks import BlockLimits

OWN = Path(__file__).resolve().parent
V5 = OWN.parent/'simulation_1000_v5'
sys.path.insert(0, str(V5))
import common
import score_main
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from simfolio_forecasting_methodology.seeds import coherent_daily_seed

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def matrix_audit(i):
    inputs = AssetInputs(V5/'inputs'); row = inputs.panel['series'][i]
    counts = sorted({2, row['observations']//2, row['observations']})
    for n in counts:
        x, w, ids = inputs.training(row, n)
        assert x.shape == (n, len(ids)) and not x.flags.writeable
    return {'series_id': row['series_id'], 'prefixes': counts, 'constituents': len(ids), 'pid': os.getpid()}


def forecast_case(task):
    i, j, replacement = task
    inputs = AssetInputs(V5/'inputs'); row = inputs.panel['series'][i]
    origins = json.loads((V5/'inputs'/row['origins_file']).read_text()); origin = origins[j]
    dates, targets = inputs.source(row)
    stop, horizon = origin['position']+1, origin['max_horizon']
    # Future targets remain outside the simulator; its input is this immutable prefix.
    started = time.perf_counter()
    history, weights, ids = inputs.training(row, stop)
    profile = json.loads((OWN/'profile.json').read_text())
    model = JointAssetBootstrap(replacement=replacement,
        limits=BlockLimits(profile['minimum_block_observations'], profile['maximum_block_observations']))
    realized = np.cumsum(targets[stop:stop+horizon]); losses = np.empty(horizon)
    seed = coherent_daily_seed(origin['origin_date'], horizon, 1000)
    cursor = 0
    for start, terminal in model.terminal_chunks(history, weights, horizon, 1000, seed):
        assert start == cursor and terminal.shape[0] == 1000
        end = start+terminal.shape[1]
        losses[start:end] = empirical_crps_by_horizon(terminal, realized[start:end]); cursor = end
    assert cursor == horizon and np.isfinite(losses).all()
    score = score_main.origin_score(losses, targets[:stop], origin['origin_date'], dates[stop:stop+horizon])
    result = {'series_index': i, 'origin_index': j, 'series_id': row['series_id'],
        'origin': origin, 'model_id': model.model_id, 'simulations': 1000,
        'constituent_ids': ids, 'forecast_and_score_seconds': time.perf_counter()-started,
        'main_origin_score': float(score), 'loss_sha256': hashlib.sha256(losses.tobytes()).hexdigest(),
        'worker_pid': os.getpid(), 'joint_asset_paths_simulated': True,
        'weights_rebalanced_inside_each_path': True, 'main_score_protocol': score_main.SCORE_PROTOCOL_ID}
    folder = OWN/'qualification/results'/model.model_id; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/f'{i:06d}_{j:03d}.npz', losses=losses,
                        metadata=np.asarray(json.dumps(result, sort_keys=True)))
    return result


def main():
    inputs = AssetInputs(V5/'inputs'); panel = inputs.panel
    assert len(panel['series']) == 1264 and panel['expected_tasks'] == 64049
    # Verify every native input checksum before qualifying an architecture change.
    for row in panel['series']:
        for path, digest in row['files'].items():
            assert sha(V5/'inputs'/path) == digest
    grouped = {}
    for i, row in enumerate(panel['series']):
        grouped.setdefault((row['asset_class'], row['family']), i)
    cases = set()
    for i in grouped.values():
        row = panel['series'][i]
        origins = json.loads((V5/'inputs'/row['origins_file']).read_text())
        rolling = [j for j, o in enumerate(origins) if o['evaluation_split'] == 'rolling']
        j = min(rolling or range(len(origins)), key=lambda j: abs(origins[j]['max_horizon']-252))
        cases.add((i, j))
    for cls in sorted({r['asset_class'] for r in panel['series']}):
        i = max((i for i, r in enumerate(panel['series']) if r['asset_class'] == cls),
                key=lambda i: panel['series'][i]['max_horizon'])
        origins = json.loads((V5/'inputs'/panel['series'][i]['origins_file']).read_text())
        cases.add((i, max(range(len(origins)), key=lambda j: origins[j]['max_horizon'])))
    tasks = [(i, j, replacement) for i, j in sorted(cases) for replacement in (True, False)]
    hashes = {p.name: sha(p) for p in sorted(OWN.glob('*.py'))}
    hashes['profile.json'] = sha(OWN/'profile.json')
    plan = {'utc': now(), 'source_hashes': hashes, 'panel_fingerprint': panel['panel_fingerprint'],
            'families': len(grouped), 'tasks': tasks, 'workers': 12, 'paths': 1000,
            'representative_qualification_only': True, 'no_reference_selected_from_smoke_scores': True}
    write(OWN/'qualification/plan.json', plan)
    active = {'utc': now(), 'pid': os.getpid(), 'status': 'running', 'phase': 'matrix_audit',
              'expected_forecasts': len(tasks), 'completed_forecasts': 0, 'worker_pids': []}
    write(OWN/'qualification/active.json', active)
    ctx = multiprocessing.get_context('spawn'); matrices, results, failures = [], [], []
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        futures = [pool.submit(matrix_audit, i) for i, r in enumerate(panel['series']) if r['panel_branch'] == 'portfolios']
        for future in as_completed(futures):
            matrices.append(future.result())
        assert len(matrices) == 542
        write(OWN/'qualification/matrix_audit.json', {'utc': now(), 'portfolios': matrices,
              'prefixes': sum(len(r['prefixes']) for r in matrices), 'failures': 0})
        active.update(phase='forecast_smoke', worker_pids=sorted({r['pid'] for r in matrices}))
        write(OWN/'qualification/active.json', active)
        futures = {pool.submit(forecast_case, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                result = future.result(); results.append(result)
            except Exception as exc:
                failures.append({'task': futures[future], 'type': type(exc).__name__, 'message': str(exc)})
            active.update(utc=now(), completed_forecasts=len(results), failures=failures,
                          worker_pids=sorted({r['worker_pid'] for r in results}))
            write(OWN/'qualification/active.json', active)
            if (len(results)+len(failures)) % 12 == 0:
                print(json.dumps({k: v for k, v in active.items() if k != 'failures'}), flush=True)
    assert all(sha(OWN/name) == digest for name, digest in hashes.items()), 'Qualification source changed'
    write(OWN/'qualification/receipt.json', {'utc': now(), 'successes': len(results),
          'expected_forecasts': len(tasks), 'failures': failures, 'source_hashes': hashes,
          'worker_pids': sorted({r['worker_pid'] for r in results}), 'families': len(grouped),
          'all542_portfolio_matrices_checked': True, 'simulations': 1000,
          'full_panel_complete': False, 'reference_selection_pending_full_pair': True, 'records': results})
    active.update(utc=now(), status='terminal', active_workers=0)
    write(OWN/'qualification/active.json', active)
    if failures:
        raise RuntimeError(f'{len(failures)} qualification failures; no full admission')


if __name__ == '__main__':
    main()
