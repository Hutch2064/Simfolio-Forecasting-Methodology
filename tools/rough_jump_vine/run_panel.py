"""Checkpointed five-model canonical run; the public scorer is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'tools/mcmc_runtime')]
import models
import numba
import numpy as np
import pyvinecopulib as pv
import pyvinecopulib.pyvinecopulib_ext as native
import scipy

from simfolio_forecasting_methodology.data import prepare_canonical_data, verify_canonical_snapshot
from simfolio_forecasting_methodology.evaluation import CellAccumulator
from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.runner import evaluate_origin_task, task_identity

REFERENCE = None
CANDIDATES = models.CANDIDATES


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def initialize(cache_root, reference, vine_threads, state_workers, model_ids):
    global REFERENCE, CANDIDATES
    CANDIDATES = tuple(c for c in models.CANDIDATES if c.model_id in model_ids)
    REFERENCE = Path(reference)
    models.initialize(cache_root, vine_threads, state_workers)


def evaluate(index, task):
    models.clear_path_cache()
    records = []
    for candidate in CANDIDATES:
        models.TIMINGS.clear()
        start = time.perf_counter()
        losses = evaluate_origin_task(candidate, task, simulations=240)
        seconds = time.perf_counter()-start
        if candidate.model_id == models.MODEL_IDS[0]:
            expected = np.load(REFERENCE/f'task-{index:04d}-model-1.npz')['losses']
            if losses.tobytes() != expected.tobytes():
                raise ValueError(f'production baseline loss-vector parity failed: {index}; '
                                 f'max difference={np.max(np.abs(losses-expected))}')
        details = dict(models.TIMINGS)
        records.append((candidate.model_id, losses, seconds, details))
    models.clear_path_cache()
    return index, records


def source_hashes():
    paths = sorted((ROOT/'src').rglob('*.py'))
    paths += sorted((ROOT/'tools/mcmc_runtime').glob('*.py'))
    paths += sorted(HERE.glob('*.py'))
    paths += sorted(HERE.glob('*.patch'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def score_completed(plan, out, digest, elapsed, model_ids):
    counts = {}
    for task in plan.tasks:
        for h in range(1, task.horizon_days+1):
            key = (task.portfolio_id, h)
            counts[key] = counts.get(key, 0)+1
    accumulators = {mid: CellAccumulator() for mid in model_ids}
    totals = {mid: 0.0 for mid in model_ids}
    timings = {mid: {} for mid in model_ids}
    hashes = {}
    for i, task in enumerate(plan.tasks):
        path = out/f'task-{i:04d}.npz'
        with np.load(path) as saved:
            assert str(saved['manifest']) == digest
            for j, mid in enumerate(model_ids):
                losses = saved[f'losses_{j}']
                assert losses.size == task.horizon_days
                accumulators[mid].add_vector(task.portfolio_id, losses)
                totals[mid] += float(saved[f'seconds_{j}'])
        record = json.loads((out/f'task-{i:04d}.json').read_text())
        for mid, values in record['timings'].items():
            for key, value in values.items():
                timings[mid][key] = timings[mid].get(key, 0)+value
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    scores = {}
    cell_arrays = {}
    keys = sorted(counts)
    cell_arrays['portfolio'] = np.array([k[0] for k in keys])
    cell_arrays['horizon'] = np.array([k[1] for k in keys])
    for j, mid in enumerate(model_ids):
        a = accumulators[mid]
        score = a.fixed_denominator_score(expected_cells=701280, expected_tasks=4080,
                                        completed_tasks=len(plan.tasks), expected_origin_counts=counts)
        scores[mid] = {'exact_empirical_crps': score, 'tasks': 4080, 'cells': 701280,
                          'summed_worker_seconds': totals[mid], 'timings': timings[mid]}
        means = a.cell_means()
        cell_arrays[f'model_{j}'] = np.array([means[k] for k in keys])
    baseline = 0.25246784959071183
    if models.MODEL_IDS[0] in scores:
        assert scores[models.MODEL_IDS[0]]['exact_empirical_crps'] == baseline
    for mid, values in scores.items():
        values['relative_improvement_vs_production'] = 1-values['exact_empirical_crps']/baseline
    np.savez_compressed(out/'paired-cell-losses.npz', **cell_arrays)
    atomic(out/'checkpoint-hashes.json', hashes)
    atomic(out/'results.json', {'scope': 'full_canonical_asset_level_ablations',
            'manifest_sha256': digest, 'all_denominator_gates_passed': True,
            'baseline_every_loss_vector_byte_exact': models.MODEL_IDS[0] in scores, 'portfolios': 80,
            'rolling_origins_per_portfolio': 48, 'temporal_holdouts_per_portfolio': 3,
            'simulations': 240, 'elapsed_seconds': elapsed, 'scores': scores,
            'paired_cells_sha256': hashlib.sha256((out/'paired-cell-losses.npz').read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--pilot-indices', default='')
    parser.add_argument('--model-id', choices=models.MODEL_IDS)
    parser.add_argument('--reuse', type=Path, help='Reuse identical model checkpoints from a stopped run')
    args = parser.parse_args()
    candidates = tuple(c for c in models.CANDIDATES if not args.model_id or c.model_id == args.model_id)
    model_ids = tuple(c.model_id for c in candidates)
    budget = max(1, 12//args.workers)
    state_workers = min(3, budget)
    vine_threads = max(1, budget//state_workers)
    global REFERENCE
    REFERENCE = args.reference.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = out.parent/'data'
    if not data.exists():
        prepare_canonical_data(destination=data)
    plan = build_experiment_plan(data)
    assert len(plan.portfolios) == 80 and len(plan.tasks) == 4080 and plan.cell_capacity == 701280
    assert str(plan.common_dates[0].date()) == '1979-12-31'
    indices = list(range(4080))
    if args.pilot_indices:
        indices = [int(i) for i in args.pilot_indices.split(',')]
    manifest = {'scope': 'pilot' if args.pilot_indices else 'full_canonical',
            'models': [asdict(m) for m in candidates], 'simulations': 240,
            'portfolio_count': 80, 'task_count': 4080, 'cell_count': 701280,
            'common_start': str(plan.common_dates[0].date()), 'common_end': str(plan.common_dates[-1].date()),
            'source_hashes': source_hashes(), 'python': platform.python_version(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'numba': numba.__version__,
            'platform': platform.platform(), 'pyvinecopulib': pv.__version__, 'workers': args.workers,
            'native_binary_sha256': hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            'state_workers': state_workers, 'vine_threads': vine_threads,
            'baseline_reference_manifest': json.loads((REFERENCE/'results.json').read_text())['manifest_sha256'],
            'scoring': 'unchanged empirical_crps_by_horizon and CellAccumulator; terminal log returns; n squared pairwise denominator; equal portfolio-horizon cells',
            'snapshot': verify_canonical_snapshot(),
            'tasks': [asdict(task_identity(t, models.MODEL_IDS[0], 240)) for t in plan.tasks]}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text()) == manifest, 'resume manifest mismatch'
    else:
        atomic(out/'manifest.json', manifest)
    if args.reuse:
        previous = json.loads((args.reuse/'manifest.json').read_text())
        old_digest = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
        old_ids = [c['model_id'] for c in previous['models']]
        for candidate in candidates:
            assert asdict(candidate) == previous['models'][old_ids.index(candidate.model_id)]
        for key in ('simulations', 'portfolio_count', 'task_count', 'cell_count',
                    'common_start', 'common_end', 'scoring', 'snapshot', 'tasks', 'native_binary_sha256'):
            assert manifest[key] == previous[key], f'reuse contract differs: {key}'
        for path, value in previous['source_hashes'].items():
            if path.startswith('tools/mcmc_runtime/') or path == 'tools/rough_jump_vine/models.py':
                assert manifest['source_hashes'][path] == value, f'reuse numerical source differs: {path}'
        for index in indices:
            target = out/f'task-{index:04d}.npz'
            source = args.reuse/f'task-{index:04d}.npz'
            metadata = source.with_suffix('.json')
            if target.exists() or not source.exists() or not metadata.exists():
                continue
            old_record = json.loads(metadata.read_text())
            arrays = {'manifest': digest}
            with np.load(source) as saved:
                assert str(saved['manifest']) == old_digest
                for j, mid in enumerate(model_ids):
                    k = old_ids.index(mid)
                    losses = saved[f'losses_{k}']
                    assert losses.size == plan.tasks[index].horizon_days and np.isfinite(losses).all()
                    arrays[f'losses_{j}'] = losses
                    arrays[f'seconds_{j}'] = saved[f'seconds_{k}']
            temporary = target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(target)
            atomic(target.with_suffix('.json'), {'task': index,
                'baseline_byte_exact': old_record['baseline_byte_exact'],
                'reused_from_manifest': old_digest,
                'timings': {mid: old_record['timings'][mid] for mid in model_ids},
                'seconds': {mid: old_record['seconds'][mid] for mid in model_ids}})
    completed = set()
    for i in indices:
        path = out/f'task-{i:04d}.npz'
        if path.exists() and (out/f'task-{i:04d}.json').exists():
            with np.load(path) as saved:
                assert str(saved['manifest']) == digest
            completed.add(i)
    started = time.perf_counter()
    milestones = iter(sorted({max(1, (len(indices)*quarter+3)//4) for quarter in range(1, 5)}))
    next_milestone = next(milestones, None)
    while next_milestone is not None and len(completed) >= next_milestone:
        next_milestone = next(milestones, None)
    atomic(out/'progress.json', {'completed': len(completed), 'required': len(indices),
            'reused': len(completed), 'models_per_task': len(model_ids), 'elapsed_seconds': 0})
    print(json.dumps({'started': True, 'models': model_ids, 'reused_tasks': len(completed),
            'remaining_tasks': len(indices)-len(completed), 'workers': args.workers}), flush=True)
    todo = iter(i for i in indices if i not in completed)
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(out.parent/'cache', REFERENCE, vine_threads, state_workers, model_ids)) as pool:
        futures = {}
        def submit():
            try:
                index = next(todo)
            except StopIteration:
                return
            futures[pool.submit(evaluate, index, plan.tasks[index])] = index
        for _ in range(args.workers):
            submit()
        while futures:
            ready, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in ready:
                index, records = future.result()
                assert futures.pop(future) == index
                arrays = {'manifest': digest}
                for j, (mid, losses, seconds, timings) in enumerate(records):
                    assert mid == model_ids[j]
                    arrays[f'losses_{j}'] = losses
                    arrays[f'seconds_{j}'] = seconds
                path = out/f'task-{index:04d}.npz'
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, **arrays)
                temp.replace(path)
                atomic(out/f'task-{index:04d}.json', {'task': index, 'baseline_byte_exact': True,
                    'timings': {mid: values for mid, _, _, values in records},
                    'seconds': {mid: seconds for mid, _, seconds, _ in records}})
                completed.add(index)
                submit()
            progress = {'completed': len(completed), 'required': len(indices),
                            'elapsed_seconds': time.perf_counter()-started,
                            'models_per_task': len(model_ids), 'pending': len(futures)}
            atomic(out/'progress.json', progress)
            if next_milestone is not None and len(completed) >= next_milestone:
                print(json.dumps(progress), flush=True)
                next_milestone = next(milestones, None)
    assert source_hashes() == manifest['source_hashes'], 'source changed during run'
    if args.pilot_indices:
        print('PILOT COMPLETED; no full-panel score claim', flush=True)
    else:
        score_completed(plan, out, digest, time.perf_counter()-started, model_ids)
        print((out/'results.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
