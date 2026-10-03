"""Cold-fit matched screening with the unchanged canonical OOS evaluator."""
import argparse
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mcmc_runtime'))
import inference
import numpy as np
import optimized_mcmc as opt
import parameter_mcmc_tuned as pm
from runtime_setup import DATA, initialize

from simfolio_forecasting_methodology.evaluation import CellAccumulator
from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
    FilteredInnovationFixedMeanMomentSV,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.runner import (
    ForecastContext,
    evaluate_origin_task,
    task_identity,
)

ORIGINS = ('rolling_09', 'rolling_29', 'train_first_quarter_test_remaining',
           'train_first_half_test_remaining', 'train_first_three_quarters_test_final_quarter')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'screen25', 'screen400', 'full'), default='screen25')
    parser.add_argument('--methods', default='laplace_is,full_rank_vi,vi_importance,hmc')
    args = parser.parse_args()
    initialize(True); opt.install(False)
    original = pm.parameter_fit
    methods = args.methods.split(',') if args.methods else []
    callbacks = {'mcmc': original}
    for method in methods:
        @lru_cache(maxsize=4096)
        def callback(data, selected=method):
            return inference.parameter_fit(data, selected)
        callbacks[method] = callback
    models = {'frontier': FilteredInnovationFixedMeanMomentSV(), 'mcmc': opt.OptimizedMCMC()}
    models.update({m: opt.OptimizedMCMC(model_id='experimental_sv_inference_' + m) for m in methods})
    plan = build_experiment_plan(DATA)
    indexed = [(i, t) for i, t in enumerate(plan.tasks) if
               (args.stage == 'full' or t.origin_label in ORIGINS) and
               (args.stage in ('full', 'screen400') or int(t.portfolio_id.rsplit('_', 1)[1]) in (1, 2, 3, 4, 7))]
    if args.stage == 'pilot':
        indexed = indexed[:1]
    def clear():
        inference.parameter_fit.cache_clear(); original.cache_clear()
        for callback in callbacks.values(): callback.cache_clear()
        opt.nodes.cache_clear(); opt.power.cache_clear()
        for cell in bd.fit_bdes_fastmap.__closure__ or ():
            if hasattr(cell.cell_contents, 'cache_clear'): cell.cell_contents.cache_clear()
    warm = indexed[0][1]
    for arm, model in models.items():
        pm.parameter_fit = callbacks.get(arm, original)
        context = ForecastContext(model.model_id, warm.portfolio_id, warm.origin_label,
                                  2, 240, warm.seed, warm.future_dates[:2], warm.origin_date)
        model.simulate_daily_log_returns(warm.training, context)
        clear()
        print(json.dumps({'warmup': arm}), flush=True)
    inference.RECORDS.clear(); inference.TRACES.clear()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in (HERE, HERE.parent / 'mcmc_runtime', HERE.parent.parent / 'src')
               for p in folder.rglob('*.py')}
    manifest = {'scope': 'full_canonical' if args.stage == 'full' else 'screen_not_full_canonical',
                'stage': args.stage, 'simulations': 240, 'model_ids': {k: v.model_id for k, v in models.items()},
                'tasks': [[task_identity(t, model.model_id, 240).__dict__ for model in models.values()]
                          for _, t in indexed], 'sources': sources, 'timing': 'local warm compiler / cold model fits',
                'diagnostics_dependency': 'arviz==0.22.0', 'blas_threads': 1}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    acc = {arm: CellAccumulator() for arm in models}; records = []; counts = {}
    for index, (canonical_index, task) in enumerate(indexed):
        for h in range(1, task.horizon_days+1): counts[(task.portfolio_id, h)] = counts.get((task.portfolio_id, h), 0)+1
        order = list(models) if index % 2 == 0 else list(models)[::-1]
        for arm in order:
            clear(); pm.parameter_fit = callbacks.get(arm, original)
            before = len(inference.RECORDS)
            start = time.perf_counter(); cpu = time.process_time()
            losses = evaluate_origin_task(models[arm], task, simulations=240)
            wall = time.perf_counter()-start; cpu = time.process_time()-cpu
            assert len(losses) == task.horizon_days and np.all(np.isfinite(losses))
            acc[arm].add_vector(task.portfolio_id, losses)
            path = args.output / f'task-{canonical_index:04d}-{arm}.npz'
            np.savez_compressed(path, losses=losses)
            record = {'canonical_task': canonical_index, 'arm': arm, 'seconds': wall, 'cpu_seconds': cpu,
                      'portfolio': task.portfolio_id, 'origin': task.origin_label,
                      'horizon': task.horizon_days, 'losses_sha256': hashlib.sha256(losses.tobytes()).hexdigest(),
                      'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'diagnostics': inference.RECORDS[before:]}
            records.append(record)
            print(json.dumps({k: record[k] for k in ('canonical_task','arm','seconds','cpu_seconds')}), flush=True)
        (args.output/'progress.json').write_text(json.dumps({'completed': index+1, 'required': len(indexed)}))
        (args.output/'records.json').write_text(json.dumps(records, indent=2)+'\n')
    keys = sorted(counts)
    arrays = {'portfolio': np.asarray([k[0] for k in keys]), 'horizon': np.asarray([k[1] for k in keys]),
              'origin_count': np.asarray([counts[k] for k in keys])}
    scores = {}
    for arm in models:
        score = acc[arm].fixed_denominator_score(expected_cells=len(keys), expected_tasks=len(indexed),
                    completed_tasks=len(indexed), expected_origin_counts=counts)
        means = acc[arm].cell_means()
        arrays[arm] = np.array([means[k] for k in keys])
        scores[arm] = {'crps': score, 'seconds': sum(r['seconds'] for r in records if r['arm']==arm),
                       'cpu_seconds': sum(r['cpu_seconds'] for r in records if r['arm']==arm),
                       'cells': len(keys), 'tasks': len(indexed)}
    np.savez_compressed(args.output/'paired-cells.npz', **arrays)
    (args.output/'results.json').write_text(json.dumps({'scope': manifest['scope'], 'scores': scores,
            'all_denominator_gates_passed': True}, indent=2)+'\n')
    print(json.dumps(scores, indent=2), flush=True)

if __name__ == '__main__':
    main()
