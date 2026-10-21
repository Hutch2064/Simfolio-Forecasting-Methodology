"""Profile the actual batched selector without changing its saved forecasts."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from unittest.mock import patch
import datetime, fcntl, hashlib, json, platform, statistics, subprocess, sys, time
import numpy as np
O = Path(__file__).resolve().parent; V = O / 'asset_level_v6'
D = V / 'candidates/asset_selector_inner_precision_batched/study1000'
sys.path.insert(0, str(V)); sys.path.insert(0, str(O))
import common, sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_selector_inner_precision_batched import candidate as c
from candidates.asset_selector_universe import bank
from candidates.asset_selector_batched_paths import paths as batching
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from verify_selector_inner_precision_batched72 import standalone_cost
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    slots = (O / 'asset_v6_forecast_slots.lock').open('a'); fcntl.flock(slots, fcntl.LOCK_EX)
    plan = read(D / 'plan.json'); summary = read(D / 'summary.json'); audit = read(D / 'verification_receipt.json')
    assert audit['verified'] and audit['summary_sha256'] == sha(D / 'summary.json')
    assert all(sha(O / p) == h for p, h in plan['source_hashes'].items())
    controls = {r['case']: r for r in summary['records'] if r['configuration'] == {'criterion': 'path', 'inner_paths': 30, 'seed_replication': 0}}
    lookup = {(r['case'], r['config_index']): r for r in summary['records']}
    paired = next(i for i, cfg in enumerate(plan['configs']) if cfg == {'criterion': 'terminal', 'inner_paths': 30, 'seed_replication': 0})
    ranked = sorted(controls, key=lambda i: (standalone_cost(controls[i], lookup[i, paired]), i))
    selected = sorted({ranked[len(ranked)//2], ranked[-1], max(controls, key=lambda i: (len(controls[i]['constituent_ids']), -i))})
    # Warm the qualified numerical signatures outside all measured requests.
    rng = np.random.default_rng(638); warm = rng.normal(0, .01, (40, 3)); weights = np.array([.2, .3, .5])
    dates = np.datetime64('2000-01-01') + np.arange(40)
    begin = time.process_time(); c.prepare(warm, weights, 4, dates, inner_paths=10)
    warmup_cpu = time.process_time() - begin
    metadata = {r['series_id']: r for r in api.load_panel(V / 'inputs', verify_files=False)['series']}
    results = []
    for number in selected:
        case = plan['cases'][number]; origin = case['origin']; row = metadata[case['series_id']]
        stop = origin['position'] + 1; horizon = origin['max_horizon']
        x, w, ids = AssetInputs(V / 'inputs').training(row, stop)
        dates, values, origins = api.load_series(V / 'inputs', row, verify_files=False); assert origin in origins
        totals = {}; calls = {}; phase = 'prepare'
        def charge(key, start):
            totals[key] = totals.get(key, 0.) + time.process_time() - start
            calls[key] = calls.get(key, 0) + 1
        def timed(fn, key):
            def wrapper(*args, **kwargs):
                label = key(*args, **kwargs) if callable(key) else key
                begin = time.process_time()
                try:
                    return fn(*args, **kwargs)
                finally:
                    charge(label, begin)
            return wrapper
        def stream(generator, key):
            while True:
                begin = time.process_time()
                try:
                    item = next(generator)
                except StopIteration:
                    charge(key, begin); return
                charge(key, begin); yield item
        original_group, original_model = batching.group_chunks, bank.model
        def group(history, weights, horizon, paths, seed, indices, prepared, **kwargs):
            spec = bank.BANK[indices[0]]
            return stream(original_group(history, weights, horizon, paths, seed, indices, prepared, **kwargs),
                          f'inner_simulation/{spec["family"]}/{spec["innovations"]}')
        def profiled_model(index, prepared):
            original = original_model(index, prepared); spec = bank.BANK[index]
            label = spec['family'] + '/' + spec['innovations'] if spec['kind'] == 'filtered' else ('parent' if spec['kind'] == 'parent' else 'raw_' + ('IID' if spec['sampling'] == 'IID' else 'replacement' if spec['replacement'] else 'history_cycles'))
            class Proxy:
                def terminal_chunks(self, *args, **kwargs):
                    return stream(original.terminal_chunks(*args, **kwargs), phase + '_simulation/' + label)
            return Proxy()
        def fit_label(history, family, *args, **kwargs):
            return ('full_refit/' if len(history) == len(x) else 'inner_fit/') + family
        def parent_label(history, *args, **kwargs):
            return ('full_refit/' if len(history) == len(x) else 'inner_fit/') + 'nested_parent'
        begin = time.process_time(); wall = time.perf_counter()
        with patch.object(batching, 'group_chunks', group), patch.object(batching, 'model', profiled_model), patch.object(c, 'model', profiled_model), \
             patch.object(bank, 'pure_prepare', timed(bank.pure_prepare, fit_label)), \
             patch.object(bank, 'parent_prepare', timed(bank.parent_prepare, parent_label)), \
             patch.object(bank.fits, 'base_fit', timed(bank.fits.base_fit, 'base_fit')), \
             patch.object(c, 'accumulate', timed(c.accumulate, 'path_coefficients')), \
             patch.object(c.energy, 'coefficients', timed(c.energy.coefficients, 'endpoint_coefficients')), \
             patch.object(c.energy, 'fit', timed(c.energy.fit, 'weight_optimization')):
            bundle = c.prepare(x, w, horizon, dates[:stop], inner_paths=30, seed_replication=0)
            prepare_cpu = time.process_time() - begin; phase = 'final'
            model = c.PathCriterionPool({'criterion': 'path', 'inner_paths': 30}, dates[:stop], bundle)
            actual = np.cumsum(values[stop:stop+horizon]); loss = np.empty(horizon); cursor = 0
            seed = api.original.coherent_daily_seed(origin['origin_date'], horizon, 1000)
            final_begin = time.process_time()
            for start, paths in model.terminal_chunks(x, w, horizon, 1000, seed):
                assert start == cursor; cursor += paths.shape[1]
                loss[start:cursor] = empirical_crps_by_horizon(paths, actual[start:cursor])
            final_cpu = time.process_time() - final_begin
        assert cursor == horizon and np.isfinite(loss).all()
        assert hashlib.sha256(loss.tobytes()).hexdigest() == controls[number]['loss_sha256']
        assert model.last_selection['path_counts'] == controls[number]['selection']['path_counts']
        positive = set.union(*(set(np.flatnonzero(b > 0)) for b in bundle['weights_by_mode'].values()))
        emitted = set.union(*(set(np.flatnonzero(c.energy.even_allocation(b, 1000))) for b in bundle['weights_by_mode'].values()))
        results.append({'case': number, 'series_id': row['series_id'], 'assets': len(ids), 'history_rows': len(x),
                        'horizon': horizon, 'inner_paths': 30, 'final_paths': 1000, 'prepare_cpu_seconds': prepare_cpu,
                        'final_simulation_and_score_cpu_seconds': final_cpu, 'total_profiled_cpu_seconds': time.process_time()-begin,
                        'observed_wall_seconds': time.perf_counter()-wall, 'stage_cpu_seconds': totals, 'stage_calls': calls,
                        'positive_weight_components': len(positive), 'components_with_any_final_path_in_either_criterion': len(emitted),
                        'positive_but_no_allocated_path_components': [int(i) for i in sorted(positive-emitted)],
                        'whole_loss_vector_byte_exact_to_saved_control': True, 'loss_sha256': controls[number]['loss_sha256']})
        api.original._atomic_json(O / 'selector_batched_family_runtime_profile.partial.json', {'records': results})
    assert all(sha(O / p) == h for p, h in plan['source_hashes'].items())
    out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'profiled_requests': len(results),
           'selection_rule': 'Median attributed CPU, maximum attributed CPU, and widest constituent count among the72completed30-inner-path seed-zero controls; duplicates collapsed; no loss-based selection',
           'warmup_cpu_seconds_excluded': warmup_cpu, 'process': {'python': sys.version, 'platform': platform.platform()},
           'source_plan_sha256': sha(D/'plan.json'), 'source_summary_sha256': sha(D/'summary.json'),
           'source_closure_unchanged': True, 'instrumented_entire_forecasts_byte_exact': True,
           'scope': 'Serial process-CPU profile on frozen research requests, including both objective optimizations and their union refits; not a whole-panel average, isolated causal speed comparison or deployed latency',
           'new_model_or_full_panel_catalogue_increment': 0, 'diagnostics_have_zero_score_or_selection_weight': True,
           'records': results, 'profiler_sha256': sha(Path(__file__))}
    api.original._atomic_json(O / 'selector_batched_family_runtime_profile.json', out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
