"""Run the new suite through the existing, unchanged canonical panel runner."""
from pathlib import Path
import hashlib
import importlib.util
import argparse
import json
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'tools/mcmc_runtime')]
import models

spec = importlib.util.spec_from_file_location('canonical_rough_runner', ROOT / 'tools/rough_jump_vine/run_panel.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
original_hashes = runner.source_hashes
original_evaluate = runner.evaluate
original_score = runner.score_completed

def source_hashes():
    hashes = original_hashes()
    for path in sorted(list(HERE.glob('*.py')) + list(HERE.glob('*.cpp'))):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    import overlay
    backend = overlay.native_filter()
    if backend is not None:
        hashes['rough_overlay_native_binary'] = hashlib.sha256(Path(backend[0].__file__).read_bytes()).hexdigest()
    best = os.environ.get('BAYESIAN_BEST_REFERENCE')
    if best:
        for name in ('manifest.json', 'paired-cell-losses.npz'):
            hashes[f'best_reference/{name}'] = hashlib.sha256((Path(best)/name).read_bytes()).hexdigest()
    return hashes

runner.models = models
runner.CANDIDATES = models.CANDIDATES
runner.source_hashes = source_hashes


def evaluate(index, task):
    models.FIT_DIAGNOSTICS.clear()
    result = original_evaluate(index, task)
    output = Path(os.environ['BAYESIAN_PANEL_OUTPUT'])
    runner.atomic(output / f'task-{index:04d}-diagnostics.json', list(models.FIT_DIAGNOSTICS.values()))
    return result


def score_completed(plan, out, digest, elapsed, model_ids):
    original_score(plan, out, digest, elapsed, model_ids)
    unique = {}
    for index in range(len(plan.tasks)):
        for record in json.loads((out / f'task-{index:04d}-diagnostics.json').read_text()):
            key = (record['model_id'], record['data_sha256'])
            if key in unique:
                assert unique[key] == record, 'shared asset fit diagnostics drifted'
            unique[key] = record
    results = json.loads((out / 'results.json').read_text())
    results['inference_diagnostics'] = {}
    for model_id in model_ids:
        rows = [r for (mid, _), r in unique.items() if mid == model_id]
        if not rows:
            continue
        results['inference_diagnostics'][model_id] = {
            'unique_asset_fits': len(rows), 'flagged_asset_fits': sum(not r['convergence_flag'] for r in rows),
            'maximum_rank_split_rhat': max(d['rank_split_rhat'] for r in rows for d in r['diagnostics']),
            'minimum_bulk_ess': min(d['bulk_ess'] for r in rows for d in r['diagnostics']),
            'minimum_ess_per_second': min(r['ess_per_second'] for r in rows)}
    best = os.environ.get('BAYESIAN_BEST_REFERENCE')
    if best:
        reference = Path(best)
        saved = json.loads((reference/'results.json').read_text())
        manifest = json.loads((reference/'manifest.json').read_text())
        assert saved['all_denominator_gates_passed'] and saved['portfolios'] == 80
        assert manifest['simulations'] == 240 and manifest['task_count'] == 4080
        current_manifest = json.loads((out/'manifest.json').read_text())
        for key in ('snapshot','common_start','common_end','portfolio_count','task_count','cell_count','simulations','tasks'):
            assert manifest[key] == current_manifest[key], f'best reference panel mismatch: {key}'
        assert hashlib.sha256((reference/'paired-cell-losses.npz').read_bytes()).hexdigest() == saved['paired_cells_sha256']
        with np.load(out/'paired-cell-losses.npz') as current, np.load(reference/'paired-cell-losses.npz') as previous:
            assert len(current['portfolio']) == len(previous['portfolio']) == 701280
            assert np.array_equal(current['portfolio'], previous['portfolio'])
            assert np.array_equal(current['horizon'], previous['horizon'])
        best_id = manifest['models'][0]['model_id']
        assert best_id == 'asset_rough_volterra_sv_dynamic_lift_bayesian'
        baseline = saved['scores'][best_id]['exact_empirical_crps']
        results['best_completed_reference'] = {'model_id':best_id, 'exact_empirical_crps':baseline,
            'manifest_sha256':saved['manifest_sha256'], 'paired_cells_sha256':saved['paired_cells_sha256'],
            'portfolio_horizon_keys_match':True}
        for record in results['scores'].values():
            record['relative_improvement_vs_best_completed'] = 1-record['exact_empirical_crps']/baseline
    runner.atomic(out / 'results.json', results)


runner.evaluate = evaluate
runner.score_completed = score_completed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-ids', nargs='+', choices=models.MODEL_IDS)
    parser.add_argument('--best-reference', type=Path)
    known, remaining = parser.parse_known_args()
    if known.model_ids:
        models.CANDIDATES = tuple(c for c in models.CANDIDATES if c.model_id in known.model_ids)
        runner.CANDIDATES = models.CANDIDATES
    if known.best_reference:
        os.environ['BAYESIAN_BEST_REFERENCE'] = str(known.best_reference.resolve())
    sys.argv = [sys.argv[0], '--output', known.output, *remaining]
    os.environ['BAYESIAN_PANEL_OUTPUT'] = str(Path(known.output).resolve())
    runner.main()
