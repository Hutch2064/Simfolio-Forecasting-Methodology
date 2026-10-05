"""Run the new suite through the existing, unchanged canonical panel runner."""
from pathlib import Path
import hashlib
import importlib.util
import argparse
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'tools/mcmc_runtime')]
import models

spec = importlib.util.spec_from_file_location('canonical_rough_runner', ROOT / 'tools/rough_jump_vine/run_panel.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original_hashes = runner.source_hashes
original_evaluate = runner.evaluate
original_score = runner.score_completed

def source_hashes():
    hashes = original_hashes()
    for path in sorted(HERE.glob('*.py')):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
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
    runner.atomic(out / 'results.json', results)


runner.evaluate = evaluate
runner.score_completed = score_completed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    known, _ = parser.parse_known_args()
    os.environ['BAYESIAN_PANEL_OUTPUT'] = str(Path(known.output).resolve())
    runner.main()
