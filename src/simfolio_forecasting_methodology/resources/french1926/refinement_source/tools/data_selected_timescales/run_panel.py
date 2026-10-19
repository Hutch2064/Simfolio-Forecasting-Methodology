"""Timescale-only adapter to the unchanged canonical M256 panel runner."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import models

spec = importlib.util.spec_from_file_location('timescale_canonical_adapter',
    models.ROOT/'tools/student_corrected_dcc_innovation_copula/run_panel.py')
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
runner = adapter.runner
original_hashes = runner.source_hashes


def source_hashes():
    hashes = original_hashes()
    hashes.update(models.source_hashes())
    return hashes


runner.source_hashes = source_hashes
original_score_completed = runner.score_completed


def score_completed(plan, out, digest, elapsed, model_ids):
    original_score_completed(plan, out, digest, elapsed, model_ids)
    result = json.loads((out/'results.json').read_text())
    baseline = json.loads((models.ROOT/'docs/results/rough-bayesian/m256-50-asset-speed-optimization.json').read_text())['canonical_panel']
    baseline_score = next(iter(baseline['scores'].values()))['exact_empirical_crps']
    for score in result['scores'].values():
        score.pop('relative_improvement_vs_production')
        score['relative_improvement_vs_saved_M256'] = 1.-score['exact_empirical_crps']/baseline_score
    result['saved_M256_reference'] = baseline
    result['baseline_reexecuted'] = False
    result['runtime_scope'] = 'same frozen panel and worker settings; separately scheduled local runs, not controlled wall-time proof'
    runner.atomic(out/'results.json', result)


runner.score_completed = score_completed
original_evaluate = runner.evaluate


def evaluate(index, task):
    result = original_evaluate(index, task)
    reference = os.environ.get('SIMFOLIO_TIMESCALE_PARITY_REFERENCE')
    if reference:
        with np.load(Path(reference)/f'task-{index:04d}.npz') as saved:
            assert len(result[1]) == 1
            assert result[1][0][1].tobytes() == saved['losses_0'].tobytes(), index
    return result


runner.evaluate = evaluate


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    args, _ = parser.parse_known_args()
    os.environ['ROUGH_INFERENCE_OUTPUT'] = str(Path(args.output).resolve())
    runner.main()
