"""Canonical origin smoke; invokes only the selected timescale alternative."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import models

from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.runner import evaluate_origin_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', default='0,24,48')
    args = parser.parse_args()
    source_hashes = models.source_hashes()
    plan = build_experiment_plan(args.data)
    assert len(plan.tasks) == 4080 and plan.cell_capacity == 701280
    models.initialize(None, 1, 1)
    records = []
    for i in map(int, args.indices.split(',')):
        task = plan.tasks[i]
        expected = None
        for cache_state in ('fresh', 'repeat'):
            models.clear_path_cache()
            models.TIMINGS.clear()
            models.FIT_DIAGNOSTICS.clear()
            start = time.perf_counter()
            losses = evaluate_origin_task(models.CANDIDATES[0], task, simulations=240)
            seconds = time.perf_counter() - start
            if expected is not None:
                assert losses.tobytes() == expected.tobytes(), 'repeat loss-vector parity'
            expected = losses.copy()
            record = {'task': i, 'portfolio': task.portfolio_id,
                'origin': str(task.origin_date), 'horizon': task.horizon_days,
                'cache_state': cache_state, 'seconds': seconds,
                'origin_crps': float(losses.mean()), 'timings': dict(models.TIMINGS),
                'loss_sha256': hashlib.sha256(losses.tobytes()).hexdigest()}
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert models.source_hashes() == source_hashes, 'source changed during smoke'
    args.output.write_text(json.dumps({'model': models.MODEL_ID,
        'scope': 'smoke_not_full_panel', 'records': records,
        'python': platform.python_version(), 'source_hashes': source_hashes}, indent=2)+'\n')


if __name__ == '__main__':
    main()
