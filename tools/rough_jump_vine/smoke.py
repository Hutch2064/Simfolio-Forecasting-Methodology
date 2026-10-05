"""Execute all five models on representative canonical tasks for score parity."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--module-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--indices", default="0,47,48")
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(args.module_dir), str(root / "tools/mcmc_runtime")]
import models
import pyvinecopulib as pv
import pyvinecopulib.pyvinecopulib_ext as native

from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.runner import evaluate_origin_task

plan = build_experiment_plan(args.data)
models.initialize(args.cache)
args.output.mkdir(parents=True, exist_ok=True)
receipt = {
    "model_source_sha256": hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest(),
    "native_binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
    "native_version": pv.__version__,
    "cases": [],
}
# Compile small numerical kernels before measuring the representative cases.
phi, w, q = models.rough_parameters([0.1, np.log(1 / 63), np.log(0.7)])
initial = np.zeros((2, 8))
normals = np.zeros((2, 4, 8))
chol = np.linalg.cholesky(q + np.eye(8) * 1e-14)
models.rough_paths(phi, w, chol, initial, normals)
models.rough_paths_reference(phi, w, chol, initial, normals)
models.poisson_inverse(np.full((2, 4), 0.5), np.ones(4) * 0.01)
models.hmm_forward_backward(np.zeros((4, 3)), np.ones((3, 3)) / 3, np.ones(3) / 3)
for index in map(int, args.indices.split(",")):
    if hasattr(models, "clear_path_cache"):
        models.clear_path_cache()
    task = plan.tasks[index]
    record = {
        "index": index,
        "training_rows": task.training.asset_log_returns.shape[0],
        "horizon": task.horizon_days,
        "simulations": 240,
        "models": {},
    }
    arrays = {}
    for j, c in enumerate(models.CANDIDATES):
        models.TIMINGS.clear()
        start = time.perf_counter()
        cpu = time.process_time()
        losses = evaluate_origin_task(c, task, simulations=240)
        cpu = time.process_time() - cpu
        elapsed = time.perf_counter() - start
        arrays[f"losses_{j}"] = losses
        record["models"][c.model_id] = {
            "seconds": elapsed,
            "cpu_seconds": cpu,
            "smoke_mean_crps": float(losses.mean()),
            "losses_sha256": hashlib.sha256(losses.tobytes()).hexdigest(),
            "timings": dict(models.TIMINGS),
        }
        print(
            index, c.model_id, elapsed, record["models"][c.model_id]["smoke_mean_crps"], flush=True
        )
    np.savez_compressed(args.output / f"task-{index:04d}.npz", **arrays)
    receipt["cases"].append(record)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("All five models evaluated for all smoke cases.", flush=True)
