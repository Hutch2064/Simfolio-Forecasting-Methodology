"""Checkpointed canonical80/single52 execution; original model adapters unchanged."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/mcmc_runtime"), str(Path(__file__).parent)]
from protocol import block_cells, single_asset_tasks
from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.runner import evaluate_origin_task, task_identity

MODELS = None
CANDIDATE = None


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize(adapter, model_id):
    global MODELS, CANDIDATE
    source = ROOT / adapter
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("expanded_selected_adapter", source)
    MODELS = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = MODELS
    spec.loader.exec_module(MODELS)
    # No persistent fit cache: bounded model LRU caches suffice, and every
    # origin is fit from its own training prefix. Numerical compiler caches
    # use the campaign's explicitly owned NUMBA_CACHE_DIR/TMPDIR.
    MODELS.initialize(None, 1, 1)
    candidates = [c for c in MODELS.CANDIDATES if not model_id or c.model_id == model_id]
    if len(candidates) != 1:
        raise ValueError("select exactly one candidate per process")
    CANDIDATE = candidates[0]


def evaluate(index, task):
    MODELS.clear_path_cache()
    MODELS.TIMINGS.clear()
    MODELS.FIT_DIAGNOSTICS.clear()
    started = time.perf_counter()
    losses = evaluate_origin_task(CANDIDATE, task, simulations=240)
    record = {"index": index, "portfolio": task.portfolio_id,
              "origin": task.origin_date, "horizon": task.horizon_days,
              "seconds": time.perf_counter() - started,
              "timings": dict(MODELS.TIMINGS),
              "diagnostics": list(MODELS.FIT_DIAGNOSTICS.values())}
    MODELS.clear_path_cache()
    return index, losses, record


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (ROOT / "src", ROOT / "tools") for p in sorted(folder.rglob("*"))
            if p.suffix in (".py", ".cpp")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--block", choices=("single52", "original80"), default="single52")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--indices", help="Smoke only; cannot produce a full-block score")
    args = parser.parse_args()
    tasks = (single_asset_tasks(args.data) if args.block == "single52"
             else build_experiment_plan(args.data).tasks)
    indices = list(range(len(tasks))) if args.indices is None else [int(i) for i in args.indices.split(",")]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(tasks) for i in indices):
        raise ValueError("invalid or duplicated task indices")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = hashes()
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "numba", "llvmlite", "pandas", "pyvinecopulib", "pybind11")}
    manifest = {"protocol": "canonical80-plus-unique52-equal-blocks-v1", "block": args.block,
        "adapter": args.adapter, "requested_model_id": args.model_id,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("SIMFOLIO_")},
        "python": platform.python_version(), "packages": versions,
        "workers": args.workers, "state_workers": 1, "simulations": 240,
        "source_hashes": source, "indices": indices,
        "source_revision": "6bfeb13157c80f5debfdd21340c69884e6a6cf99",
        "data_manifest_sha256": hashlib.sha256((args.data / "canonical_data_manifest.json").read_bytes()).hexdigest(),
        "task_identities": [asdict(task_identity(t, args.model_id or args.adapter, 240)) for t in tasks]}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (out / "manifest.json").exists():
        if json.loads((out / "manifest.json").read_text()) != manifest:
            raise ValueError("resume manifest mismatch")
    else:
        atomic(out / "manifest.json", manifest)
    completed = set()
    for i in indices:
        checkpoint = out / f"task-{i:04d}.npz"
        if checkpoint.exists() and checkpoint.with_suffix(".json").exists():
            with np.load(checkpoint) as saved:
                if str(saved["manifest"]) != digest:
                    raise ValueError("checkpoint manifest mismatch")
            completed.add(i)
    started = time.perf_counter()
    todo = iter(i for i in indices if i not in completed)
    portfolio_indices = {}
    for i, task in enumerate(tasks):
        portfolio_indices.setdefault(task.portfolio_id, set()).add(i)
    print(json.dumps({"started": True, "tasks": len(indices), "reused": len(completed),
                      "block": args.block, "workers": args.workers}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize,
            initargs=(args.adapter, args.model_id)) as pool:
        pending = {}
        def submit():
            index = next(todo, None)
            if index is not None:
                pending[pool.submit(evaluate, index, tasks[index])] = index
        for _ in range(args.workers):
            submit()
        last_portfolios = set()
        while pending:
            ready, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in ready:
                expected_index = pending.pop(future)
                try:
                    index, losses, record = future.result()
                except Exception as error:
                    atomic(out / f"failure-{expected_index:04d}.json", {
                        "index": expected_index, "error": type(error).__name__, "message": str(error)})
                    raise
                if index != expected_index:
                    raise ValueError("task completion identity mismatch")
                path = out / f"task-{index:04d}.npz"
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, manifest=digest, losses=losses)
                temporary.replace(path)
                atomic(path.with_suffix(".json"), record)
                completed.add(index)
                submit()
            portfolios = {name for name, group in portfolio_indices.items()
                          if group.issubset(completed)}
            progress = {"completed_tasks": len(completed), "required_tasks": len(indices),
                "complete_portfolios": len(portfolios), "elapsed_seconds": time.perf_counter() - started}
            atomic(out / "progress.json", progress)
            if portfolios != last_portfolios:
                print(json.dumps(progress), flush=True)
                last_portfolios = portfolios
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in source.items()):
        raise ValueError("source changed during run")
    if args.indices is not None:
        print("SMOKE COMPLETE; no full-block score", flush=True)
        return
    losses = [np.load(out / f"task-{i:04d}.npz")["losses"] for i in range(len(tasks))]
    names, cells, counts = block_cells(tasks, losses)
    required = (52, 2652, 455832) if args.block == "single52" else (80, 4080, 701280)
    if (len(names), len(tasks), cells.size) != required:
        raise ValueError("block denominator mismatch")
    np.savez_compressed(out / "cells.npz", portfolio=names, losses=cells, counts=counts)
    seconds = sum(json.loads((out / f"task-{i:04d}.json").read_text())["seconds"]
                  for i in range(len(tasks)))
    result = {"block": args.block, "all_denominator_gates_passed": True,
              "exact_empirical_crps": float(cells.mean()), "portfolios": len(names),
              "tasks": len(tasks), "cells": cells.size, "manifest_sha256": digest,
              "elapsed_seconds": time.perf_counter() - started, "summed_worker_seconds": seconds,
              "portfolio_scores": dict(zip(names, cells.mean(axis=1).tolist())),
              "cells_sha256": hashlib.sha256((out / "cells.npz").read_bytes()).hexdigest()}
    atomic(out / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
