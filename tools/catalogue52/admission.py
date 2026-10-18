"""Isolated exact-origin execution before admitting a catalogue implementation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/mcmc_runtime"),
                str(ROOT / "tools/expanded_panel")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    from adapters import model_specs
    spec = next(row for row in model_specs() if row["canonical_rank"] == args.rank)
    os.environ.update(spec["environment"])
    started = time.perf_counter()
    if spec["adapter"] is None:
        from simfolio_forecasting_methodology.models.registry import build_model
        model, module = build_model(spec["model_id"]), None
    else:
        source = ROOT / spec["adapter"]
        sys.path.insert(0, str(source.parent))
        loader = importlib.util.spec_from_file_location("catalogue52_admission_adapter", source)
        module = importlib.util.module_from_spec(loader)
        sys.modules[loader.name] = module
        loader.loader.exec_module(module)
        module.initialize(None, 1, 1)
        selected = [candidate for candidate in module.CANDIDATES
                    if candidate.model_id == spec["model_id"]]
        if len(selected) != 1:
            raise ValueError("exact catalogue identity missing or ambiguous in retained adapter")
        model = selected[0]
    initialization_seconds = time.perf_counter() - started
    from dataclasses import asdict

    import numpy as np
    from protocol import single_asset_tasks

    from simfolio_forecasting_methodology.runner import evaluate_origin_task, task_identity
    task = single_asset_tasks(args.data)[args.index]
    if module is not None and callable(getattr(module, "clear_path_cache", None)):
        module.clear_path_cache()
    started = time.perf_counter()
    losses = evaluate_origin_task(model, task, simulations=240)
    seconds = time.perf_counter() - started
    result = {"canonical_rank": args.rank, "model_id": model.model_id,
              "adapter": spec["adapter"], "environment": spec["environment"],
              "task": asdict(task_identity(task, model.model_id, 240)),
              "index": args.index, "simulations": 240,
              "initialization_seconds": initialization_seconds, "seconds": seconds,
              "loss_count": len(losses), "mean_origin_loss": float(losses.mean()),
              "losses_sha256": hashlib.sha256(losses.tobytes()).hexdigest(),
              "full_panel_score": False}
    if args.repeat:
        if module is not None and callable(getattr(module, "clear_path_cache", None)):
            module.clear_path_cache()
        repeated = evaluate_origin_task(model, task, simulations=240)
        result["cached_repeat_array_equal"] = bool(np.array_equal(losses, repeated))
        if not result["cached_repeat_array_equal"]:
            raise ValueError("same-origin cached repeat changed the output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
