"""Compare compiler-only execution to every completed original checkpoint."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import legacy_exact_kernels as fast
import numpy as np
from campaign import atomic, digest, read, verified_process
from worker import checkpoint_losses, shared_tasks

from simfolio_forecasting_methodology.models.registry import build_model
from simfolio_forecasting_methodology.runner import evaluate_origin_task


def verify(campaign, names):
    config = read(campaign / "campaign.json")
    evidence = []
    fast.prepare()
    for name in names:
        output = campaign / "models" / name
        state = read(output / "process.json")
        if verified_process(state):
            raise ValueError("reference must finish its current task and stop before auditing")
        spec = read(output / "registration.json")
        tasks, cache = shared_tasks(campaign, spec, config)
        manifest = read(output / "manifest.json")
        binding = digest(manifest)
        count = read(output / "progress.json")["completed_tasks"]
        model = fast.ExactKernelModel(build_model(spec["model_id"]))
        hashes = []
        for index in range(count):
            reference = checkpoint_losses(output, index, binding, tasks[index])
            if reference is None:
                raise ValueError("reference prefix contains a missing checkpoint")
            actual = evaluate_origin_task(model, tasks[index], simulations=240)
            if not np.array_equal(actual, reference):
                raise ValueError(f"compiler-only loss vector differs at {name} task {index}")
            hashes.append(hashlib.sha256(actual.tobytes()).hexdigest())
        evidence.append({"reference_execution": name, "model_id": model.model_id,
                         "reference_manifest_sha256": binding,
                         "reference_source_sha256": spec["source_sha256"],
                         "optimized_sources": fast.source_version(),
                         "complete_reference_prefix_tasks": count,
                         "all_loss_vectors_byte_equal": True, "loss_vector_sha256": hashes,
                         "cache_sha256": cache["sha256"], "simulations": 240})
    return {"scope": "every_completed_original_origin_before_compiler_only_supersession",
            "full_panel_score": False, "models": evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic(args.output, verify(args.campaign.resolve(), args.models))
