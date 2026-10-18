"""One independent model, exact frozen single52 tasks, atomic checkpoints.

Adapter files follow expanded_panel: initialize(None, 1, 1), CANDIDATES, and
optional clear_path_cache/TIMINGS/FIT_DIAGNOSTICS. With no adapter the verified
canonical registry factory is used. No inference settings or seeds are changed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import io
import multiprocessing
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

from campaign import (
    EXPECTED,
    PROTOCOL,
    atomic,
    digest,
    file_hash,
    lock,
    origin_worker_count,
    read,
    storage_ok,
    verify_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/mcmc_runtime"),
                str(ROOT / "tools/expanded_panel")]
CACHE_REFERENCE_MODEL = "catalogue52-task-cache-reference"
_ORIGIN_STATE = None


def parent_watchdog(parent_pid):
    """An owned group cannot be reused while this member remains alive."""
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, signal.SIG_IGN)
    group = os.getpgrp()
    if group != parent_pid or os.getppid() != parent_pid:
        raise RuntimeError("origin watchdog requires its leader's owned process group")
    print("origin-watchdog-ready", flush=True)
    while os.getppid() == parent_pid:
        time.sleep(0.2)
    if os.getpgrp() == group:
        os.killpg(group, signal.SIGKILL)


def model_identity(model):
    from simfolio_forecasting_methodology.runner import _numerical_environment_identity
    return {"model_id": model.model_id,
            "native_seed_contract": getattr(model, "seed_contract", "source-bound native model seed derivation"),
            "native_forecast_semantics": getattr(model, "forecast_output_semantics", "daily_log_return_paths"),
            "specification_fingerprint": getattr(model, "specification_fingerprint", None),
            "model_environment": {k: v for k, v in os.environ.items() if k.startswith("SIMFOLIO_")},
            "numerical_environment": _numerical_environment_identity()}


def initialize_origins(registration, binding, expected_cache, expected_model, parent_pid):
    global _ORIGIN_STATE
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    output = Path(registration).resolve().parent
    spec = read(registration)
    for name in list(os.environ):
        if name.startswith("SIMFOLIO_"):
            del os.environ[name]
    os.environ.update(spec["environment"])
    runtime = output / "runtime" / f"origin-{os.getpid()}"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update({"NUMBA_CACHE_DIR": str(runtime / "numba"), "TMPDIR": str(runtime)})
    tasks, receipt = shared_tasks(output.parent.parent, spec, read(output.parent.parent / "campaign.json"))
    if receipt != expected_cache:
        raise ValueError("origin child task-cache identity mismatch")
    model, adapter = load_model(spec)
    if model_identity(model) != expected_model:
        raise ValueError("origin child model/environment identity mismatch")
    if os.getppid() != parent_pid or os.getpgrp() != parent_pid:
        raise RuntimeError("origin child lost its owned leader")
    _ORIGIN_STATE = tasks, model, adapter, output, binding, parent_pid
    atomic(output / "runtime" / f"origin-process-{os.getpid()}.json",
           {"pid": os.getpid(), "parent": parent_pid, "pgid": os.getpgrp()})


def evaluate_origin_index(index):
    from simfolio_forecasting_methodology.runner import evaluate_origin_task
    tasks, model, adapter, output, binding, parent_pid = _ORIGIN_STATE
    if os.getppid() != parent_pid:
        raise RuntimeError("origin child lost its leader before dispatch")
    if adapter is not None:
        getattr(adapter, "clear_path_cache", lambda: None)()
        getattr(adapter, "TIMINGS", {}).clear()
        getattr(adapter, "FIT_DIAGNOSTICS", {}).clear()
    started = time.perf_counter()
    try:
        losses = evaluate_origin_task(model, tasks[index], simulations=240)
        write_checkpoint(output, index, binding, tasks[index], losses,
                         time.perf_counter() - started, adapter)
    except Exception as error:
        atomic(output / f"failure-{index:04d}.json", {"index": index,
               "manifest_sha256": binding, "error": type(error).__name__, "message": str(error)})
        raise
    finally:
        if adapter is not None:
            getattr(adapter, "clear_path_cache", lambda: None)()
    return index


def parallel_origins(registration, tasks, task_cache, model, binding, indices, completed,
                     workers, stop_requested, progress, active_tasks):
    """Bounded index-only dispatch; drain every submitted task before returning."""
    if os.getpgrp() != os.getpid():
        raise RuntimeError("parallel origins require an isolated process-group leader")
    output = Path(registration).resolve().parent
    campaign = output.parent.parent
    config = read(campaign / "campaign.json")
    context = multiprocessing.get_context("spawn")
    # A subprocess has no multiprocessing interpreter auto-join. It remains
    # alive through leader exit, including errors and TERM-ignoring descendants.
    watchdog = subprocess.Popen([sys.executable, "-s", str(ROOT / "tools/catalogue52/worker.py"),
                                 "--watch-parent", str(os.getpid())], stdout=subprocess.PIPE, text=True)
    pool = None
    failure = None
    status = "running"
    next_storage_check = 0
    try:
        if watchdog.stdout.readline().strip() != "origin-watchdog-ready":
            raise RuntimeError("origin parent-death watchdog failed to initialize")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=initialize_origins,
                                   initargs=(str(registration), binding, task_cache,
                                             model_identity(model), os.getpid()))
        todo = iter(i for i in indices if i not in completed)
        pending = {}
        exhausted = False
        while pending or not exhausted:
            if stop_requested and stop_requested():
                status = "paused"
            if watchdog.poll() is not None:
                failure = failure or RuntimeError("origin parent-death watchdog exited unexpectedly")
            if status == "running" and time.monotonic() >= next_storage_check:
                storage_file = campaign / "storage.json"
                storage = read(storage_file) if storage_file.exists() else {}
                available = (storage["available"] if time.time() - storage.get("time", 0) < 120
                             else storage_ok(campaign, config, additional_bytes=workers * 2 * 1024 ** 2))
                if not available:
                    status = "storage_paused"
                next_storage_check = time.monotonic() + 60
            if failure is None and status == "running":
                while len(pending) < workers and not exhausted:
                    if stop_requested and stop_requested():
                        status = "paused"
                        break
                    index = next(todo, None)
                    if index is None:
                        exhausted = True
                    else:
                        pending[pool.submit(evaluate_origin_index, index)] = index
            else:
                exhausted = True
            active_tasks[:] = sorted(pending.values())
            progress("stopping" if status != "running" or failure else "running")
            if not pending:
                break
            done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    if future.result() != index or checkpoint_losses(output, index, binding, tasks[index]) is None:
                        raise ValueError("origin child completion/checkpoint identity mismatch")
                    completed.add(index)
                except Exception as error:  # noqa: BLE001 - re-raised after draining valid in-flight tasks
                    # A completed atomic checkpoint survives a child/IPC crash.
                    if checkpoint_losses(output, index, binding, tasks[index]) is not None:
                        completed.add(index)
                    atomic(output / f"failure-{index:04d}.json", {"index": index,
                           "manifest_sha256": binding, "error": type(error).__name__, "message": str(error)})
                    failure = failure or error
    except Exception as error:  # noqa: BLE001 - re-raised after pool and watchdog cleanup
        failure = failure or error
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        watchdog.stdout.close()
        active_tasks.clear()
    if failure is not None:
        progress("failed")
        raise failure
    return status


def runtime_packages():
    return {package.metadata["Name"]: package.version
            for package in importlib.metadata.distributions()}


def prepared_hashes(data):
    result = {}
    for path in sorted(Path(data).rglob("*")):
        if path.is_symlink():
            raise ValueError("prepared input symlinks cannot enter the shared task cache")
        if path.is_file():
            result[str(path.relative_to(data))] = file_hash(path)
    return result


def dump_task_views(tasks, stream):
    """Store each physical root buffer once and preserve every ndarray view."""
    import numpy as np
    roots, positions = [], {}
    class ViewPickler(pickle.Pickler):
        def persistent_id(self, value):
            if not isinstance(value, np.ndarray):
                return None
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            if not (root.flags.c_contiguous or root.flags.f_contiguous):
                raise ValueError("task cache requires contiguous physical root buffers")
            if id(root) not in positions:
                positions[id(root)] = len(roots)
                roots.append(root.ravel(order="K"))
            return (positions[id(root)], value.dtype.str, value.shape, value.strides,
                    value.ctypes.data - root.ctypes.data, value.flags.writeable)
    metadata = io.BytesIO()
    ViewPickler(metadata, protocol=5).dump(tasks)
    pickle.dump((roots, metadata.getvalue()), stream, protocol=5)


def load_task_views(stream):
    import numpy as np
    roots, metadata = pickle.load(stream)
    class ViewUnpickler(pickle.Unpickler):
        def persistent_load(self, reference):
            index, dtype, shape, strides, offset, writable = reference
            result = np.ndarray(shape, dtype=dtype, buffer=roots[index], offset=offset, strides=strides)
            result.flags.writeable = writable
            return result
    return ViewUnpickler(io.BytesIO(metadata)).load()


def verify_task_roundtrip(original, restored, identities):
    from simfolio_forecasting_methodology.runner import task_identity
    if len(original) != EXPECTED["tasks"] or len(restored) != len(original):
        raise ValueError("task-cache roundtrip denominator mismatch")
    restored_ids = [asdict(task_identity(task, CACHE_REFERENCE_MODEL, 240)) for task in restored]
    if restored_ids != identities:
        raise ValueError("task-cache roundtrip changed task identities")
    array_count = 0
    original_groups, restored_groups = {}, {}
    for before, after in zip(original, restored):
        if before.training.policy != after.training.policy:
            raise ValueError("task-cache roundtrip changed holding policy")
        arrays = ((before.training.portfolio_log_returns, after.training.portfolio_log_returns),
                  (before.training.asset_log_returns, after.training.asset_log_returns),
                  (before.training.training_dates, after.training.training_dates),
                  (before.realized_future_daily_log_returns, after.realized_future_daily_log_returns),
                  (before.future_dates, after.future_dates))
        for left, right in arrays:
            if left is None or right is None:
                if left is not right:
                    raise ValueError("task-cache roundtrip changed optional arrays")
                continue
            if (left.dtype != right.dtype or left.shape != right.shape or left.strides != right.strides
                    or left.flags.writeable != right.flags.writeable or left.tobytes() != right.tobytes()):
                raise ValueError("task-cache roundtrip changed float/date array bytes")
            for value, groups in ((left, original_groups), (right, restored_groups)):
                root = value
                while isinstance(root.base, type(value)):
                    root = root.base
                groups.setdefault(id(root), []).append((array_count, value.ctypes.data - root.ctypes.data))
            array_count += 1
    if sorted(original_groups.values()) != sorted(restored_groups.values()):
        raise ValueError("task-cache roundtrip changed buffer alias groups or offsets")
    return {"all_task_identities_equal": True, "all_float_date_arrays_byte_equal": True,
            "all_strides_and_alias_groups_equal": True,
            "tasks": len(original), "arrays": array_count}


def shared_tasks(campaign, spec, config):
    """Reuse only an exact, locally constructed and exhaustively checked tuple.

    Task generation depends on src resources/code and expanded_panel/protocol;
    adding an unrelated tool adapter does not invalidate these frozen inputs.
    The mutable owner data directory is content-hashed on every load.
    """
    from protocol import single_asset_tasks

    from simfolio_forecasting_methodology.runner import (
        _numerical_environment_identity,
        task_identity,
    )
    snapshot = read(ROOT / "snapshot.json")
    if snapshot["sha256"] != spec["source_sha256"]:
        raise ValueError("shared task cache source snapshot mismatch")
    source = {name: value for name, value in snapshot["files"].items()
              if name.startswith("src/") or name == "tools/expanded_panel/protocol.py"}
    binding = {"format": "single52-exact-rootbuffer-task-views-pickle5-v2", "protocol": PROTOCOL,
               "sources": source, "prepared_files": prepared_hashes(Path(config["data"])),
               "python": platform.python_version(), "packages": runtime_packages(),
               "numerical_environment": _numerical_environment_identity()}
    key = digest(binding)
    parent = Path(campaign) / "input-cache"
    parent.mkdir(exist_ok=True)
    cache = parent / key
    started = time.monotonic()
    with lock(parent / f"{key}.lock"):
        if not cache.exists():
            if not storage_ok(campaign, config, additional_bytes=512 * 1024 ** 2):
                raise ValueError("shared task cache exceeds storage reserve/free-space gate")
            temporary = parent / f"{key}.tmp-{os.getpid()}"
            temporary.mkdir()
            try:
                tasks = single_asset_tasks(Path(config["data"]))
                identities = [asdict(task_identity(task, CACHE_REFERENCE_MODEL, 240)) for task in tasks]
                payload = temporary / "tasks.pickle"
                with payload.open("wb") as stream:
                    dump_task_views(tasks, stream)
                with payload.open("rb") as stream:
                    restored = load_task_views(stream)
                parity = verify_task_roundtrip(tasks, restored, identities)
                if prepared_hashes(Path(config["data"])) != binding["prepared_files"]:
                    raise ValueError("prepared inputs changed while constructing task cache")
                receipt = {"binding": binding, "sha256": key, "payload_sha256": file_hash(payload),
                           "payload_bytes": payload.stat().st_size, "parity": parity,
                           "reference_identities_sha256": digest(identities),
                           "reference_model": CACHE_REFERENCE_MODEL,
                           "construction_seconds": time.monotonic() - started}
                atomic(temporary / "manifest.json", receipt)
                del tasks, restored
                for path in temporary.iterdir():
                    path.chmod(0o444)
                temporary.rename(cache)
                cache.chmod(0o555)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        receipt = read(cache / "manifest.json")
        if receipt["binding"] != binding or receipt["sha256"] != key or digest(receipt["binding"]) != key:
            raise ValueError("shared task cache binding mismatch")
        if (not receipt["parity"].get("all_task_identities_equal")
                or not receipt["parity"].get("all_float_date_arrays_byte_equal")
                or not receipt["parity"].get("all_strides_and_alias_groups_equal")
                or receipt["parity"].get("tasks") != EXPECTED["tasks"]):
            raise ValueError("shared task cache lacks complete roundtrip parity")
        payload = cache / "tasks.pickle"
        if receipt["payload_sha256"] != file_hash(payload):
            raise ValueError("shared task cache payload checksum mismatch")
        with payload.open("rb") as stream:
            tasks = load_task_views(stream)
        if not isinstance(tasks, tuple) or len(tasks) != EXPECTED["tasks"]:
            raise ValueError("shared task cache task denominator mismatch")
    return tasks, {"sha256": key, "payload_sha256": receipt["payload_sha256"],
                   "payload_bytes": receipt["payload_bytes"], "parity": receipt["parity"],
                   "reference_identities_sha256": receipt["reference_identities_sha256"]}


def verify_environment(expected):
    if platform.python_version() != expected["python"]:
        raise ValueError("Python version differs from saved campaign environment")
    for package, version in expected["packages"].items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            raise ValueError(f"{package} version differs from saved environment: {installed} != {version}")


def load_model(spec):
    if spec["adapter"] is None:
        from simfolio_forecasting_methodology.models.registry import build_model
        return build_model(spec["model_id"]), None
    source = ROOT / spec["adapter"]
    sys.path.insert(0, str(source.parent))
    module_spec = importlib.util.spec_from_file_location("catalogue52_selected_adapter", source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    module.initialize(None, 1, 1)
    selected = [item for item in module.CANDIDATES if item.model_id == spec["model_id"]]
    if len(selected) != 1:
        raise ValueError("adapter must expose exactly one matching model identity")
    return selected[0], module


def checkpoint_losses(output, index, binding, task):
    import numpy as np
    path = output / f"task-{index:04d}.npz"
    record = path.with_suffix(".json")
    if not path.exists() or not record.exists():
        return None
    receipt = read(record)
    if receipt.get("index") != index or receipt.get("manifest_sha256") != binding:
        raise ValueError("checkpoint receipt identity mismatch")
    if receipt.get("sha256") != file_hash(path):
        raise ValueError("checkpoint checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        losses = saved["losses"]
        if str(saved["manifest"]) != binding:
            raise ValueError("checkpoint manifest mismatch")
    if losses.shape != (task.horizon_days,) or not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("checkpoint loss-vector shape/finiteness mismatch")
    return losses


def write_checkpoint(output, index, binding, task, losses, seconds, adapter):
    import numpy as np
    losses = np.asarray(losses, dtype=np.float64)
    if losses.shape != (task.horizon_days,) or not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("invalid task loss vector")
    path = output / f"task-{index:04d}.npz"
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, manifest=binding, losses=losses)
    temporary.replace(path)
    atomic(path.with_suffix(".json"), {"index": index, "manifest_sha256": binding,
           "sha256": file_hash(path), "portfolio": task.portfolio_id,
           "origin": task.origin_date, "horizon": task.horizon_days, "seed": int(task.seed),
           "seconds": seconds, "timings": dict(getattr(adapter, "TIMINGS", {})),
           "diagnostics": list(getattr(adapter, "FIT_DIAGNOSTICS", {}).values())})


def finalize(output, tasks, binding, *, smoke=False):
    """Publish a full score only after all exact task/cell/origin gates pass."""
    import numpy as np
    from protocol import block_cells
    if smoke:
        return None
    if len(tasks) != EXPECTED["tasks"]:
        raise ValueError("single52 task denominator mismatch")
    losses = [checkpoint_losses(output, index, binding, task) for index, task in enumerate(tasks)]
    if any(vector is None for vector in losses):
        raise ValueError("incomplete single52 tasks; no full score")
    names, cells, counts = block_cells(tasks, losses)
    if (len(names), len(tasks), cells.size) != tuple(EXPECTED[key] for key in ("portfolios", "tasks", "cells")):
        raise ValueError("single52 cell denominator mismatch")
    temporary = output / "cells.tmp.npz"
    np.savez_compressed(temporary, portfolio=names, losses=cells, counts=counts)
    temporary.replace(output / "cells.npz")
    result = {"block": "single52", "all_denominator_gates_passed": True,
              "exact_empirical_crps": float(cells.mean()), **EXPECTED,
              "simulations": 240, "manifest_sha256": binding,
              "portfolio_scores": dict(zip(names, cells.mean(axis=1).tolist())),
              "summed_worker_seconds": sum(read(output / f"task-{i:04d}.json")["seconds"]
                                           for i in range(len(tasks))),
              "cells_sha256": file_hash(output / "cells.npz")}
    atomic(output / "results.json", result)
    return result


def execute(registration, *, stop_requested=None):
    from simfolio_forecasting_methodology.runner import (
        _numerical_environment_identity,
        evaluate_origin_task,
        task_identity,
    )
    output = Path(registration).resolve().parent
    archived = output / "archive-receipt.json"
    if archived.exists() and read(archived).get("status") == "archived":
        raise ValueError("archived model requires explicit restoration before execution")
    startup_started = time.monotonic()
    startup = {"stage": "source_environment_verification", "stage_seconds": {}}
    atomic(output / "startup.json", startup)
    spec = read(registration)
    origin_workers = origin_worker_count(spec["environment"])
    campaign = output.parent.parent
    config = read(campaign / "campaign.json")
    if config["protocol"] != PROTOCOL or config["expected"] != EXPECTED or config["simulations"] != 240:
        raise ValueError("campaign protocol differs from frozen single52 contract")
    if digest(config) != spec["campaign_sha256"]:
        raise ValueError("campaign binding changed")
    verify_snapshot(ROOT, spec["source_sha256"])
    verify_environment(config["environment"])
    data = Path(config["data"])
    if file_hash(data / "canonical_data_manifest.json") != config["data_manifest_sha256"]:
        raise ValueError("frozen data manifest changed")
    startup["stage_seconds"]["source_environment_verification"] = time.monotonic() - startup_started
    startup["stage"] = "shared_task_input_cache"
    atomic(output / "startup.json", startup)
    phase_started = time.monotonic()
    tasks, task_cache = shared_tasks(campaign, spec, config)
    startup["stage_seconds"]["shared_task_input_cache"] = time.monotonic() - phase_started
    startup["stage"] = "task_identities"
    atomic(output / "startup.json", startup)
    phase_started = time.monotonic()
    identities = [asdict(task_identity(task, spec["model_id"], 240)) for task in tasks]
    if len(identities) != EXPECTED["tasks"] or len({item["task_id"] for item in identities}) != len(tasks):
        raise ValueError("single52 task identity denominator mismatch")
    startup["stage_seconds"]["task_identities"] = time.monotonic() - phase_started
    startup["stage"] = "model_initialization"
    atomic(output / "startup.json", startup)
    phase_started = time.monotonic()
    model, adapter = load_model(spec)
    if model.model_id != spec["model_id"]:
        raise ValueError("factory returned a different model identity")
    startup["stage_seconds"]["model_initialization"] = time.monotonic() - phase_started
    startup["stage"] = "complete"
    startup["total_seconds"] = time.monotonic() - startup_started
    atomic(output / "startup.json", startup)
    manifest = {"protocol": PROTOCOL, "registration": spec, "campaign": config,
                "task_cache": task_cache,
                "task_identities": identities,
                "task_seed_forwarding": "origin_task.seed_to_forecast_context.seed.v1",
                "native_seed_contract": getattr(model, "seed_contract", "source-bound native model seed derivation"),
                "model_environment": {k: v for k, v in os.environ.items() if k.startswith("SIMFOLIO_")},
                "forecast_semantics": "terminal_log_return_ensemble_by_horizon",
                "native_forecast_semantics": getattr(model, "forecast_output_semantics", "daily_log_return_paths"),
                "runtime_packages": runtime_packages(),
                "numerical_environment": _numerical_environment_identity(),
                "specification_fingerprint": getattr(model, "specification_fingerprint", None)}
    manifest_file = output / "manifest.json"
    if manifest_file.exists() and read(manifest_file) != manifest:
        raise ValueError("resume manifest mismatch")
    atomic(manifest_file, manifest)
    binding = digest(manifest)
    indices = list(range(len(tasks))) if spec["indices"] is None else spec["indices"]
    completed = {i for i in indices if checkpoint_losses(output, i, binding, tasks[i]) is not None}
    started = time.monotonic()
    next_storage_check = started
    current_task = None
    active_tasks = []
    def progress(status):
        value = {"status": status, "completed_tasks": len(completed),
               "required_tasks": len(tasks), "scheduled_tasks": len(indices),
               "full_score_eligible": spec["indices"] is None,
               "current_task": current_task,
               "manifest_sha256": binding, "elapsed_seconds": time.monotonic() - started}
        if origin_workers > 1:
            value.update({"origin_workers": origin_workers, "active_tasks": list(active_tasks)})
        atomic(output / "progress.json", value)
    progress("running")
    if origin_workers > 1:
        status = parallel_origins(registration, tasks, task_cache, model, binding, indices,
                                  completed, origin_workers, stop_requested, progress, active_tasks)
        if status != "running":
            progress(status)
            return
        verify_snapshot(ROOT, spec["source_sha256"])
        finalize(output, tasks, binding, smoke=spec["indices"] is not None)
        progress("complete" if spec["indices"] is None else "smoke_complete")
        return
    for index in indices:
        if index in completed:
            continue
        if stop_requested and stop_requested():
            progress("paused")
            return
        if time.monotonic() >= next_storage_check:
            storage_file = campaign / "storage.json"
            storage = read(storage_file) if storage_file.exists() else {}
            available = (storage["available"] if time.time() - storage.get("time", 0) < 120
                         else storage_ok(campaign, config))
            if not available:
                progress("storage_paused")
                return
            next_storage_check = time.monotonic() + 60
        if adapter is not None:
            getattr(adapter, "clear_path_cache", lambda: None)()
            getattr(adapter, "TIMINGS", {}).clear()
            getattr(adapter, "FIT_DIAGNOSTICS", {}).clear()
        task_started = time.perf_counter()
        current_task = index
        progress("running")
        try:
            losses = evaluate_origin_task(model, tasks[index], simulations=240)
            write_checkpoint(output, index, binding, tasks[index], losses,
                             time.perf_counter() - task_started, adapter)
        except Exception as error:
            atomic(output / f"failure-{index:04d}.json", {"index": index,
                   "manifest_sha256": binding, "error": type(error).__name__, "message": str(error)})
            progress("failed")
            raise
        completed.add(index)
        current_task = None
        progress("running")
        if adapter is not None:
            getattr(adapter, "clear_path_cache", lambda: None)()
    verify_snapshot(ROOT, spec["source_sha256"])
    finalize(output, tasks, binding, smoke=spec["indices"] is not None)
    progress("complete" if spec["indices"] is None else "smoke_complete")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--watch-parent":
        parent_watchdog(int(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--token", required=True)
    parser.add_argument("--nice", type=int, default=10)
    args = parser.parse_args()
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    os.nice(args.nice)
    output = args.registration.resolve().parent
    with lock(output / "worker.lock", nonblocking=True):
        atomic(output / "ready.json", {"token": args.token, "pid": os.getpid()})
        try:
            execute(args.registration, stop_requested=lambda: stopping)
        except Exception as error:
            atomic(output / "error.json", {"error": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
