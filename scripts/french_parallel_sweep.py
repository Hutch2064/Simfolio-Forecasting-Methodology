"""Shared CPU slots for the unchanged, checkpoint-compatible French origin tasks.

Each spawn imports one methodology for its whole lifetime. The parent alone owns
queues, in-flight assignments, aggregation and status; processes never share
native model hooks. Controls use the original models/workers/stop JSON schema.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import statistics
import time

import numpy as np

from simfolio_forecasting_methodology.french_panel import runner as original
from simfolio_forecasting_methodology.french_panel.sweep import (
    _identity, exclusive_lock, model_directory, read_control,
)

BASELINE = "asset_m256_data_timescales_existing_sv_rate"


def task_identity(fingerprint, series_id, origin):
    return {"execution_fingerprint": fingerprint, "series_id": series_id,
            "origin": asdict(origin), "seed": original.coherent_daily_seed(
                origin.origin_date, origin.max_horizon, original.SIMULATIONS)}


def task_paths(root, task):
    stem = f"{task[0]:06d}/{task[1]:03d}"
    return root / "tasks" / f"{stem}.npz", root / "errors" / f"{stem}.json"


def prepare_manifest(model, spec, panel, origins, root):
    """The original manifest contract, using its unchanged source/runtime helpers.

    Kept outside the package so scheduler changes do not invalidate saved losses.
    The tests compare this manifest directly with original.run_panel's manifest.
    """
    identity = {
        "protocol_id": original.PROTOCOL_ID,
        "panel_fingerprint": panel["panel_fingerprint"],
        "series_ids": [s["series_id"] for s in panel["series"]],
        "model_id": model.model_id,
        "implementation_digest": spec["implementation_digest"],
        "runtime": original.dependency_identity(model),
        "simulations_per_origin": original.SIMULATIONS,
        "source_closure": original._source_closure(),
        "origins": [asdict(o) for o in origins],
        "seed_contract": "coherent_daily_seed(origin_date,max_horizon,240)",
        "aggregation": "origin_mean_per_series_horizon_then_equal_cell_mean",
        "future_dates": "synthetic_weekdays_after_origin_actual_dates_for_train_and_targets",
    }
    return original._manifest(root, identity)


def execute_task(model, panel, dates, returns, origins, root, fingerprint, task):
    """Exactly the original origin operation and file/seed/error contracts."""
    i, j = task
    sid, origin = panel["series"][i]["series_id"], origins[j]
    identity = task_identity(fingerprint, sid, origin)
    checkpoint, error = task_paths(root, task)
    if checkpoint.exists():
        original._read_checkpoint(checkpoint, identity)
        return "done"
    try:
        losses, diagnostics = original._forecast(
            model, sid, returns[i], dates, origin, identity["seed"])
    except Exception as exc:
        original._atomic_json(error, {"identity": identity,
            "exception_type": type(exc).__name__, "message": str(exc)})
        return "failed"
    original._write_checkpoint(checkpoint, identity, losses, diagnostics)
    error.unlink(missing_ok=True)
    return "done"


def _continuing(path, mid, spec):
    try:
        control = read_control(path)
        current = control["models"].get(mid)
        return bool(not control["stop"] and current and current["enabled"]
                    and _identity(current) == _identity(spec))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def worker(connection, mid, spec, panel_root, root, control_path, workers_lock):
    """One methodology per process, with a fresh control check at each boundary."""
    try:
        with ExitStack() as stack:
            for path in (workers_lock, root / ".run.lock"):
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open("a+"))
                fcntl.flock(handle, fcntl.LOCK_SH)
            if not _continuing(control_path, mid, spec):
                connection.send(("paused", None))
                return
            module, factory = spec["factory"].split(":")
            model = getattr(importlib.import_module(module), factory)()
            if getattr(model, "model_id", None) != mid:
                raise ValueError("factory model_id differs from control")
            panel = original.load_panel(panel_root)
            origins = tuple(original.schedule(panel_root))
            dates, returns = original.load_dates(panel_root), original.load_returns(panel_root)
            fingerprint = prepare_manifest(model, spec, panel, origins, root)
            connection.send(("ready", fingerprint))
            while True:
                task = connection.recv()
                if task is None:
                    return
                if not _continuing(control_path, mid, spec):
                    connection.send(("paused", task))
                    return
                result = execute_task(model, panel, dates, returns, origins,
                                      root, fingerprint, tuple(task))
                connection.send((result, task))
    except (EOFError, BrokenPipeError):
        pass  # Parent died: finish only the already assigned origin, then exit.
    except Exception as exc:
        try:
            connection.send(("fatal", f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


def start_worker(*args):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, *args))
    process.start()
    child.close()
    return process, parent


class ModelProgress:
    def __init__(self, mid, spec, panel, origins, root):
        self.mid, self.spec, self.panel = mid, spec, panel
        self.origins, self.root = origins, root
        self.fingerprint = None
        self.queue = deque()
        self.completed = set()
        self.failures = set()
        self.times = {}
        self.series_sums = {}
        self.error = None
        self.resumed = self.attempted = 0
        self.published_summary = None
        self.capacity = max(o.max_horizon for o in origins)
        self.expected = len(panel["series"]) * len(origins)

    def identity(self, task):
        return task_identity(self.fingerprint,
                             self.panel["series"][task[0]]["series_id"], self.origins[task[1]])

    def scan(self, fingerprint):
        self.fingerprint = fingerprint
        # Existing failures are retried once on this explicit controller run,
        # like original.run_panel; failures this run are never hot-looped.
        for i in range(len(self.panel["series"])):
            for j in range(len(self.origins)):
                task = i, j
                checkpoint, _ = task_paths(self.root, task)
                if checkpoint.exists():
                    self.accept(task, aggregate=False)
                    self.resumed += 1
                else:
                    self.queue.append(task)
            self.aggregate_series(i)

    def accept(self, task, *, aggregate=True):
        checkpoint, _ = task_paths(self.root, task)
        _, diagnostics = original._read_checkpoint(checkpoint, self.identity(task))
        self.completed.add(task)
        self.failures.discard(task)
        self.times[task] = diagnostics["forecast_and_score_seconds"]
        if aggregate:
            self.aggregate_series(task[0])

    def aggregate_series(self, i):
        if i in self.series_sums or not all((i, j) in self.completed for j in range(len(self.origins))):
            return
        sums = np.zeros(self.capacity, dtype=np.float64)
        counts = np.zeros(self.capacity, dtype=np.int64)
        # Preserve the original summation order irrespective of worker completion.
        for j, origin in enumerate(self.origins):
            task = i, j
            losses, _ = original._read_checkpoint(task_paths(self.root, task)[0], self.identity(task))
            sums[:origin.max_horizon] += losses
            counts[:origin.max_horizon] += 1
        self.series_sums[i] = float(np.sum(sums / counts))

    def summary(self, paused=False):
        cell_sum = 0.0
        indices = sorted(self.series_sums)
        for i in indices:
            cell_sum += self.series_sums[i]
        cells = len(indices) * self.capacity
        score = cell_sum / cells if cells else None
        complete = len(self.completed) == self.expected and not self.failures and not self.error
        seconds = sum(self.times[t] for t in sorted(self.times))
        return {
            "protocol_id": original.PROTOCOL_ID, "execution_fingerprint": self.fingerprint,
            "panel_fingerprint": self.panel["panel_fingerprint"], "model_id": self.mid,
            "status": "complete" if complete else "incomplete", "paused": paused,
            "simulations_per_origin": original.SIMULATIONS,
            "expected_tasks": self.expected, "completed_tasks": len(self.completed),
            "attempted_tasks_this_call": self.attempted, "resumed_tasks": self.resumed,
            "completed_forecast_and_score_seconds": seconds,
            "timing_scope": "sum of saved completed-origin forecast and scoring worker compute",
            "failed_tasks": len(self.failures),
            "pending_tasks": self.expected - len(self.completed) - len(self.failures),
            "expected_cells": len(self.panel["series"]) * self.capacity,
            "complete_series": [self.panel["series"][i]["series_id"] for i in indices],
            "complete_series_cells": cells,
            "final_score": score if complete else None,
            "partial_complete_series_score": None if complete else score,
            "partial_score_scope": "matching complete series subset only; compare identical series IDs",
            "diagnostics": "per-origin final-horizon CAGR stored in task checkpoints; no exclusions",
        }


def choose_model(candidates, active_counts, estimates, progress):
    """Admit every waiting model before lending additional slots."""
    unserved = [m for m in candidates if active_counts.get(m, 0) == 0]
    if unserved:
        return unserved[0]
    def priority(mid):
        item = progress[mid]
        estimate = estimates.get(mid, {}).get("mean_origin_extrapolation_worker_seconds")
        total = estimate if estimate is not None else (
            statistics.mean(item.times.values()) * item.expected if item.times else item.expected)
        fraction = (item.expected - len(item.completed) - len(item.failures)) / item.expected
        return total * fraction / active_counts[mid]
    return max(candidates, key=priority)


class ParallelController:
    def __init__(self, panel_root, output_root, control_path, *, starter=start_worker,
                 estimates=None):
        self.panel_root, self.output_root = Path(panel_root), Path(output_root)
        self.control_path = Path(control_path)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.panel = original.load_panel(self.panel_root)
        self.origins = tuple(original.schedule(self.panel_root))
        self.progress, self.active = {}, {}
        self.sequence = 0
        self.starter = starter
        self.estimates = estimates or {}
        self.last_control = {"models": {}, "workers": 12, "stop": True}
        self.control_error = None
        self.workers_lock = self.output_root / ".parallel-workers.lock"

    def enabled(self, mid, control):
        spec = control["models"].get(mid)
        return bool(not control["stop"] and spec and spec["enabled"]
                    and not self.progress[mid].error
                    and _identity(spec) == _identity(self.progress[mid].spec))

    def counts(self):
        result = {}
        for slot in self.active.values():
            result[slot["mid"]] = result.get(slot["mid"], 0) + 1
        return result

    def _return_task(self, item, task, *, crash=False):
        if task is None:
            return
        path, error = task_paths(item.root, task)
        if path.exists():
            item.accept(task)
        elif crash:
            original._atomic_json(error, {"identity": item.identity(task),
                "exception_type": "WorkerProcessExit", "message": "worker exited before reporting origin"})
            item.failures.add(task)
        else:
            item.queue.appendleft(task)

    def poll(self):
        self.control_error = None
        try:
            control = read_control(self.control_path)
            self.last_control = control
        except (OSError, ValueError) as exc:
            self.control_error = str(exc)
            control = {**self.last_control, "stop": True}
        for mid, spec in control["models"].items():
            if mid not in self.progress:
                self.progress[mid] = ModelProgress(mid, spec, self.panel, self.origins,
                                                   model_directory(self.output_root, mid))
            elif _identity(spec) != _identity(self.progress[mid].spec):
                self.progress[mid].error = "factory or implementation digest changed; use a new model ID"
        for key, slot in list(self.active.items()):
            item, connection, process = self.progress[slot["mid"]], slot["connection"], slot["process"]
            try:
                while connection.poll():
                    kind, payload = connection.recv()
                    if kind == "ready":
                        if item.fingerprint is None:
                            item.scan(payload)
                        elif payload != item.fingerprint:
                            raise ValueError("worker execution identity mismatch")
                        slot["ready"] = True
                    elif kind == "done":
                        item.accept(tuple(payload))
                        slot.update(task=None, ready=True)
                    elif kind == "failed":
                        item.failures.add(tuple(payload))
                        slot.update(task=None, ready=True)
                    elif kind == "paused":
                        self._return_task(item, slot["task"])
                        slot.update(task=None, ready=False, stopping=True)
                    elif kind == "fatal":
                        item.error = payload
                        self._return_task(item, slot["task"], crash=True)
                        slot.update(task=None, ready=False, stopping=True)
            except EOFError:
                pass
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
                if not slot["stopping"]:
                    try:
                        connection.send(None)
                    except (OSError, EOFError):
                        pass
                slot.update(ready=False, stopping=True)
            if not process.is_alive():
                process.join()
                if slot["task"] is None and not slot["stopping"]:
                    item.error = f"worker exited before ready (exit code {process.exitcode})"
                try:
                    self._return_task(item, slot["task"], crash=True)
                except Exception as exc:
                    item.error = f"{type(exc).__name__}: {exc}"
                connection.close()
                del self.active[key]
        allowed_keys = list(self.active)[:control["workers"]]
        for key, slot in self.active.items():
            if not slot["ready"]:
                continue
            item = self.progress[slot["mid"]]
            if key not in allowed_keys or not self.enabled(item.mid, control) or not item.queue:
                slot["connection"].send(None)
                slot.update(ready=False, stopping=True)
            else:
                task = item.queue.popleft()
                item.attempted += 1
                slot["connection"].send(task)
                slot.update(task=task, ready=False)
        while len(self.active) < control["workers"] and not control["stop"]:
            counts = self.counts()
            candidates = [mid for mid, item in self.progress.items()
                          if self.enabled(mid, control) and
                          (item.fingerprint is None or len(item.queue) > sum(
                              s["mid"] == mid and s["task"] is None and not s["stopping"]
                              for s in self.active.values()))]
            if not candidates:
                break
            mid = choose_model(candidates, counts, self.estimates, self.progress)
            item = self.progress[mid]
            # Old whole-model workers own exclusive locks; never overlap them.
            if not counts.get(mid):
                try:
                    with exclusive_lock(item.root / ".run.lock"):
                        pass
                except RuntimeError as exc:
                    item.error = str(exc)
                    continue
            process, connection = self.starter(mid, item.spec, self.panel_root, item.root,
                                               self.control_path, self.workers_lock)
            self.sequence += 1
            self.active[self.sequence] = dict(mid=mid, process=process, connection=connection,
                                              ready=False, task=None, stopping=False)
        return self.publish(control)

    def publish(self, control):
        counts = self.counts()
        models, summaries = {}, {}
        for mid, item in self.progress.items():
            enabled = self.enabled(mid, control)
            summary = item.summary(paused=not enabled)
            summaries[mid] = summary
            if item.fingerprint is not None and summary != item.published_summary:
                original._atomic_json(item.root / "summary.json", summary)
                original._atomic_json(item.root / "progress.json", summary)
                item.published_summary = summary
            state = ("failed" if item.error else "completed" if summary["status"] == "complete"
                     else "active" if counts.get(mid) else "waiting" if enabled else "paused")
            models[mid] = {**summary, "state": state, "enabled": bool(control["models"].get(mid, {}).get("enabled")),
                          "active_workers": counts.get(mid, 0), "ready_queue": len(item.queue),
                          "running_origins": sum(s["mid"] == mid and s["task"] is not None
                                                 for s in self.active.values()),
                          "in_flight_origins": [list(s["task"]) for s in self.active.values()
                                                if s["mid"] == mid and s["task"] is not None],
                          "error": item.error, "output_directory": str(item.root)}
        baseline = self.progress.get(BASELINE)
        for mid, item in self.progress.items():
            common = sorted(set(item.series_sums) & set(baseline.series_sums)) if baseline else []
            a = b = 0.0
            for i in common:
                a += item.series_sums[i]
                b += baseline.series_sums[i]
            models[mid]["matched_m261_complete_series"] = len(common)
            models[mid]["matched_m261_series_ids"] = [self.panel["series"][i]["series_id"] for i in common]
            models[mid]["percent_better_m261_matching_series"] = 100 * (1 - a / b) if b else None
        samples = [s["completed_forecast_and_score_seconds"] for s in summaries.values()
                   if s["status"] == "complete"]
        eta = {"completed_model_sample_count": len(samples),
               "kind": "workload_projection_not_measured_wall_eta",
               "uncertainty_interval_seconds": None,
               "assumptions": "Broad-market profile scaled by remaining task fraction / slots; origin and series fit costs differ. No parallel contention, setup or IO model; no calibrated uncertainty band.",
               "mean_seconds": None, "median_seconds": None,
               "scope": "enabled unfinished work / shared worker slots; excludes setup, IO, contention and current origin remainder",
               "projection_source": "saved stratified origin extrapolation; not a measured full sweep",
               "projection_model_count": sum(mid in self.estimates for mid in models),
               "projection_sample_scope": "stored broad-market stratified origins; see frozen remaining_runtime_estimate.json"}
        enabled = [mid for mid in models if self.enabled(mid, control) and models[mid]["status"] != "complete"]
        for label, function in (("mean", statistics.mean), ("median", statistics.median)):
            remaining = []
            for mid in enabled:
                item = self.progress[mid]
                estimate = self.estimates.get(mid, {}).get(f"{label}_origin_extrapolation_worker_seconds")
                total = estimate if estimate is not None else (function(samples) if samples else None)
                if total is None:
                    remaining = None
                    break
                fraction = (item.expected - len(item.completed) - len(item.failures)) / item.expected
                remaining.append(total * fraction)
            eta[f"{label}_seconds"] = sum(remaining) / control["workers"] if remaining is not None else None
        status = {"models": models, "workers": control["workers"],
                  "active_workers": len(self.active), "free_slots": control["workers"] - len(self.active),
                  "ready_queue": sum(m["ready_queue"] for m in models.values()),
                  "stop_requested": control["stop"] and self.control_error is None,
                  "control_error": self.control_error, "eta": eta,
                  "scheduler": "shared_origin_queues_v1", "counts": {
                      state: sum(m["state"] == state for m in models.values())
                      for state in ("active", "waiting", "paused", "completed", "failed")},
                  "ranking": sorted(({"model_id": mid, "final_score": m["final_score"]}
                     for mid, m in models.items() if m["final_score"] is not None and m["enabled"]),
                     key=lambda m: m["final_score"])}
        original._atomic_json(self.output_root / "status.json", status)
        original._atomic_json(self.output_root / "parallel-state.json", status)
        return status


def run_controller(panel_root, output_root, control_path, *, poll_seconds=.1):
    output_root = Path(output_root)
    estimate_path = Path(original.__file__).parents[1] / "resources/french1926/remaining_runtime_estimate.json"
    estimates = json.loads(estimate_path.read_text())["models"] if estimate_path.exists() else {}
    with exclusive_lock(output_root / ".controller.lock"):
        # A previous parent may have died during a forecast. Its shared worker
        # lock remains held until that origin is saved and its process exits.
        with exclusive_lock(output_root / ".parallel-workers.lock"):
            pass
        original._atomic_json(output_root / "scheduler-provenance.json", {
            "scheduler": "shared_origin_queues_v1",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "statistical_source_identity": "unchanged original execution manifests"})
        controller = ParallelController(panel_root, output_root, control_path, estimates=estimates)
        while True:
            status = controller.poll()
            if status["stop_requested"] and not controller.active:
                return status
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    run_controller(args.panel_root, args.output_root, args.control)


if __name__ == "__main__":
    main()
