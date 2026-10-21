"""Opt-in dynamic French-panel sweeps. No model is registered by this module.

Edit the JSON control file atomically while the controller is running. Workers
finish their current origin before pausing; checkpoints and excluded history stay
in place. Run with ``python -m ...french_panel.sweep --help``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import multiprocessing
import re
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

from .data import load_panel, schedule
from .runner import _atomic_json, run_panel


def read_control(path: Path) -> dict:
    """Validate controls without importing any factory or adapter."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != {"models", "workers", "stop"}:
        raise ValueError("control requires exactly models, workers, and stop")
    if type(value["workers"]) is not int or not 1 <= value["workers"] <= 12:
        raise ValueError("workers must be an integer from 1 through 12")
    if type(value["stop"]) is not bool or not isinstance(value["models"], dict):
        raise ValueError("stop must be boolean and models must be an object")
    for model_id, spec in value["models"].items():
        if (not model_id.strip() or len(model_id) > 2048 or not model_id.isprintable()):
            raise ValueError("model IDs must be nonempty printable strings up to 2048 characters")
        if not isinstance(spec, dict) or set(spec) != {
            "enabled", "factory", "implementation_digest"
        }:
            raise ValueError(f"invalid specification for {model_id}")
        if type(spec["enabled"]) is not bool:
            raise ValueError("enabled must be boolean")
        if (not isinstance(spec["factory"], str)
                or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*",
                                    spec["factory"])):
            raise ValueError("factory must be module:factory")
        if (not isinstance(spec["implementation_digest"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", spec["implementation_digest"])):
            raise ValueError("implementation_digest must be a lowercase SHA-256")
    return value


def model_directory(output_root: Path, model_id: str) -> Path:
    """Preserve arbitrary historical IDs without treating them as filesystem paths."""
    return Path(output_root) / "models" / hashlib.sha256(model_id.encode("utf-8")).hexdigest()


def _identity(spec: dict) -> dict:
    return {key: spec[key] for key in ("factory", "implementation_digest")}


@contextmanager
def exclusive_lock(path: Path):
    """Kernel-owned lock: a crashed owner cannot leave a stale locked sweep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _is_locked(path: Path) -> bool:
    try:
        with exclusive_lock(path):
            return False
    except RuntimeError:
        return True


def worker_should_continue(control_path: Path, admission_path: Path,
                           model_id: str, identity: dict) -> bool:
    """Fail closed on malformed edits, exclusion, stop, or implementation drift."""
    try:
        control = read_control(control_path)
        spec = control["models"].get(model_id)
        admitted = json.loads(admission_path.read_text())["models"]
        return bool(not control["stop"] and spec and spec["enabled"]
                    and _identity(spec) == identity and model_id in admitted)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _worker(model_id: str, spec: dict, panel_root: Path, model_root: Path,
            control_path: Path, admission_path: Path) -> None:
    """Spawn target; factories are resolved only for explicitly admitted work."""
    acquired = False
    try:
        with exclusive_lock(model_root / ".run.lock"):
            acquired = True
            identity = _identity(spec)
            continuing = lambda: worker_should_continue(
                control_path, admission_path, model_id, identity
            )
            if not continuing():
                _atomic_json(model_root / "worker-result.json", {"state": "paused", "model_id": model_id})
                return
            module, factory = spec["factory"].split(":")
            model = getattr(importlib.import_module(module), factory)()
            if getattr(model, "model_id", None) != model_id:
                raise ValueError("factory model_id must match the explicit control model ID")
            result = run_panel(
                model, panel_root, model_root,
                implementation_digest=spec["implementation_digest"],
                should_continue=continuing,
                progress_callback=lambda progress: _atomic_json(model_root / "progress.json", progress),
            )
            state = ("completed" if result["status"] == "complete" else
                     "paused" if result.get("paused") else "failed")
            _atomic_json(model_root / "worker-result.json", {"state": state, "model_id": model_id, "result": result})
    except Exception as exc:  # noqa: BLE001 - isolate adapter failures from other workers.
        if not acquired:
            return  # Never overwrite the result of a different live lock owner.
        _atomic_json(model_root / "worker-result.json", {
            "state": "failed", "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"
        })


def _start_worker(*args):
    process = multiprocessing.get_context("spawn").Process(target=_worker, args=args)
    process.start()
    return process


def _read_json(path: Path, default: dict) -> dict:
    return json.loads(path.read_text()) if path.exists() else default


def build_status(records: dict, control: dict, expected_tasks: int) -> dict:
    """Model ETA is unavailable until at least one whole model has finished.

    Samples are full saved forecast+score compute sums, not overlapping wall time.
    Mean/median assume comparable model cost; each enabled remaining model's saved
    compute is subtracted. Greedy slot loading estimates the remaining makespan;
    one model occupies one worker and cannot split across idle workers. IO, setup,
    current unsaved origins, contention and model heterogeneity are not predicted.
    """
    models = {}
    counts = dict.fromkeys(("active", "waiting", "paused", "completed", "failed"), 0)
    samples = []
    remaining = []
    for model_id, record in records.items():
        spec = control["models"].get(model_id)
        enabled = bool(spec and spec["enabled"])
        admitted = enabled and not control["stop"]
        state = record["state"]
        if state not in {"active", "completed", "failed"}:
            state = "waiting" if admitted else "paused"
        progress = record.get("progress", {})
        seconds = progress.get("completed_forecast_and_score_seconds", 0.0)
        tasks = progress.get("completed_tasks", 0)
        counts[state] += 1
        models[model_id] = {
            "state": state, "enabled": enabled, "completed_tasks": tasks,
            "expected_tasks": expected_tasks,
            "completed_forecast_and_score_seconds": seconds,
            "error": record.get("error"), "output_directory": record.get("output_directory"),
            "final_score": progress.get("final_score") if state == "completed" else None,
        }
        if state == "completed":
            samples.append(seconds)
        elif admitted and state != "failed":
            remaining.append(seconds)
    mean = statistics.mean(samples) if samples else None
    median = statistics.median(samples) if samples else None
    # A zero-time synthetic run cannot support a real execution-time estimate.
    def makespan(sample):
        if sample is None or sample <= 0:
            return None
        slots = [0.0] * control["workers"]
        for used in remaining:
            slot = min(range(len(slots)), key=slots.__getitem__)
            slots[slot] += max(0.0, sample - used)
        return max(slots)

    mean_eta, median_eta = makespan(mean), makespan(median)
    return {
        "models": models, "counts": counts, "workers": control["workers"],
        "stop_requested": control["stop"],
        "eta": {
            "mean_seconds": mean_eta, "median_seconds": median_eta,
            "completed_model_sample_count": len(samples),
            "mean_completed_model_compute_seconds": mean,
            "median_completed_model_compute_seconds": median,
            "enabled_remaining_model_count": len(remaining),
            "scope": "enabled unfinished models only; paused and failed models excluded",
            "assumptions": build_status.__doc__.strip(),
        },
        "completed_history": [
            {"model_id": key, "enabled": item["enabled"], "final_score": item["final_score"]}
            for key, item in models.items() if item["state"] == "completed"
        ],
        "ranking": sorted(
            ({"model_id": key, "final_score": item["final_score"]}
             for key, item in models.items()
             if item["state"] == "completed" and item["enabled"]
             and item["final_score"] is not None),
            key=lambda item: item["final_score"],
        ),
    }


class SweepController:
    """One poll admits/reaps work; run_controller owns the exclusive writer lock."""

    def __init__(self, panel_root: Path, output_root: Path, control_path: Path, *,
                 expected_tasks: int | None = None, start_worker=_start_worker):
        self.panel_root, self.output_root = Path(panel_root), Path(output_root)
        self.control_path = Path(control_path)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.admission_path = self.output_root / "admission.json"
        self.expected_tasks = expected_tasks
        if self.expected_tasks is None:
            self.expected_tasks = len(load_panel(self.panel_root)["series"]) * len(
                tuple(schedule(self.panel_root)))
        self.records = _read_json(self.output_root / "state.json", {"models": {}})["models"]
        # An interrupted controller never assumes the old process still belongs to it.
        for record in self.records.values():
            if record["state"] == "active":
                record["state"] = "paused"
        self.active = {}
        self.start_worker = start_worker
        self.last_control = {"models": {}, "workers": 1, "stop": False}

    def poll(self) -> dict:
        control_error = None
        try:
            control = read_control(self.control_path)
            self.last_control = control
        except (OSError, ValueError) as exc:
            # A partially saved edit pauses admission until a valid edit arrives.
            control_error = f"{type(exc).__name__}: {exc}"
            control = {**self.last_control, "stop": True}
        for model_id, spec in control["models"].items():
            record = self.records.setdefault(model_id, {
                "identity": _identity(spec), "state": "waiting", "progress": {},
                "output_directory": str(model_directory(self.output_root, model_id)),
            })
            if record["identity"] != _identity(spec):
                record.update(state="failed", error="factory or implementation digest changed; "
                              "use a new model ID and output directory")
        # Reattach checkpoint-boundary admission to children left by an interrupted
        # controller. A held model lock is authoritative; never launch a duplicate.
        for model_id, record in self.records.items():
            root = model_directory(self.output_root, model_id)
            if model_id not in self.active and _is_locked(root / ".run.lock"):
                self.active[model_id] = None
                if record["state"] != "failed":
                    record["state"] = "active"
            elif model_id not in self.active and record["state"] != "failed":
                outcome = _read_json(root / "worker-result.json", {})
                if outcome:
                    record.update(state=outcome["state"], error=outcome.get("error"))
                    if "result" in outcome:
                        record["progress"] = outcome["result"]
        # Revoking admission pauses excess workers at their next origin boundary.
        allowed = [key for key in self.active
                   if key in control["models"] and control["models"][key]["enabled"]
                   and self.records[key]["state"] != "failed"][:control["workers"]]
        if control["stop"]:
            allowed = []
        _atomic_json(self.admission_path, {"models": allowed})
        for model_id, process in list(self.active.items()):
            root = model_directory(self.output_root, model_id)
            record = self.records[model_id]
            progress = _read_json(root / "progress.json", record.get("progress", {}))
            # During resume the runner scans old checkpoints before reaching new work.
            if progress.get("completed_tasks", 0) >= record.get("progress", {}).get(
                    "completed_tasks", 0):
                record["progress"] = progress
            if (process.is_alive() if process is not None else _is_locked(root / ".run.lock")):
                continue
            if process is not None:
                process.join()
            result = _read_json(root / "worker-result.json", {
                "state": "failed", "error": "worker exited without result "
                f"({process.exitcode if process is not None else 'external worker'})"
            })
            if record["state"] != "failed":
                record.update(state=result["state"], error=result.get("error"))
                if "result" in result:
                    record["progress"] = result["result"]
            del self.active[model_id]
        if not control["stop"]:
            for model_id, spec in control["models"].items():
                record = self.records[model_id]
                if len(self.active) >= control["workers"]:
                    break
                if (not spec["enabled"] or model_id in self.active
                        or record["state"] in {"completed", "failed"}):
                    continue
                root = model_directory(self.output_root, model_id)
                root.mkdir(parents=True, exist_ok=True)
                (root / "worker-result.json").unlink(missing_ok=True)
                # Publish admission before starting a child; no startup race.
                _atomic_json(self.admission_path, {"models": allowed + [model_id]})
                self.active[model_id] = self.start_worker(
                    model_id, spec, self.panel_root, root, self.control_path, self.admission_path
                )
                allowed.append(model_id)
                record["state"] = "active"
        status = build_status(self.records, control, self.expected_tasks)
        status["control_error"] = control_error
        if control_error:
            status["stop_requested"] = False
        _atomic_json(self.output_root / "state.json", {"models": self.records})
        _atomic_json(self.output_root / "status.json", status)
        return status


def run_controller(panel_root: Path, output_root: Path, control_path: Path, *,
                   poll_seconds: float = 1.0) -> dict:
    """Remain available for additions until stop=true; never terminate a forecast."""
    if poll_seconds < 1:
        raise ValueError("poll_seconds must be at least one second")
    with exclusive_lock(Path(output_root) / ".controller.lock"):
        controller = SweepController(panel_root, output_root, control_path)
        while True:
            status = controller.poll()
            if status["stop_requested"] and not controller.active:
                return status
            time.sleep(poll_seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--status", action="store_true", help="print current status without running work")
    args = parser.parse_args(argv)
    if args.status:
        print((args.output_root / "status.json").read_text(), end="")
    else:
        if args.panel_root is None or args.control is None:
            parser.error("--panel-root and --control are required to run the controller")
        run_controller(args.panel_root, args.output_root, args.control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
