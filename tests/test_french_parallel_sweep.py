"""Synthetic CPU-slot and exact loss/score oracles; no native research fitting."""
import json
from pathlib import Path
from dataclasses import astuple
import sys
import time

import numpy as np
import pytest
from test_french_panel_runner import ZeroOracle, panel  # noqa: F401
from simfolio_forecasting_methodology.french_panel import runner

# Make the standalone script importable both in pytest and spawned children.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import french_parallel_sweep as parallel  # noqa: E402


def spec(factory="synthetic_parallel:make_model"):
    return {"enabled": True, "factory": factory, "implementation_digest": "a" * 64}


def control(path, models, workers=3, stop=False):
    path.write_text(json.dumps({"models": models, "workers": workers, "stop": stop}))


def test_manifest_and_reverse_completion_losses_scores_exactly_match_serial(panel):
    mid = ZeroOracle.model_id
    serial = runner.run_panel(ZeroOracle(), panel[0], panel[1], implementation_digest="a" * 64)
    root = panel[1].parent / "parallel"
    fingerprint = parallel.prepare_manifest(ZeroOracle(), spec(), panel[5], panel[4], root)
    assert (root / "execution.json").read_bytes() == (panel[1] / "execution.json").read_bytes()
    item = parallel.ModelProgress(mid, spec(), panel[5], panel[4], root)
    item.scan(fingerprint)
    for task in reversed(list(item.queue)):
        parallel.execute_task(ZeroOracle(), panel[5], panel[3], panel[2], panel[4], root, fingerprint, task)
        item.accept(task)
        expected = panel[1] / "tasks" / f"{task[0]:06d}/{task[1]:03d}.npz"
        actual = parallel.task_paths(root, task)[0]
        with np.load(expected) as a, np.load(actual) as b:
            assert a["losses"].tobytes() == b["losses"].tobytes()
            assert json.loads(str(a["metadata"].item()))["payload"]["identity"] == json.loads(str(b["metadata"].item()))["payload"]["identity"]
    summary = item.summary()
    assert summary["final_score"] == serial["final_score"]
    assert summary["complete_series"] == serial["complete_series"]
    saved = {p: p.read_bytes() for p in root.rglob("*.npz")}
    resumed = parallel.ModelProgress(mid, spec(), panel[5], panel[4], root)
    resumed.scan(fingerprint)
    assert resumed.resumed == 4 and not resumed.queue
    assert resumed.summary()["final_score"] == serial["final_score"]
    assert all(p.read_bytes() == data for p, data in saved.items())


class FakeConnection:
    def __init__(self):
        self.events, self.sent = [], []
    def poll(self):
        return bool(self.events)
    def recv(self):
        return self.events.pop(0)
    def send(self, value):
        self.sent.append(value)
    def close(self):
        pass


class FakeProcess:
    alive = True
    exitcode = 0
    def is_alive(self):
        return self.alive
    def join(self):
        pass


@pytest.fixture
def controller(panel, tmp_path):
    path = tmp_path / "control.json"
    control(path, {"one": spec(), "two": spec()}, workers=3)
    spawned = []
    def starter(*args):
        process, connection = FakeProcess(), FakeConnection()
        spawned.append((args, process, connection))
        return process, connection
    instance = parallel.ParallelController(panel[0], tmp_path / "outputs", path, starter=starter,
        estimates={"one": {"mean_origin_extrapolation_worker_seconds": 100},
                   "two": {"mean_origin_extrapolation_worker_seconds": 10}})
    return instance, path, spawned


def ready(instance, spawned, panel):
    for args, _, connection in spawned:
        mid, selected, _, root, *_ = args
        model = ZeroOracle()
        model.model_id = mid
        fingerprint = parallel.prepare_manifest(model, selected, panel[5], panel[4], root)
        connection.events.append(("ready", fingerprint))


def complete_slot(instance, key, panel):
    slot = instance.active[key]
    item = instance.progress[slot["mid"]]
    model = ZeroOracle()
    model.model_id = item.mid
    task = slot["task"]
    parallel.execute_task(model, panel[5], panel[3], panel[2], panel[4],
                          item.root, item.fingerprint, task)
    slot["connection"].events.append(("done", task))


def test_work_conservation_initial_admission_and_lending(controller, panel):
    instance, _, spawned = controller
    status = instance.poll()
    assert status["active_workers"] == 3
    assert {args[0] for args, *_ in spawned} == {"one", "two"}
    ready(instance, spawned, panel)
    instance.poll()
    tasks = [(s["mid"], s["task"]) for s in instance.active.values()]
    assert len(tasks) == len(set(tasks)) == 3
    # Exhaust the cheap model first. Its freed slot must lend to the costly one.
    cheap_key = next(k for k, s in instance.active.items() if s["mid"] == "two")
    while instance.progress["two"].queue or instance.active[cheap_key]["task"]:
        complete_slot(instance, cheap_key, panel)
        instance.poll()
        if instance.active[cheap_key]["stopping"]:
            break
    instance.active[cheap_key]["process"].alive = False
    instance.poll()
    assert len(instance.active) == 3
    assert all(slot["mid"] == "one" for slot in instance.active.values())
    # A single queued task can take a new slot while other workers are busy.
    assert len(spawned) == 4


def test_pause_remove_resume_and_no_duplicate_inflight(controller, panel):
    instance, path, spawned = controller
    instance.poll()
    ready(instance, spawned, panel)
    instance.poll()
    checkpoint_data = {}
    control(path, {"one": spec()}, workers=2, stop=True)
    for key in list(instance.active):
        complete_slot(instance, key, panel)
    instance.poll()
    assert all(s["stopping"] for s in instance.active.values())
    for _, process, _ in spawned:
        process.alive = False
    assert instance.poll()["active_workers"] == 0
    for p in instance.output_root.rglob("*.npz"):
        checkpoint_data[p] = p.read_bytes()
    control(path, {"one": spec(), "two": spec()}, workers=3)
    instance.poll()
    assert len(instance.active) == 3
    assert all(p.read_bytes() == data for p, data in checkpoint_data.items())
    assigned = [(s["mid"], s["task"]) for s in instance.active.values() if s["task"]]
    assert len(assigned) == len(set(assigned))


def test_worker_crash_recovers_saved_checkpoint_else_records_one_failure(controller, panel):
    instance, _, spawned = controller
    instance.poll()
    ready(instance, spawned, panel)
    instance.poll()
    keys = list(instance.active)
    complete_slot(instance, keys[0], panel)
    # Lose the event, as if the worker died just after its atomic save.
    instance.active[keys[0]]["connection"].events.clear()
    saved_mid = instance.active[keys[0]]["mid"]
    saved_task = instance.active[keys[0]]["task"]
    instance.active[keys[0]]["process"].alive = False
    lost_mid = instance.active[keys[1]]["mid"]
    lost_task = instance.active[keys[1]]["task"]
    instance.active[keys[1]]["process"].alive = False
    instance.poll()
    assert saved_task in instance.progress[saved_mid].completed
    assert lost_task in instance.progress[lost_mid].failures
    assert lost_task not in instance.progress[lost_mid].queue
    assert parallel.task_paths(instance.progress[lost_mid].root, lost_task)[1].exists()
    assert instance.progress[lost_mid].summary()["final_score"] is None


def test_identity_corruption_and_controls_fail_closed(controller, panel):
    instance, path, spawned = controller
    instance.poll()
    ready(instance, spawned, panel)
    instance.poll()
    path.write_text('{"models":')
    for key in list(instance.active):
        complete_slot(instance, key, panel)
    status = instance.poll()
    assert status["control_error"] and all(s["stopping"] for s in instance.active.values())
    assert not parallel._continuing(path, "one", spec())
    changed = {**spec(), "implementation_digest": "b" * 64}
    control(path, {"one": changed})
    assert instance.poll()["models"]["one"]["state"] == "failed"


def test_origin_failure_and_reuse_do_not_refit(panel):
    class Fail(ZeroOracle):
        def simulate_daily_log_returns(self, *_):
            raise RuntimeError("synthetic failure")
    root = panel[1]
    fingerprint = parallel.prepare_manifest(Fail(), spec(), panel[5], panel[4], root)
    args = (panel[5], panel[3], panel[2], panel[4], root, fingerprint, (0, 0))
    assert parallel.execute_task(Fail(), *args) == "failed"
    assert parallel.task_paths(root, (0, 0))[1].exists()
    assert parallel.execute_task(ZeroOracle(), *args) == "done"
    assert not parallel.task_paths(root, (0, 0))[1].exists()
    saved = parallel.task_paths(root, (0, 0))[0].read_bytes()
    assert parallel.execute_task(Fail(), *args) == "done"
    assert parallel.task_paths(root, (0, 0))[0].read_bytes() == saved


def test_real_spawn_parallel_checkpoint_reuse_and_bitwise_score(panel, tmp_path, monkeypatch):
    # A tiny child-importable adapter patches only test data loaders. Production
    # still uses the original immutable data and methodology factory unchanged.
    module = tmp_path / "synthetic_parallel.py"
    origins = repr([astuple(o) for o in panel[4]])
    module.write_text('''import numpy as np
from simfolio_forecasting_methodology.french_panel import runner
from simfolio_forecasting_methodology.origins import OriginDescriptor
class Oracle:
    model_id = "synthetic-zero-oracle"
    def simulate_daily_log_returns(self, training, context):
        return np.zeros((context.simulations, context.horizon_days))
def make_model():
    runner.load_panel = lambda _: ''' + repr(panel[5]) + '''
    runner.load_dates = lambda _: np.array(''' + repr(panel[3].astype(str).tolist()) + ''', dtype="datetime64[D]")
    runner.load_returns = lambda _: np.array(''' + repr(panel[2].tolist()) + ''')
    runner.schedule = lambda _: tuple(OriginDescriptor(*o) for o in ''' + origins + ''')
    runner._source_closure = lambda: {"source": "synthetic-frozen"}
    runner._dependency_identity = lambda *_: {"runtime": "synthetic"}
    return Oracle()
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "control.json"
    control(path, {ZeroOracle.model_id: spec()}, workers=3)
    # Seed a genuine original-runner checkpoint, then verify parallel skips it.
    root = parallel.model_directory(tmp_path / "spawn", ZeroOracle.model_id)
    runner.run_panel(ZeroOracle(), panel[0], root, implementation_digest="a" * 64, max_tasks=1)
    saved_path = root / "tasks/000000/000.npz"
    saved = saved_path.read_bytes()
    serial = runner.run_panel(ZeroOracle(), panel[0], panel[1], implementation_digest="a" * 64)
    instance = parallel.ParallelController(panel[0], tmp_path / "spawn", path)
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            status = instance.poll()
            if status["models"][ZeroOracle.model_id]["status"] == "complete":
                break
            time.sleep(.02)
        else:
            pytest.fail(f"spawn completion timeout: {status}")
        actual = status["models"][ZeroOracle.model_id]
        assert actual["final_score"] == serial["final_score"]
        assert actual["completed_tasks"] == 4 and actual["resumed_tasks"] == 1
        assert saved_path.read_bytes() == saved
        for p in root.rglob("*.npz"):
            with np.load(p) as a, np.load(panel[1] / p.relative_to(root)) as b:
                assert a["losses"].tobytes() == b["losses"].tobytes()
    finally:
        control(path, {ZeroOracle.model_id: spec()}, stop=True)
        while instance.active and time.monotonic() < deadline:
            instance.poll()
            time.sleep(.02)
        for slot in instance.active.values():
            slot["process"].terminate()
            slot["process"].join()


def test_restart_scans_only_missing_and_corruption_never_overwritten(panel):
    root = panel[1]
    model = ZeroOracle()
    saved = runner.run_panel(model, panel[0], root, implementation_digest="a" * 64, max_tasks=2)
    fingerprint = saved["execution_fingerprint"]
    checkpoint = root / "tasks/000000/000.npz"
    before = checkpoint.read_bytes()
    item = parallel.ModelProgress(model.model_id, spec(), panel[5], panel[4], root)
    item.scan(fingerprint)
    assert list(item.queue) == [(1, 0), (1, 1)]
    assert len(item.completed) == item.resumed == 2
    assert checkpoint.read_bytes() == before
    checkpoint.write_bytes(before[:-10])
    corrupt = checkpoint.read_bytes()
    from simfolio_forecasting_methodology.results.checkpoint import CheckpointCorruptError
    with pytest.raises(CheckpointCorruptError):
        parallel.ModelProgress(model.model_id, spec(), panel[5], panel[4], root).scan(fingerprint)
    assert checkpoint.read_bytes() == corrupt


def test_initialization_crash_fails_closed_without_restart_loop(controller):
    instance, _, spawned = controller
    instance.poll()
    first = next(iter(instance.active.values()))
    mid = first["mid"]
    first["process"].alive = False
    first["process"].exitcode = 99
    status = instance.poll()
    assert status["models"][mid]["state"] == "failed"
    count = len(spawned)
    instance.poll()
    assert len(spawned) == count


def test_coordinator_and_orphan_worker_locks_prevent_duplicate_parent(tmp_path):
    import fcntl
    lock = tmp_path / ".parallel-workers.lock"
    lock.touch()
    with lock.open("a+") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_SH)
        with pytest.raises(RuntimeError, match="another process"):
            with parallel.exclusive_lock(lock):
                pytest.fail("orphan worker lock was ignored")
    with parallel.exclusive_lock(lock):
        pass


def test_matched_complete_series_delta_uses_identical_cells(panel, tmp_path):
    path = tmp_path / "control.json"
    models = {parallel.BASELINE: spec(), "candidate": spec()}
    control(path, models, stop=True)
    instance = parallel.ParallelController(panel[0], tmp_path / "out", path)
    for mid in models:
        instance.progress[mid] = parallel.ModelProgress(mid, models[mid], panel[5], panel[4],
                                                        parallel.model_directory(instance.output_root, mid))
    instance.progress[parallel.BASELINE].series_sums = {0: 4., 1: 40.}
    instance.progress["candidate"].series_sums = {1: 20.}
    status = instance.poll()
    assert status["models"]["candidate"]["matched_m261_complete_series"] == 1
    assert status["models"]["candidate"]["matched_m261_series_ids"] == ["s1"]
    assert status["models"]["candidate"]["percent_better_m261_matching_series"] == 50.
    assert status["models"]["candidate"]["final_score"] is None
