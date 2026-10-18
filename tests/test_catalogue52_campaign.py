"""Bounded process lifecycle tests and fail-closed single52 publication gates.

The lifecycle subprocess is a tiny control fixture, never a forecasting run.
The scoring checks use the real origin-first block reducer with synthetic loss
vectors; they do not claim model fidelity or benchmark evidence.
"""
from __future__ import annotations

import io
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/catalogue52"))
import campaign
import worker

CONTROL_WORKER = '''
import argparse, json, os, signal, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--registration'); p.add_argument('--token'); p.add_argument('--nice')
a = p.parse_args(); out = Path(a.registration).parent
stopping = False
def stop(signum, frame):
    global stopping
    stopping = True
signal.signal(signal.SIGUSR1, stop)
signal.signal(signal.SIGTERM, stop)
def atomic(name, value):
    path = out / name; temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value)); temp.replace(path)
atomic('ready.json', {'token': a.token, 'pid': os.getpid()})
completed = len(list(out.glob('fixture-task-*.json')))
for index in range(completed, 8):
    if stopping:
        break
    atomic('active.json', {'index': index})
    time.sleep(0.15)
    atomic('fixture-task-%04d.json' % index, {'index': index})
    completed += 1
    atomic('progress.json', {'status': 'running', 'completed_tasks': completed})
atomic('progress.json', {'status': 'complete' if completed == 8 else 'paused',
                         'completed_tasks': completed})
'''


@pytest.fixture
def controlled(tmp_path):
    source = tmp_path / "source"
    (source / "tools/catalogue52").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "src/numerical.py").write_text("VERSION = 1\n")
    (source / "tools/catalogue52/worker.py").write_text(CONTROL_WORKER)
    data = tmp_path / "data"
    data.mkdir()
    (data / "canonical_data_manifest.json").write_text("{}")
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"python": "test-only", "packages": {"numpy": "test-only"}}))
    root = tmp_path / "campaign"
    campaign.initialize(root, data, environment, min_free_gib=0, budget_gib=1)
    controller = campaign.Controller(root, max_models=2, python=sys.executable, nice=0)
    yield root, source, controller
    # Only this fixture's verified process groups may be signaled.
    for process_file in (root / "models").glob("*/process.json"):
        state = campaign.read(process_file)
        campaign.request_stop(state)
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if not any(campaign.verified_process(campaign.read(path))
                   for path in (root / "models").glob("*/process.json")):
            break
        for child in controller.children.values():
            child.poll()
        time.sleep(0.02)
    for process_file in (root / "models").glob("*/process.json"):
        state = campaign.read(process_file)
        if campaign.verified_process(state):
            os.killpg(state["pid"], signal.SIGKILL)
    for child in controller.children.values():
        child.wait(timeout=3)


def await_condition(controller, predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        controller.tick()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("control fixture did not reach expected state")


def test_live_add_remove_resume_without_restarting_other_models(controlled):
    root, source, controller = controlled
    campaign.add_model(root, "first", source_root=source)
    campaign.add_model(root, "second", source_root=source)
    controller.tick()
    first = campaign.read(root / "models/first/process.json")
    second = campaign.read(root / "models/second/process.json")
    await_condition(controller, lambda: (root / "models/first/active.json").exists()
                    and (root / "models/second/active.json").exists())
    campaign.add_model(root, "late", source_root=source)
    controller.tick()
    assert not (root / "models/late/process.json").exists()  # Bounded concurrency.
    campaign.remove_model(root, "first")
    await_condition(controller, lambda: (root / "models/late/process.json").exists())
    assert campaign.read(root / "models/first/progress.json")["status"] == "paused"
    partial = list((root / "models/first").glob("fixture-task-*.json"))
    assert 0 < len(partial) < 8  # In-flight task was checkpointed before stopping.
    assert campaign.read(root / "models/second/process.json")["pid"] == second["pid"]
    assert campaign.read(root / "models/second/process.json")["token"] == second["token"]
    campaign.add_model(root, "first", source_root=source)
    await_condition(controller, lambda: campaign.read(root / "models/first/process.json")["token"] != first["token"])
    await_condition(controller, lambda: all(
        (root / "models" / name / "progress.json").exists()
        and campaign.read(root / "models" / name / "progress.json")["status"] == "complete"
        for name in ("first", "second", "late")))
    assert len(list((root / "models/late").glob("fixture-task-*.json"))) == 8
    assert len(list((root / "models/first").glob("fixture-task-*.json"))) == 8


def test_source_is_pinned_per_model_and_resume_ignores_live_edits(controlled):
    root, source, _ = controlled
    first = campaign.add_model(root, "first", source_root=source)
    (source / "src/numerical.py").write_text("VERSION = 2\n")
    second = campaign.add_model(root, "second", source_root=source)
    assert first["source_sha256"] != second["source_sha256"]
    assert (Path(first["snapshot"]) / "src/numerical.py").read_text() == "VERSION = 1\n"
    assert campaign.add_model(root, "first", source_root=source) == first
    with pytest.raises(ValueError, match="binding differs"):
        campaign.add_model(root, "first", environment={"SIMFOLIO_VARIANT": "changed"}, source_root=source)
    target = Path(first["snapshot"]) / "src/numerical.py"
    target.chmod(0o644)
    target.write_text("VERSION = 3\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        campaign.verify_snapshot(first["snapshot"], first["source_sha256"])


def test_controller_restart_adopts_verified_pid_and_rejects_forged_identity(controlled):
    root, source, controller = controlled
    campaign.add_model(root, "first", source_root=source)
    controller.tick()
    await_condition(controller, lambda: (root / "models/first/active.json").exists())
    original = campaign.read(root / "models/first/process.json")
    adopted = campaign.Controller(root, max_models=2, python=sys.executable, nice=0)
    adopted.tick()
    assert campaign.read(root / "models/first/process.json") == original
    forged = dict(original, token="different-token")
    assert not campaign.verified_process(forged)
    assert not campaign.request_stop(forged)
    assert campaign.verified_process(original)


def test_crash_retains_checkpoints_and_requires_explicit_resume(controlled):
    root, source, controller = controlled
    campaign.add_model(root, "first", source_root=source)
    await_condition(controller, lambda: (root / "models/first/fixture-task-0000.json").exists())
    state = campaign.read(root / "models/first/process.json")
    assert campaign.verified_process(state)
    os.killpg(state["pid"], signal.SIGKILL)
    controller.children["first"].wait(timeout=3)
    controller.tick()
    assert campaign.read(root / "models/first/process.json")["token"] == state["token"]
    retained = (root / "models/first/fixture-task-0000.json").read_bytes()
    campaign.add_model(root, "first", source_root=source)
    await_condition(controller, lambda: campaign.read(root / "models/first/progress.json")["status"] == "complete")
    assert (root / "models/first/fixture-task-0000.json").read_bytes() == retained


def test_checkpoint_binding_checksum_and_shape_fail_closed(tmp_path):
    task = SimpleNamespace(horizon_days=2, portfolio_id="single_TEST", origin_date="2000-01-01", seed=7)
    worker.write_checkpoint(tmp_path, 0, "binding", task, np.array([1., 2.]), 1., None)
    np.testing.assert_array_equal(worker.checkpoint_losses(tmp_path, 0, "binding", task), [1., 2.])
    with pytest.raises(ValueError, match="identity mismatch"):
        worker.checkpoint_losses(tmp_path, 0, "other-binding", task)
    with pytest.raises(ValueError, match="shape/finiteness"):
        worker.checkpoint_losses(tmp_path, 0, "binding", SimpleNamespace(horizon_days=3))
    path = tmp_path / "task-0000.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        worker.checkpoint_losses(tmp_path, 0, "binding", task)


def test_partial_and_smoke_can_never_publish_full_score(tmp_path, monkeypatch):
    assert worker.finalize(tmp_path, [], "binding", smoke=True) is None
    with pytest.raises(ValueError, match="task denominator"):
        worker.finalize(tmp_path, [SimpleNamespace(horizon_days=1)], "binding")
    tasks = [SimpleNamespace(horizon_days=8766)] * 2652
    with pytest.raises(ValueError, match="incomplete single52"):
        worker.finalize(tmp_path, tasks, "binding")
    assert not (tmp_path / "results.json").exists()


def test_full_reduction_requires_exact_52_by_8766_cells(tmp_path, monkeypatch):
    tasks = [SimpleNamespace(portfolio_id=f"single_{asset:02d}", horizon_days=8766)
             for asset in range(52) for origin in range(51)]
    vector = np.ones(8766)
    monkeypatch.setattr(worker, "checkpoint_losses", lambda *args: vector)
    monkeypatch.setattr(worker, "read", lambda path: {"seconds": 1.0})
    result = worker.finalize(tmp_path, tasks, "binding")
    assert result["tasks"] == 2652 and result["cells"] == 455832
    assert result["exact_empirical_crps"] == 1.0
    with np.load(tmp_path / "cells.npz") as cells:
        assert np.all(cells["counts"] == 51)
    (tmp_path / "results.json").unlink()
    short = np.ones(8765)
    monkeypatch.setattr(worker, "checkpoint_losses", lambda *args: short)
    for task in tasks:
        task.horizon_days = 8765
    with pytest.raises(ValueError, match="cell denominator"):
        worker.finalize(tmp_path, tasks, "binding")
    assert not (tmp_path / "results.json").exists()


def test_saved_environment_and_override_allowlist_are_enforced(controlled):
    root, source, _ = controlled
    with pytest.raises(ValueError, match="Python version"):
        worker.verify_environment({"python": "0.0.0", "packages": {}})
    with pytest.raises(ValueError, match="SIMFOLIO_"):
        campaign.add_model(root, "invalid", environment={"PYTHONPATH": "mutable"}, source_root=source)
    with pytest.raises(ValueError, match="fit caches"):
        campaign.add_model(root, "invalid", environment={"SIMFOLIO_RESEARCH_FIT_CACHE": "cache"}, source_root=source)
    with pytest.raises(ValueError, match="smoke indices"):
        campaign.add_model(root, "invalid", indices=[0, 0], source_root=source)


def test_worker_stops_at_checkpoint_then_resumes_native_seed_and_240_paths(controlled, monkeypatch):
    import protocol

    from simfolio_forecasting_methodology.runner import OriginTask, TrainingData

    root, source, _ = controlled
    spec = campaign.add_model(root, "fixture", source_root=source, indices=[0, 1])
    training = TrainingData(portfolio_log_returns=np.array([.01, -.01, .005]))
    tasks = tuple(OriginTask("single_FIXTURE", f"origin_{i}", training,
                             np.array([.002, -.003]), seed=100 + i)
                  for i in range(2652))
    calls = []
    class Model:
        model_id = "fixture"
        seed_contract = "fixture-native-seed"
        def simulate_daily_log_returns(self, training, context):
            calls.append((context.seed, context.simulations, context.horizon_days))
            return np.zeros((context.simulations, context.horizon_days))
    monkeypatch.setattr(protocol, "single_asset_tasks", lambda data: tasks)
    monkeypatch.setattr(worker, "ROOT", Path(spec["snapshot"]))
    monkeypatch.setattr(worker, "verify_environment", lambda expected: None)
    monkeypatch.setattr(worker, "load_model", lambda spec: (Model(), None))
    output = root / "models/fixture"
    worker.execute(output / "registration.json", stop_requested=lambda: bool(calls))
    assert campaign.read(output / "progress.json")["status"] == "paused"
    assert calls == [(100, 240, 2)]
    checkpoint = (output / "task-0000.npz").read_bytes()
    worker.execute(output / "registration.json", stop_requested=lambda: False)
    assert calls == [(100, 240, 2), (101, 240, 2)]
    assert (output / "task-0000.npz").read_bytes() == checkpoint
    assert campaign.read(output / "progress.json")["status"] == "smoke_complete"
    assert campaign.read(output / "manifest.json")["native_seed_contract"] == "fixture-native-seed"
    assert not (output / "results.json").exists()


def test_force_removal_only_signals_verified_group_after_explicit_request(controlled, monkeypatch):
    root, _, _ = controlled
    output = root / "force-fixture"
    output.mkdir()
    campaign.atomic(output / "progress.json", {"current_task": 13, "completed_tasks": 12})
    signals = []
    monkeypatch.setattr(campaign, "verified_process", lambda state: state["token"] == "verified")
    monkeypatch.setattr(campaign.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    state = {"pid": 321, "token": "verified"}
    assert not campaign.force_stop(dict(state, token="unrelated"), output, "explicit removal", now=100)
    assert not signals
    assert campaign.force_stop(state, output, "explicit removal", now=100)
    assert signals == [(321, signal.SIGTERM)]
    campaign.force_stop(state, output, "explicit removal", now=114)
    assert len(signals) == 1
    campaign.force_stop(state, output, "explicit removal", now=115)
    assert signals == [(321, signal.SIGTERM), (321, signal.SIGKILL)]
    receipt = campaign.read(output / "interruption.json")
    assert receipt["interrupted_task"] == 13 and receipt["completed_tasks"] == 12
    assert receipt["full_score_eligible"] is False
    with pytest.raises(ValueError, match="explicit reason"):
        campaign.remove_model(root, "anything", force=True)


def test_shared_task_cache_reuses_exact_closure_and_rejects_corruption(controlled, monkeypatch):
    import protocol

    from simfolio_forecasting_methodology.runner import OriginTask, TrainingData

    root, source, _ = controlled
    dates = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    training = TrainingData(np.array([.01, -.01]), np.array([[.01], [-.01]]),
                            training_dates=dates)
    # A policy is required when asset-level inputs are present.
    from simfolio_forecasting_methodology.runner import PortfolioPolicy
    training = TrainingData(training.portfolio_log_returns, training.asset_log_returns,
                            PortfolioPolicy(("FIXTURE",), (1.0,), "none"), dates)
    tasks = tuple(OriginTask("single_FIXTURE", f"origin_{i}", training,
                             np.array([.002]), seed=i, future_dates=dates[:1])
                  for i in range(2652))
    calls = []
    def materialize(data):
        calls.append(str(data))
        return tasks
    monkeypatch.setattr(protocol, "single_asset_tasks", materialize)
    first = campaign.add_model(root, "first", source_root=source)
    monkeypatch.setattr(worker, "ROOT", Path(first["snapshot"]))
    config = campaign.read(root / "campaign.json")
    restored, receipt = worker.shared_tasks(root, first, config)
    assert receipt["parity"] == {"all_task_identities_equal": True,
                                  "all_float_date_arrays_byte_equal": True,
                                  "all_strides_and_alias_groups_equal": True,
                                  "tasks": 2652, "arrays": 13260}
    assert len(restored) == 2652 and len(calls) == 1
    (source / "tools/new_model.py").write_text("NEW_ADAPTER = True\n")
    second = campaign.add_model(root, "second", source_root=source)
    assert second["source_sha256"] != first["source_sha256"]
    monkeypatch.setattr(worker, "ROOT", Path(second["snapshot"]))
    _, reused = worker.shared_tasks(root, second, config)
    assert reused == receipt and len(calls) == 1
    payload = root / "input-cache" / receipt["sha256"] / "tasks.pickle"
    payload.chmod(0o644)
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="payload checksum"):
        worker.shared_tasks(root, second, config)
    assert len(calls) == 1


def test_shared_task_cache_invalidates_changed_prepared_bytes(controlled, monkeypatch):
    import protocol

    from simfolio_forecasting_methodology.runner import OriginTask, TrainingData

    root, source, _ = controlled
    tasks = tuple(OriginTask("single_FIXTURE", f"origin_{i}",
                             TrainingData(np.array([.01])), np.array([.002]), seed=i)
                  for i in range(2652))
    calls = []
    monkeypatch.setattr(protocol, "single_asset_tasks", lambda data: calls.append(data) or tasks)
    spec = campaign.add_model(root, "first", source_root=source)
    monkeypatch.setattr(worker, "ROOT", Path(spec["snapshot"]))
    config = campaign.read(root / "campaign.json")
    _, first = worker.shared_tasks(root, spec, config)
    extra = Path(config["data"]) / "frozen-values.bin"
    extra.write_bytes(b"new prepared source bytes")
    _, second = worker.shared_tasks(root, spec, config)
    assert first["sha256"] != second["sha256"] and len(calls) == 2
    assert first["payload_sha256"] == second["payload_sha256"]


def test_task_cache_preserves_strided_views_aliases_and_writeability():
    root = np.arange(24, dtype=np.float64).reshape(6, 4)
    views = (root[:, 1], root[1:5, 1:3], root[::-1, 2], root.T)
    views[0].flags.writeable = False
    stream = io.BytesIO()
    worker.dump_task_views(views, stream)
    stream.seek(0)
    restored = worker.load_task_views(stream)
    for before, after in zip(views, restored):
        assert before.strides == after.strides
        assert before.flags.writeable == after.flags.writeable
        assert before.tobytes() == after.tobytes()
    assert np.shares_memory(restored[0], restored[1])
    restored[1][0, 0] = -100
    assert restored[0][1] == -100


def test_archived_model_requires_restoration_before_add_or_worker(controlled):
    root, source, _ = controlled
    campaign.add_model(root, "first", source_root=source)
    output = root / "models/first"
    campaign.atomic(output / "archive-receipt.json", {"status": "archived"})
    with pytest.raises(ValueError, match="explicit restoration"):
        campaign.add_model(root, "first", source_root=source)
    with pytest.raises(ValueError, match="explicit restoration"):
        worker.execute(output / "registration.json")
