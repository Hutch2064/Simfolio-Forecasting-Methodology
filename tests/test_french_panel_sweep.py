"""Synthetic control, checkpoint, and timing oracles; no research model is run."""

import json
from types import SimpleNamespace

import pytest
from test_french_panel_runner import ZeroOracle, panel  # noqa: F401 - shared synthetic fixture

from simfolio_forecasting_methodology.french_panel import runner, sweep


def spec(enabled=True, digest="a" * 64):
    return {"enabled": enabled, "factory": "synthetic_adapter:make_model",
            "implementation_digest": digest}


def write_control(path, models=None, workers=2, stop=False):
    value = {"models": models or {}, "workers": workers, "stop": stop}
    path.write_text(json.dumps(value))
    return value


class FakeProcess:
    def __init__(self):
        self.alive = True
        self.joined = False
        self.exitcode = 0

    def is_alive(self):
        return self.alive

    def join(self):
        self.joined = True


@pytest.fixture
def controller(tmp_path):
    control = tmp_path / "control.json"
    write_control(control)
    calls = []

    def start(*args):
        process = FakeProcess()
        calls.append((args, process))
        return process

    instance = sweep.SweepController(tmp_path / "panel", tmp_path / "out", control,
                                     expected_tasks=10, start_worker=start)
    return instance, control, calls


def finish(controller, model_id, *, state="completed", seconds=100, tasks=10):
    process = controller.active[model_id]
    result = {"state": state, "result": {
        "status": "complete" if state == "completed" else "incomplete",
        "completed_tasks": tasks,
        "completed_forecast_and_score_seconds": seconds,
        "final_score": .25 if state == "completed" else None,
    }}
    sweep._atomic_json(sweep.model_directory(controller.output_root, model_id) / "worker-result.json", result)
    process.alive = False
    return process


def test_empty_control_never_starts_or_imports_model(controller, monkeypatch):
    instance, _, calls = controller
    monkeypatch.setattr(sweep.importlib, "import_module", lambda _: pytest.fail("unexpected import"))
    status = instance.poll()
    assert not calls
    assert status["counts"] == dict.fromkeys(
        ("active", "waiting", "paused", "completed", "failed"), 0)
    assert status["eta"]["mean_seconds"] is None
    assert status["eta"]["median_seconds"] is None
    assert json.loads((instance.output_root / "status.json").read_text()) == status


@pytest.mark.parametrize("models,workers,stop", [
    ({"": spec()}, 1, False), ({"control\ncharacter": spec()}, 1, False),
    ({"safe": spec(digest="not-a-sha")}, 1, False),
    ({"safe": {**spec(), "factory": "not:a.factory"}}, 1, False),
    ({"safe": {**spec(), "enabled": 1}}, 1, False),
    ({}, 0, False), ({}, 13, False), ({}, True, False), ({}, 1, "false"),
])
def test_invalid_controls_fail_before_scheduling(tmp_path, models, workers, stop):
    path = tmp_path / "control.json"
    write_control(path, models, workers, stop)
    with pytest.raises(ValueError):
        sweep.read_control(path)


def test_live_add_remove_pause_resume_and_complete_no_restart(controller):
    instance, control, calls = controller
    write_control(control, {"one": spec(), "two": spec(), "three": spec()})
    assert instance.poll()["counts"] == {
        "active": 2, "waiting": 1, "paused": 0, "completed": 0, "failed": 0}
    first_process = instance.active["one"]
    second_process = instance.active["two"]
    write_control(control, {"one": spec(), "three": spec(), "four": spec()})
    instance.poll()
    assert instance.active["one"] is first_process
    assert instance.active["two"] is second_process
    assert not sweep.worker_should_continue(control, instance.admission_path, "two", sweep._identity(spec()))
    finish(instance, "two", state="paused", seconds=25, tasks=2)
    status = instance.poll()
    assert status["models"]["two"]["state"] == "paused"
    assert status["models"]["two"]["completed_tasks"] == 2
    assert set(instance.active) == {"one", "three"}
    finish(instance, "one")
    status = instance.poll()
    assert status["models"]["one"]["state"] == "completed"
    assert set(instance.active) == {"three", "four"}
    assert first_process.joined
    # Reinsert an excluded model; it resumes after a slot is free.
    write_control(control, {"one": spec(), "two": spec(), "three": spec(), "four": spec()})
    finish(instance, "three")
    instance.poll()
    assert "two" in instance.active
    assert sum(args[0] == "one" for args, _ in calls) == 1


def test_disable_stop_and_worker_reduction_revoke_next_origin_admission(controller):
    instance, control, _ = controller
    write_control(control, {"one": spec(), "two": spec()}, workers=2)
    instance.poll()
    write_control(control, {"one": spec(), "two": spec()}, workers=1)
    instance.poll()
    assert sweep.worker_should_continue(control, instance.admission_path, "one", sweep._identity(spec()))
    assert not sweep.worker_should_continue(control, instance.admission_path, "two", sweep._identity(spec()))
    write_control(control, {"one": spec(False), "two": spec()}, workers=1, stop=True)
    instance.poll()
    assert not sweep.worker_should_continue(control, instance.admission_path, "one", sweep._identity(spec()))
    assert not sweep.worker_should_continue(control, instance.admission_path, "two", sweep._identity(spec()))
    finish(instance, "one", state="paused", seconds=2, tasks=1)
    finish(instance, "two", state="paused", seconds=3, tasks=1)
    status = instance.poll()
    assert status["stop_requested"]
    assert not instance.active
    assert status["counts"]["paused"] == 2


@pytest.mark.parametrize("change", ["implementation_digest", "factory"])
def test_identity_change_fails_closed_and_preserves_old_results(controller, change):
    instance, control, calls = controller
    write_control(control, {"one": spec()})
    instance.poll()
    finish(instance, "one", state="paused", seconds=50, tasks=5)
    write_control(control, {"one": spec(False)})
    instance.poll()
    changed = spec()
    changed[change] = "b" * 64 if change == "implementation_digest" else "other:make_model"
    write_control(control, {"one": changed})
    status = instance.poll()
    assert status["models"]["one"]["state"] == "failed"
    assert status["models"]["one"]["completed_tasks"] == 5
    assert len(calls) == 1
    assert "changed" in status["models"]["one"]["error"]
    restarted = sweep.SweepController(instance.panel_root, instance.output_root, control,
                                      expected_tasks=10, start_worker=instance.start_worker)
    assert restarted.poll()["models"]["one"]["state"] == "failed"
    assert len(calls) == 1


def test_live_progress_survives_resume_scanning_and_ranking_uses_complete_only(controller):
    instance, control, _ = controller
    write_control(control, {"one": spec()})
    instance.poll()
    root = sweep.model_directory(instance.output_root, "one")
    sweep._atomic_json(root / "progress.json", {
        "completed_tasks": 4, "completed_forecast_and_score_seconds": 15, "final_score": .1})
    status = instance.poll()
    assert status["models"]["one"]["completed_tasks"] == 4
    assert status["ranking"] == []
    assert status["eta"]["mean_seconds"] is None
    sweep._atomic_json(root / "progress.json", {
        "completed_tasks": 2, "completed_forecast_and_score_seconds": 7})
    assert instance.poll()["models"]["one"]["completed_tasks"] == 4
    finish(instance, "one")
    assert instance.poll()["ranking"] == [{"model_id": "one", "final_score": .25}]


def test_mean_and_median_eta_use_completed_model_compute_not_wall_time():
    records = {}
    for model_id, state, seconds, tasks in [
        ("a", "completed", 100, 10), ("b", "completed", 200, 10),
        ("c", "completed", 900, 10), ("d", "active", 50, 3),
        ("e", "waiting", 300, 4), ("excluded", "paused", 15, 1),
    ]:
        records[model_id] = {"state": state, "progress": {
            "completed_tasks": tasks, "completed_forecast_and_score_seconds": seconds}}
    control = {"models": {key: spec(key != "excluded") for key in records},
               "workers": 2, "stop": False}
    eta = sweep.build_status(records, control, 10)["eta"]
    assert eta["completed_model_sample_count"] == 3
    assert eta["mean_completed_model_compute_seconds"] == 400
    assert eta["median_completed_model_compute_seconds"] == 200
    assert eta["mean_seconds"] == 350
    assert eta["median_seconds"] == 150
    assert eta["enabled_remaining_model_count"] == 2
    assert "saved" in eta["assumptions"]
    records["a"]["progress"]["completed_forecast_and_score_seconds"] = 0
    only_zero = sweep.build_status({"a": records["a"]}, control, 10)["eta"]
    assert only_zero["mean_seconds"] is None
    assert only_zero["median_seconds"] is None


def test_malformed_live_control_pauses_worker_and_controller_does_not_start(controller):
    instance, control, calls = controller
    write_control(control, {"one": spec()})
    instance.poll()
    control.write_text('{"models":')
    assert not sweep.worker_should_continue(control, instance.admission_path, "one", sweep._identity(spec()))
    status = instance.poll()
    assert status["control_error"]
    assert not status["stop_requested"]
    assert json.loads(instance.admission_path.read_text())["models"] == []
    assert len(calls) == 1
    write_control(control, {"one": spec(), "two": spec()})
    assert instance.poll()["control_error"] is None
    assert len(calls) == 2


def test_locks_are_exclusive_and_release(tmp_path):
    path = tmp_path / "lock"
    with (sweep.exclusive_lock(path),
          pytest.raises(RuntimeError, match="another process"),
          sweep.exclusive_lock(path)):
        pytest.fail("duplicate owner")
    with sweep.exclusive_lock(path):
        pass


def test_spawn_worker_imports_only_after_admission_and_checks_model_id(tmp_path, monkeypatch):
    control, admission = tmp_path / "control.json", tmp_path / "admission.json"
    root = tmp_path / "model"
    write_control(control, {"one": spec(False)})
    sweep._atomic_json(admission, {"models": ["one"]})
    imports = []
    monkeypatch.setattr(sweep.importlib, "import_module", lambda name: imports.append(name))
    sweep._worker("one", spec(), tmp_path, root, control, admission)
    assert not imports
    assert json.loads((root / "worker-result.json").read_text())["state"] == "paused"
    write_control(control, {"one": spec()})
    monkeypatch.setattr(sweep.importlib, "import_module", lambda name: SimpleNamespace(
        make_model=lambda: SimpleNamespace(model_id="wrong")))
    monkeypatch.setattr(sweep, "run_panel", lambda *a, **k: pytest.fail("wrong model entered runner"))
    sweep._worker("one", spec(), tmp_path, root, control, admission)
    result = json.loads((root / "worker-result.json").read_text())
    assert result["state"] == "failed"
    assert "model_id" in result["error"]


def test_runner_pauses_after_saved_origin_then_resumes_exact_scores(panel):  # noqa: F811
    model = ZeroOracle()
    progress = []
    result = runner.run_panel(
        model, panel[0], panel[1], implementation_digest="fixture-v1",
        should_continue=lambda: not model.calls, progress_callback=progress.append)
    assert len(model.calls) == 1
    assert result["paused"]
    assert result["completed_tasks"] == 1
    assert result["final_score"] is None
    assert progress[-1]["completed_tasks"] == 1
    checkpoint = panel[1] / "tasks/000000/000.npz"
    saved = checkpoint.read_bytes()
    resumed = runner.run_panel(model, panel[0], panel[1], implementation_digest="fixture-v1")
    assert resumed["status"] == "complete"
    assert resumed["resumed_tasks"] == 1
    assert checkpoint.read_bytes() == saved
    assert len(model.calls) == 4


def test_runner_pause_still_loads_all_preexisting_checkpoints(panel):  # noqa: F811
    model = ZeroOracle()
    complete = runner.run_panel(model, panel[0], panel[1], implementation_digest="fixture-v1")
    result = runner.run_panel(model, panel[0], panel[1], implementation_digest="fixture-v1",
                              should_continue=lambda: False)
    assert result["status"] == "complete"
    assert result["final_score"] == complete["final_score"]
    assert result["completed_forecast_and_score_seconds"] == complete[
        "completed_forecast_and_score_seconds"]
    assert len(model.calls) == 4


def test_spawn_context_is_explicit(monkeypatch):
    seen = []
    process = FakeProcess()
    process.start = lambda: seen.append("started")

    def context(method):
        seen.append(method)
        return SimpleNamespace(Process=lambda **kwargs: process)

    monkeypatch.setattr(sweep.multiprocessing, "get_context", context)
    assert sweep._start_worker("synthetic") is process
    assert seen == ["spawn", "started"]


def test_historical_long_pipe_ids_use_hash_paths_without_traversal(controller):
    instance, control, calls = controller
    model_id = "expanding_sample_mean|constant_sample_volatility|" + "filtered_empirical_tail" * 12
    write_control(control, {model_id: spec(), "../../escape": spec()})
    status = instance.poll()
    assert len(calls) == 2
    assert calls[0][0][0] == model_id
    for key in (model_id, "../../escape"):
        path = sweep.model_directory(instance.output_root, key)
        assert path.parent == instance.output_root / "models"
        assert len(path.name) == 64
        assert path.exists()
        assert status["models"][key]["output_directory"] == str(path)


def test_ranking_excludes_disabled_and_removed_preserving_completed_history():
    records = {key: {"state": "completed", "progress": {
        "completed_tasks": 10, "completed_forecast_and_score_seconds": 60,
        "final_score": score}}
        for key, score in (("active", .3), ("disabled", .2), ("removed", .1))}
    control = {"models": {"active": spec(), "disabled": spec(False)},
               "workers": 12, "stop": False}
    status = sweep.build_status(records, control, 10)
    assert status["ranking"] == [{"model_id": "active", "final_score": .3}]
    assert len(status["completed_history"]) == 3
    assert status["models"]["removed"]["final_score"] == .1
    assert status["eta"]["completed_model_sample_count"] == 3


def test_model_eta_never_splits_one_model_across_idle_workers():
    complete = {"state": "completed", "progress": {
        "completed_tasks": 10, "completed_forecast_and_score_seconds": 60}}
    waiting = {"state": "waiting", "progress": {}}
    records = {"complete": complete, "one": waiting}
    control = {"models": {key: spec() for key in records}, "workers": 12, "stop": False}
    eta = sweep.build_status(records, control, 10)["eta"]
    assert eta["mean_seconds"] == eta["median_seconds"] == 60
    # Queue packing for three equal jobs on two workers needs two model durations.
    records.update(two=waiting, three=waiting)
    control.update(models={key: spec() for key in records}, workers=2)
    eta = sweep.build_status(records, control, 10)["eta"]
    assert eta["mean_seconds"] == eta["median_seconds"] == 120


def test_restart_reconciles_external_lock_owner_without_duplicate_or_result_overwrite(controller):
    instance, control, calls = controller
    write_control(control, {"one": spec()})
    instance.poll()
    root = sweep.model_directory(instance.output_root, "one")
    with sweep.exclusive_lock(root / ".run.lock"):
        restarted = sweep.SweepController(instance.panel_root, instance.output_root, control,
                                          expected_tasks=10, start_worker=instance.start_worker)
        status = restarted.poll()
        assert status["models"]["one"]["state"] == "active"
        assert restarted.active["one"] is None
        assert len(calls) == 1
        assert sweep.worker_should_continue(control, restarted.admission_path, "one", sweep._identity(spec()))
        # The old child publishes its final result before releasing its run lock.
        finish(instance, "one")
        result_path = root / "worker-result.json"
        saved = result_path.read_bytes()
        assert restarted.poll()["models"]["one"]["state"] == "active"
        assert result_path.read_bytes() == saved
    status = restarted.poll()
    assert status["models"]["one"]["state"] == "completed"
    assert result_path.read_bytes() == saved
    assert not restarted.active
    assert len(calls) == 1


def test_worker_lock_collision_never_overwrites_owner_result(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    outcome = root / "worker-result.json"
    outcome.write_text('{"owner": "original"}')
    with sweep.exclusive_lock(root / ".run.lock"):
        sweep._worker("one", spec(), tmp_path, root, tmp_path / "missing-control", tmp_path / "missing-admission")
    assert json.loads(outcome.read_text()) == {"owner": "original"}
