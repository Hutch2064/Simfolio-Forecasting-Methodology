from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology.results.aggregate import aggregate_fixed_denominator
from simfolio_forecasting_methodology.results.checkpoint import (
    CheckpointCorruptError,
    CheckpointIdentityError,
    TaskResult,
    stable_digest,
)
from simfolio_forecasting_methodology.runner import (
    OriginTask,
    TrainingData,
    build_execution_manifest,
    execute_model_checkpointed,
    task_identity,
)

MODEL_ID = "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced"


class ZeroPathModel:
    model_id = MODEL_ID

    def simulate_daily_log_returns(self, training, context):
        del training
        return np.zeros((context.simulations, context.horizon_days), dtype=np.float64)


class FailingPathModel(ZeroPathModel):
    def simulate_daily_log_returns(self, training, context):
        if context.origin_label == "bad":
            raise RuntimeError("source model failed for this task")
        return super().simulate_daily_log_returns(training, context)


class GlobalRngModel(ZeroPathModel):
    """Legacy model shape that would collide if tasks shared one RNG stream."""

    def simulate_daily_log_returns(self, training, context):
        del training
        return np.random.random((context.simulations, context.horizon_days))


class FingerprintedModel(ZeroPathModel):
    specification_fingerprint = "a" * 64
    implementation_digest = "b" * 64


def _tasks() -> list[OriginTask]:
    training = TrainingData(portfolio_log_returns=np.array([0.0, 0.01, -0.01]))
    return [
        OriginTask("p1", "good", training, np.array([0.1, 0.2]), 11),
        OriginTask("p1", "bad", training, np.array([0.3]), 12),
    ]


def test_checkpoint_resume_is_deterministic_and_rejects_changed_identity(tmp_path: Path):
    tasks = _tasks()
    checkpoint = tmp_path / "run"
    first = execute_model_checkpointed(
        ZeroPathModel(),
        tasks,
        simulations=4,
        checkpoint_dir=checkpoint,
        workers=2,
    )
    resumed = execute_model_checkpointed(
        ZeroPathModel(),
        tasks,
        simulations=4,
        checkpoint_dir=checkpoint,
        workers=1,
        resume=True,
    )

    assert first.status == resumed.status == "completed"
    assert first.score == resumed.score
    assert first.completed_count == resumed.completed_count == 2
    assert resumed.resumed_completed_count == 2
    assert (checkpoint / "summary.json").is_file()
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["execution_variant"] == "canonical"
    assert manifest["expected_cell_caps"] == [["p1", 2]]
    assert not (checkpoint / "index.json").exists()
    task_payload = json.loads(next((checkpoint / "tasks").glob("*.json")).read_text())
    assert task_payload["manifest_fingerprint"] == manifest["manifest_fingerprint"]

    changed = list(tasks)
    changed[0] = OriginTask("p1", "good", tasks[0].training, np.array([0.1, 0.25]), 11)
    with pytest.raises(CheckpointIdentityError, match="manifest fingerprint differs"):
        execute_model_checkpointed(
            ZeroPathModel(),
            changed,
            simulations=4,
            checkpoint_dir=checkpoint,
            resume=True,
        )


def test_manifest_id_mutation_is_rejected_even_after_rehash(tmp_path: Path):
    checkpoint = tmp_path / "mutated-manifest"
    execute_model_checkpointed(
        ZeroPathModel(),
        _tasks(),
        simulations=4,
        checkpoint_dir=checkpoint,
    )
    manifest_path = checkpoint / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["tasks"][0]["task_id"] = "0" * 64
    identity_payload = {
        key: value for key, value in payload.items() if key != "manifest_fingerprint"
    }
    payload["manifest_fingerprint"] = stable_digest(identity_payload)
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(CheckpointIdentityError, match="manifest fingerprint differs"):
        execute_model_checkpointed(
            ZeroPathModel(),
            _tasks(),
            simulations=4,
            checkpoint_dir=checkpoint,
            resume=True,
        )


def test_task_result_digest_detects_corruption_on_resume(tmp_path: Path):
    checkpoint = tmp_path / "mutated-task"
    execute_model_checkpointed(
        ZeroPathModel(),
        _tasks(),
        simulations=4,
        checkpoint_dir=checkpoint,
    )
    task_path = next((checkpoint / "tasks").glob("*.json"))
    payload = json.loads(task_path.read_text())
    payload["losses"][0] = float(payload["losses"][0]) + 1.0
    task_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(CheckpointCorruptError, match="task result digest mismatch"):
        execute_model_checkpointed(
            ZeroPathModel(),
            _tasks(),
            simulations=4,
            checkpoint_dir=checkpoint,
            resume=True,
        )


def test_failed_tasks_are_visible_and_never_aggregated(tmp_path: Path):
    summary = execute_model_checkpointed(
        FailingPathModel(),
        _tasks(),
        simulations=3,
        checkpoint_dir=tmp_path / "failed",
        workers=2,
    )

    assert summary.status == "failed"
    assert summary.completed_count == 1
    assert summary.failed_count == 1
    assert summary.score is None
    assert summary.failures[0]["error_type"] == "RuntimeError"
    assert "source model failed" in summary.failures[0]["error_message"]


def test_global_rng_model_is_forced_to_one_worker_with_seeded_tasks(tmp_path: Path):
    events = []
    requested_parallel = execute_model_checkpointed(
        GlobalRngModel(),
        _tasks(),
        simulations=3,
        checkpoint_dir=tmp_path / "parallel-request",
        workers=2,
        progress_callback=events.append,
    )
    serial = execute_model_checkpointed(
        GlobalRngModel(),
        _tasks(),
        simulations=3,
        checkpoint_dir=tmp_path / "serial",
        workers=1,
    )

    assert requested_parallel.effective_workers == 1
    assert requested_parallel.parallelism_note is not None
    assert requested_parallel.score == serial.score
    assert any(event.status == "parallelism_restricted" for event in events)


def test_noncanonical_execution_variant_is_explicit_in_manifest_and_summary(tmp_path: Path):
    checkpoint = tmp_path / "smoke"
    summary = execute_model_checkpointed(
        ZeroPathModel(),
        _tasks(),
        simulations=2,
        checkpoint_dir=checkpoint,
        execution_variant="smoke_noncanonical",
    )

    summary_payload = summary.to_dict()
    assert summary_payload["result_kind"] == "new_execution_noncanonical"
    assert summary_payload["execution_variant"] == "smoke_noncanonical"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["execution_variant"] == "smoke_noncanonical"
    assert manifest["experiment_id"].endswith("::smoke_noncanonical")
    assert manifest["protocol_id"].endswith("::smoke_noncanonical")


def test_aggregation_uses_manifest_task_order_and_fixed_cell_denominator():
    training = TrainingData(portfolio_log_returns=np.array([0.0, 0.01]))
    tasks = [
        OriginTask("p1", label, training, np.array([0.0]), seed)
        for label, seed in (("first", 1), ("second", 2), ("third", 3))
    ]
    manifest = build_execution_manifest(MODEL_ID, tasks, simulations=1)
    identities = [task_identity(task, MODEL_ID, 1) for task in tasks]
    records = {
        identities[2].task_id: TaskResult(identities[2].task_id, "completed", (-(10**16),)),
        identities[1].task_id: TaskResult(identities[1].task_id, "completed", (1.0,)),
        identities[0].task_id: TaskResult(identities[0].task_id, "completed", ((10**16),)),
    }

    aggregate = aggregate_fixed_denominator(manifest, records)

    assert aggregate.cell_count == 1
    assert aggregate.task_count == 3
    assert aggregate.score == ((float(10**16) + 1.0) - float(10**16)) / 3.0


def test_changed_specification_or_implementation_cannot_resume(tmp_path: Path):
    checkpoint = tmp_path / "fingerprinted"
    execute_model_checkpointed(
        FingerprintedModel(),
        _tasks(),
        simulations=2,
        checkpoint_dir=checkpoint,
    )
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["specification_fingerprint"] == "a" * 64
    assert manifest["implementation_digest"] == "b" * 64
    assert manifest["seed_contract"] == "origin_task.seed_to_forecast_context.seed.v1"
    assert manifest["dependency_identity"]["python"]
    assert manifest["dependency_identity"]["numpy"]

    class ChangedSpec(FingerprintedModel):
        specification_fingerprint = "c" * 64

    with pytest.raises(CheckpointIdentityError, match="manifest fingerprint differs"):
        execute_model_checkpointed(
            ChangedSpec(),
            _tasks(),
            simulations=2,
            checkpoint_dir=checkpoint,
            resume=True,
        )

    class ChangedImplementation(FingerprintedModel):
        implementation_digest = "d" * 64

    with pytest.raises(CheckpointIdentityError, match="manifest fingerprint differs"):
        execute_model_checkpointed(
            ChangedImplementation(),
            _tasks(),
            simulations=2,
            checkpoint_dir=checkpoint,
            resume=True,
        )
def test_checkpoint_implementation_identity_includes_numerical_helpers(tmp_path, monkeypatch):
    from simfolio_forecasting_methodology import runner
    from simfolio_forecasting_methodology.models.asset_level.frontier import HistoricalFrontierModel

    helper = tmp_path / "numerical_helper.py"
    helper.write_text("BOUND = 1\n")
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "runner.py"))
    before = runner._implementation_digest(HistoricalFrontierModel(), {})
    helper.write_text("BOUND = 2\n")
    after = runner._implementation_digest(HistoricalFrontierModel(), {})
    assert before != after
