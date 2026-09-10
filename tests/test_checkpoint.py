from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology.results.checkpoint import CheckpointIdentityError
from simfolio_forecasting_methodology.runner import (
    OriginTask,
    TrainingData,
    execute_model_checkpointed,
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
