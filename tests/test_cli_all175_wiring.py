from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simfolio_forecasting_methodology import cli
from simfolio_forecasting_methodology.catalogue import load_canonical_models
from simfolio_forecasting_methodology.models.registry import build_model as registry_build_model
from simfolio_forecasting_methodology.runner import OriginTask, TrainingData


@dataclass(frozen=True)
class _StubSummary:
    model_id: str
    checkpoint_dir: str
    task_count: int
    simulations: int
    execution_variant: str

    status: str = "completed"
    completed_count: int = 1
    failed_count: int = 0
    score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "result_kind": "new_execution",
            "model_id": self.model_id,
            "checkpoint_dir": self.checkpoint_dir,
            "task_count": self.task_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "pending_count": 0,
            "aggregate_score": self.score,
            "status": self.status,
            "execution_variant": self.execution_variant,
        }


def test_canonical_cli_default_wires_all_175_real_factories(monkeypatch, tmp_path: Path):
    expected_ids = tuple(row["public_model_id"] for row in load_canonical_models())
    built_ids: list[str] = []
    plan_calls: list[dict[str, object]] = []
    execution_calls: list[dict[str, object]] = []
    task = OriginTask(
        portfolio_id="cli-wiring-stub",
        origin_label="one",
        training=TrainingData(np.array([0.0, 0.01, -0.01])),
        realized_future_daily_log_returns=np.array([0.01]),
        seed=17,
    )
    data_root = tmp_path / ".simfolio-oos-data"
    data_root.mkdir()
    monkeypatch.chdir(tmp_path)
    output_root = tmp_path / "canonical-output"

    def recording_build_model(model_id: str):
        model = registry_build_model(model_id)
        built_ids.append(model_id)
        assert model.model_id == model_id
        return model

    def stub_plan(
        data_root_arg: Path,
        *,
        portfolio_limit: int | None = None,
        rolling_origins: int,
    ):
        plan_calls.append(
            {
                "data_root": Path(data_root_arg).resolve(),
                "portfolio_limit": portfolio_limit,
                "rolling_origins": rolling_origins,
            }
        )
        return SimpleNamespace(tasks=(task,))

    def stub_execute(
        model,
        tasks,
        *,
        simulations: int,
        checkpoint_dir: Path,
        workers: int,
        resume: bool,
        progress_callback,
        execution_variant: str,
    ):
        del workers, resume, progress_callback
        task_list = tuple(tasks)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        execution_calls.append(
            {
                "model_id": model.model_id,
                "tasks": task_list,
                "simulations": simulations,
                "checkpoint_dir": checkpoint_dir,
                "execution_variant": execution_variant,
            }
        )
        return _StubSummary(
            model_id=model.model_id,
            checkpoint_dir=str(checkpoint_dir),
            task_count=len(task_list),
            simulations=simulations,
            execution_variant=execution_variant,
        )

    monkeypatch.setattr(cli, "build_model", recording_build_model)
    monkeypatch.setattr(cli, "build_experiment_plan", stub_plan)
    monkeypatch.setattr(cli, "execute_model_checkpointed", stub_execute)

    assert (
        cli.main(
            [
                "canonical-175",
                "--output",
                str(output_root),
            ]
        )
        == 0
    )

    assert tuple(built_ids) == expected_ids
    assert len(execution_calls) == len(expected_ids) == 175
    assert plan_calls == [
        {
            "data_root": data_root,
            "portfolio_limit": None,
            "rolling_origins": 48,
        }
    ]
    assert {call["simulations"] for call in execution_calls} == {240}
    assert {call["execution_variant"] for call in execution_calls} == {"canonical"}
    assert all(call["tasks"] == (task,) for call in execution_calls)
    assert all(
        call["checkpoint_dir"]
        == output_root / model_id.replace("|", "__")
        for call, model_id in zip(execution_calls, expected_ids)
    )
    assert all(
        (output_root / model_id.replace("|", "__")).is_dir()
        for model_id in expected_ids
    )
