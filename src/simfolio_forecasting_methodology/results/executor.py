"""Bounded, resumable execution over a constructed canonical task list."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .aggregate import IncompleteExecutionError, aggregate_fixed_denominator
from .checkpoint import CheckpointStore, ExecutionManifest, TaskResult


MAX_WORKERS = 32


@dataclass(frozen=True)
class ExecutionSummary:
    """Serializable execution status; ``score`` is absent for incomplete runs."""

    model_id: str
    checkpoint_dir: str
    manifest_fingerprint: str
    task_count: int
    completed_count: int
    failed_count: int
    pending_count: int
    resumed_completed_count: int
    cell_count: int | None
    score: float | None
    status: str
    failures: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result_kind": "new_execution",
            "model_id": self.model_id,
            "checkpoint_dir": self.checkpoint_dir,
            "manifest_fingerprint": self.manifest_fingerprint,
            "task_count": self.task_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "pending_count": self.pending_count,
            "resumed_completed_count": self.resumed_completed_count,
            "cell_count": self.cell_count,
            "aggregate_score": self.score,
            "status": self.status,
            "failures": list(self.failures),
        }
        return payload


def _task_result(task_id: str, model: Any, task: Any, simulations: int) -> TaskResult:
    try:
        # Imported lazily so runner can expose the public orchestration function
        # without a module import cycle.
        from ..runner import evaluate_origin_task

        losses = np.asarray(
            evaluate_origin_task(model, task, simulations=int(simulations)), dtype=np.float64
        ).reshape(-1)
        if not np.all(np.isfinite(losses)):
            raise ValueError("model returned nonfinite horizon losses")
        return TaskResult(
            task_id=task_id,
            status="completed",
            losses=tuple(float(item) for item in losses),
        )
    except Exception as exc:  # noqa: BLE001 - task failures are durable and visible
        return TaskResult(
            task_id=task_id,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def execute_checkpointed(
    model: Any,
    tasks: Iterable[Any],
    *,
    manifest: ExecutionManifest,
    checkpoint_dir: Path,
    simulations: int,
    workers: int = 1,
    resume: bool = False,
) -> ExecutionSummary:
    """Execute at most ``workers`` tasks concurrently and persist each outcome."""

    if int(workers) < 1 or int(workers) > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    task_list = list(tasks)
    if len(task_list) != manifest.task_count:
        raise ValueError(
            f"task constructor returned {len(task_list)} tasks; "
            f"manifest expects {manifest.task_count}"
        )
    # Resolve IDs through the public runner identity function.  The executor
    # therefore works with immutable OriginTask instances and does not mutate
    # the task constructor's objects just to attach storage metadata.
    from ..runner import task_identity

    task_ids = [task_identity(task, model.model_id, int(simulations)).task_id for task in task_list]
    if set(task_ids) != {item.task_id for item in manifest.tasks}:
        raise ValueError("task constructor identity differs from the execution manifest")
    task_pairs = dict(zip(task_ids, task_list))

    store = CheckpointStore(Path(checkpoint_dir), manifest, resume=bool(resume))
    initial_statuses = store.statuses()
    resumed_completed_count = sum(status == "completed" for status in initial_statuses.values())
    pending_ids = [task_id for task_id in task_ids if initial_statuses[task_id] != "completed"]

    def submit_next(
        iterator: Iterable[str],
        pool: ThreadPoolExecutor,
        futures: dict[Future[TaskResult], str],
    ) -> None:
        try:
            task_id = next(iterator)  # type: ignore[arg-type]
        except StopIteration:
            return
        futures[
            pool.submit(_task_result, task_id, model, task_pairs[task_id], simulations)
        ] = task_id

    pending_iterator = iter(pending_ids)
    if pending_ids:
        with ThreadPoolExecutor(
            max_workers=int(workers), thread_name_prefix="canonical-oos"
        ) as pool:
            futures: dict[Future[TaskResult], str] = {}
            for _ in range(min(int(workers), len(pending_ids))):
                submit_next(pending_iterator, pool, futures)
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: futures[item]):
                    task_id = futures.pop(future)
                    result = future.result()
                    store.write_result(result)
                    submit_next(pending_iterator, pool, futures)

    records = store.results()
    counts = store.summary()
    score: float | None = None
    cell_count: int | None = None
    status = "completed"
    failures: tuple[dict[str, str], ...] = ()
    try:
        aggregate = aggregate_fixed_denominator(manifest, records)
        score = aggregate.score
        cell_count = aggregate.cell_count
    except IncompleteExecutionError as exc:
        status = "failed" if counts["failed"] else "incomplete"
        failed_records = [
            result
            for result in records.values()
            if result.status == "failed"
        ]
        failures = tuple(
            {
                "task_id": result.task_id,
                "error_type": result.error_type or "TaskFailure",
                "error_message": result.error_message or str(exc),
            }
            for result in sorted(failed_records, key=lambda item: item.task_id)
        )
        if not failures:
            failures = (
                {
                    "task_id": "execution",
                    "error_type": "IncompleteExecutionError",
                    "error_message": str(exc),
                },
            )

    summary = ExecutionSummary(
        model_id=manifest.model_id,
        checkpoint_dir=str(Path(checkpoint_dir)),
        manifest_fingerprint=manifest.manifest_fingerprint,
        task_count=manifest.task_count,
        completed_count=counts["completed"],
        failed_count=counts["failed"],
        pending_count=counts["pending"],
        resumed_completed_count=resumed_completed_count,
        cell_count=cell_count,
        score=score,
        status=status,
        failures=failures,
    )
    store.write_summary(summary.to_dict())
    return summary


__all__ = ["ExecutionSummary", "MAX_WORKERS", "execute_checkpointed"]
