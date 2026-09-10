"""Bounded, resumable execution over a constructed canonical task list."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    requested_workers: int
    effective_workers: int
    parallelism_note: str | None
    execution_variant: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result_kind": (
                "new_execution"
                if self.execution_variant == "canonical"
                else "new_execution_noncanonical"
            ),
            "model_id": self.model_id,
            "execution_variant": self.execution_variant,
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
            "requested_workers": self.requested_workers,
            "effective_workers": self.effective_workers,
            "parallelism_note": self.parallelism_note,
        }
        return payload


@dataclass(frozen=True)
class ProgressEvent:
    """One durable task transition suitable for a CLI or structured logger."""

    task_id: str
    status: str
    processed_count: int
    completed_count: int
    failed_count: int
    pending_count: int
    task_count: int
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "processed_count": self.processed_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "pending_count": self.pending_count,
            "task_count": self.task_count,
            "resumed": self.resumed,
        }


def _task_result(task_id: str, model: Any, task: Any, simulations: int) -> TaskResult:
    try:
        # Imported lazily so runner can expose the public orchestration function
        # without a module import cycle.
        from ..runner import evaluate_origin_task

        # Keep serial fallback deterministic even for legacy implementations
        # that still consume NumPy's process-global stream instead of using
        # ForecastContext.seed directly.
        np.random.seed(int(task.seed) % (2**32))
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


def _process_task_result(
    factory: Callable[[], Any],
    expected_model_id: str,
    task_id: str,
    task: Any,
    simulations: int,
) -> TaskResult:
    """Construct a fresh model in a child process for one seeded task."""

    try:
        # OriginTask.seed is the only task-local random identity. Set the
        # legacy NumPy stream explicitly before construction and evaluation so
        # a process worker cannot inherit a parent's advanced global stream.
        np.random.seed(int(task.seed) % (2**32))
        model = factory()
        if getattr(model, "model_id", None) != expected_model_id:
            raise ValueError("process worker factory returned a different model ID")
        return _task_result(task_id, model, task, simulations)
    except Exception as exc:  # noqa: BLE001 - serialize worker failures visibly
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
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> ExecutionSummary:
    """Execute tasks with bounded, deterministic worker isolation.

    Models without an explicitly verified, importable ``process_worker_factory``
    are forced onto one worker. This prevents shared model state and global RNG
    streams from crossing task boundaries. A factory-enabled model is built
    afresh for each process task and receives the task seed before evaluation.
    """

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
    current_statuses = dict(initial_statuses)
    status_counts = {"pending": 0, "completed": 0, "failed": 0}
    for status in initial_statuses.values():
        if status not in status_counts:
            raise ValueError(f"checkpoint contains unknown task status: {status}")
        status_counts[status] += 1
    resumed_completed_count = sum(status == "completed" for status in initial_statuses.values())
    pending_ids = [task_id for task_id in task_ids if initial_statuses[task_id] != "completed"]

    requested_workers = int(workers)
    process_factory = getattr(model, "process_worker_factory", None)
    process_enabled = requested_workers > 1 and callable(process_factory)
    effective_workers = requested_workers if process_enabled else 1
    parallelism_note = None
    if requested_workers > 1 and not process_enabled:
        parallelism_note = (
            "one worker enforced: model has no verified process_worker_factory; "
            "shared/global RNG state is kept task-serial"
        )

    def emit(task_id: str, status: str, *, resumed: bool = False) -> None:
        if progress_callback is None:
            return
        progress_callback(
            ProgressEvent(
                task_id=task_id,
                status=status,
                processed_count=status_counts["completed"] + status_counts["failed"],
                completed_count=status_counts["completed"],
                failed_count=status_counts["failed"],
                pending_count=status_counts["pending"],
                task_count=manifest.task_count,
                resumed=resumed,
            )
        )

    def persist_result(result: TaskResult) -> None:
        previous_status = current_statuses.get(result.task_id)
        if previous_status not in status_counts:
            raise ValueError(f"checkpoint contains unknown task status for {result.task_id}")
        store.write_result(result)
        status_counts[previous_status] -= 1
        status_counts[result.status] += 1
        current_statuses[result.task_id] = result.status

    if parallelism_note and progress_callback is not None:
        progress_callback(
            ProgressEvent(
                task_id="executor",
                status="parallelism_restricted",
                processed_count=status_counts["completed"] + status_counts["failed"],
                completed_count=status_counts["completed"],
                failed_count=status_counts["failed"],
                pending_count=status_counts["pending"],
                task_count=manifest.task_count,
            )
        )
    if progress_callback is not None and resumed_completed_count:
        progress_callback(
            ProgressEvent(
                task_id="executor",
                status="resumed",
                processed_count=status_counts["completed"] + status_counts["failed"],
                completed_count=status_counts["completed"],
                failed_count=status_counts["failed"],
                pending_count=status_counts["pending"],
                task_count=manifest.task_count,
                resumed=True,
            )
        )

    def submit_next(iterator: Iterable[str], pool: Any, futures: dict[Future[TaskResult], str]) -> None:
        try:
            task_id = next(iterator)  # type: ignore[arg-type]
        except StopIteration:
            return
        if process_enabled:
            futures[
                pool.submit(
                    _process_task_result,
                    process_factory,
                    model.model_id,
                    task_id,
                    task_pairs[task_id],
                    simulations,
                )
            ] = task_id
        else:
            futures[pool.submit(_task_result, task_id, model, task_pairs[task_id], simulations)] = task_id

    pending_iterator = iter(pending_ids)
    if pending_ids and effective_workers > 1:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures: dict[Future[TaskResult], str] = {}
            for _ in range(min(effective_workers, len(pending_ids))):
                submit_next(pending_iterator, pool, futures)
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: futures[item]):
                    task_id = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - process failures are durable
                        result = TaskResult(
                            task_id=task_id,
                            status="failed",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    persist_result(result)
                    emit(task_id, result.status)
                    submit_next(pending_iterator, pool, futures)
    elif pending_ids:
        for task_id in pending_ids:
            result = _task_result(task_id, model, task_pairs[task_id], simulations)
            persist_result(result)
            emit(task_id, result.status)

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
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        parallelism_note=parallelism_note,
        execution_variant=manifest.execution_variant,
    )
    store.write_summary(summary.to_dict())
    return summary


__all__ = [
    "MAX_WORKERS",
    "ExecutionSummary",
    "ProgressEvent",
    "execute_checkpointed",
]
