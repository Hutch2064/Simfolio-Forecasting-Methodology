"""Fixed-denominator aggregation for completed canonical task cells."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .checkpoint import ExecutionManifest, TaskResult


class IncompleteExecutionError(RuntimeError):
    """Raised when a score would omit an expected task or portfolio-horizon cell."""


@dataclass(frozen=True)
class AggregatedScore:
    score: float
    cell_count: int
    task_count: int


def aggregate_fixed_denominator(
    manifest: ExecutionManifest,
    records: Mapping[str, TaskResult],
) -> AggregatedScore:
    """Aggregate all expected cells and reject partial/failing execution."""

    expected_task_ids = {item.task_id for item in manifest.tasks}
    if set(records) != expected_task_ids:
        missing = sorted(expected_task_ids - set(records))
        extra = sorted(set(records) - expected_task_ids)
        raise IncompleteExecutionError(
            f"execution is incomplete: missing task results={missing[:3]} extra={extra[:3]}"
        )
    failed = [record for record in records.values() if record.status != "completed"]
    if failed:
        details = "; ".join(
            (
                f"{record.task_id}: {record.error_type or 'failure'}: "
                f"{record.error_message or ''}"
            ).strip()
            for record in sorted(failed, key=lambda item: item.task_id)
        )
        raise IncompleteExecutionError(f"execution contains failed tasks: {details}")

    identities = {item.task_id: item for item in manifest.tasks}
    sums: dict[tuple[str, int], list[float | int]] = {}
    for task_id in sorted(expected_task_ids):
        identity = identities[task_id]
        result = records[task_id]
        if len(result.losses) != identity.horizon_days:
            raise IncompleteExecutionError(
                f"task {task_id} returned {len(result.losses)} losses; "
                f"expected {identity.horizon_days}"
            )
        for offset, loss in enumerate(result.losses, start=1):
            if not math.isfinite(float(loss)):
                raise IncompleteExecutionError(f"task {task_id} returned a nonfinite loss")
            key = (identity.portfolio_id, offset)
            current = sums.setdefault(key, [0.0, 0])
            current[0] = float(current[0]) + float(loss)
            current[1] = int(current[1]) + 1

    actual_cells = set(sums)
    expected_cells = set(manifest.expected_cells)
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        extra = sorted(actual_cells - expected_cells)
        raise IncompleteExecutionError(
            f"fixed cell denominator changed: missing={missing[:3]} extra={extra[:3]}"
        )
    means = [float(total) / int(count) for total, count in (sums[key] for key in sorted(sums))]
    if not means:
        raise IncompleteExecutionError("execution produced no expected cells")
    return AggregatedScore(
        score=math.fsum(means) / len(means),
        cell_count=len(means),
        task_count=len(manifest.tasks),
    )


__all__ = ["AggregatedScore", "IncompleteExecutionError", "aggregate_fixed_denominator"]
