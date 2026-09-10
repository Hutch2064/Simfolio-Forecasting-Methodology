"""Fixed-denominator aggregation for completed canonical task cells."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

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
    failed = [
        records[item.task_id]
        for item in manifest.tasks
        if records[item.task_id].status != "completed"
    ]
    if failed:
        details = "; ".join(
            (
                f"{record.task_id}: {record.error_type or 'failure'}: "
                f"{record.error_message or ''}"
            ).strip()
            for record in failed
        )
        raise IncompleteExecutionError(f"execution contains failed tasks: {details}")

    sums: dict[tuple[str, int], list[float | int]] = {}
    cell_order: list[tuple[str, int]] = []
    for identity in manifest.tasks:
        task_id = identity.task_id
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
            if key not in sums:
                cell_order.append(key)
            current = sums.setdefault(key, [0.0, 0])
            current[0] = float(current[0]) + float(loss)
            current[1] = int(current[1]) + 1

    actual_caps: dict[str, int] = {}
    for portfolio, horizon in sums:
        actual_caps[portfolio] = max(actual_caps.get(portfolio, 0), horizon)
    expected_caps = dict(manifest.expected_cell_caps)
    if actual_caps != expected_caps:
        missing = sorted(set(expected_caps) - set(actual_caps))
        extra = sorted(set(actual_caps) - set(expected_caps))
        raise IncompleteExecutionError(
            f"fixed cell caps changed: missing={missing[:3]} extra={extra[:3]}"
        )
    means = [float(sums[key][0]) / int(sums[key][1]) for key in cell_order]
    if not means:
        raise IncompleteExecutionError("execution produced no expected cells")
    return AggregatedScore(
        # The source aggregation is arithmetic cell means followed by the
        # ordinary NumPy mean in manifest/task order. Keep that order and
        # operation explicit so a resume has the same floating-point path.
        score=float(np.mean(np.asarray(means, dtype=np.float64))),
        cell_count=len(means),
        task_count=len(manifest.tasks),
    )


__all__ = ["AggregatedScore", "IncompleteExecutionError", "aggregate_fixed_denominator"]
