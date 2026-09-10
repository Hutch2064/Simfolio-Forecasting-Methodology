"""Durable execution records for canonical forecasting runs.

The result package stores task identity and execution state separately from the
retained historical score registry.  A stored score can therefore be inspected
without being mistaken for a newly executed forecast.
"""

from .aggregate import IncompleteExecutionError, aggregate_fixed_denominator
from .checkpoint import (
    CheckpointCorruptError,
    CheckpointIdentityError,
    CheckpointStore,
    ExecutionManifest,
    TaskIdentity,
    TaskResult,
)
from .executor import ExecutionSummary, ProgressEvent, execute_checkpointed
from .retained import retained_score_report

__all__ = [
    "CheckpointCorruptError",
    "CheckpointIdentityError",
    "CheckpointStore",
    "ExecutionManifest",
    "ExecutionSummary",
    "IncompleteExecutionError",
    "ProgressEvent",
    "TaskIdentity",
    "TaskResult",
    "aggregate_fixed_denominator",
    "execute_checkpointed",
    "retained_score_report",
]
