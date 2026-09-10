"""Memory-bounded primitives for dense daily-horizon OOS evaluation."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np


def empirical_crps_by_horizon(
    terminal_samples: np.ndarray,
    realized_terminal: np.ndarray,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Exact empirical CRPS independently at every daily horizon.

    Parameters
    ----------
    terminal_samples:
        Shape ``(n_simulations, n_horizons)``. Each column is the simulated
        terminal log-return distribution at that horizon.
    realized_terminal:
        Shape ``(n_horizons,)`` containing realized terminal log returns.
    """

    samples = np.asarray(terminal_samples, dtype=np.float64)
    realized = np.asarray(realized_terminal, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("terminal_samples must be a two-dimensional matrix")
    if realized.ndim != 1 or samples.shape[1] != realized.size:
        raise ValueError("realized_terminal must align with sample horizons")
    if samples.shape[0] < 1:
        raise ValueError("at least one simulation is required")

    valid = np.isfinite(realized) & np.all(np.isfinite(samples), axis=0)
    if strict and not np.all(valid):
        raise ValueError("fixed-denominator CRPS requires finite samples at every horizon")
    output = np.full(realized.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return output

    selected = samples[:, valid]
    observation_term = np.mean(np.abs(selected - realized[valid][None, :]), axis=0)
    ordered = np.sort(selected, axis=0)
    n = float(selected.shape[0])
    ranks = np.arange(1.0, n + 1.0, dtype=np.float64)
    pairwise_sum = 2.0 * np.sum((2.0 * ranks - n - 1.0)[:, None] * ordered, axis=0)
    pairwise_mean = pairwise_sum / (n * n)
    output[valid] = np.maximum(0.0, observation_term - 0.5 * pairwise_mean)
    return output


@dataclass
class CellAccumulator:
    """Stream origin losses into portfolio-horizon cells without row explosion."""

    loss_sum: dict[tuple[Hashable, int], float] = field(default_factory=dict)
    origin_count: dict[tuple[Hashable, int], int] = field(default_factory=dict)

    def add_vector(
        self,
        portfolio_id: Hashable,
        horizon_losses: np.ndarray,
        *,
        first_horizon_day: int = 1,
    ) -> None:
        losses = np.asarray(horizon_losses, dtype=np.float64).reshape(-1)
        for offset, loss in enumerate(losses):
            if not np.isfinite(loss):
                continue
            horizon = int(first_horizon_day) + offset
            key = (portfolio_id, horizon)
            self.loss_sum[key] = self.loss_sum.get(key, 0.0) + float(loss)
            self.origin_count[key] = self.origin_count.get(key, 0) + 1

    def cell_means(self) -> dict[tuple[Hashable, int], float]:
        if set(self.loss_sum) != set(self.origin_count):
            raise RuntimeError("cell accumulator state is inconsistent")
        result: dict[tuple[Hashable, int], float] = {}
        for key, total in self.loss_sum.items():
            count = self.origin_count[key]
            if count <= 0:
                raise RuntimeError(f"nonpositive origin count for cell {key!r}")
            result[key] = total / float(count)
        return result

    def aggregate_score(self) -> float:
        means = self.cell_means()
        if not means:
            raise ValueError("no finite portfolio-horizon cells were accumulated")
        return float(np.mean(np.fromiter(means.values(), dtype=np.float64)))

    def assert_complete(
        self,
        *,
        expected_cells: int,
        expected_tasks: int,
        completed_tasks: int,
        failed_tasks: int = 0,
    ) -> None:
        """Apply the canonical fail-closed task and cell denominator gates."""
        if int(failed_tasks) != 0:
            raise ValueError(f"fixed-denominator evaluation has {int(failed_tasks)} failed tasks")
        if int(completed_tasks) != int(expected_tasks):
            raise ValueError(
                f"fixed-denominator evaluation requires {int(expected_tasks)} tasks; "
                f"observed {int(completed_tasks)}"
            )
        if self.cell_count != int(expected_cells):
            raise ValueError(
                f"fixed-denominator evaluation requires {int(expected_cells)} cells; "
                f"observed {self.cell_count}"
            )

    def fixed_denominator_score(
        self,
        *,
        expected_cells: int,
        expected_tasks: int,
        completed_tasks: int,
        failed_tasks: int = 0,
    ) -> float:
        self.assert_complete(
            expected_cells=expected_cells,
            expected_tasks=expected_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
        )
        return self.aggregate_score()

    @property
    def cell_count(self) -> int:
        return len(self.loss_sum)
