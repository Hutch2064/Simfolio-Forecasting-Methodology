"""Exact distributional scoring for the canonical dense OOS protocol."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ScoredOrigin:
    portfolio_id: Hashable
    horizon_days: int
    crps: float


def empirical_crps(samples: Sequence[float] | np.ndarray, realized: float) -> float:
    """Return exact empirical CRPS for equally weighted forecast samples."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("at least one forecast sample is required")
    if not np.all(np.isfinite(x)) or not np.isfinite(realized):
        raise ValueError("CRPS inputs must be finite")
    x = np.sort(x)
    n = x.size
    observation_term = float(np.mean(np.abs(x - float(realized))))
    ranks = np.arange(1, n + 1, dtype=np.float64)
    half_pairwise_term = float(np.dot(2.0 * ranks - n - 1.0, x) / (n * n))
    return observation_term - half_pairwise_term


def aggregate_equal_portfolio_horizon(rows: Iterable[ScoredOrigin]) -> float:
    """Average origins within cells, then equally weight portfolio-horizon cells."""
    cells: dict[tuple[Hashable, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.horizon_days <= 0 or not np.isfinite(row.crps):
            raise ValueError("invalid scored origin")
        cells[(row.portfolio_id, int(row.horizon_days))].append(float(row.crps))
    if not cells:
        raise ValueError("at least one scored origin is required")
    return float(np.mean([np.mean(values) for values in cells.values()]))
