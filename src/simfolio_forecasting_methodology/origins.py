"""Origin-selection primitives for the canonical dense OOS protocol."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")
ROLLING_ORIGINS_PER_PORTFOLIO = 48
TEMPORAL_SPLITS: tuple[tuple[str, float], ...] = (
    ("train_first_quarter_test_remaining", 0.25),
    ("train_first_half_test_remaining", 0.50),
    ("train_first_three_quarters_test_final_quarter", 0.75),
)


@dataclass(frozen=True)
class TemporalOrigin:
    split_name: str
    train_fraction: float
    position: int
    max_horizon: int


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    n = int(length)
    target = int(count)
    if n <= 0 or target <= 0:
        return []
    if n <= target:
        return list(range(n))
    raw = np.linspace(0, n - 1, target)
    selected = sorted({int(round(float(index))) for index in raw})
    while len(selected) < target:
        for index in range(n):
            if index not in selected:
                selected.append(index)
                selected.sort()
                break
    return selected[:target]


def select_full_history_even(eligible_origins: Sequence[T], count: int = ROLLING_ORIGINS_PER_PORTFOLIO) -> list[T]:
    return [eligible_origins[index] for index in evenly_spaced_indices(len(eligible_origins), count)]


def temporal_holdouts(n_observations: int) -> list[TemporalOrigin]:
    n = int(n_observations)
    if n < 81:
        raise ValueError("at least 81 observations are required")
    output: list[TemporalOrigin] = []
    for split_name, train_fraction in TEMPORAL_SPLITS:
        position = int(floor(float(n) * float(train_fraction))) - 1
        position = max(79, min(position, n - 2))
        output.append(TemporalOrigin(split_name, train_fraction, position, n - position - 1))
    return output


def expected_origin_tasks(portfolios: int = 80, rolling_origins: int = 48) -> int:
    return int(portfolios) * (int(rolling_origins) + len(TEMPORAL_SPLITS))
