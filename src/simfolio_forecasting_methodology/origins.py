"""Origin and horizon rules recovered from the pinned research harness."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")
ROLLING_ORIGINS_PER_PORTFOLIO = 48
ROLLING_MIN_TRAINING_OBSERVATIONS = 504
ORIGIN_SELECTION_HORIZONS_DAYS = (
    21,
    42,
    63,
    126,
    189,
    252,
    378,
    504,
    756,
    1008,
    1260,
    1512,
    1764,
    2016,
    2268,
    2520,
    3780,
    5040,
    7560,
)
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
    """Match the pinned ``np.linspace`` plus Python ``round`` selection."""
    n = int(length)
    target = int(count)
    if n <= 0 or target <= 0:
        return []
    if n <= target:
        return list(range(n))
    raw = np.linspace(0, n - 1, target)
    selected = sorted({round(float(index)) for index in raw})
    while len(selected) < target:
        for index in range(n):
            if index not in selected:
                selected.append(index)
                selected.sort()
                break
    return selected[:target]


def select_full_history_even(
    eligible_origins: Sequence[T],
    count: int = ROLLING_ORIGINS_PER_PORTFOLIO,
) -> list[T]:
    return [eligible_origins[index] for index in evenly_spaced_indices(len(eligible_origins), count)]


def _validated_dates(dates: Sequence[object] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if index.empty:
        raise ValueError("at least one date is required")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("dates must be unique and sorted ascending")
    return index


def _horizon_is_eligible(position: int, n_observations: int, horizons: Sequence[int]) -> bool:
    return any(
        int(horizon) > 0
        and int(position) + int(horizon) < int(n_observations)
        and int(position) + 1 >= 4 * int(horizon)
        for horizon in horizons
    )


def eligible_quarter_end_positions(
    dates: Sequence[object] | pd.DatetimeIndex,
    *,
    horizons: Sequence[int] = ORIGIN_SELECTION_HORIZONS_DAYS,
    min_training_observations: int = ROLLING_MIN_TRAINING_OBSERVATIONS,
) -> list[int]:
    """Return source-equivalent eligible quarter-end positions.

    The source groups by calendar quarter, retains each quarter's last row,
    requires position >= 504, and accepts an origin when at least one of the
    configured selection horizons has both a future realization and four
    training observations per forecast day.
    """
    index = _validated_dates(dates)
    minimum = int(min_training_observations)
    if minimum < 0:
        raise ValueError("minimum training observations must be nonnegative")
    horizon_values = tuple(int(value) for value in horizons if int(value) > 0)
    if not horizon_values:
        raise ValueError("at least one positive selection horizon is required")
    positions: list[int] = []
    periods = index.to_period("Q")
    for period in periods.unique():
        candidates = np.flatnonzero(periods == period)
        if candidates.size == 0:
            continue
        position = int(candidates[-1])
        if position >= minimum and _horizon_is_eligible(position, len(index), horizon_values):
            positions.append(position)
    return positions


def select_rolling_origin_positions(
    dates: Sequence[object] | pd.DatetimeIndex,
    *,
    max_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
    horizons: Sequence[int] = ORIGIN_SELECTION_HORIZONS_DAYS,
    min_training_observations: int = ROLLING_MIN_TRAINING_OBSERVATIONS,
    policy: str = "full_history_even",
) -> list[int]:
    eligible = eligible_quarter_end_positions(
        dates,
        horizons=horizons,
        min_training_observations=min_training_observations,
    )
    normalized = str(policy).strip().lower()
    if normalized == "full_history_even":
        return select_full_history_even(eligible, count=int(max_origins))
    if normalized in {"recent", "last"}:
        count = max(0, int(max_origins))
        return eligible[-count:] if count else []
    if normalized in {"last_80_20", "last80_20"}:
        count = max(0, int(max_origins))
        start = max(0, len(eligible) - max(count, 1) * 4)
        return select_full_history_even(eligible[start:], count=count)
    raise ValueError(f"unsupported rolling-origin policy: {policy!r}")


def rolling_horizon_cap(
    position: int,
    n_observations: int,
    *,
    explicit_cap: int = 0,
) -> int:
    """Return daily rolling horizons 1..min(forward, floor(training/4))."""
    pos = int(position)
    n = int(n_observations)
    if pos < 0 or pos >= n - 1:
        raise ValueError("origin position must leave at least one future observation")
    result = min(n - pos - 1, (pos + 1) // 4)
    if int(explicit_cap) > 0:
        result = min(result, int(explicit_cap))
    return max(0, int(result))


def temporal_holdouts(n_observations: int) -> list[TemporalOrigin]:
    n = int(n_observations)
    if n < 81:
        raise ValueError("at least 81 observations are required")
    output: list[TemporalOrigin] = []
    for split_name, train_fraction in TEMPORAL_SPLITS:
        position = floor(float(n) * float(train_fraction)) - 1
        position = max(79, min(position, n - 2))
        output.append(TemporalOrigin(split_name, train_fraction, position, n - position - 1))
    return output


def expected_origin_tasks(portfolios: int = 80, rolling_origins: int = 48) -> int:
    return int(portfolios) * (int(rolling_origins) + len(TEMPORAL_SPLITS))


def expected_dense_cell_count(
    n_observations: int,
    *,
    portfolios: int = 80,
    rolling_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
    horizons: Sequence[int] = ORIGIN_SELECTION_HORIZONS_DAYS,
    min_training_observations: int = ROLLING_MIN_TRAINING_OBSERVATIONS,
) -> int:
    """Count the unique portfolio/horizon cells implied by actual rules.

    Temporal origins expose their complete future, while rolling origins are
    capped by the four-observations-per-day rule. Since the temporal quarter
    split has the largest future, the canonical 80-panel identity is 80 *
    8,766 = 701,280 cells.
    """
    n = int(n_observations)
    dates = pd.bdate_range("2000-01-03", periods=n)
    rolling = select_rolling_origin_positions(
        dates,
        max_origins=int(rolling_origins),
        horizons=horizons,
        min_training_observations=int(min_training_observations),
    )
    rolling_max = max((rolling_horizon_cap(pos, n) for pos in rolling), default=0)
    temporal_max = max((item.max_horizon for item in temporal_holdouts(n)), default=0)
    return int(portfolios) * max(rolling_max, temporal_max)


@dataclass(frozen=True)
class OriginDescriptor:
    """Value-free descriptor for one canonical forecast task."""

    origin_label: str
    evaluation_split: str
    position: int
    origin_date: str
    max_horizon: int
    train_fraction: float | None = None

    @property
    def horizon_days(self) -> tuple[int, ...]:
        return tuple(range(1, int(self.max_horizon) + 1))

    def horizon_mask(self, n_observations: int) -> np.ndarray:
        """Return a boolean mask over future rows for this descriptor."""
        forward = int(n_observations) - int(self.position) - 1
        if forward < 0 or int(self.max_horizon) > forward:
            raise ValueError("descriptor horizon exceeds the supplied calendar")
        mask = np.zeros(forward, dtype=bool)
        mask[: int(self.max_horizon)] = True
        return mask


def origin_schedule(
    dates: Sequence[object] | pd.DatetimeIndex,
    *,
    max_rolling_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
    selection_horizons: Sequence[int] = ORIGIN_SELECTION_HORIZONS_DAYS,
    min_training_observations: int = ROLLING_MIN_TRAINING_OBSERVATIONS,
    policy: str = "full_history_even",
) -> tuple[OriginDescriptor, ...]:
    """Build the canonical value-free origin/date/horizon schedule."""
    index = _validated_dates(dates)
    rolling_positions = select_rolling_origin_positions(
        index,
        max_origins=int(max_rolling_origins),
        horizons=selection_horizons,
        min_training_observations=int(min_training_observations),
        policy=policy,
    )
    descriptors: list[OriginDescriptor] = []
    for ordinal, position in enumerate(rolling_positions, start=1):
        descriptors.append(
            OriginDescriptor(
                origin_label=f"rolling_{ordinal:02d}",
                evaluation_split="rolling_origin",
                position=int(position),
                origin_date=pd.Timestamp(index[position]).date().isoformat(),
                max_horizon=rolling_horizon_cap(int(position), len(index)),
            )
        )
    for holdout in temporal_holdouts(len(index)):
        descriptors.append(
            OriginDescriptor(
                origin_label=holdout.split_name,
                evaluation_split=holdout.split_name,
                position=int(holdout.position),
                origin_date=pd.Timestamp(index[holdout.position]).date().isoformat(),
                max_horizon=int(holdout.max_horizon),
                train_fraction=float(holdout.train_fraction),
            )
        )
    return tuple(descriptors)
