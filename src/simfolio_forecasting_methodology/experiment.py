"""Canonical schedule construction and value-materialization helpers.

The schedule is reproducible from the frozen calendar and panel metadata alone.
Value-bearing training tasks are materialized only from a caller-provided,
verified return frame; no source arrays are packaged or copied into task
records.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CANONICAL_TICKERS,
    canonical_calendar,
    load_canonical_engine_inputs,
)
from .origins import (
    ORIGIN_SELECTION_HORIZONS_DAYS,
    ROLLING_ORIGINS_PER_PORTFOLIO,
    OriginDescriptor,
    origin_schedule,
)
from .panel import PortfolioSpec, generate_equal_class_history_panel
from .runner import OriginTask, PortfolioPolicy, TrainingData, evaluate_model
from .seeds import coherent_daily_seed

BASE_SEED = 20260528
EXPECTED_PORTFOLIOS = 80
EXPECTED_ORIGINS_PER_PORTFOLIO = 51
EXPECTED_CELLS_PER_MODEL = 701_280


@dataclass(frozen=True)
class ScheduledTask:
    """Value-free task identity; arrays are supplied only during materialization."""

    portfolio_id: str
    portfolio_index: int
    descriptor: OriginDescriptor

    @property
    def origin_label(self) -> str:
        return self.descriptor.origin_label

    @property
    def origin_date(self) -> str:
        return self.descriptor.origin_date

    @property
    def position(self) -> int:
        return self.descriptor.position

    @property
    def max_horizon(self) -> int:
        return self.descriptor.max_horizon


@dataclass(frozen=True)
class ExperimentSchedule:
    """Complete origin/date/mask schedule without financial values."""

    common_dates: pd.DatetimeIndex
    portfolios: tuple[PortfolioSpec, ...]
    tasks: tuple[ScheduledTask, ...]

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def cell_capacity(self) -> int:
        by_portfolio: dict[str, int] = {}
        for task in self.tasks:
            by_portfolio[task.portfolio_id] = max(
                by_portfolio.get(task.portfolio_id, 0), task.max_horizon
            )
        return int(sum(by_portfolio.values()))

    def horizon_mask(self, task: ScheduledTask) -> np.ndarray:
        return task.descriptor.horizon_mask(len(self.common_dates))


@dataclass(frozen=True)
class ExperimentPlan:
    common_dates: pd.DatetimeIndex
    portfolios: tuple[PortfolioSpec, ...]
    tasks: tuple[OriginTask, ...]
    schedule: ExperimentSchedule | None = None

    @property
    def cell_capacity(self) -> int:
        if self.schedule is not None:
            return self.schedule.cell_capacity
        by_portfolio: dict[str, int] = {}
        for task in self.tasks:
            by_portfolio[task.portfolio_id] = max(
                by_portfolio.get(task.portfolio_id, 0), task.horizon_days
            )
        return int(sum(by_portfolio.values()))

    @property
    def task_count(self) -> int:
        return len(self.tasks)


def _validate_calendar(dates: pd.DatetimeIndex, *, require_canonical: bool) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("experiment dates must be unique and sorted ascending")
    if require_canonical:
        expected = canonical_calendar()
        if not index.equals(expected):
            raise ValueError("canonical input dates do not match the frozen calendar")
    return index


def build_experiment_schedule(
    dates: pd.DatetimeIndex | None = None,
    *,
    portfolio_limit: int | None = None,
    rolling_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
    require_canonical: bool = True,
) -> ExperimentSchedule:
    """Construct exact task descriptors without loading financial arrays."""
    index = canonical_calendar() if dates is None else _validate_calendar(
        dates, require_canonical=require_canonical
    )
    panel = generate_equal_class_history_panel()
    selected = panel if portfolio_limit is None else panel[: max(1, int(portfolio_limit))]
    descriptors = origin_schedule(
        index,
        max_rolling_origins=int(rolling_origins),
        selection_horizons=ORIGIN_SELECTION_HORIZONS_DAYS,
    )
    tasks = tuple(
        ScheduledTask(spec.name, portfolio_index, descriptor)
        for portfolio_index, spec in enumerate(selected, start=1)
        for descriptor in descriptors
    )
    schedule = ExperimentSchedule(index, tuple(selected), tasks)
    if portfolio_limit is None and int(rolling_origins) == ROLLING_ORIGINS_PER_PORTFOLIO:
        if len(schedule.portfolios) != EXPECTED_PORTFOLIOS:
            raise AssertionError("canonical portfolio count drifted")
        if schedule.task_count != EXPECTED_PORTFOLIOS * EXPECTED_ORIGINS_PER_PORTFOLIO:
            raise AssertionError("canonical origin-task count drifted")
        if schedule.cell_capacity != EXPECTED_CELLS_PER_MODEL:
            raise AssertionError(f"canonical cell capacity drifted: {schedule.cell_capacity}")
    return schedule


# Explicit alias for callers that only need the value-free protocol proof.
build_protocol_schedule = build_experiment_schedule


def _portfolio_log_returns(
    simple_returns: pd.DataFrame, weights: tuple[float, ...], frequency: str
) -> pd.Series:
    target = np.asarray(weights, dtype=np.float64)
    holdings = target.copy()
    output = np.empty(len(simple_returns), dtype=np.float64)
    prior_period: object | None = None
    normalized = str(frequency).lower()
    for row_index, (date, row) in enumerate(simple_returns.iterrows()):
        previous = max(float(np.sum(holdings)), 1e-300)
        holdings *= 1.0 + np.asarray(row, dtype=np.float64)
        ending = max(float(np.sum(holdings)), 1e-300)
        output[row_index] = np.log(ending / previous)
        if normalized == "monthly":
            period: object = (date.year, date.month)
        elif normalized == "quarterly":
            period = (date.year, (date.month - 1) // 3)
        elif normalized in {"annually", "annual", "yearly"}:
            period = date.year
        else:
            period = None
        if period is not None and prior_period is not None and period != prior_period:
            holdings = ending * target
        prior_period = period
    return pd.Series(output, index=simple_returns.index)


def _materialize_tasks_from_logs(
    spec: PortfolioSpec,
    common_dates: pd.DatetimeIndex,
    asset_log_frame: pd.DataFrame,
    portfolio_log_series: pd.Series,
    descriptors: tuple[OriginDescriptor, ...],
) -> list[OriginTask]:
    asset_values = asset_log_frame.loc[common_dates, list(spec.tickers)].to_numpy(
        dtype=np.float64, copy=False
    )
    portfolio_values = portfolio_log_series.loc[common_dates].to_numpy(
        dtype=np.float64, copy=False
    )
    if not np.all(np.isfinite(asset_values)) or not np.all(np.isfinite(portfolio_values)):
        raise ValueError(f"nonfinite source-derived returns for {spec.name}")
    dates = common_dates.to_numpy(dtype="datetime64[ns]", copy=True)
    policy = PortfolioPolicy(spec.tickers, spec.weights, spec.rebalance)
    tasks: list[OriginTask] = []
    for descriptor in descriptors:
        position = int(descriptor.position)
        stop = position + 1 + int(descriptor.max_horizon)
        seed = coherent_daily_seed(descriptor.origin_date, descriptor.max_horizon, 240)
        training = TrainingData(
            # Every origin shares these two per-portfolio arrays through views;
            # task construction does not copy a history 4,080 times.
            portfolio_log_returns=portfolio_values[: position + 1],
            asset_log_returns=asset_values[: position + 1, :],
            policy=policy,
            training_dates=dates[: position + 1],
        )
        # The realized target is indexed by observed source dates.  The
        # forecast policy calendar follows the historical Frontier adapter:
        # business days synthesized from the last training date.  Holidays
        # therefore affect target indexing and policy rebalancing separately.
        policy_dates = pd.bdate_range(
            start=pd.Timestamp(descriptor.origin_date) + pd.offsets.BDay(1),
            periods=int(descriptor.max_horizon),
        ).to_numpy(dtype="datetime64[ns]")
        values = {
            "portfolio_id": spec.name,
            "origin_label": descriptor.origin_label,
            "training": training,
            "realized_future_daily_log_returns": portfolio_values[position + 1 : stop],
            "seed": seed,
            "future_dates": policy_dates,
        }
        # The shared runner integration adds this optional field.  Keeping the
        # constructor compatible with the pre-integration checkout lets the
        # schedule/data commit be tested independently and carries the exact
        # origin identity once that runner field is present.
        if "origin_date" in getattr(OriginTask, "__dataclass_fields__", {}):
            values["origin_date"] = descriptor.origin_date
        tasks.append(OriginTask(**values))
    return tasks


def _materialize_portfolio_tasks(
    spec: PortfolioSpec,
    returns: pd.DataFrame,
    descriptors: tuple[OriginDescriptor, ...],
) -> list[OriginTask]:
    assets = returns.loc[:, list(spec.tickers)]
    asset_values = assets.to_numpy(dtype=np.float64, copy=False)
    if not np.all(np.isfinite(asset_values)) or np.any(asset_values <= -1.0):
        raise ValueError(f"nonfinite or invalid simple returns for {spec.name}")
    asset_log = pd.DataFrame(
        np.log1p(asset_values), index=assets.index, columns=assets.columns
    )
    portfolio_series = _portfolio_log_returns(assets, spec.weights, spec.rebalance)
    return _materialize_tasks_from_logs(
        spec, assets.index, asset_log, portfolio_series, descriptors
    )


def build_experiment_plan_from_returns(
    returns: pd.DataFrame,
    *,
    portfolio_limit: int | None = None,
    rolling_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
    require_canonical: bool = False,
) -> ExperimentPlan:
    """Materialize tasks from caller-provided simple returns.

    The caller owns the values. This helper is also the bounded synthetic-fixture
    entry point for tests; canonical runs require the verified data loader.
    """
    raw = returns.copy(deep=False)
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if require_canonical:
        if set(raw.columns) != set(CANONICAL_TICKERS):
            raise ValueError("canonical return columns do not match the frozen 52-series identity")
        common = _validate_calendar(raw.index, require_canonical=True)
    else:
        common = _validate_calendar(raw.index, require_canonical=False)
    panel = generate_equal_class_history_panel()
    selected = panel if portfolio_limit is None else panel[: max(1, int(portfolio_limit))]
    required = sorted({ticker for item in selected for ticker in item.tickers})
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"return frame is missing panel tickers: {missing}")
    common_frame = raw.loc[common, required]
    if common_frame.isna().any().any():
        raise ValueError("return frame contains missing values in the scored panel")
    schedule = build_experiment_schedule(
        common,
        portfolio_limit=portfolio_limit,
        rolling_origins=rolling_origins,
        require_canonical=require_canonical,
    )
    tasks: list[OriginTask] = []
    for spec in selected:
        tasks.extend(_materialize_portfolio_tasks(spec, common_frame, tuple(
            task.descriptor for task in schedule.tasks if task.portfolio_id == spec.name
        )))
    return ExperimentPlan(common, tuple(selected), tuple(tasks), schedule)


def build_experiment_plan(
    data_root: Path,
    *,
    portfolio_limit: int | None = None,
    rolling_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
) -> ExperimentPlan:
    """Load a verified cache and construct source-equivalent value-bearing tasks."""
    asset_logs, portfolio_logs = load_canonical_engine_inputs(Path(data_root))
    schedule = build_experiment_schedule(
        asset_logs.index,
        portfolio_limit=portfolio_limit,
        rolling_origins=rolling_origins,
        require_canonical=True,
    )
    tasks: list[OriginTask] = []
    for spec in schedule.portfolios:
        descriptors = tuple(
            task.descriptor for task in schedule.tasks if task.portfolio_id == spec.name
        )
        portfolio_log = portfolio_logs.get(spec.name)
        if portfolio_log is None:
            raise ValueError(f"verified cache is missing portfolio log series {spec.name}")
        tasks.extend(
            _materialize_tasks_from_logs(
                spec,
                schedule.common_dates,
                asset_logs,
                portfolio_log,
                descriptors,
            )
        )
    return ExperimentPlan(schedule.common_dates, schedule.portfolios, tuple(tasks), schedule)


def run_model(model, plan: ExperimentPlan, *, simulations: int) -> dict[str, object]:
    accumulator = evaluate_model(model, list(plan.tasks), simulations=int(simulations))
    score = accumulator.aggregate_score()
    return {
        "model_id": model.model_id,
        "exact_empirical_crps": float(score),
        "cells": int(accumulator.cell_count),
        "origin_tasks": len(plan.tasks),
        "portfolios": len(plan.portfolios),
        "simulations": int(simulations),
        "first_date": str(plan.common_dates.min().date()),
        "last_date": str(plan.common_dates.max().date()),
    }


def write_result(path: Path, result: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def iter_smoke_tasks(
    plan: ExperimentPlan, count: int = 2, horizon: int = 63
) -> Iterable[OriginTask]:
    for task in plan.tasks[: max(1, int(count))]:
        realized = np.asarray(task.realized_future_daily_log_returns)[: int(horizon)]
        dates = None if task.future_dates is None else np.asarray(task.future_dates)[: int(horizon)]
        values = {
            "portfolio_id": task.portfolio_id,
            "origin_label": task.origin_label + "_smoke",
            "training": task.training,
            "realized_future_daily_log_returns": realized,
            "seed": task.seed,
            "future_dates": dates,
        }
        if "origin_date" in getattr(OriginTask, "__dataclass_fields__", {}):
            values["origin_date"] = getattr(task, "origin_date", None)
        yield OriginTask(**values)
