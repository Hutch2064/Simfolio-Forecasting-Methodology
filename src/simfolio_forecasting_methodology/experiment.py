"""End-to-end construction and execution of the dense OOS experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import CANONICAL_END, CANONICAL_START, load_canonical_returns
from .origins import ROLLING_ORIGINS_PER_PORTFOLIO, select_full_history_even, temporal_holdouts
from .panel import PortfolioSpec, generate_equal_class_history_panel
from .runner import OriginTask, PortfolioPolicy, TrainingData, evaluate_model
from .seeds import deterministic_seed

BASE_SEED = 20260528
EXPECTED_PORTFOLIOS = 80
EXPECTED_ORIGINS_PER_PORTFOLIO = 51
EXPECTED_CELLS_PER_MODEL = 701_280


@dataclass(frozen=True)
class ExperimentPlan:
    common_dates: pd.DatetimeIndex
    portfolios: tuple[PortfolioSpec, ...]
    tasks: tuple[OriginTask, ...]

    @property
    def cell_capacity(self) -> int:
        by_portfolio: dict[str, int] = {}
        for task in self.tasks:
            by_portfolio[task.portfolio_id] = max(
                by_portfolio.get(task.portfolio_id, 0), task.horizon_days
            )
        return int(sum(by_portfolio.values()))


def _portfolio_log_returns(
    simple_returns: pd.DataFrame, weights: tuple[float, ...], frequency: str
) -> pd.Series:
    target = np.asarray(weights, dtype=np.float64)
    holdings = target.copy()
    output = np.empty(len(simple_returns), dtype=np.float64)
    prior_period: object | None = None
    for row_index, (date, row) in enumerate(simple_returns.iterrows()):
        previous = max(float(np.sum(holdings)), 1e-300)
        holdings *= 1.0 + np.asarray(row, dtype=np.float64)
        ending = max(float(np.sum(holdings)), 1e-300)
        output[row_index] = np.log(ending / previous)
        normalized = str(frequency).lower()
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


def _portfolio_tasks(
    spec: PortfolioSpec,
    returns: pd.DataFrame,
    *,
    rolling_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
) -> list[OriginTask]:
    assets = returns.loc[:, list(spec.tickers)]
    asset_log = np.log1p(assets.to_numpy(dtype=np.float64))
    portfolio_log = _portfolio_log_returns(assets, spec.weights, spec.rebalance).to_numpy()
    n = len(assets)
    holdouts = temporal_holdouts(n)
    first_position = holdouts[0].position
    # The dense protocol uses the same 25%-training boundary as the earliest
    # temporal holdout.  Rolling origins are evenly distributed over the
    # admissible region from that boundary through the penultimate observation.
    rolling_positions = select_full_history_even(
        list(range(first_position, n - 1)), count=int(rolling_origins)
    )
    labeled_positions: list[tuple[str, int]] = [
        (f"rolling_{index + 1:02d}", position)
        for index, position in enumerate(rolling_positions)
    ]
    labeled_positions.extend((item.split_name, item.position) for item in holdouts)
    policy = PortfolioPolicy(spec.tickers, spec.weights, spec.rebalance)
    tasks: list[OriginTask] = []
    for label, position in labeled_positions:
        seed = deterministic_seed(BASE_SEED, spec.name, label, int(position))
        training = TrainingData(
            portfolio_log_returns=portfolio_log[: position + 1].copy(),
            asset_log_returns=asset_log[: position + 1, :].copy(),
            policy=policy,
        )
        tasks.append(
            OriginTask(
                portfolio_id=spec.name,
                origin_label=label,
                training=training,
                realized_future_daily_log_returns=portfolio_log[position + 1 :].copy(),
                seed=seed,
            )
        )
    return tasks


def build_experiment_plan(
    data_root: Path,
    *,
    portfolio_limit: int | None = None,
    rolling_origins: int = ROLLING_ORIGINS_PER_PORTFOLIO,
) -> ExperimentPlan:
    raw = load_canonical_returns(Path(data_root))
    panel = generate_equal_class_history_panel()
    required = sorted({ticker for item in panel for ticker in item.tickers})
    common = raw.loc[CANONICAL_START:CANONICAL_END, required].dropna(how="any")
    if len(common) < 10_000:
        raise ValueError(
            f"canonical public panel has only {len(common)} common observations; expected long history"
        )
    selected = panel if portfolio_limit is None else panel[: max(1, int(portfolio_limit))]
    tasks: list[OriginTask] = []
    for spec in selected:
        tasks.extend(_portfolio_tasks(spec, common, rolling_origins=rolling_origins))
    plan = ExperimentPlan(common.index, tuple(selected), tuple(tasks))
    if portfolio_limit is None and rolling_origins == ROLLING_ORIGINS_PER_PORTFOLIO:
        if len(plan.portfolios) != EXPECTED_PORTFOLIOS:
            raise AssertionError("canonical portfolio count drifted")
        if len(plan.tasks) != EXPECTED_PORTFOLIOS * EXPECTED_ORIGINS_PER_PORTFOLIO:
            raise AssertionError("canonical origin-task count drifted")
        # Public-source files are periodically refreshed.  Exact historical data
        # should yield 701,280 cells; rebuilt public data is allowed a small
        # calendar difference but the value is written into every run manifest.
        if abs(plan.cell_capacity - EXPECTED_CELLS_PER_MODEL) > EXPECTED_PORTFOLIOS * 5:
            raise AssertionError(
                f"canonical cell capacity drifted materially: {plan.cell_capacity}"
            )
    return plan


def run_model(
    model,
    plan: ExperimentPlan,
    *,
    simulations: int,
) -> dict[str, object]:
    accumulator = evaluate_model(model, list(plan.tasks), simulations=int(simulations))
    score = accumulator.model_score()
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


def iter_smoke_tasks(plan: ExperimentPlan, count: int = 2, horizon: int = 63) -> Iterable[OriginTask]:
    for task in plan.tasks[: max(1, int(count))]:
        realized = np.asarray(task.realized_future_daily_log_returns)[: int(horizon)]
        yield OriginTask(
            portfolio_id=task.portfolio_id,
            origin_label=task.origin_label + "_smoke",
            training=task.training,
            realized_future_daily_log_returns=realized,
            seed=task.seed,
        )
