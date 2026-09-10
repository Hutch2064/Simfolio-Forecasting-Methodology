"""Catalogue-independent execution interfaces for dense OOS evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .evaluation import CellAccumulator, empirical_crps_by_horizon


@dataclass(frozen=True)
class PortfolioPolicy:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    rebalance: str

    def validate(self) -> None:
        if not self.tickers:
            raise ValueError("portfolio policy requires at least one ticker")
        if len(self.tickers) != len(self.weights):
            raise ValueError("ticker and weight lengths differ")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-10):
            raise ValueError("portfolio weights must sum to one")


@dataclass(frozen=True)
class TrainingData:
    portfolio_log_returns: np.ndarray
    asset_log_returns: np.ndarray | None = None
    policy: PortfolioPolicy | None = None
    training_dates: np.ndarray | None = None

    def validate(self) -> None:
        portfolio = np.asarray(self.portfolio_log_returns, dtype=np.float64)
        if portfolio.ndim != 1 or portfolio.size < 1 or not np.all(np.isfinite(portfolio)):
            raise ValueError("portfolio_log_returns must be a finite one-dimensional series")
        if self.training_dates is not None:
            dates = np.asarray(self.training_dates, dtype="datetime64[ns]")
            if dates.shape != portfolio.shape or np.any(np.isnat(dates)):
                raise ValueError("training_dates must align with portfolio history")
            if np.any(dates[1:] <= dates[:-1]):
                raise ValueError("training_dates must be strictly increasing")
        if self.asset_log_returns is not None:
            assets = np.asarray(self.asset_log_returns, dtype=np.float64)
            if assets.ndim != 2 or assets.shape[0] != portfolio.size:
                raise ValueError("asset_log_returns must align row-wise with portfolio history")
            if not np.all(np.isfinite(assets)):
                raise ValueError("asset_log_returns must be finite")
            if self.policy is None:
                raise ValueError("asset-level training data requires a portfolio policy")
            self.policy.validate()
            if assets.shape[1] != len(self.policy.tickers):
                raise ValueError("asset columns must align with policy tickers")


@dataclass(frozen=True)
class ForecastContext:
    model_id: str
    portfolio_id: str
    origin_label: str
    horizon_days: int
    simulations: int
    seed: int
    future_dates: np.ndarray | None = None
    origin_date: str | None = None


@runtime_checkable
class ForecastModel(Protocol):
    model_id: str

    def simulate_daily_log_returns(
        self,
        training: TrainingData,
        context: ForecastContext,
    ) -> np.ndarray:
        """Return a matrix shaped ``(simulations, horizon_days)``."""
        ...


@dataclass(frozen=True)
class OriginTask:
    portfolio_id: str
    origin_label: str
    training: TrainingData
    realized_future_daily_log_returns: np.ndarray
    seed: int
    future_dates: np.ndarray | None = None

    @property
    def horizon_days(self) -> int:
        return int(np.asarray(self.realized_future_daily_log_returns).size)

    def validate(self) -> None:
        self.training.validate()
        realized = np.asarray(self.realized_future_daily_log_returns, dtype=np.float64)
        if realized.ndim != 1 or realized.size < 1 or not np.all(np.isfinite(realized)):
            raise ValueError("realized future returns must be a finite nonempty vector")
        if self.future_dates is not None and len(self.future_dates) != realized.size:
            raise ValueError("future_dates must align with realized future returns")


def evaluate_origin_task(
    model: ForecastModel,
    task: OriginTask,
    *,
    simulations: int,
) -> np.ndarray:
    """Generate one coherent path matrix and score every daily horizon."""
    task.validate()
    context = ForecastContext(
        model_id=model.model_id,
        portfolio_id=task.portfolio_id,
        origin_label=task.origin_label,
        horizon_days=task.horizon_days,
        simulations=int(simulations),
        seed=int(task.seed),
        future_dates=None if task.future_dates is None else np.asarray(task.future_dates),
    )
    daily_paths = np.asarray(model.simulate_daily_log_returns(task.training, context), dtype=np.float64)
    expected_shape = (int(simulations), task.horizon_days)
    if daily_paths.shape != expected_shape:
        raise ValueError(f"model returned path shape {daily_paths.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(daily_paths)):
        raise ValueError("model returned nonfinite forecast paths")
    terminal_samples = np.cumsum(daily_paths, axis=1, dtype=np.float64)
    realized_terminal = np.cumsum(
        np.asarray(task.realized_future_daily_log_returns, dtype=np.float64), dtype=np.float64
    )
    return empirical_crps_by_horizon(terminal_samples, realized_terminal)


def evaluate_model(
    model: ForecastModel,
    tasks: list[OriginTask],
    *,
    simulations: int,
) -> CellAccumulator:
    """Evaluate a model over origin tasks using streaming cell-first aggregation."""
    accumulator = CellAccumulator()
    for task in tasks:
        losses = evaluate_origin_task(model, task, simulations=simulations)
        accumulator.add_vector(task.portfolio_id, losses)
    return accumulator
