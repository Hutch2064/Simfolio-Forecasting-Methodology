"""Preserve canonical80 and add its unique tickers as a separate scoring block."""
from dataclasses import dataclass

import numpy as np

from simfolio_forecasting_methodology.data import CANONICAL_TICKERS, load_canonical_engine_inputs
from simfolio_forecasting_methodology.experiment import _materialize_tasks_from_logs
from simfolio_forecasting_methodology.origins import origin_schedule
from simfolio_forecasting_methodology.panel import generate_equal_class_history_panel


@dataclass(frozen=True)
class SingleAsset:
    name: str
    tickers: tuple[str, ...]
    weights: tuple[float, ...] = (1.0,)
    rebalance: str = "none"


def single_asset_tasks(data_root):
    """Use exact frozen price-derived logs, not the raw daily_return CSV field.

    A 100% holding without fees or rebalancing has the asset's own log return.
    No synthetic starting-capital roundoff is introduced into that identity.
    Original80 portfolio targets, policies and dates are never reconstructed.
    """
    original = generate_equal_class_history_panel()
    tickers = sorted({ticker for portfolio in original for ticker in portfolio.tickers})
    if tuple(tickers) != CANONICAL_TICKERS:
        raise ValueError("original80 unique ticker membership drifted")
    assets, _ = load_canonical_engine_inputs(data_root)
    descriptors = origin_schedule(assets.index)
    tasks = []
    for ticker in tickers:
        spec = SingleAsset("single_" + ticker, (ticker,))
        tasks.extend(_materialize_tasks_from_logs(
            spec, assets.index, assets, assets[ticker], descriptors
        ))
    if len(tasks) != 2652 or max(t.horizon_days for t in tasks) != 8766:
        raise ValueError("singleton origin/horizon contract drifted")
    return tuple(tasks)


def block_cells(tasks, losses):
    """Deterministic origin-first reduction with complete-denominator gates."""
    if len(tasks) != len(losses):
        raise ValueError("missing tasks")
    names = sorted({task.portfolio_id for task in tasks})
    positions = {name: index for index, name in enumerate(names)}
    width = max(task.horizon_days for task in tasks)
    sums = np.zeros((len(names), width))
    counts = np.zeros(sums.shape, dtype=np.int64)
    expected = np.zeros_like(counts)
    for task, vector in zip(tasks, losses):
        row = positions[task.portfolio_id]
        expected[row, :task.horizon_days] += 1
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (task.horizon_days,) or not np.isfinite(vector).all():
            raise ValueError("missing, malformed or nonfinite origin loss vector")
        sums[row, :len(vector)] += vector
        counts[row, :len(vector)] += 1
    if not np.array_equal(counts, expected) or np.any(counts == 0):
        raise ValueError("incomplete portfolio-horizon denominator")
    return names, sums / counts, counts


def combined_score(original80, single52):
    if not np.isfinite([original80, single52]).all():
        raise ValueError("both complete block scores are required")
    return float(0.5 * original80 + 0.5 * single52)
