"""Bounded research tasks made only from the packaged synthetic fixture."""

from importlib.resources import files

import numpy as np
import pandas as pd

from .models.asset_level.frontier import _historical_rebalance_dates
from .models.numerical.dynamic_gaussian import rebalanced_portfolio_log_paths
from .runner import OriginTask, PortfolioPolicy, TrainingData
from .seeds import deterministic_seed


def fixture_smoke_tasks(*, origins: int = 2, horizon: int = 63) -> list[OriginTask]:
    """Create at most two origins, one synthetic portfolio, and 63 daily horizons."""
    if not 1 <= origins <= 2 or not 1 <= horizon <= 63:
        raise ValueError("fixture smoke requires 1–2 origins and 1–63 horizons")
    resource = files("simfolio_forecasting_methodology").joinpath(
        "resources/test_fixtures/frontier/source_reference.npz"
    )
    with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as fixture:
        assets = fixture["asset_log_returns"].copy()
        weights = fixture["weights"].copy()
        dates = pd.DatetimeIndex(fixture["training_dates"])
    policy = PortfolioPolicy(("ALPHA", "BETA", "GAMMA", "DELTA"), tuple(weights), "monthly")
    rebalance_dates = _historical_rebalance_dates(dates, policy.rebalance)
    mask = np.asarray([date in rebalance_dates for date in dates], dtype=bool)
    portfolio = rebalanced_portfolio_log_paths(assets[None, :, :], weights, mask)[0]
    tasks = []
    for ordinal in range(origins):
        stop = len(dates) - horizon - ordinal
        origin_date = dates[stop - 1].date().isoformat()
        tasks.append(OriginTask(
            portfolio_id="synthetic-fixture-not-canonical-data",
            origin_label=f"fixture_smoke_{ordinal + 1}",
            training=TrainingData(
                portfolio[:stop], assets[:stop], policy, dates[:stop].to_numpy()
            ),
            realized_future_daily_log_returns=portfolio[stop:stop + horizon],
            seed=deterministic_seed("synthetic-fixture-smoke", origin_date, horizon),
            future_dates=pd.bdate_range(
                dates[stop - 1] + pd.offsets.BDay(1), periods=horizon
            ).to_numpy(),
            origin_date=origin_date,
        ))
    return tasks
