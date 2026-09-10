import numpy as np
import pandas as pd

from simfolio_forecasting_methodology.data import CANONICAL_TICKERS
from simfolio_forecasting_methodology.experiment import build_experiment_plan_from_returns


def test_synthetic_task_constructor_is_bounded_and_aligned():
    dates = pd.bdate_range("2000-01-03", periods=365)
    values = np.zeros((len(dates), len(CANONICAL_TICKERS)), dtype=float)
    returns = pd.DataFrame(values, index=dates, columns=CANONICAL_TICKERS)
    plan = build_experiment_plan_from_returns(
        returns,
        portfolio_limit=1,
        rolling_origins=48,
    )
    assert len(plan.portfolios) == 1
    assert len(plan.tasks) == 3
    task = plan.tasks[0]
    assert task.training.training_dates is not None
    assert len(task.training.training_dates) == len(task.training.portfolio_log_returns)
    assert task.future_dates is not None
    assert len(task.future_dates) == task.horizon_days
    expected_first_policy_date = pd.Timestamp(task.training.training_dates[-1]) + pd.offsets.BDay(1)
    assert pd.Timestamp(task.future_dates[0]) == expected_first_policy_date
    assert task.training.asset_log_returns is not None
    assert task.training.asset_log_returns.base is not None
