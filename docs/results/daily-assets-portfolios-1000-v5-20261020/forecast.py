"""Versioned1,000-path runner; exact original empirical CRPS, horizon batches."""
from time import perf_counter
import numpy as np
import pandas as pd
import streaming
from simfolio_forecasting_methodology.french_panel import runner as original
from simfolio_forecasting_methodology.runner import empirical_crps_by_horizon


def forecast(model, series_id, values, dates, origin, seed):
    stop, horizon = int(origin.position)+1, int(origin.max_horizon)
    history = original._readonly_copy(np.asarray(values[:stop], dtype=np.float64))
    # Keep the same input and synthetic-policy-calendar validation as native.
    training = original.TrainingData(portfolio_log_returns=history,
        asset_log_returns=history.reshape(-1, 1),
        policy=original.PortfolioPolicy((series_id,), (1.,), 'none'),
        training_dates=original._readonly_copy(dates[:stop]))
    training.validate()
    future = pd.bdate_range(start=pd.Timestamp(origin.origin_date)+pd.offsets.BDay(1),
                           periods=horizon).to_numpy(dtype='datetime64[ns]')
    context = original.ForecastContext(model_id=model.model_id, portfolio_id=series_id,
        origin_label=origin.origin_label, horizon_days=horizon, simulations=streaming.PATHS,
        seed=seed, future_dates=original._readonly_copy(future), origin_date=origin.origin_date)
    started = perf_counter()
    realized = np.cumsum(np.asarray(values[stop:stop+horizon], dtype=np.float64))
    losses = np.empty(horizon, dtype=np.float64)
    cursor = 0
    for start, terminal in model.iter_terminal_log_returns(training, context):
        if start != cursor or terminal.shape[0] != context.simulations:
            raise ValueError('Streaming forecast coverage/path count differs')
        end = start+terminal.shape[1]
        losses[start:end] = empirical_crps_by_horizon(terminal, realized[start:end])
        cursor = end
    if cursor != horizon or not np.isfinite(losses).all():
        raise ValueError('Incomplete or nonfinite forecast losses')
    elapsed = perf_counter()-started
    with np.errstate(over='ignore', invalid='ignore'):
        cagr = np.expm1(terminal[:, -1]*(252./horizon))
        diagnostics = {'forecast_and_score_seconds': elapsed,
            'timing_scope': 'training-only native-law forecast and exact1,000-member CRPS; excludes checkpoint IO',
            'horizon_days': horizon, 'simulations': context.simulations,
            'horizon_chunk': streaming.WIDTH, 'all_paths_scored_at_every_horizon': True,
            'annualization_days': 252,
            'forecast_mean_cagr': original._finite_float(np.mean(cagr)),
            'forecast_median_cagr': original._finite_float(np.median(cagr)),
            'training_historical_cagr': original._finite_float(np.expm1(np.mean(history)*252.)),
            'nonfinite_cagr_paths': int(np.count_nonzero(~np.isfinite(cagr))),
            'diagnostic_only_no_exclusions': True, 'execution_version': streaming.NAMESPACE}
    return losses, diagnostics
