"""Train only on constituent prefixes; score all1,000 terminal portfolio paths."""
from time import perf_counter
import numpy as np
from inputs import AssetInputs
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon


def forecast(model, panel_root, row, values, dates, origin, seed):
    stop, horizon = origin.position+1, origin.max_horizon
    started = perf_counter()
    history, weights, ids = AssetInputs(panel_root).training(row, stop)
    realized = np.cumsum(values[stop:stop+horizon])
    loss = np.empty(horizon); cursor = 0
    for start, terminal in model.terminal_chunks(history, weights, horizon, 1000, seed):
        if start != cursor or terminal.shape[0] != 1000:
            raise ValueError('Asset forecast path count or coverage differs')
        end = start+terminal.shape[1]
        loss[start:end] = empirical_crps_by_horizon(terminal, realized[start:end])
        cursor = end
    if cursor != horizon or not np.isfinite(loss).all():
        raise ValueError('Incomplete or nonfinite asset forecast loss')
    diagnostics = {'forecast_and_score_seconds': perf_counter()-started,
        'timing_scope': 'causal asset interval preparation, joint simulation and exact1000-member CRPS; excludes checkpoint IO',
        'simulations': 1000, 'all_paths_scored_at_every_horizon': True,
        'horizon_days': horizon, 'horizon_chunk': 64, 'constituent_ids': ids,
        'joint_asset_paths_simulated': True, 'portfolio_weights_applied_inside_paths': True,
        'forecast_level': 'joint_assets', 'native_training_rows': len(history)}
    return loss, diagnostics
