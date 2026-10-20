"""Current mean regularization and raw block half, multi-timescale FHS half."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
import harch
retained = parent.retained
MODEL_ID = 'model_testing_equal_bootstrap_HARCH_1_5_22_FHS_smooth_Gaussian_mean_pool_v1'


class MultiscaleComponent:
    model_id = 'HARCH_1_5_22_empirical_FHS_smooth_Gaussian_mean_v1'

    def simulate_daily_log_returns(self, training, context):
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared history')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        fitted = harch.fit(history)
        indices = args['rng'].integers(0, len(history), size=(context.simulations, context.horizon_days))
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        se = retained.gaussian_sd(df, scale)
        center = parent.regularized_mean(fitted['mean'], se)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)
        daily, clips, floors = harch.native_paths(fitted['pool'], indices, center, fitted['target_variance'],
                                                fitted['alpha'], fitted['last_squared_normalized'], delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite HARCH native forecast')
        self.last_metadata = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items() if k not in ['pool', 'last_squared_normalized']}
        self.last_metadata.update(regularized_mean=center, Gaussian_sd_unchanged=se, parameter_seed=seed,
            batch_scale=scale, batch_df=df, batch_count=batches, batch_long_run_variance=lrv,
            pool_mean=float(fitted['pool'].mean()), pool_variance=float(np.mean(fitted['pool']**2)),
            daily_clips=int(clips), wealth_floor_events=int(floors))
        return daily


class MultiscalePool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (parent.ShrunkComponent('WR'), MultiscaleComponent())

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'HARCH_source_sha256': hashlib.sha256((Path(__file__).parent/'harch.py').read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'replace_GARCH_FHS_half_by_variance_targeted_HARCH_1_5_22_FHS',
            'lags': list(harch.LAGS), 'stationarity_margin': harch.MARGIN, 'fit_starts': list(harch.STARTS),
            'prehistory_squared_normalized_residual': 1., 'mean_law_and_WR_unchanged': True,
            'empirical_training_innovations': 'centered_population_variance_normalized_filtered_residuals',
            'paths': 240, 'weights': [.5, .5], 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return MultiscalePool()
