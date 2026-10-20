"""Current shrunk-mean pool; replace only the FHS volatility estimation objective."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
from crps_fit import fit_crps
retained = parent.retained
MODEL_ID = 'model_testing_equal_bootstrap_CRPS_fit_FHS_smooth_Gaussian_mean_pool_v1'


class CrpsFitComponent:
    model_id = 'Gaussian_CRPS_fitted_empirical_FHS_smooth_mean'

    def simulate_daily_log_returns(self, training, context):
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared training history')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        fit = fit_crps(history)
        indices = args['rng'].integers(0, len(history), size=(context.simulations, context.horizon_days))
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        se = retained.gaussian_sd(df, scale)
        mu = float(history.mean())
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)+(parent.regularized_mean(mu, se)-mu)
        daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Invalid CRPS-fit native forecast')
        self.last_metadata = {k: v for k, v in fit.items() if k != 'pool'}
        self.last_metadata.update(parameter_seed=seed, parameter_namespace=law.NAMESPACE,
            batch_scale=scale, batch_df=df, batch_count=batches, batch_long_run_variance=lrv,
            Gaussian_sd=se, regularized_mean=parent.regularized_mean(mu, se),
            daily_clips=int(clips), wealth_floor_events=int(floors))
        return daily


class CrpsFitPool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (parent.ShrunkComponent('WR'), CrpsFitComponent())

    @property
    def dependency_identity(self):
        own = Path(__file__).resolve().parent
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fitter_sha256': hashlib.sha256((own/'crps_fit.py').read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'conditional_Gaussian_CRPS_for_variance_targeted_GARCH_training_fit_only',
            'constant_variance_boundary_member': True, 'fit_starts': [[.05, .90], [.10, .50]],
            'mean_law_weights_shocks_volatility_recursion_seeds_accounting_unchanged': True,
            'empirical_FHS_not_Gaussian_forecast_innovations': True,
            'surrogate_training_objective_not_full_multihorizon_CRPS': True,
            'paths': 240, 'weights': [.5, .5], 'fit_prefix_only': True,
            'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return CrpsFitPool()
