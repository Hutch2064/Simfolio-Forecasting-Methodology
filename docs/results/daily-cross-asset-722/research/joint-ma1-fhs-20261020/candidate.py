"""Current native mean pool with a coherent conditional MA1-GARCH FHS component."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts', Path(__file__).resolve().parent.parent):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
import joint_fit
retained = parent.retained
MODEL_ID = 'model_testing_equal_bootstrap_conditional_MA1_GARCH_FHS_smooth_mean_pool_v1'


class MovingAverageComponent:
    model_id = 'conditional_MA1_GARCH_FHS_smooth_Gaussian_mean'

    def simulate_daily_log_returns(self, training, context):
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared training history')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        fit = joint_fit.fit_joint(history)
        indices = args['rng'].integers(0, len(history), size=(context.simulations, context.horizon_days))
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        se = retained.gaussian_sd(df, scale)
        mu = float(history.mean())
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)+(parent.regularized_mean(mu, se)-mu)
        daily, clips, floors = joint_fit.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite conditional MA1-GARCH native forecast')
        self.last_metadata = {k: v for k, v in fit.items() if k != 'pool'}
        self.last_metadata.update(parameter_seed=seed, parameter_namespace=law.NAMESPACE,
            batch_scale=scale, batch_df=df, batch_count=batches, batch_long_run_variance=lrv,
            Gaussian_sd=se, regularized_mean=parent.regularized_mean(mu, se),
            daily_clips=int(clips), wealth_floor_events=int(floors))
        return daily


class MovingAveragePool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (parent.ShrunkComponent('WR'), MovingAverageComponent())

    @property
    def dependency_identity(self):
        own = Path(__file__).resolve().parent
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fitter_sha256': hashlib.sha256((own/'joint_fit.py').read_bytes()).hexdigest(),
            'MA_seed_source_sha256': hashlib.sha256((own.parent/'ma1_qualification/ma1.py').read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'joint_conditional_MA1_and_variance_targeted_GARCH_FHS',
            'mean_equation': 'r_t=mu+epsilon_t+theta*epsilon_t_minus_1',
            'pre_sample_innovation': 0., 'initial_conditional_variance': 'training_innovation_population_variance',
            'sample_mean_mean_uncertainty_WR_weights_seeds_native_accounting_unchanged': True,
            'Gaussian_quasi_likelihood_empirical_forecast_innovations': True,
            'exact_stationary_joint_likelihood_or_fully_Bayesian_claim': False,
            'MA_GARCH_coefficient_uncertainty_integrated': False, 'paths': 240, 'weights': [.5, .5],
            'fit_prefix_only': True, 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return MovingAveragePool()
