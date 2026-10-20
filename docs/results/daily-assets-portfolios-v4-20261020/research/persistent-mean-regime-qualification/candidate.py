"""Parent pool plus one training-fitted future mean-regime variance component."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
import random_effects
MODEL_ID = 'model_testing_equal_bootstrap_FHS_Paule_Mandel_mean_regime_uncertainty_v1'


def mean_draws(df, scale, paths, seed, tau_squared):
    original = parent.retained.parameter_draws(df, scale, paths, seed)
    if tau_squared == 0:
        return original
    se = parent.retained.gaussian_sd(df, scale)
    if not np.isfinite(tau_squared) or tau_squared < 0 or se <= 0:
        raise ValueError('Positive finite reference scale required for a nonzero regime component')
    return original*(np.hypot(se, np.sqrt(tau_squared))/se)


class MeanRegimeComponent(parent.ShrunkComponent):
    def simulate_daily_log_returns(self, training, context):
        retained = parent.retained
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared training history')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        source = np.clip(np.expm1(history), -.999999, np.finfo(float).max/10)
        if self.kind == 'WR':
            indices = retained.parent.incumbent.bounded_public.bounded_block_indices(
                len(history), context.horizon_days, context.simulations, args['rng'])
            fit = None
        else:
            fit = retained.FHS.base.fit_variance_target(history)
            indices = args['rng'].integers(0, len(history), size=(context.simulations, context.horizon_days))
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        se = retained.gaussian_sd(df, scale)
        sample_mean = float(history.mean())
        center = parent.regularized_mean(sample_mean, se)
        regime = random_effects.fit_training_batches(history)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = mean_draws(df, scale, context.simulations, seed, regime['tau_squared'])+(center-sample_mean)
        if self.kind == 'WR':
            daily, clips, floors = prior.shifted_native_paths(history, source, indices, delta)
        else:
            daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite native mean-regime forecast')
        self.last_metadata = {'sample_mean': sample_mean, 'regularized_mean': center,
            'mean_shift': center-sample_mean, 'parent_Gaussian_mean_sd': se,
            'future_mean_regime_tau_squared': regime['tau_squared'],
            'combined_Gaussian_mean_sd': float(np.hypot(se, np.sqrt(regime['tau_squared']))),
            'parameter_seed': seed, 'parameter_namespace': law.NAMESPACE,
            'batch_scale': scale, 'batch_df': df, 'batch_count': batches,
            'batch_long_run_variance': lrv, 'daily_clips': int(clips), 'wealth_floor_events': int(floors)}
        return daily


class MeanRegimePool(parent.ShrunkPool):
    model_id = MODEL_ID
    def __init__(self):
        self.components = (MeanRegimeComponent('WR'), MeanRegimeComponent('FHS'))

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'random_effects_source_sha256': hashlib.sha256(Path(random_effects.__file__).read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'training_Paule_Mandel_between_batch_mean_variance_added_to_parent_Gaussian_mean_variance',
            'batches': 'existing_floor_sqrt_n_nonoverlapping_all_observations',
            'within_variance': 'sample_variance_ddof1_divided_by_batch_n',
            'working_martingale_batch_sampling_approximation': True,
            'current_mean_center_shocks_volatility_seed_pool_accounting_unchanged': True,
            'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return MeanRegimePool()
