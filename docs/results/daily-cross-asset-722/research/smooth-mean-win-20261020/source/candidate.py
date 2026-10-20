"""Regularize historical drift; preserve the current winner's uncertainty law."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_gaussian_equal_pool_live import module as retained, build_gaussian_equal_pool

MODEL_ID = 'model_testing_equal_bootstrap_FHS_smooth_mean_shrinkage_Gaussian_uncertainty_v1'


def regularized_mean(mu, se):
    """mu*t^2/(1+t^2), evaluated without overflowing or dividing by zero."""
    if not np.isfinite(mu) or not np.isfinite(se) or se < 0:
        raise ValueError('Finite mean and nonnegative finite standard error required')
    if se == 0 or mu == 0:
        return float(mu)
    if abs(mu) >= se:
        weight = 1/(1+(se/mu)**2)
    else:
        ratio = (mu/se)**2
        weight = ratio/(1+ratio)
    return float(mu*weight)


class ShrunkComponent:
    def __init__(self, kind):
        self.kind = kind
        self.model_id = 'smooth_shrunk_Gaussian_uncertainty_'+kind

    def simulate_daily_log_returns(self, training, context):
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
        center = regularized_mean(sample_mean, se)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)+(center-sample_mean)
        if self.kind == 'WR':
            daily, clips, floors = prior.shifted_native_paths(history, source, indices, delta)
        else:
            daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite regularized-mean native forecast')
        self.last_metadata = {'sample_mean': sample_mean, 'regularized_mean': center,
            'mean_shift': center-sample_mean, 'Gaussian_sd_unchanged': se,
            'parameter_seed': seed, 'parameter_namespace': law.NAMESPACE,
            'batch_scale': scale, 'batch_df': df, 'batch_count': batches,
            'batch_long_run_variance': lrv, 'daily_clips': int(clips), 'wealth_floor_events': int(floors)}
        return daily


class ShrunkPool(retained.GaussianPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (ShrunkComponent('WR'), ShrunkComponent('FHS'))

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parent': build_gaussian_equal_pool().dependency_identity,
            'change': 'sample_mean_times_t_squared_over_one_plus_t_squared',
            't_standard_error': 'unchanged_training_batch_matched_Gaussian_mean_sd',
            'mean_parameter_uncertainty_variance_unchanged': True,
            'volatility_fits_resampling_namespaces_weights_native_accounting_unchanged': True,
            'posterior_optimality_claim': False, 'paths': 240, 'weights': [.5, .5],
            'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return ShrunkPool()
