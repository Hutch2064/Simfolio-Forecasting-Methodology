"""Regularize historical drift; preserve the current winner's uncertainty law."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
from location import estimate as location_estimate, C
retained = parent.retained

MODEL_ID = 'model_testing_equal_bootstrap_FHS_Huber_location_smooth_Gaussian_uncertainty_v1'


class RobustComponent:
    def __init__(self, kind):
        self.kind = kind
        self.model_id = 'Huber_location_smooth_Gaussian_uncertainty_'+kind

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
        robust_center = location_estimate(history)
        center = parent.regularized_mean(robust_center, se)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)+(center-sample_mean)
        if self.kind == 'WR':
            daily, clips, floors = prior.shifted_native_paths(history, source, indices, delta)
        else:
            daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite regularized-mean native forecast')
        self.last_metadata = {'sample_mean': sample_mean, 'Huber_location': robust_center, 'regularized_mean': center,
            'mean_shift': center-sample_mean, 'Gaussian_sd_unchanged': se,
            'parameter_seed': seed, 'parameter_namespace': law.NAMESPACE,
            'batch_scale': scale, 'batch_df': df, 'batch_count': batches,
            'batch_long_run_variance': lrv, 'daily_clips': int(clips), 'wealth_floor_events': int(floors)}
        return daily


class RobustPool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (RobustComponent('WR'), RobustComponent('FHS'))

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'location_source_sha256': hashlib.sha256((Path(__file__).parent/'location.py').read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'Huber_training_location_then_unchanged_smooth_regularizer',
            'Huber_constant': C, 'scale': 'training_MAD_over_Normal_q75_sample_SD_if_zero_MAD',
            'raw_empirical_tail_observations_kept': True, 'batch_mean_uncertainty_and_volatility_fit_unchanged': True,
            'asymmetric_distribution_location_not_unbiased_mean_claim': True,
            'indices_namespaces_weights_native_accounting_unchanged': True,
            'paths': 240, 'weights': [.5, .5], 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return RobustPool()
