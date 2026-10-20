"""Current regularized mean, with its normal-normal working variance contraction."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
retained = parent.retained

MODEL_ID = 'model_testing_equal_bootstrap_FHS_smooth_mean_contracted_Gaussian_uncertainty_v1'


def contraction_factor(mu, se):
    if not np.isfinite(mu) or not np.isfinite(se) or se < 0:
        raise ValueError('Finite mean and nonnegative finite standard error required')
    if se == 0:
        return 1.0
    if mu == 0:
        return 0.0
    if abs(mu) >= se:
        weight = 1/(1+(se/mu)**2)
    else:
        ratio = (mu/se)**2
        weight = ratio/(1+ratio)
    return float(np.sqrt(weight))


class ContractedComponent:
    def __init__(self, kind):
        self.kind = kind
        self.model_id = 'smooth_shrunk_contracted_Gaussian_'+kind

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
        mu = float(history.mean())
        center = parent.regularized_mean(mu, se)
        factor = contraction_factor(mu, se)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)*factor+(center-mu)
        if self.kind == 'WR':
            daily, clips, floors = prior.shifted_native_paths(history, source, indices, delta)
        else:
            daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite contracted-uncertainty native forecast')
        self.last_metadata = {'sample_mean': mu, 'regularized_mean_unchanged': center,
            'prior_Gaussian_sd': se, 'new_Gaussian_sd': se*factor,
            'contraction_factor': factor, 'parameter_seed': seed,
            'parameter_namespace': law.NAMESPACE, 'batch_scale': scale,
            'batch_df': df, 'batch_count': batches, 'batch_long_run_variance': lrv,
            'daily_clips': int(clips), 'wealth_floor_events': int(floors)}
        return daily


class ContractedPool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (ContractedComponent('WR'), ContractedComponent('FHS'))

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'Gaussian_mean_draws_times_sqrt_t_squared_over_one_plus_t_squared',
            'regularized_mean_volatility_fits_indices_namespaces_weights_accounting_unchanged': True,
            'working_prior_variance': 'observed_training_sample_mean_squared',
            'prior_scale_uncertainty_integrated': False, 'fully_Bayesian_claim': False,
            'paths': 240, 'weights': [.5, .5], 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return ContractedPool()
