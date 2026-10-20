"""Retained mean pool, integrating stationary FHS coefficient uncertainty only."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
import posterior
retained = parent.retained
MODEL_ID = 'model_testing_equal_bootstrap_stationary_GARCH_posterior_FHS_smooth_mean_pool_v1'
COEFFICIENT_NAMESPACE = 'stationary_GARCH_coefficient_posterior_v1'


def basis(history):
    mean = float(history.mean())
    centered = history-mean
    target = float(np.mean(centered*centered))
    return mean, target, centered/np.sqrt(target)


def conditional_fit(prepared, a, b):
    mean, target, y = prepared
    _, _, variances, next_v = retained.FHS.base.filter_loss(y, a, b)
    pool = y/np.sqrt(variances)
    pool = pool-pool.mean()
    pool = pool/np.sqrt(np.mean(pool*pool))
    return {'mean': mean, 'target_variance': target, 'alpha': float(a), 'beta': float(b),
            'next_variance_normalized': float(next_v), 'pool': pool}


class PosteriorComponent:
    model_id = 'stationary_GARCH_coefficient_posterior_empirical_FHS_smooth_mean'

    def simulate_daily_log_returns(self, training, context):
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared training history')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        coefficient_seed = prior.deterministic_seed(COEFFICIENT_NAMESPACE, int(context.seed))
        coefficients, diagnostic = posterior.parameter_draws(history, context.simulations, coefficient_seed)
        indices = args['rng'].integers(0, len(history), size=(context.simulations, context.horizon_days))
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        se = retained.gaussian_sd(df, scale)
        mu = float(history.mean())
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)+(parent.regularized_mean(mu, se)-mu)
        prepared = basis(history)
        daily = np.empty((context.simulations, context.horizon_days))
        clips = floors = 0
        maximum_pool_mean = maximum_pool_variance_error = 0.
        for p, (a, b) in enumerate(coefficients):
            fit = conditional_fit(prepared, a, b)
            maximum_pool_mean = max(maximum_pool_mean, abs(float(fit['pool'].mean())))
            maximum_pool_variance_error = max(maximum_pool_variance_error, abs(float(np.mean(fit['pool']**2))-1.))
            result, c, f = retained.FHS.parameter_paths(fit, indices[p:p+1], delta[p:p+1])
            daily[p] = result[0]
            clips += c
            floors += f
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite coefficient-posterior native forecast')
        self.last_metadata = {'coefficient_posterior': diagnostic, 'coefficient_seed': coefficient_seed,
            'coefficient_namespace': COEFFICIENT_NAMESPACE, 'sample_mean': mu,
            'regularized_mean': parent.regularized_mean(mu, se), 'Gaussian_mean_sd': se,
            'parameter_seed': seed, 'parameter_namespace': law.NAMESPACE,
            'batch_scale': scale, 'batch_df': df, 'batch_count': batches, 'batch_long_run_variance': lrv,
            'maximum_absolute_innovation_pool_mean': maximum_pool_mean,
            'maximum_innovation_pool_variance_error': maximum_pool_variance_error,
            'daily_clips': int(clips), 'wealth_floor_events': int(floors)}
        return daily


class PosteriorPool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (parent.ShrunkComponent('WR'), PosteriorComponent())

    @property
    def dependency_identity(self):
        own = Path(__file__).resolve().parent
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'posterior_source_sha256': hashlib.sha256((own/'posterior.py').read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'stationary_GARCH_coefficient_working_posterior_predictive_in_FHS_half',
            'prior': 'flat_alpha_beta_on_trimmed_stationary_simplex_Dirichlet_1_1_1',
            'fixed_training_sample_mean_and_variance': True, 'mean_uncertainty_law_unchanged': True,
            'coefficient_namespace': COEFFICIENT_NAMESPACE, 'quadrature_powers': list(posterior.NODE_POWERS),
            'quadrature_precision': posterior.PRECISION, 'importance_proposal_Student_not_financial_law': True,
            'fully_Bayesian_empirical_shock_claim': False, 'native_kernel_accounting_unchanged': True,
            'paths': 240, 'weights': [.5, .5], 'fit_prefix_only': True,
            'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return PosteriorPool()
