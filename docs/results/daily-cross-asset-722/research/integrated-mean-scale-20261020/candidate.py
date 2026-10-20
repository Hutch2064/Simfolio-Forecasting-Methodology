"""Integrate a half-Cauchy local mean scale; retain the exact two native components."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from scipy.special import dawsn
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
retained = parent.retained
MODEL_ID = 'model_testing_equal_bootstrap_FHS_horseshoe_mean_uncertainty_v1'
NAMESPACE = 'horseshoe_local_mean_scale_v1'


def local_scale_cdf(u, z):
    """CDF of u=sqrt(w): density proportional to exp(z^2*u^2/2), 0<=u<=1."""
    u = np.asarray(u, dtype=float)
    if not np.isfinite(z) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('Finite standardized mean and unit interval nodes required')
    x = abs(float(z))/np.sqrt(2.)
    if x == 0:
        return u.copy()
    return np.exp((x*x)*(u*u-1.))*dawsn(x*u)/dawsn(x)


def local_scale_quantile(probability, z):
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Unit interval probabilities required')
    lo, hi = np.zeros_like(probability), np.ones_like(probability)
    for _ in range(54):
        mid = (lo+hi)/2.
        below = local_scale_cdf(mid, z) < probability
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(probability == 0, 0., np.where(probability == 1, 1., (lo+hi)/2.))


def paired_local_weights(mu, se, paths, seed):
    if paths < 2 or paths % 2 or not np.isfinite(mu) or not np.isfinite(se) or se < 0:
        raise ValueError('Even ensemble and finite mean/nonnegative standard error required')
    if se == 0:
        return np.ones(paths)
    prior = retained.parent.incumbent.prior
    rng = np.random.default_rng(prior.deterministic_seed(NAMESPACE, int(seed)))
    half = paths//2
    p = (np.arange(half, dtype=float)+1.-rng.random(half))/half
    weights = local_scale_quantile(p, mu/se)**2
    weights = weights[rng.permutation(half)]
    gaussian_rng = np.random.default_rng(seed)
    gaussian_rng.random(half)
    gaussian_order = gaussian_rng.permutation(paths)
    return np.concatenate((weights, weights))[gaussian_order]


def mean_offsets(mu, df, scale, paths, seed):
    se = retained.gaussian_sd(df, scale)
    weights = paired_local_weights(mu, se, paths, seed)
    gaussian = retained.parameter_draws(df, scale, paths, seed)
    return weights*mu+np.sqrt(weights)*gaussian-mu


class HorseshoeComponent:
    def __init__(self, kind):
        self.kind = kind
        self.model_id = 'horseshoe_mean_uncertainty_'+kind

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
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = mean_offsets(mu, df, scale, context.simulations, seed)
        if self.kind == 'WR':
            daily, clips, floors = prior.shifted_native_paths(history, source, indices, delta)
        else:
            daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite integrated local-scale native forecast')
        self.last_metadata = {'sample_mean': mu, 'working_likelihood_mean_SE': se,
            'mean_offset_sample_mean': float(delta.mean()), 'mean_offset_sample_sd': float(delta.std(ddof=1)),
            'parameter_seed': seed, 'parameter_namespace': law.NAMESPACE,
            'local_scale_namespace': NAMESPACE, 'batch_scale': scale,
            'batch_df': df, 'batch_count': batches, 'batch_long_run_variance': lrv,
            'daily_clips': int(clips), 'wealth_floor_events': int(floors)}
        return daily


class HorseshoePool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (HorseshoeComponent('WR'), HorseshoeComponent('FHS'))

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'working_mean_likelihood': 'sample_mean_given_theta_Normal_theta_batch_SE_squared',
            'mean_prior': 'theta_given_lambda_Normal_0_batch_SE_squared_times_lambda_squared',
            'local_prior': 'lambda_half_Cauchy_0_1', 'global_scale_in_SE_units': 1.,
            'conditional_mean_law': 'Normal_w_times_sample_mean_w_times_batch_SE_squared',
            'local_scale_uncertainty_integrated': True, 'batch_SE_uncertainty_integrated': False,
            'conditional_normal_pairs_share_independent_stratified_local_scale': True,
            'volatility_fits_indices_component_namespaces_weights_accounting_unchanged': True,
            'fully_Bayesian_or_financial_dominance_claim': False,
            'paths': 240, 'weights': [.5, .5], 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return HorseshoePool()
