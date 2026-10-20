"""Current portfolio mixture with sign-asymmetric empirical FHS volatility."""
from pathlib import Path
import hashlib
import numpy as np
import streaming as s
from research_stable_wealth import adapter as stable, kernels
from . import gjr_fit

MODEL_ID = 'model_testing_equal_bootstrap_GJR_FHS_smooth_mean_Gaussian_uncertainty_v1'


class AsymmetricPool:
    model_id = MODEL_ID

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID,
            'parent': stable.build().dependency_identity,
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fit_sha256': hashlib.sha256(Path(gjr_fit.__file__).read_bytes()).hexdigest(),
            'change': 'Sign-asymmetric GARCH QMLE in the FHS half; empirical innovation moment variance targeting',
            'raw_WR_component_mean_Gaussian_draws_namespaces_weights_unchanged': True,
            'paths_per_forecast': 1000, 'component_paths': 500,
            'fitting_only_training': True, 'OOS_or_family_horizon_route': False,
            'stationarity': 'beta+max(alpha_positive,alpha_negative)<1',
            'symmetric_boundary': 'Exact native v5 best when parent symmetric fit selected',
            'diagnostic_selection': False}

    def iter_terminal_log_returns(self, training, context):
        training.validate()
        if context.simulations != 1000:
            raise ValueError('Frozen v5 ensemble requires1,000 total paths')
        simple = np.expm1(training.portfolio_log_returns)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared historical returns')
        history = np.log1p(np.clip(simple, -.999999, np.finfo(float).max/10))
        source = np.clip(np.expm1(history), -.999999, np.finfo(float).max/10)
        fit = gjr_fit.fit(history)
        parent = s.incumbent.retained
        prior = parent.parent.incumbent.prior
        scale, df, _, _ = parent.parent.incumbent.base.batch_mean_scale(history)
        se = parent.gaussian_sd(df, scale)
        center = s.incumbent.regularized_mean(float(history.mean()), se)
        seeds = [prior.deterministic_seed(ns, int(context.seed)) for ns in parent.parent.NAMESPACES]
        deltas = [parent.parameter_draws(df, scale, 500,
            prior.deterministic_seed(parent.parent.incumbent.base.NAMESPACE, int(seed)))
            +(center-float(history.mean())) for seed in seeds]
        streams = [s.BlockStream(len(history), context.horizon_days, 500, seeds[0]),
                   s.FhsStream(len(history), 500, seeds[1])]
        states = [(np.ones(500), np.zeros(500), np.zeros(500, dtype=bool),
                   np.zeros(500), np.zeros(500)) for _ in streams]
        variance = np.full(500, fit['next_variance_normalized'])
        self.last_metadata = {k:v for k,v in fit.items() if k != 'pool'}
        for start in range(0, context.horizon_days, s.WIDTH):
            count = min(s.WIDTH, context.horizon_days-start)
            first = kernels.raw_block(history, source, streams[0].take(count), deltas[0], *states[0])[0]
            second = gjr_fit.terminal_block(fit['pool'], streams[1].take(count),
                fit['mean']+deltas[1], fit['target_variance'], fit['alpha_positive'],
                fit['alpha_negative'], fit['beta'], fit['omega_forecast'], variance, *states[1])
            terminal = np.concatenate((first, second), axis=0)
            if not np.isfinite(terminal).all():
                raise ValueError('Nonfinite asymmetric forecast')
            yield start, terminal


def build():
    return AsymmetricPool()
