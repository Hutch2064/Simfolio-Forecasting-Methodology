"""Identical current model and seeds; explicitly repaired wealth execution."""
from pathlib import Path
import hashlib
import numpy as np
import streaming as s
import retained_models
from . import kernels

MODEL_ID = 'model_testing_equal_bootstrap_FHS_smooth_mean_Gaussian_stable_wealth_v1'


class StablePool:
    model_id = MODEL_ID

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'parent': retained_models.build_best().dependency_identity,
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'kernel_sha256': hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
            'change': 'Switch to log wealth only after positive native product overflow',
            'clipping_floor_absorbing_underflow_unchanged': True,
            'mean_fit_variance_fit_sampling_seeds_path_counts_unchanged': True,
            'all_regular_native_paths_bitwise_unchanged': True, 'paths_per_forecast': 1000,
            'no_outcome_fit_or_arbitrary_tail_cap': True}

    def iter_terminal_log_returns(self, training, context):
        training.validate()
        if context.simulations != 1000:
            raise ValueError('Frozen v5 ensemble requires1,000 paths')
        simple = np.expm1(training.portfolio_log_returns)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite historical source returns')
        history = np.log1p(np.clip(simple, -.999999, np.finfo(float).max/10))
        source = np.clip(np.expm1(history), -.999999, np.finfo(float).max/10)
        parent = s.incumbent.retained
        prior = parent.parent.incumbent.prior
        scale, df, _, _ = parent.parent.incumbent.base.batch_mean_scale(history)
        se = parent.gaussian_sd(df, scale)
        center = s.incumbent.regularized_mean(float(history.mean()), se)
        seeds = [prior.deterministic_seed(ns, int(context.seed)) for ns in parent.parent.NAMESPACES]
        delta = [parent.parameter_draws(df, scale, 500,
            prior.deterministic_seed(parent.parent.incumbent.base.NAMESPACE, int(seed)))
            +(center-float(history.mean())) for seed in seeds]
        fit = parent.FHS.base.fit_variance_target(history)
        streams = [s.BlockStream(len(history), context.horizon_days, 500, seeds[0]),
                   s.FhsStream(len(history), 500, seeds[1])]
        states = [(np.ones(500), np.zeros(500), np.zeros(500, dtype=np.bool_),
                   np.zeros(500), np.zeros(500)) for _ in streams]
        variance = np.full(500, fit['next_variance_normalized'])
        overflows = 0
        for start in range(0, context.horizon_days, s.WIDTH):
            count = min(s.WIDTH, context.horizon_days-start)
            a, oa = kernels.raw_block(history, source, streams[0].take(count), delta[0], *states[0])
            b, ob = kernels.fhs_block(fit['pool'], streams[1].take(count), fit['mean']+delta[1],
                fit['target_variance'], fit['alpha'], fit['beta'], variance, *states[1])
            terminal = np.concatenate((a, b), axis=0)
            if not np.isfinite(terminal).all():
                raise ValueError('Nonfinite overflow-safe terminal forecast')
            overflows += oa+ob
            self.last_metadata = {'positive_wealth_overflow_transitions': int(overflows),
                                  'no_new_return_or_variance_cap': True}
            yield start, terminal


def build():
    return StablePool()
