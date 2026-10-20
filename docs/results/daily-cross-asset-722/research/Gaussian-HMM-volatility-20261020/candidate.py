"""Half raw random blocks, half persistent Gaussian regimes; fixed mean treatment."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from expanded_public_smooth_mean_live import module as parent, build_smooth_mean_pool
import hmm
retained = parent.retained
MODEL_ID = 'model_testing_equal_bootstrap_MAP_Gaussian_HMM_smooth_mean_uncertainty_v1'
STATE_NAMESPACE = 'two_state_volatility_MAP_state_path_v1'
SHOCK_NAMESPACE = 'two_state_volatility_MAP_Normal_shocks_v1'


class RegimeComponent:
    model_id = 'MAP_two_state_Gaussian_volatility_smooth_mean_uncertainty_v1'

    def simulate_daily_log_returns(self, training, context):
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared history')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        fitted = hmm.fit(history)
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        se = retained.gaussian_sd(df, scale)
        center = parent.regularized_mean(fitted['mean'], se)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)
        state_seed = prior.deterministic_seed(STATE_NAMESPACE, int(context.seed))
        shock_seed = prior.deterministic_seed(SHOCK_NAMESPACE, int(context.seed))
        uniforms = np.random.default_rng(state_seed).random((context.simulations, context.horizon_days+1))
        shocks = np.random.default_rng(shock_seed).standard_normal((context.simulations, context.horizon_days))
        daily, clips, floors = hmm.native_paths(fitted['variance'], fitted['transition'], fitted['filtered'],
                                               center, delta, uniforms, shocks)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite HMM native forecast')
        self.last_metadata = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items()}
        self.last_metadata.update(regularized_mean=center, Gaussian_sd_unchanged=se,
            parameter_seed=seed, state_seed=state_seed, shock_seed=shock_seed,
            batch_scale=scale, batch_df=df, batch_count=batches, batch_long_run_variance=lrv,
            Gaussian_emissions=True, parameter_point_MAP_state_paths_integrated=True,
            daily_clips=int(clips), wealth_floor_events=int(floors))
        return daily


class RegimePool(parent.ShrunkPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (parent.ShrunkComponent('WR'), RegimeComponent())

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'HMM_source_sha256': hashlib.sha256((Path(__file__).parent/'hmm.py').read_bytes()).hexdigest(),
            'parent': build_smooth_mean_pool().dependency_identity,
            'change': 'replace_FHS_half_by_two_state_fixed_mean_Gaussian_HMM_MAP',
            'variance_prior': 'IG_shape2_scale3_times_training_population_variance',
            'transition_prior': 'independent_Dirichlet_2_2_rows', 'initial_historical_state': [.5, .5],
            'starts': list(hmm.STARTS), 'parameter_posterior_integrated': False,
            'future_state_paths_integrated': True, 'mean_law_and_WR_unchanged': True,
            'paths': 240, 'weights': [.5, .5], 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return RegimePool()
