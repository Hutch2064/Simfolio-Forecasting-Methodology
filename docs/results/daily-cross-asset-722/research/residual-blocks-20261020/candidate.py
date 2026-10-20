"""Keep fitted volatility; retain measured dependence in filtered innovations."""
from pathlib import Path
import hashlib
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
from numba import njit
from expanded_public_gaussian_equal_pool_live import module as retained, build_gaussian_equal_pool

MODEL_ID = 'model_testing_equal_bootstrap_stationary_filtered_blocks_Gaussian_mean_pool_v1'
BLOCK_NAMESPACE = 'FHS_filtered_residual_stationary_restart_v1'


@njit(cache=True, fastmath=False)
def coupled_indices(iid_indices, restarts, n):
    out = iid_indices.copy()
    for path in range(out.shape[0]):
        for h in range(1, out.shape[1]):
            if not restarts[path, h-1]:
                out[path, h] = (out[path, h-1]+1) % n
    return out


def residual_indices(pool, paths, horizon, iid_rng, restart_seed, block=None):
    prior = retained.parent.incumbent.prior
    if block is None:
        centered = pool-float(pool.mean())
        block = min(len(pool), max(prior._politis_white_block_length(pool),
                                  prior._politis_white_block_length(centered**2)))
    if not np.isfinite(block) or block < 1:
        raise ValueError('Invalid fitted residual block length')
    iid = iid_rng.integers(0, len(pool), size=(paths, horizon))
    flags = np.random.default_rng(restart_seed).random((paths, max(horizon-1, 0))) < 1/block
    return coupled_indices(iid, flags, len(pool)), float(block)


class ResidualBlockComponent:
    model_id = 'stationary_block_filtered_innovations_Gaussian_mean'

    def simulate_daily_log_returns(self, training, context):
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared training returns')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        fit = retained.FHS.base.fit_variance_target(history)
        restart_seed = prior.deterministic_seed(BLOCK_NAMESPACE, int(context.seed))
        indices, block = residual_indices(fit['pool'], context.simulations, context.horizon_days,
                                          args['rng'], restart_seed)
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)
        daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Nonfinite blocked-residual FHS forecast')
        self.last_metadata = {k: v for k, v in fit.items() if k != 'pool'}
        self.last_metadata.update(filtered_residual_average_block=block, restart_seed=restart_seed,
            restart_namespace=BLOCK_NAMESPACE, parameter_seed=seed, batch_scale=scale, batch_df=df,
            batch_count=batches, batch_long_run_variance=lrv, daily_clips=int(clips), wealth_floor_events=int(floors),
            IID_GARCH_unconditional_variance_identity_not_asserted=True)
        return daily


class ResidualBlockPool(retained.GaussianPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (retained.GaussianComponent('WR'), ResidualBlockComponent())

    @property
    def dependency_identity(self):
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parent': build_gaussian_equal_pool().dependency_identity,
            'change': 'FHS_innovation_indices_stationary_blocks_instead_of_IID',
            'block_fit': 'max_Politis_White_signed_and_squared_filtered_training_innovations',
            'restart_namespace': BLOCK_NAMESPACE, 'IID_block_one_source_identical': True,
            'volatility_fit_mean_law_weights_accounting_unchanged': True,
            'no_stationary_unconditional_variance_equality_claim_under_dependent_innovations': True,
            'paths': 240, 'weights': [.5, .5], 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return ResidualBlockPool()
