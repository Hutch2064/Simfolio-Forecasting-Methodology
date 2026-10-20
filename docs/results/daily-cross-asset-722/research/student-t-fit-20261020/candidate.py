"""Retained Gaussian mean pool, replacing only the FHS training likelihood."""
from pathlib import Path
import hashlib
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent / 'french551-full-20261019/code'
for p in (ROOT, CODE/'src', CODE/'scripts', Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))
import numpy as np
from expanded_public_gaussian_equal_pool_live import module as retained, build_gaussian_equal_pool
from t_fit import fit_student

MODEL_ID = 'model_testing_equal_bootstrap_student_fit_FHS_Gaussian_mean_pool_v1'


class StudentFitComponent:
    model_id = 'student_likelihood_fitted_empirical_FHS_Gaussian_mean'

    def simulate_daily_log_returns(self, training, context):
        prior = retained.parent.incumbent.prior
        args = prior.control_adapters.native_control_arguments(training, context)
        simple = args['asset_returns'].to_numpy(dtype=float, copy=False)
        if not np.isfinite(simple).all():
            raise ValueError('Nonfinite prepared training history')
        history = np.log1p(np.clip(simple[:, 0], -.999999, np.finfo(float).max/10))
        fit = fit_student(history)
        indices = args['rng'].integers(0, len(history), size=(context.simulations, context.horizon_days))
        law = retained.parent.incumbent.base
        scale, df, lrv, batches = law.batch_mean_scale(history)
        seed = prior.deterministic_seed(law.NAMESPACE, int(context.seed))
        delta = retained.parameter_draws(df, scale, context.simulations, seed)
        daily, clips, floors = retained.FHS.parameter_paths(fit, indices, delta)
        if not np.isfinite(daily).all():
            raise ValueError('Invalid Student-fit native forecast')
        self.last_metadata = {k: v for k, v in fit.items() if k != 'pool'}
        self.last_metadata.update(parameter_seed=seed, parameter_namespace=law.NAMESPACE,
            batch_scale=scale, batch_df=df, batch_count=batches, batch_long_run_variance=lrv,
            Gaussian_sd=retained.gaussian_sd(df, scale), daily_clips=int(clips), wealth_floor_events=int(floors))
        return daily


class StudentFitPool(retained.GaussianPool):
    model_id = MODEL_ID

    def __init__(self):
        self.components = (retained.GaussianComponent('WR'), StudentFitComponent())

    @property
    def dependency_identity(self):
        own = Path(__file__).resolve().parent
        return {'method': MODEL_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fitter_sha256': hashlib.sha256((own/'t_fit.py').read_bytes()).hexdigest(),
                'parent': build_gaussian_equal_pool().dependency_identity,
                'change': 'Student_t_likelihood_for_variance_targeted_GARCH_fit_only',
                'Student_df_bounds': [2.05, 500.0], 'Gaussian_boundary_member': True,
                'empirical_shocks_mean_law_weights_seeds_unchanged': True,
                'parametric_Student_forecast_innovations': False, 'paths': 240, 'weights': [.5, .5],
                'fit_prefix_only': True, 'oos_fit': False, 'family_or_horizon_routes': False}


def build_candidate():
    return StudentFitPool()
