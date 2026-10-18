"""M261 mean with Student GJR-GARCH filtered historical innovations.

Training rows remain aligned across assets. The PPW-selected stationary row
bootstrap supplies their joint future shocks; no copula or rough/SV fit runs.
"""
import hashlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from arch import arch_model
from arch.bootstrap import optimal_block_length

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/mcmc_runtime')]
spec = importlib.util.spec_from_file_location('garch_private_m261',
    ROOT/'tools/data_selected_timescales/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
parent.VARIANT = 'existing_sv_rate'
shell = parent.shell
sys.path.insert(0, str(HERE))
from gjr_filtered_kernels import forecast_paths
from streamed_paths import rejoin
from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves

MODEL_ID = 'asset_m261_mean_gjr_garch_student_filtered_history'
TIMINGS = {}
FIT_DIAGNOSTICS = {}


def timed(name, started):
    TIMINGS[name] = TIMINGS.get(name, 0.) + time.perf_counter() - started


def feasible_parameters(raw_params, tolerance=1e-9):
    """Repair only solver precision at the same declared GJR boundaries."""
    if not np.isfinite(raw_params).all():
        raise ArithmeticError('nonfinite GJR Student likelihood parameters')
    omega, alpha, gamma, beta, nu = map(float, raw_params)
    if min(alpha, alpha+gamma, beta) < -tolerance:
        raise ArithmeticError('GJR Student fit has substantive positivity violation')
    raw_persistence = alpha+gamma/2.+beta
    if raw_persistence > 1.+tolerance:
        raise ArithmeticError('GJR Student fit has substantive stationarity violation')
    if alpha < 0.:
        gamma += alpha
        alpha = 0.
    beta = max(beta, 0.)
    if alpha+gamma < 0.:
        gamma = -alpha
    positive_coefficients = np.array([alpha, gamma, beta])
    persistence = alpha+gamma/2.+beta
    if persistence >= 1.:
        target = np.nextafter(1., 0.)
        factor = target/persistence
        alpha, gamma, beta = alpha*factor, gamma*factor, beta*factor
        # The rounded three-term sum can still equal one after one scaling.
        while alpha+gamma/2.+beta >= 1.:
            factor = np.nextafter(factor, 0.)
            alpha, gamma, beta = positive_coefficients*factor
    used = np.array([omega, alpha, gamma, beta, nu])
    correction = float(np.max(np.abs(used-raw_params)))
    if correction > tolerance:
        raise ArithmeticError('GJR feasibility projection exceeds solver tolerance')
    persistence = alpha+gamma/2.+beta
    if not (omega > 0. and alpha >= 0. and beta >= 0. and alpha+gamma >= 0.
            and 0. <= persistence < 1. and nu > 2.):
        raise ArithmeticError('GJR Student fit violates stationary finite-variance constraints')
    return used, correction, raw_persistence


@lru_cache(maxsize=64)
def fit_asset(data):
    x = np.frombuffer(data, np.float64)
    # Percent units improve optimizer conditioning; output is converted back
    # to decimal-return variance. Every available training row is fitted.
    center = float(x.mean())
    residuals = (x - center)*100.
    model = arch_model(residuals, mean='Zero', vol='GARCH', p=1, o=1,
        q=1, dist='StudentsT', rescale=False)
    result = model.fit(disp='off', show_warning=False,
            options={'maxiter': 1000, 'ftol': 1e-9})
    if result.convergence_flag != 0 or not result.optimization_result.success:
        raise ArithmeticError(f'GJR Student likelihood failed: {result.optimization_result.message}')
    raw_params = result.params.to_numpy().copy()
    feasibility_tolerance = 1e-9
    projected_params, correction, raw_persistence = feasible_parameters(raw_params, feasibility_tolerance)
    omega, alpha, gamma, beta, nu = map(float, projected_params)
    original_variance = np.asarray(result.conditional_volatility)**2
    original_loglikelihood = float(result.loglikelihood)
    if correction:
        result = model.fix(projected_params)
    persistence = alpha + gamma/2. + beta
    if not (np.isfinite(result.params).all() and omega > 0. and alpha >= 0.
            and beta >= 0. and alpha+gamma >= 0. and 0. <= persistence < 1. and nu > 2.):
        raise ArithmeticError('GJR Student fit violates stationary finite-variance constraints')
    variance = np.asarray(result.conditional_volatility)**2
    if not np.isfinite(variance).all() or np.any(variance <= 0.):
        raise ArithmeticError('invalid fitted GJR conditional variance')
    pool = residuals/np.sqrt(variance)
    pool -= pool.mean()
    scale = float(np.sqrt(np.mean(pool*pool)))
    if not np.isfinite(scale) or scale <= 0.:
        raise ArithmeticError('invalid empirical GJR standardized innovations')
    pool /= scale
    params = np.array([omega/10000., alpha, gamma, beta])
    receipt = {'estimator': 'full_training_GJR_GARCH_1_1_unit_variance_Student_t_likelihood',
        'omega': float(params[0]), 'alpha': alpha, 'gamma': gamma, 'beta': beta,
        'nu': nu, 'persistence': persistence, 'converged': True,
        'training_rows': len(x), 'historical_center': center,
        'loglikelihood_percent_units': float(result.loglikelihood),
        'empirical_pool_center': float(pool.mean()),
        'empirical_pool_second_moment': float(np.mean(pool*pool)),
        'empirical_negative_shock_second_moment': float(np.mean(pool*pool*(pool < 0.))),
        'empirical_iid_variance_persistence': float(alpha+gamma*np.mean(pool*pool*(pool < 0.))+beta),
        'terminal_variance': float(variance[-1]/10000.),
        'terminal_residual': float(residuals[-1]/100.),
        'solver_feasibility_tolerance': feasibility_tolerance,
        'optimizer_persistence': raw_persistence,
        'open_stationarity_boundary_projection': raw_persistence >= 1.,
        'stationarity_boundary_interpretation': 'approximate_open_constraint_GJR_not_a_separate_IGARCH_law',
        'optimizer_loglikelihood_percent_units': original_loglikelihood,
        'optimizer_params_percent_units': raw_params.tolist(),
        'used_params_percent_units': projected_params.tolist(),
        'maximum_parameter_feasibility_correction': correction,
        'maximum_variance_refilter_change_percent_units': float(np.max(np.abs(variance-original_variance)))}
    return params, float(variance[-1]/10000.), float(residuals[-1]/100.), pool, receipt


def canonical_mean(data, horizon):
    # This is the original Gaussian fit retained by M261, not its replacement
    # Student/Laplace conventional fit; existing_sv_rate changes its components.
    original = parent.parent.body.learned.student.original_fit(data)
    return moment_return_curves(original, horizon)[0]


@lru_cache(maxsize=8)
def configuration(shape, data):
    assets = np.frombuffer(data, np.float64).reshape(shape)
    fits = [fit_asset(np.ascontiguousarray(x).tobytes()) for x in assets.T]
    pool = np.column_stack([fit[3] for fit in fits])
    # Include levels and squares so PPW can detect shock dependence as well as
    # remaining volatility dependence; max protects every asset's selected lag.
    selected = optimal_block_length(np.column_stack((pool, pool*pool)))
    lengths = selected['stationary'].to_numpy(dtype=float)
    if not np.isfinite(lengths).all() or np.any(lengths < 0.):
        raise ArithmeticError('PPW stationary block selection failed')
    block_length = max(1., float(lengths.max()))
    receipt = {'selector': 'arch.bootstrap.optimal_block_length_PPW',
        'selector_inputs': 'aligned_standardized_innovations_and_their_squares',
        'stationary_lengths': lengths.tolist(), 'expected_block_length': block_length,
        'joint_row_resampling': True, 'training_rows': shape[0],
        'innovation_matrix_sha256': hashlib.sha256(pool.tobytes()).hexdigest()}
    return fits, np.ascontiguousarray(pool), block_length, receipt


def stationary_indices(rows, sims, horizon, block_length, seed):
    rng = np.random.default_rng(seed)
    indices = np.empty((sims, horizon), dtype=np.int64)
    indices[:, 0] = rng.integers(rows, size=sims)
    for day in range(1, horizon):
        restart = rng.random(sims) < 1./block_length
        starts = rng.integers(rows, size=sims)
        indices[:, day] = np.where(restart, starts, (indices[:, day-1]+1) % rows)
    return indices




@dataclass(frozen=True)
class Candidate:
    model_id: str = MODEL_ID

    def simulate_daily_log_returns(self, training, context):
        training.validate()
        past, future = shell._validate_calendar(training, context)
        assets = np.ascontiguousarray(training.asset_log_returns, dtype=np.float64)
        sims, horizon = context.simulations, context.horizon_days
        started = time.perf_counter()
        fits, pool, block_length, bootstrap_receipt = configuration(assets.shape, assets.tobytes())
        means = np.column_stack([canonical_mean(np.ascontiguousarray(x).tobytes(), horizon)
                                 for x in assets.T])
        timed('fit_phase_wall', started)
        seed = shell.bd.deterministic_seed('gjr_filtered_history_stationary_rows',
            str(context.origin_date), horizon, sims)
        started = time.perf_counter()
        indices = stationary_indices(len(pool), sims, horizon, block_length, seed)
        paths = forecast_paths(np.array([fit[0] for fit in fits]),
            np.array([fit[1] for fit in fits]), np.array([fit[2] for fit in fits]),
            pool, indices, means)
        timed('asset_predictive_paths', started)
        for x, fit in zip(assets.T, fits):
            identity = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            FIT_DIAGNOSTICS[self.model_id, identity] = {'model_id': self.model_id,
                'data_sha256': identity, 'gjr_student_fit': fit[4],
                'stationary_bootstrap': bootstrap_receipt,
                'return_mean_curve': 'exact_M261_Gaussian_original_fit_existing_sv_rate',
                'parameter_uncertainty': False, 'rough_fit': False,
                'innovation_law': 'centered_unit_second_moment_empirical_training_residuals',
                'baseline_daily_log_return_clip': [-1., 1.]}
        dates = shell._historical_rebalance_dates(past.append(future), training.policy.rebalance)
        started = time.perf_counter()
        output = rejoin(paths, training.policy.weights, np.array([date in dates for date in future]))
        timed('portfolio_rejoin', started)
        return output


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    if cache_root is not None:
        raise ValueError('persistent fit caches are disabled for this candidate')
    parent.initialize(None, vine_threads, state_workers)
    forecast_paths(np.array([[1e-5, .05, .05, .8]]), np.array([1e-4]),
        np.array([.01]), np.array([[-1.], [1.]]), np.zeros((1, 2), np.int64), np.zeros((2, 1)))


def clear_path_cache():
    parent.clear_path_cache()
    configuration.cache_clear()


CANDIDATES = (Candidate(),)
MODEL_IDS = (parent.MODEL_IDS[0], MODEL_ID)
