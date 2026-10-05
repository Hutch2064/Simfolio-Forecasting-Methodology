"""Asset-level tempered rough Volterra SV without a conventional SV anchor.

The Gaussian log-square observation law is the existing research QML law.
Level is sampled jointly with H, log(kappa), and log(eta). Historical drift,
the dynamic Gaussian return copula, calendar and policy rejoin are retained.
Each predictive draw uses its own rough-filtered empirical innovation law.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
from numba import njit
from scipy.special import log_ndtr, ndtri_exp

import overlay
from dynamic import selected_kernel


def log_target(y, theta, maximum_lag):
    prior = overlay.log_prior(theta[:3])
    lower, upper = np.quantile(y, [.01, .99])
    if not np.isfinite(prior) or not lower - 4 <= theta[3] <= upper + 4:
        return -np.inf
    prior -= .5 * ((theta[3] - y.mean()) / 4) ** 2
    phi, weights, covariance = overlay.configuration(theta[:3], f'dynamic:{maximum_lag}')
    return prior + overlay.filter_rough(y, phi, weights, covariance, theta[3])[0]


def normal_interval_logmass(mean, sd, lower, upper):
    a,b = (lower-mean)/sd,(upper-mean)/sd
    if a >= 0:
        a,b = -b,-a
    lo,hi = float(log_ndtr(a)),float(log_ndtr(b))
    return hi+math.log1p(-math.exp(lo-hi))


def truncated_normal(mean, sd, lower, upper, uniform):
    a,b = (lower-mean)/sd,(upper-mean)/sd
    reflect = a >= 0
    if reflect:
        a,b,uniform = -b,-a,1-uniform
    lo,hi = float(log_ndtr(a)),float(log_ndtr(b))
    probability = lo if uniform == 0 else hi+math.log(uniform+(1-uniform)*math.exp(lo-hi))
    draw = float(ndtri_exp(probability))
    return mean+sd*(-draw if reflect else draw)


def parameter_chain(y, maximum_lag, seed, burn, kept, resume=None):
    """Warmup-only adaptive Metropolis; latent rough states integrated out."""
    rng = np.random.default_rng(seed)
    theta = np.array([rng.uniform(.08, .15), math.log(1 / 63), math.log(.7), y.mean()])
    root = np.diag([.01, .1, .05, .05])
    if resume:
        theta, root = resume['theta'].copy(), resume['root'].copy()
        rng.bit_generator.state = resume['rng_state']
    # Observation-only prior constants are calculated once per chain.
    lower, upper = np.quantile(y, [.01, .99]) + np.array([-4, 4])
    center = float(y.mean())
    def target(point):
        prior = overlay.log_prior(point[:3])
        if not np.isfinite(prior) or not lower <= point[3] <= upper:
            return -np.inf
        phi, weights, covariance = overlay.configuration(point[:3], f'dynamic:{maximum_lag}')
        return (prior - .5 * ((point[3] - center) / 4) ** 2
                + overlay.filter_rough(y, phi, weights, covariance, point[3])[0])
    value = target(theta)
    mean, m2 = np.zeros(4), np.zeros((4, 4))
    draws = np.empty((kept, 5))
    for iteration in range(burn + kept):
        proposed = theta + root @ rng.normal(size=4)
        likelihood = target(proposed)
        if math.log(rng.random()) < likelihood - value:
            theta, value = proposed, likelihood
        if iteration < burn:
            delta = theta - mean
            mean += delta / (iteration + 1)
            m2 += np.outer(delta, theta - mean)
            if iteration >= 63 and (iteration + 1) % 32 == 0:
                root = np.linalg.cholesky(2.38 ** 2 / 4 * (m2 / iteration + np.eye(4) * 1e-6))
        else:
            draws[iteration - burn, :4] = theta
            draws[iteration - burn, 4] = value
    return draws, {'theta': theta.copy(), 'root': root.copy(), 'rng_state': rng.bit_generator.state}


@lru_cache(maxsize=16)
def fit(data, maximum_lag, simulations, burn, kept, maximum):
    import models as shell
    identity = hashlib.sha256(data).hexdigest()
    settings = f'standalone-v1:{maximum_lag}:{simulations}:{burn}:{kept}:{maximum}'
    source = hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
        ('standalone.py', 'overlay.py', 'dynamic.py', 'native.py', 'overlay_filter.cpp'))
        + Path(shell.bd.__file__).read_bytes()).hexdigest()
    contract = source + ':' + settings
    key = hashlib.sha256((identity + contract).encode()).hexdigest()
    def build():
        started = time.perf_counter()
        x = np.frombuffer(data, np.float64)
        y, _ = shell.bd._sv_observed_log_variance(x, float(x.mean()))
        chains = [parameter_chain(y, maximum_lag,
            shell.bd.deterministic_seed('rough_standalone_fit', identity, settings, j), burn, kept)
            for j in range(2)]
        while True:
            trace = np.array([c[0] for c in chains])
            diagnostics = shell.diagnostics(trace)
            converged = all(d['rank_split_rhat'] < 1.05 and d['bulk_ess'] >= 100 for d in diagnostics)
            actual = trace.shape[1]
            if converged or actual >= maximum:
                break
            extension = min(actual, maximum - actual)
            for j, (draws, state) in enumerate(chains):
                extra, state = parameter_chain(y, maximum_lag, 0, 0, extension, state)
                chains[j] = (np.concatenate([draws, extra]), state)
        all_draws = np.concatenate([c[0][:, :4] for c in chains])
        rng = np.random.default_rng(shell.bd.deterministic_seed('rough_standalone_joint_draws', identity, settings))
        parameters = all_draws[rng.integers(0, len(all_draws), size=simulations)]
        records = [selected_kernel(float(t[0]), math.exp(t[1]), maximum_lag)[2] for t in parameters]
        seconds = time.perf_counter() - started
        return {'parameters': parameters, 'trace': trace, 'diagnostics': diagnostics,
            'diagnostic_columns': ['H', 'log_kappa', 'log_eta', 'level', 'log_posterior'],
            'convergence_flag': converged, 'kept_per_chain': actual, 'fit_seconds': seconds,
            'ess_per_second': min(d['bulk_ess'] for d in diagnostics) / seconds,
            'data_sha256': identity, 'contract': contract,
            'kernel': {'selection': 'per_parameter_dynamic_positive_integer_order',
                'minimum_factors': min(r['factors'] for r in records),
                'maximum_factors': max(r['factors'] for r in records),
                'maximum_autocorrelation_error_upper_bound': max(r['autocorrelation_error_upper_bound'] for r in records),
                'tolerance': .001, 'maximum_daily_lag': maximum_lag}}
    return shell.controls.cache('bayesian_standalone_asset', key, build)


def _fit_asset(arguments):
    return fit(*arguments)


@lru_cache(maxsize=2)
def fitted_assets(data, maximum_lag, simulations, burn, kept, maximum, workers, cache_root):
    arguments = [(x, maximum_lag, simulations, burn, kept, maximum) for x in data]
    if workers == 1:
        return tuple(_fit_asset(a) for a in arguments)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=overlay._initialize_asset_fit_worker, initargs=(cache_root,)) as pool:
        return tuple(pool.map(_fit_asset, arguments))


@lru_cache(maxsize=16)
def predictive_states(data, maximum_lag, parameter_bytes, contract, simulations):
    import models as shell
    identity = hashlib.sha256(data + parameter_bytes + repr((maximum_lag, contract, simulations)).encode()).hexdigest()
    def build():
        x = np.frombuffer(data, np.float64)
        y, eps = shell.bd._sv_observed_log_variance(x, float(x.mean()))
        prepared, unique = [], {}
        for theta in np.frombuffer(parameter_bytes, np.float64).reshape(-1, 4):
            key = theta.tobytes()
            if key not in unique:
                phi, weights, covariance = overlay.configuration(theta[:3], f'dynamic:{maximum_lag}')
                _, state, p, history = shell.controls.filter_rough(y, phi, weights, covariance, theta[3], True)
                pool = shell.bd._standardized_empirical_innovation_pool(eps * np.exp(-.5 * history), clip=None, method='mean_std')
                nodes = np.quantile(pool, np.linspace(.5 / simulations, 1 - .5 / simulations, simulations))
                nodes -= nodes.mean()
                nodes /= np.sqrt(np.mean(nodes * nodes))
                values, vectors = np.linalg.eigh((p + p.T) / 2)
                initial_root = vectors * np.sqrt(np.maximum(values, 0))
                # The dynamic spectral lift has independent OU innovations.
                assert np.array_equal(covariance, np.diag(np.diag(covariance)))
                innovation_sd = np.sqrt(np.diag(covariance))
                unique[key] = (phi, weights, state, initial_root, innovation_sd, float(theta[3]), nodes)
            prepared.append(unique[key])
        return prepared
    return shell.controls.cache('rough_standalone_predictive_states', identity, build)


@njit(cache=True, nogil=True)
def map_shocks(uniforms, nodes):
    result = np.empty(uniforms.size)
    for t in range(result.size):
        position = min(max(uniforms[t], 0.), 1.) * float(nodes.size - 1)
        low = math.floor(position)
        high = min(low + 1, nodes.size - 1)
        fraction = position - low
        result[t] = nodes[low] * (1 - fraction) + nodes[high] * fraction
    return result


@njit(cache=True, nogil=True)
def volatility_path(phi, weights, innovation_sd, initial, normals, level):
    """Advance before each return: exp(log variance / 2), in decimal units."""
    state = initial.copy()
    result = np.empty(normals.shape[0])
    for t in range(result.size):
        for j in range(phi.size):
            state[j] = phi[j] * state[j] + innovation_sd[j] * normals[t, j]
        result[t] = math.exp(.5 * (level + np.dot(weights, state))) / 100
    return result


@dataclass(frozen=True)
class StandaloneCandidate:
    model_id: str = 'asset_rough_volterra_sv_dynamic_standalone_bayesian'
    burn: int = 2048
    kept: int = 8192
    max_kept: int = 65536

    def simulate_daily_log_returns(self, training, context):
        import models as shell
        training.validate()
        past, future = shell._validate_calendar(training, context)
        assets = np.asarray(training.asset_log_returns, np.float64)
        sims, horizon = context.simulations, context.horizon_days
        maximum_lag = len(assets) + horizon - 1
        data = tuple(assets[:, a].tobytes() for a in range(assets.shape[1]))
        workers = min(assets.shape[1], max(1, shell.controls.STATE_WORKERS))
        started = time.perf_counter()
        posteriors = fitted_assets(data, maximum_lag, sims, self.burn, self.kept,
            self.max_kept, workers, shell.controls.CACHE_ROOT)
        shell.controls.timed('rough_standalone_fit_phase_wall', started)
        seed = shell.bd.deterministic_seed('copula_alternatives', shell.bd.FRONTIER_DEPENDENCE_ID,
            str(context.origin_date), horizon, sims)
        uniforms = shell.controls.gaussian_uniforms(assets.shape, assets.tobytes(), sims, horizon, seed).copy()
        for a, posterior in enumerate(posteriors):
            shell.FIT_DIAGNOSTICS[(self.model_id, posterior['data_sha256'])] = {
                name: posterior[name] for name in ('data_sha256', 'diagnostics', 'diagnostic_columns',
                    'convergence_flag', 'kept_per_chain', 'fit_seconds', 'ess_per_second', 'kernel')}
            shell.FIT_DIAGNOSTICS[(self.model_id, posterior['data_sha256'])].update(
                model_id=self.model_id, burn_per_chain=self.burn)
            started = time.perf_counter()
            prepared = predictive_states(data[a], maximum_lag, posterior['parameters'].tobytes(),
                posterior['contract'], sims)
            rng = np.random.default_rng(shell.bd.deterministic_seed('rough_standalone_prediction',
                posterior['data_sha256'], str(context.origin_date), horizon, sims))
            mean = float(assets[:, a].mean())
            for s, (phi, weights, state, initial_root, innovation_sd, level, nodes) in enumerate(prepared):
                # One conditional innovation distribution per joint parameter draw.
                shock = map_shocks(uniforms[s, :, a], nodes)
                initial = state + initial_root @ rng.normal(size=phi.size)
                sd = volatility_path(phi, weights, innovation_sd, initial,
                    rng.normal(size=(horizon, phi.size)), level)
                uniforms[s, :, a] = np.clip(mean + sd * shock, -1, 1)
            shell.controls.timed('rough_standalone_predictive_paths', started)
        dates = shell._historical_rebalance_dates(past.append(future), training.policy.rebalance)
        mask = np.asarray([date in dates for date in future])
        return shell.rejoin(uniforms, training.policy.weights, mask)
