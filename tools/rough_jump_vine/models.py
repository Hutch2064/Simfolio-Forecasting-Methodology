"""Asset-level rough-SV, compensated Poisson jumps, and HMM R-vine ablations.

Rough SV is an eight-factor daily Gaussian Volterra approximation. All fitting
uses the origin's training data. Production mean/variance anchors and policy
rejoin are retained; rough volatility is simulated pathwise, not compressed
into deterministic moment curves. Jump events are removed from the diffusion
innovation pool before adding the compensated compound-Poisson component.
"""
from __future__ import annotations

import hashlib
import math
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import optimized_mcmc as opt
import parameter_mcmc_tuned as pm
import pyvinecopulib as pv
from numba import njit
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import rankdata
from streamed_paths import map_asset_inplace, rejoin

from simfolio_forecasting_methodology.models.asset_level.frontier import (
    _historical_rebalance_dates,
    _validate_calendar,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg

MODEL_IDS = (
    'sv_parameter_mcmc_twochain_sixteen_node_moment_mixture',
    'asset_rough_volterra_sv_eight_factor',
    'asset_compensated_poisson_jump_sv',
    'asset_hmm_three_state_full_rvine_sv',
    'asset_rough_jump_hmm_rvine_sv',
)
VINE_THREADS = 3
STATE_WORKERS = 3
CACHE_ROOT: Path | None = None
TIMINGS = {}
_VINE_DRAW_CACHE = None


def timed(key, started):
    TIMINGS[key] = TIMINGS.get(key, 0.0) + time.perf_counter() - started


def cache(namespace, identity, builder):
    if CACHE_ROOT is None:
        return builder()
    import fcntl
    folder = CACHE_ROOT / namespace
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identity + '.pkl')
    with (folder / (identity + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            with path.open('rb') as handle:
                return pickle.load(handle)
        value = builder()
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(value, handle, protocol=5)
        temporary.replace(path)
        return value


@lru_cache(maxsize=64)
def asset_fit(data):
    start = time.perf_counter()
    fit = cache('production_asset', hashlib.sha256(data).hexdigest(),
                lambda: pm.parameter_fit(data))
    timed('production_asset_fit', start)
    return fit


def initialize(cache_root=None, vine_threads=3, state_workers=3):
    global CACHE_ROOT, VINE_THREADS, STATE_WORKERS
    from runtime_setup import initialize as source_initialize
    source_initialize(True)
    opt.install(False)
    CACHE_ROOT = None if cache_root is None else Path(cache_root)
    VINE_THREADS, STATE_WORKERS = vine_threads, state_workers
    from numba import set_num_threads
    set_num_threads(1)
    # Full fits live in the shared content-addressed cache; avoid retaining
    # thousands of sixteen-node histories separately in every worker process.
    original = pm.parameter_fit
    pm.parameter_fit = lru_cache(maxsize=32)(original.__wrapped__)


@njit(cache=True, nogil=True)
def filter_rough(y, phi, weights, covariance, level, keep_path=False):
    dimension = len(phi)
    state = np.zeros(dimension)
    stationary = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            stationary[i, j] = covariance[i, j] / (1.0 - phi[i] * phi[j])
    posterior = stationary.copy()
    path = np.empty(len(y))
    loglik = 0.0
    fixed_covariance = False
    for t in range(len(y)):
        if not fixed_covariance:
            predicted = posterior * phi[:, None] * phi[None, :] + covariance
            pw = predicted @ weights
            variance = 4.934802200544679 + np.dot(weights, pw)
            gain = pw / variance
            log_variance = math.log(2 * math.pi * variance)
        predicted_state = phi * state
        innovation = y[t] - level - np.dot(weights, predicted_state)
        state = predicted_state + gain * innovation
        if not fixed_covariance:
            updated = predicted - np.outer(pw, pw) / variance
            fixed_covariance = np.array_equal(updated.view(np.uint64), posterior.view(np.uint64))
            posterior = updated
        loglik -= 0.5 * (log_variance + innovation**2 / variance)
        path[t] = level + np.dot(weights, state)
    return loglik, state, posterior, path


def rough_parameters(theta):
    hurst, kappa, scale = theta[0], math.exp(theta[1]), math.exp(theta[2])
    edges = np.geomspace(1.0 / (252.0 * 50), 4.0, 9)
    rates = np.sqrt(edges[:-1] * edges[1:])
    weights = (edges[1:]**(0.5-hurst)-edges[:-1]**(0.5-hurst))/(0.5-hurst)
    rates += kappa
    phi = np.exp(-rates)
    sums = rates[:, None] + rates[None, :]
    raw = -np.expm1(-sums) / sums
    stationary = 1 / sums
    normalizer = np.sqrt(weights @ stationary @ weights)
    weights /= normalizer
    covariance = scale**2 * raw
    return phi, weights, covariance


@lru_cache(maxsize=64)
def rough_fit(data):
    def build():
        x = np.frombuffer(data, np.float64)
        y, _ = bd._sv_observed_log_variance(x, float(x.mean()))
        level = float(y.mean())
        def objective(theta):
            phi, weights, covariance = rough_parameters(theta)
            likelihood = filter_rough(y, phi, weights, covariance, level)[0]
            # Weak scale/kappa regularization and bounded H; fixed before OOS.
            penalty = 0.5*((theta[1]-math.log(1/63))/2.0)**2
            penalty += 0.5*((theta[2]-math.log(0.7))/1.5)**2
            return -likelihood + penalty
        bounds = [(0.03, 0.49), (math.log(1/2520), math.log(1/2)),
                  (math.log(0.05), math.log(3.0))]
        result = minimize(objective, [0.1, math.log(1/63), math.log(0.7)],
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 80, 'ftol': 1e-9, 'gtol': 1e-5})
        if not np.isfinite(result.fun):
            raise ValueError('nonfinite rough-SV fit')
        phi, weights, covariance = rough_parameters(result.x)
        _, state, posterior, _path = filter_rough(y, phi, weights, covariance, level)
        return {'phi': phi, 'weights': weights, 'covariance': covariance,
                    'state': state, 'posterior': posterior, 'theta': result.x,
                    'level': level,
                    'converged': bool(result.success), 'iterations': int(result.nit),
                    'objective': float(result.fun)}
    start = time.perf_counter()
    value = cache('rough_asset', hashlib.sha256(data).hexdigest(), build)
    timed('rough_asset_fit', start)
    return value


@njit(cache=True, nogil=True)
def rough_paths_reference(phi, weights, covariance, initial, normals):
    sims, horizon, dimension = normals.shape
    states = initial.copy()
    paths = np.empty((sims, horizon))
    for t in range(horizon):
        for sim in range(sims):
            total = 0.0
            for i in range(dimension):
                total += weights[i]*states[sim, i]
            paths[sim, t] = total
            innovation = np.empty(dimension)
            for i in range(dimension):
                total = 0.0
                for j in range(dimension):
                    total += covariance[i, j]*normals[sim, t, j]
                innovation[i] = total
            for i in range(dimension):
                states[sim, i] = phi[i]*states[sim, i]+innovation[i]
    return paths


@njit(cache=True, nogil=True)
def rough_paths(phi, weights, covariance, initial, normals):
    # Reference operation order is preserved for every element.
    sims, horizon, dimension = normals.shape
    states = initial.copy()
    paths = np.empty((sims, horizon))
    innovation = np.empty(dimension)
    for s in range(sims):
        for t in range(horizon):
            total = 0.0
            for i in range(dimension):
                total += weights[i]*states[s, i]
            paths[s, t] = total
            for i in range(dimension):
                total = 0.0
                for j in range(dimension):
                    total += covariance[i, j]*normals[s, t, j]
                innovation[i] = total
            for i in range(dimension):
                states[s, i] = phi[i]*states[s, i]+innovation[i]
    return paths


def rough_multiplier(fit, sims, horizon, seed):
    phi, weights = fit['phi'], fit['weights']
    rng = np.random.default_rng(seed)
    posterior = (fit['posterior'] + fit['posterior'].T) / 2
    eigen, vectors = np.linalg.eigh(posterior)
    root = vectors @ np.diag(np.sqrt(np.maximum(eigen, 0.0)))
    initial = fit['state'][None, :] + rng.normal(size=(sims, 8)) @ root.T
    normals = rng.normal(size=(sims, horizon, 8))
    chol = np.linalg.cholesky(fit['covariance'] + np.eye(8)*1e-14)
    initial = initial*phi[None, :] + rng.normal(size=(sims, 8))@chol.T
    latent = rough_paths(phi, weights, chol, initial, normals)
    mean = np.empty(horizon)
    variance = np.empty(horizon)
    state = phi*fit['state']
    p = posterior*phi[:, None]*phi[None, :]+fit['covariance']
    for t in range(horizon):
        mean[t] = weights@state
        variance[t] = weights@p@weights
        state *= phi
        p = p*phi[:, None]*phi[None, :]+fit['covariance']
    # Preserve production's conditional mean and expected daily variance;
    # the stochastic multiplier carries the rough temporal dependence.
    return np.exp(0.5*(latent-mean[None, :])-0.25*variance[None, :])


@lru_cache(maxsize=6)
def cached_rough_multiplier(data, sims, horizon, seed):
    return rough_multiplier(rough_fit(data), sims, horizon, seed)


def clear_path_cache():
    global _VINE_DRAW_CACHE
    _VINE_DRAW_CACHE = None
    cached_rough_multiplier.cache_clear()
    gaussian_uniforms.cache_clear()
    asset_nodes.cache_clear()


@lru_cache(maxsize=2)
def gaussian_uniforms(shape, data, sims, horizon, seed):
    assets = np.frombuffer(data, np.float64).reshape(shape)
    dependency = dg.fit_dynamic_gaussian_factor_model(assets)
    return dg.simulate_future_gaussian_uniforms(
        dependency, sims, horizon, np.random.default_rng(seed))


@lru_cache(maxsize=64)
def jump_fit(data):
    fit = asset_fit(data)
    pools = np.vstack([c['innovation_pool'] for c in fit['parameter_posterior_children']])
    pool = pools.reshape(-1)
    event = np.abs(pools) > 3.0
    jumps = pool[np.abs(pool) > 3.0]
    body = pool[np.abs(pool) <= 3.0].copy()
    body -= body.mean()
    body /= np.sqrt(np.mean(body**2))
    # Beta(1,99) regularization is applied to the posterior-averaged event
    # sequence, so sixteen SV draws do not count as sixteen observations.
    probability = (event.mean(axis=0).sum()+1)/(pools.shape[1]+100)
    rate = -math.log1p(-probability)
    recent = probability
    for observed in event.mean(axis=0):
        recent = (20/21)*recent + observed/21
    recent = float(np.clip(recent, 1e-8, 1-1e-8))
    return {'rate': rate, 'recent': -math.log1p(-recent),
                'location': float(jumps.mean()) if len(jumps) else 0.0,
                'variance': float(jumps.var()) if len(jumps) else 16.0,
                'body': body, 'event_threshold': 3.0}


@njit(cache=True, nogil=True)
def interpolate_nodes(uniforms, nodes):
    sims, horizon = uniforms.shape
    for sim in range(sims):
        for day in range(horizon):
            position = min(max(uniforms[sim, day], 0.), 1.)*float(sims-1)
            low = math.floor(position)
            high = min(low+1, sims-1)
            fraction = position-low
            uniforms[sim, day] = nodes[low]*(1-fraction)+nodes[high]*fraction


@njit(cache=True, nogil=True)
def poisson_inverse(uniforms, rates):
    result = np.empty(uniforms.shape, np.int64)
    for s in range(uniforms.shape[0]):
        for t in range(uniforms.shape[1]):
            mass = math.exp(-rates[t])
            cumulative = mass
            count = 0
            while uniforms[s, t] > cumulative:
                count += 1
                mass *= rates[t]/count
                previous = cumulative
                cumulative += mass
                if cumulative == previous:
                    raise ValueError('Poisson inverse exhausted numerical probability mass')
            result[s, t] = count
    return result


@njit(cache=True, nogil=True)
def apply_jumps(standardized, counts, normals, location, variance, compensation, normalizer):
    for s in range(standardized.shape[0]):
        for t in range(standardized.shape[1]):
            mark = counts[s, t]*location + math.sqrt(counts[s, t]*variance)*normals[s, t]
            standardized[s, t] += mark-compensation[t]
            standardized[s, t] /= normalizer[t]


@njit(cache=True, nogil=True)
def hmm_forward_backward(log_emissions, transition, initial):
    n, states = log_emissions.shape
    offset = np.empty(n)
    for t in range(n):
        offset[t] = np.max(log_emissions[t])
    emission = np.exp(log_emissions-offset[:, None])
    alpha = np.empty((n, states))
    scales = np.empty(n)
    alpha[0] = initial*emission[0]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]
    for t in range(1, n):
        alpha[t] = (alpha[t-1]@transition)*emission[t]
        scales[t] = alpha[t].sum()
        alpha[t] /= scales[t]
    beta = np.ones((n, states))
    counts = np.zeros((states, states))
    for t in range(n-2, -1, -1):
        beta[t] = transition@(emission[t+1]*beta[t+1])/scales[t+1]
        pair = alpha[t][:, None]*transition*(emission[t+1]*beta[t+1])[None, :]
        counts += pair/pair.sum()
    posterior = alpha*beta
    for t in range(n):
        posterior[t] /= posterior[t].sum()
    return float(np.sum(np.log(scales)+offset)), posterior, counts, alpha[-1]


def gaussian_emissions(z, correlations):
    result = np.empty((len(z), 3))
    for k, correlation in enumerate(correlations):
        inverse = np.linalg.inv(correlation)-np.eye(z.shape[1])
        result[:, k] = -0.5*np.linalg.slogdet(correlation)[1]
        result[:, k] -= 0.5*np.sum((z@inverse)*z, axis=1)
    return result


def fit_hmm_vines(uniforms):
    u = np.asfortranarray(np.clip(uniforms, 1e-8, 1-1e-8))
    z = ndtri(u)
    n, d = z.shape
    transition = np.full((3, 3), 0.025)
    np.fill_diagonal(transition, 0.95)
    initial = np.ones(3)/3
    correlations = [np.eye(d)*0.95+np.ones((d, d))*0.05,
                    np.eye(d)*0.65+np.ones((d, d))*0.35,
                    np.eye(d)*0.25+np.ones((d, d))*0.75]
    for _ in range(20):
        _, posterior, counts, terminal = hmm_forward_backward(
            gaussian_emissions(z, correlations), transition, initial)
        for k in range(3):
            weighted = z.T@(z*posterior[:, k, None])/posterior[:, k].sum()
            diagonal = np.sqrt(np.diag(weighted))
            weighted /= diagonal[:, None]*diagonal[None, :]
            correlations[k] = 0.95*weighted+0.05*np.eye(d)
        transition = counts+0.5
        transition /= transition.sum(axis=1)[:, None]
        initial = posterior[0]
    families = [pv.BicopFamily.indep, pv.BicopFamily.gaussian,
                pv.BicopFamily.student, pv.BicopFamily.clayton,
                pv.BicopFamily.gumbel, pv.BicopFamily.frank]
    def select(k):
        controls = pv.FitControlsVinecop(
            family_set=families, selection_criterion='bic', weights=posterior[:, k],
            num_threads=VINE_THREADS, allow_rotations=True, select_trunc_lvl=False)
        return pv.Vinecop.from_data(u, controls=controls)
    def density(vine):
        return np.log(np.maximum(vine.pdf(u), 1e-300))
    def refit(item):
        k, vine = item
        vine.fit(u, controls=pv.FitControlsBicop(weights=posterior[:, k]), num_threads=VINE_THREADS)
    history = []
    with ThreadPoolExecutor(max_workers=STATE_WORKERS) as pool:
        vines = list(pool.map(select, range(3)))
        for iteration in range(12):
            emissions = np.column_stack(list(pool.map(density, vines)))
            likelihood, posterior, counts, terminal = hmm_forward_backward(
                emissions, transition, initial)
            history.append(likelihood)
            if iteration and abs(history[-1]-history[-2])/(1+abs(history[-2])) < 1e-5:
                break
            transition = counts+0.5
            transition /= transition.sum(axis=1)[:, None]
            initial = posterior[0]
            list(pool.map(refit, enumerate(vines)))
        # Terminal probabilities must correspond to the final fitted emissions.
        emissions = np.column_stack(list(pool.map(density, vines)))
    likelihood, _, _, terminal = hmm_forward_backward(emissions, transition, initial)
    return {'vines': [v.to_json() for v in vines], 'transition': transition,
                'terminal': terminal, 'loglikelihood': likelihood, 'likelihood_history': history,
                'iterations': len(history), 'dimension': d, 'training_rows': n}


@lru_cache(maxsize=4)
def dependence_fit(shape, data):
    def build():
        assets = np.frombuffer(data, np.float64).reshape(shape)
        uniforms = np.empty_like(assets)
        for a in range(assets.shape[1]):
            fit = asset_fit(assets[:, a].tobytes())
            pools = np.vstack([c['innovation_pool'] for c in fit['parameter_posterior_children']])
            residual = pools.mean(axis=0)
            uniforms[:, a] = rankdata(residual)/(len(residual)+1.0)
        return fit_hmm_vines(uniforms)
    start = time.perf_counter()
    identity = hashlib.sha256(str(shape).encode()+data).hexdigest()
    fit = cache('hmm_vine', identity, build)
    timed('hmm_vine_fit', start)
    return fit


@njit(cache=True, nogil=True)
def sample_states(transition, initial, draws):
    sims, horizon = draws.shape
    states = np.empty((sims, horizon), np.int64)
    for s in range(sims):
        probabilities = initial.copy()
        for t in range(horizon):
            if t:
                probabilities = transition[states[s, t-1]]
            cumulative = 0.0
            state = len(initial)-1
            for k in range(len(initial)):
                cumulative += probabilities[k]
                if draws[s, t] < cumulative:
                    state = k
                    break
            states[s, t] = state
    return states


def vine_uniforms(fit, sims, horizon, seed, streams=1):
    global _VINE_DRAW_CACHE
    key = (tuple(fit['vines']), fit['transition'].tobytes(), fit['terminal'].tobytes(),
           sims, horizon, seed)
    def transform(independent, flat_states, vines):
        transformed = np.empty_like(independent)
        work = []
        for k, vine in enumerate(vines):
            indices = np.flatnonzero(flat_states == k)
            for start in range(0, len(indices), 65536):
                work.append((vine, indices[start:start+65536]))
        def invert(item):
            vine, rows = item
            transformed[rows] = vine.inverse_rosenblatt(
                    np.asfortranarray(independent[rows]), num_threads=VINE_THREADS)
        with ThreadPoolExecutor(max_workers=STATE_WORKERS) as pool:
            list(pool.map(invert, work))
        return transformed.reshape(sims, horizon, fit['dimension'])
    if _VINE_DRAW_CACHE is None or _VINE_DRAW_CACHE[0] != key:
        rng = np.random.default_rng(seed)
        # The first future regime follows one transition from the filtered state.
        states = sample_states(fit['transition'], fit['terminal']@fit['transition'],
                               rng.random((sims, horizon))).reshape(-1)
        vines = [pv.Vinecop.from_json(text) for text in fit['vines']]
        first = transform(rng.random((sims*horizon, fit['dimension'])), states, vines)
        _VINE_DRAW_CACHE = (key, first, states, vines, rng.bit_generator.state)
    _, first, states, vines, rng_state = _VINE_DRAW_CACHE
    # Marginal mapping mutates its uniforms; retain the original shared stream.
    output = [first.copy()]
    rng = np.random.default_rng(0)
    rng.bit_generator.state = rng_state
    for _ in range(1, streams):
        independent = rng.random((sims*horizon, fit['dimension']))
        output.append(transform(independent, states, vines))
    return output


def empirical_nodes(pool, simulations):
    nodes = np.quantile(pool, np.linspace(.5/simulations, 1-.5/simulations, simulations))
    nodes -= nodes.mean()
    nodes /= np.sqrt(np.mean(nodes**2))
    return nodes


@lru_cache(maxsize=12)
def asset_nodes(data, simulations, jumps):
    if jumps:
        pool = jump_fit(data)['body']
    else:
        fit = asset_fit(data)
        pool = np.concatenate([c['innovation_pool'] for c in fit['parameter_posterior_children']])
    return empirical_nodes(pool, simulations)


@dataclass(frozen=True)
class Candidate:
    model_id: str
    rough: bool = False
    jumps: bool = False
    vine: bool = False

    def simulate_daily_log_returns(self, training, context):
        training.validate()
        past, future = _validate_calendar(training, context)
        assets = np.asarray(training.asset_log_returns, np.float64)
        sims, horizon = context.simulations, context.horizon_days
        seed = bd.deterministic_seed('copula_alternatives', bd.FRONTIER_DEPENDENCE_ID,
                                     str(context.origin_date), horizon, sims)
        data = [assets[:, a].tobytes() for a in range(assets.shape[1])]
        fits = [asset_fit(value) for value in data]
        started = time.perf_counter()
        if self.vine:
            dependency = dependence_fit(assets.shape, assets.tobytes())
            streams = vine_uniforms(dependency, sims, horizon, seed, 2 if self.jumps else 1)
            uniforms = streams[0]
            count_uniforms = streams[1] if self.jumps else None
        else:
            raw = assets.tobytes()
            uniforms = gaussian_uniforms(assets.shape, raw, sims, horizon, seed).copy()
            count_uniforms = (gaussian_uniforms(
                assets.shape, raw, sims, horizon, (seed+71237) % 2**32)
                if self.jumps else None)
        timed('dependence_paths', started)
        for a, (value, fit) in enumerate(zip(data, fits)):
            started = time.perf_counter()
            mean, sd = opt.mixture_curves(fit, horizon)
            if self.jumps:
                jump = jump_fit(value)
            nodes = asset_nodes(value, sims, self.jumps)
            # Generate standardized empirical innovations using the exact
            # production marginal interpolation kernel.
            if not self.rough and not self.jumps:
                map_asset_inplace(uniforms[:, :, a], mean, sd, nodes)
                timed("asset_predictive_paths", started)
                continue
            interpolate_nodes(uniforms[:, :, a], nodes)
            standardized = uniforms[:, :, a]
            if self.jumps:
                rates = jump['rate']+(jump['recent']-jump['rate'])*np.exp(-np.arange(1, horizon+1)/21)
                counts = poisson_inverse(count_uniforms[:, :, a], rates)
                rng = np.random.default_rng(bd.deterministic_seed('jump_marks', hashlib.sha256(value).hexdigest(),
                                                                 str(context.origin_date), horizon, sims))
                normalizer = np.sqrt(1+rates*(jump['variance']+jump['location']**2))
                apply_jumps(standardized, counts, rng.normal(size=counts.shape),
                            jump['location'], jump['variance'], rates*jump['location'], normalizer)
            if self.rough:
                rough_seed = bd.deterministic_seed('rough_paths', hashlib.sha256(value).hexdigest(),
                                                  str(context.origin_date), horizon, sims)
                standardized *= cached_rough_multiplier(value, sims, horizon, rough_seed)
            uniforms[:, :, a] = np.clip(mean[None, :]+sd[None, :]*standardized, -1, 1)
            timed('asset_predictive_paths', started)
        started = time.perf_counter()
        dates = _historical_rebalance_dates(past.append(future), training.policy.rebalance)
        mask = np.asarray([date in dates for date in future])
        result = rejoin(uniforms, training.policy.weights, mask)
        timed('portfolio_rejoin', started)
        return result


CANDIDATES = (
    Candidate(MODEL_IDS[0]), Candidate(MODEL_IDS[1], rough=True),
    Candidate(MODEL_IDS[2], jumps=True), Candidate(MODEL_IDS[3], vine=True),
    Candidate(MODEL_IDS[4], rough=True, jumps=True, vine=True),
)
