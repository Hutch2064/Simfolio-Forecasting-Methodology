"""Incremental Bayesian upgrades of the winning production-anchored overlay.

The eight-factor arm changes parameter uncertainty alone. The adaptive arm
approximates the original tempered fractional Volterra covariance, with its
unresolved zero-lag variance retained as an analytically sized white component.
Both retain the original Gaussian log-square quasi likelihood and empirical
return innovations, rather than changing the production variance anchor.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from numba import njit
from scipy.special import gamma, hyperu

NODES, GAUSS = np.polynomial.legendre.leggauss(3)


def exact_covariance(hurst, kappa, lags):
    alpha = hurst + .5
    lags = np.asarray(lags, float)
    zero = gamma(2 * hurst) / (2 * kappa) ** (2 * hurst) / gamma(alpha) ** 2
    out = np.ones_like(lags)
    positive = lags > 0
    t = lags[positive]
    out[positive] = (np.exp(-kappa * t) * t ** (2 * hurst)
                    * hyperu(alpha, 1 + 2 * hurst, 2 * kappa * t) / gamma(alpha) / zero)
    return out


def lifted_covariance(hurst, kappa, bins):
    alpha = hurst + .5
    edges = np.linspace(math.log(1e-11), math.log(1e9), bins + 1)
    widths = np.diff(edges) / 2
    locations = (edges[:-1] + edges[1:]) / 2
    rates = np.exp((locations[:, None] + widths[:, None] * NODES).ravel())
    mass = (widths[:, None] * GAUSS * rates.reshape(-1, 3) ** (1 - alpha)).ravel()
    mass /= gamma(alpha) * gamma(1 - alpha)
    low = 1e-11 ** (1 - alpha) / (1 - alpha) / (gamma(alpha) * gamma(1 - alpha))
    mass, rates = np.r_[low, mass], np.r_[.5e-11, rates] + kappa
    covariance = np.outer(mass, mass) / (rates[:, None] + rates[None, :])
    zero = gamma(2 * hurst) / (2 * kappa) ** (2 * hurst) / gamma(alpha) ** 2
    raw_zero = covariance.sum()
    covariance *= min(1., zero / raw_zero) / zero
    phi = np.exp(-rates)
    fast = phi <= 1e-15
    slow = ~fast
    if np.any(fast):
        count = int(slow.sum())
        combined = np.empty((count + 1, count + 1))
        combined[:count, :count] = covariance[np.ix_(slow, slow)]
        cross = covariance[np.ix_(slow, fast)].sum(axis=1)
        combined[:count, count] = cross
        combined[count, :count] = cross
        combined[count, count] = covariance[np.ix_(fast, fast)].sum()
        covariance, phi = combined, np.r_[phi[slow], 0.]
    # The exact diagonal retains sub-daily rough variance omitted by the
    # finite Laplace range. Nonzero lag error is tested independently below.
    remainder = max(0., 1 - covariance.sum())
    enlarged = np.zeros((phi.size + 1, phi.size + 1))
    enlarged[:-1, :-1] = covariance
    enlarged[-1, -1] = remainder
    return np.r_[phi, 0.], enlarged


@lru_cache(maxsize=4)
def accuracy_grid(tolerance=.001):
    lags = np.unique(np.r_[np.arange(64), np.geomspace(64, 25200, 256).astype(int)])
    for bins in (8, 12, 16, 24, 32, 48, 64):
        error = 0.
        for h in np.linspace(.03, .49, 17):
            for k in (1 / 2520, 1 / 63, .5):
                phi, covariance = lifted_covariance(h, k, bins)
                approximate = (phi[None, :] ** lags[:, None]) @ covariance.sum(axis=1)
                expected = exact_covariance(h, k, lags)
                error = max(error, float(np.max(np.abs(approximate - expected))))
        if error <= tolerance:
            return bins, {'tolerance': tolerance, 'maximum_absolute_autocorrelation_error': error,
                          'factors': len(phi), 'log_rate_bins': bins,
                          'diagonal_variance': 'analytic_exact_with_unresolved_white_remainder'}
    raise ValueError('tempered fractional covariance did not meet error budget')


def configuration(theta, adaptive):
    import models as shell
    if not adaptive:
        return shell.controls.rough_parameters(theta)
    bins, _ = accuracy_grid()
    phi, stationary = lifted_covariance(theta[0], math.exp(theta[1]), bins)
    covariance = math.exp(2 * theta[2]) * stationary * (1 - phi[:, None] * phi[None, :])
    return phi, np.ones(phi.size), covariance


def log_prior(theta):
    if not (.03 < theta[0] < .49 and math.log(1 / 2520) < theta[1] < math.log(.5)
            and math.log(.05) < theta[2] < math.log(3)):
        return -np.inf
    return (-.5 * ((theta[1] - math.log(1 / 63)) / 2) ** 2
            - .5 * ((theta[2] - math.log(.7)) / 1.5) ** 2)


@njit(cache=True, nogil=True)
def filter_rough(y, phi, weights, covariance, level):
    """Collapsed Gaussian filter without allocations inside the history loop."""
    n = phi.size
    state = np.zeros(n)
    posterior = covariance / (1 - phi[:, None] * phi[None, :])
    predicted = np.empty((n, n))
    pw, gain = np.empty(n), np.empty(n)
    fixed = False
    loglik = 0.
    for t in range(y.size):
        if not fixed:
            for i in range(n):
                total = 0.
                for j in range(n):
                    predicted[i, j] = posterior[i, j] * phi[i] * phi[j] + covariance[i, j]
                    total += predicted[i, j] * weights[j]
                pw[i] = total
            variance = 4.934802200544679 + np.dot(weights, pw)
            gain[:] = pw / variance
            log_variance = math.log(2 * math.pi * variance)
        prediction = 0.
        for i in range(n):
            state[i] *= phi[i]
            prediction += weights[i] * state[i]
        innovation = y[t] - level - prediction
        for i in range(n):
            state[i] += gain[i] * innovation
        if not fixed:
            fixed = True
            for i in range(n):
                for j in range(n):
                    value = predicted[i, j] - pw[i] * pw[j] / variance
                    if value != posterior[i, j]:
                        fixed = False
                    posterior[i, j] = value
        loglik -= .5 * (log_variance + innovation ** 2 / variance)
    return loglik, state, posterior


def parameter_chain(y, level, adaptive, seed, burn, kept, resume=None):
    rng = np.random.default_rng(seed)
    theta = np.array([rng.uniform(.08, .15), math.log(1 / 63), math.log(.7)])
    root = np.diag([.01, .1, .05])
    if resume:
        theta, root = resume['theta'].copy(), resume['root'].copy()
        rng.bit_generator.state = resume['rng_state']
    def target(point):
        prior = log_prior(point)
        if not np.isfinite(prior):
            return -np.inf
        phi, weights, covariance = configuration(point, adaptive)
        return prior + filter_rough(y, phi, weights, covariance, level)[0]
    value = target(theta)
    mean, m2 = np.zeros(3), np.zeros((3, 3))
    draws = np.empty((kept, 4))
    for iteration in range(burn + kept):
        proposed = theta + root @ rng.normal(size=3)
        likelihood = target(proposed)
        if math.log(rng.random()) < likelihood - value:
            theta, value = proposed, likelihood
        if iteration < burn:
            delta = theta - mean
            mean += delta / (iteration + 1)
            m2 += np.outer(delta, theta - mean)
            if iteration >= 63 and (iteration + 1) % 32 == 0:
                root = np.linalg.cholesky(2.38 ** 2 / 3 * (m2 / iteration + np.eye(3) * 1e-6))
        else:
            draws[iteration - burn] = np.r_[theta, value]
    return draws, {'theta': theta.copy(), 'root': root.copy(), 'rng_state': rng.bit_generator.state}


@lru_cache(maxsize=16)
def fit(data, adaptive, simulations, burn=2048, kept=8192, maximum=65536):
    import models as shell
    identity = hashlib.sha256(data).hexdigest()
    settings = f'overlay-v1:{adaptive}:{simulations}:{burn}:{kept}:{maximum}'
    source = hashlib.sha256(Path(__file__).read_bytes() + Path(shell.controls.__file__).read_bytes()).hexdigest()
    contract = source + ':' + settings
    key = hashlib.sha256((identity + contract).encode()).hexdigest()
    def build():
        started = time.perf_counter()
        x = np.frombuffer(data, np.float64)
        y, _ = shell.bd._sv_observed_log_variance(x, float(x.mean()))
        level = float(y.mean())
        chains = [parameter_chain(y, level, adaptive,
                  shell.bd.deterministic_seed('rough_overlay_fit', identity, settings, j), burn, kept)
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
                extra, state = parameter_chain(y, level, adaptive, 0, 0, extension, state)
                chains[j] = (np.concatenate([draws, extra]), state)
        all_draws = np.concatenate([c[0][:, :3] for c in chains])
        rng = np.random.default_rng(shell.bd.deterministic_seed('rough_overlay_joint_draws', identity, settings))
        selected = rng.integers(0, len(all_draws), size=simulations)
        seconds = time.perf_counter() - started
        return {'parameters': all_draws[selected], 'trace': trace, 'level': level,
                'diagnostics': diagnostics, 'convergence_flag': converged,
                'kept_per_chain': actual, 'fit_seconds': seconds,
                'ess_per_second': min(d['bulk_ess'] for d in diagnostics) / seconds,
                'data_sha256': identity, 'contract': contract,
                'kernel': accuracy_grid()[1] if adaptive else {'factors': 8, 'baseline_kernel_unchanged': True}}
    started = time.perf_counter()
    result = shell.controls.cache('bayesian_overlay_asset', key, build)
    shell.controls.timed('bayesian_overlay_fit', started)
    return result


@njit(cache=True, nogil=True)
def multiplier_path(phi, weights, covariance, initial, posterior, normals):
    state, mean_state = initial.copy(), np.zeros(phi.size)
    # Caller shifts the random initial state around its posterior mean.
    mean_state[:] = normals[0, :phi.size]
    result = np.empty(normals.shape[0] - 1)
    variance = posterior.copy()
    root = covariance
    q = root @ root.T
    for t in range(result.size):
        h = np.dot(weights, state)
        mean = np.dot(weights, mean_state)
        v = np.dot(weights, variance @ weights)
        result[t] = math.exp(.5 * (h - mean) - .25 * v)
        state = phi * state + root @ normals[t + 1]
        mean_state *= phi
        variance = variance * phi[:, None] * phi[None, :] + q
    return result


@dataclass(frozen=True)
class Candidate:
    model_id: str
    adaptive: bool = False
    burn: int = 2048
    kept: int = 8192
    max_kept: int = 65536

    def simulate_daily_log_returns(self, training, context):
        import models as shell
        training.validate()
        past, future = shell._validate_calendar(training, context)
        assets = np.asarray(training.asset_log_returns, np.float64)
        sims, horizon = context.simulations, context.horizon_days
        seed = shell.bd.deterministic_seed('copula_alternatives', shell.bd.FRONTIER_DEPENDENCE_ID,
                                          str(context.origin_date), horizon, sims)
        uniforms = shell.controls.gaussian_uniforms(assets.shape, assets.tobytes(), sims, horizon, seed).copy()
        for a in range(assets.shape[1]):
            data = assets[:, a].tobytes()
            posterior = fit(data, self.adaptive, sims, self.burn, self.kept, self.max_kept)
            shell.FIT_DIAGNOSTICS[(self.model_id, posterior['data_sha256'])] = {
                'model_id': self.model_id, 'data_sha256': posterior['data_sha256'],
                'convergence_flag': posterior['convergence_flag'], 'diagnostics': posterior['diagnostics'],
                'fit_seconds': posterior['fit_seconds'], 'ess_per_second': posterior['ess_per_second'],
                'kernel': posterior['kernel'], 'burn_per_chain': self.burn,
                'kept_per_chain': posterior['kept_per_chain']}
            started = time.perf_counter()
            production = shell.controls.asset_fit(data)
            mean, sd = shell.opt.mixture_curves(production, horizon)
            nodes = shell.controls.asset_nodes(data, sims, False)
            shell.controls.interpolate_nodes(uniforms[:, :, a], nodes)
            y, _ = shell.bd._sv_observed_log_variance(assets[:, a], float(assets[:, a].mean()))
            rng = np.random.default_rng(shell.bd.deterministic_seed('rough_overlay_prediction',
                  posterior['data_sha256'], str(context.origin_date), horizon, sims))
            for s, theta in enumerate(posterior['parameters']):
                phi, weights, covariance = configuration(theta, self.adaptive)
                _, state, p = filter_rough(y, phi, weights, covariance, posterior['level'])
                values, vectors = np.linalg.eigh((p + p.T) / 2)
                initial = state + (vectors * np.sqrt(np.maximum(values, 0))) @ rng.normal(size=phi.size)
                values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
                root = vectors * np.sqrt(np.maximum(values, 0))
                normals = np.empty((horizon + 1, phi.size))
                initial = phi * initial + root @ rng.normal(size=phi.size)
                p = p * phi[:, None] * phi[None, :] + covariance
                normals[0] = phi * state
                normals[1:] = rng.normal(size=(horizon, phi.size))
                uniforms[s, :, a] *= multiplier_path(phi, weights, root, initial, p, normals)
            uniforms[:, :, a] = np.clip(mean[None, :] + sd[None, :] * uniforms[:, :, a], -1, 1)
            shell.controls.timed('asset_predictive_paths', started)
        dates = shell._historical_rebalance_dates(past.append(future), training.policy.rebalance)
        mask = np.asarray([date in dates for date in future])
        return shell.rejoin(uniforms, training.policy.weights, mask)
