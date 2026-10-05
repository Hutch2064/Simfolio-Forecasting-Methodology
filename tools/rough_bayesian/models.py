"""Research-only asset volatility replacements in the unchanged public shell."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import ndtri, stdtrit
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'tools/mcmc_runtime')]
spec = importlib.util.spec_from_file_location('rough_controls', ROOT / 'tools/rough_jump_vine/models.py')
controls = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controls
spec.loader.exec_module(controls)

import optimized_mcmc as opt
from inference import chain, configuration, heston_path
from kernels import grid
from streamed_paths import rejoin
from simfolio_forecasting_methodology.models.asset_level.frontier import (
    _historical_rebalance_dates, _validate_calendar,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

TIMINGS = controls.TIMINGS
FIT_DIAGNOSTICS = {}
FIT_SOURCE_SHA = hashlib.sha256(b''.join((HERE / name).read_bytes()
    for name in ('kernels.py', 'inference.py', 'models.py'))).hexdigest()
MODEL_IDS = (
    controls.MODEL_IDS[0], controls.MODEL_IDS[1],
    'asset_bayesian_lifted_rfsv_ess',
    'asset_bayesian_lifted_rfsv_tight_ess',
    'asset_bayesian_lifted_rfsv_ess_terminal_pgas',
    'asset_bayesian_lifted_rfsv_leverage_ess',
    'asset_bayesian_lifted_rough_heston_hmc',
    'asset_bayesian_lifted_rfsv_student_ess',
)


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    controls.initialize(cache_root, vine_threads, state_workers)
    grid(.01)
    grid(.001)


def clear_path_cache():
    controls.clear_path_cache()


def diagnostics(traces):
    """Rank-normalized split R-hat and initial-positive-sequence ESS.

    Both rank and folded rank diagnostics are retained. Fixed-budget research
    fits may be flagged; a flagged chain is never described as converged.
    """
    half = traces.shape[1] // 2
    split = np.concatenate([traces[:, :half], traces[:, -half:]], axis=0)
    output = []
    for j in range(split.shape[2]):
        raw = split[:, :, j]
        transformed = []
        for values in (raw, np.abs(raw - np.median(raw))):
            ranks = rankdata(values.ravel()).reshape(values.shape)
            transformed.append(ndtri((ranks - .375) / (ranks.size + .25)))
        rhats = []
        for values in transformed:
            w = float(np.mean(np.var(values, axis=1, ddof=1)))
            b = half * float(np.var(values.mean(axis=1), ddof=1))
            variance = (half - 1) / half * w + b / half
            rhats.append(math.sqrt(variance / w) if w > 0 else float('inf'))
        values = transformed[0]
        centered = values - values.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(centered, n=2 * half, axis=1)
        ac = np.fft.irfft(spectrum * spectrum.conjugate(), axis=1)[:, :half]
        ac /= np.arange(half, 0, -1)[None, :]
        w = float(np.mean(ac[:, 0]))
        b = half * float(np.var(values.mean(axis=1), ddof=1))
        variance = (half - 1) / half * w + b / half
        correlation = 1 - (w - ac.mean(axis=0)) / max(variance, 1e-30)
        pairs = correlation[1:-1:2] + correlation[2::2]
        positive = []
        for pair in pairs:
            if pair <= 0:
                break
            positive.append(min(pair, positive[-1]) if positive else pair)
        ess = min(raw.size, raw.size / max(1., 1 + 2 * sum(positive)))
        output.append({'rank_split_rhat': max(rhats), 'bulk_ess': ess})
    return output


@lru_cache(maxsize=16)
def fit(data, kind, tolerance, leverage, pgas, burn=2048, kept=8192, max_kept=65536, simulations=240):
    identity = hashlib.sha256(data).hexdigest()
    sampling_contract = f'v2:{kind}:{tolerance}:{leverage}:{pgas}:{burn}:{kept}:{max_kept}:{simulations}'
    contract = f'{FIT_SOURCE_SHA}:{sampling_contract}'
    key = hashlib.sha256((identity + contract).encode()).hexdigest()
    def build():
        started = time.perf_counter()
        x = np.frombuffer(data, np.float64)
        y = (x - float(x.mean())) * 100
        chains = [chain(y, bd.deterministic_seed('rough_bayesian_fit', identity, sampling_contract, j),
                        tolerance, kind, leverage, pgas, burn, kept) for j in range(2)]
        # rho is constant for non-leverage models, and is not a sampled
        # parameter. Exclude it from convergence diagnostics in that case.
        columns = [0, 1, 2, 3] + ([4] if leverage else []) + [5, 6]
        if kind == 'fou_t':
            columns.append(7)
        while True:
            trace = np.array([c['diagnostic_trace'] for c in chains])
            diagnostic = diagnostics(trace[:, :, columns])
            converged = all(d['rank_split_rhat'] < 1.05 and d['bulk_ess'] >= 100 for d in diagnostic)
            actual_kept = trace.shape[1]
            if converged or actual_kept >= max_kept:
                break
            extension = min(actual_kept, max_kept - actual_kept)
            for j, current in enumerate(chains):
                extra = chain(y, 0, tolerance, kind, leverage, pgas, 0, extension, resume=current['state'])
                for name in ('parameters', 'terminal', 'diagnostic_trace'):
                    current[name] = np.concatenate([current[name], extra[name]])
                current['parameter_acceptance'] = (
                    current['parameter_acceptance'] * (burn + actual_kept)
                    + extra['parameter_acceptance'] * extension) / (burn + actual_kept + extension)
                current['likelihood_evaluations'] += extra['likelihood_evaluations']
                current['state'] = extra['state']
        parameters = np.concatenate([c['parameters'] for c in chains])
        terminal = np.concatenate([c['terminal'] for c in chains])
        # Select the actual joint draw for every predictive path from ALL
        # retained draws. Storing those selected states is an exact cache
        # compression for the configured ensemble, not posterior node fitting.
        rng = np.random.default_rng(bd.deterministic_seed('rough_joint_draw_selection', identity, sampling_contract))
        indices = rng.integers(0, parameters.shape[0], size=simulations)
        predictive_parameters, predictive_terminal = parameters[indices], terminal[indices]
        # Keep all posterior draws. Prediction samples a draw per path, rather
        # than substituting posterior means or sixteen parameter nodes.
        seconds = time.perf_counter() - started
        return {'parameters': predictive_parameters, 'terminal': predictive_terminal, 'trace': trace,
                'diagnostics': diagnostic,
                'diagnostic_columns': columns,
                'convergence_flag': converged,
                'fit_seconds': seconds, 'ess_per_second': min(d['bulk_ess'] for d in diagnostic) / seconds,
                'parameter_acceptance': [c['parameter_acceptance'] for c in chains],
                'likelihood_evaluations': [c['likelihood_evaluations'] for c in chains],
                'kernel': chains[0]['kernel'], 'burn_per_chain': burn, 'kept_per_chain': actual_kept,
                'predictive_draw_indices': indices, 'total_retained_draws': 2 * actual_kept,
                'data_sha256': identity, 'contract': contract}
    started = time.perf_counter()
    result = controls.cache('bayesian_rough_asset', key, build)
    controls.timed('bayesian_asset_fit', started)
    return result


@njit(cache=True, nogil=True)
def predictive_fou(phi, innovation, weights, terminal, noise, return_shock, mean, level, scale):
    result = np.empty(noise.size)
    state = terminal.copy()
    for t in range(noise.size):
        h = level + scale * np.dot(weights, state)
        result[t] = min(1., max(-1., mean[t] + math.exp(h / 2) / 100 * return_shock[t]))
        for j in range(state.size):
            state[j] = phi[j] * state[j] + innovation[j] * noise[t]
    return result


@njit(cache=True, nogil=True)
def predictive_heston(phi, weights, step, terminal, noise, return_shock, mean, level, kappa, eta):
    logv, _ = heston_path(phi, weights, step, terminal, noise, level, kappa, eta)
    result = np.empty(noise.size)
    for t in range(noise.size):
        result[t] = min(1., max(-1., mean[t] + math.exp(logv[t] / 2) / 100 * return_shock[t]))
    return result


@dataclass(frozen=True)
class Candidate:
    model_id: str
    kind: str = 'fou'
    tolerance: float = .01
    leverage: bool = False
    pgas: bool = False
    burn: int = 2048
    kept: int = 8192
    max_kept: int = 65536

    def simulate_daily_log_returns(self, training, context):
        training.validate()
        past, future = _validate_calendar(training, context)
        assets = np.asarray(training.asset_log_returns, np.float64)
        sims, horizon = context.simulations, context.horizon_days
        seed = bd.deterministic_seed('copula_alternatives', bd.FRONTIER_DEPENDENCE_ID,
                                     str(context.origin_date), horizon, sims)
        started = time.perf_counter()
        uniforms = controls.gaussian_uniforms(assets.shape, assets.tobytes(), sims, horizon, seed)
        shocks = ndtri(np.clip(uniforms, 1e-12, 1 - 1e-12))
        paths = np.empty_like(shocks)
        controls.timed('dependence_paths', started)
        for a in range(assets.shape[1]):
            data = assets[:, a].tobytes()
            posterior = fit(data, self.kind, self.tolerance, self.leverage, self.pgas, self.burn, self.kept,
                            self.max_kept, sims)
            FIT_DIAGNOSTICS[(self.model_id, posterior['data_sha256'])] = {
                'model_id': self.model_id, 'data_sha256': posterior['data_sha256'],
                'convergence_flag': posterior['convergence_flag'],
                'diagnostics': posterior['diagnostics'], 'diagnostic_columns': posterior['diagnostic_columns'],
                'fit_seconds': posterior['fit_seconds'], 'ess_per_second': posterior['ess_per_second'],
                'kernel': posterior['kernel'], 'burn_per_chain': self.burn,
                'kept_per_chain': posterior['kept_per_chain']}
            started = time.perf_counter()
            production = controls.asset_fit(data)
            mean, _ = opt.mixture_curves(production, horizon)
            # The predictive random stream is shared across numerical/sampler
            # arms, independently of the draw count and kernel factor count.
            rng = np.random.default_rng(bd.deterministic_seed('rough_bayesian_predictive',
                hashlib.sha256(data).hexdigest(), str(context.origin_date), horizon, sims))
            private_noise = rng.normal(size=(sims, horizon))
            for s, index in enumerate(range(sims)):
                theta = posterior['parameters'][index]
                config, _ = configuration(theta, self.tolerance, self.kind, with_root=False)
                rho = math.tanh(theta[4]) if self.leverage else 0.
                noise = rho * shocks[s, :, a] + math.sqrt(1 - rho * rho) * private_noise[s]
                state = posterior['terminal'][index]
                return_shock = shocks[s, :, a]
                if self.kind == 'fou_t':
                    nu = 2 + math.exp(theta[5])
                    return_shock = stdtrit(nu, np.clip(uniforms[s, :, a], 1e-12, 1 - 1e-12))
                    return_shock *= math.sqrt((nu - 2) / nu)
                if self.kind.startswith('fou'):
                    paths[s, :, a] = predictive_fou(*config[:3], state, noise, return_shock, mean,
                                                   *config[4:])
                else:
                    paths[s, :, a] = predictive_heston(*config[:3], state, noise, shocks[s, :, a], mean,
                                                      *config[4:])
            controls.timed('asset_predictive_paths', started)
        started = time.perf_counter()
        dates = _historical_rebalance_dates(past.append(future), training.policy.rebalance)
        mask = np.asarray([date in dates for date in future])
        result = rejoin(paths, training.policy.weights, mask)
        controls.timed('portfolio_rejoin', started)
        return result


CANDIDATES = (
    controls.Candidate(MODEL_IDS[0]), controls.Candidate(MODEL_IDS[1], rough=True),
    Candidate(MODEL_IDS[2]), Candidate(MODEL_IDS[3], tolerance=.001),
    Candidate(MODEL_IDS[4], pgas=True), Candidate(MODEL_IDS[5], leverage=True),
    Candidate(MODEL_IDS[6], kind='heston', leverage=True),
    Candidate(MODEL_IDS[7], kind='fou_t'),
)
