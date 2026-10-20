"""Bounded-memory execution of the four retained portfolio forecast laws.

All paths enter every empirical CRPS calculation. Only horizon columns are
batched. WR/cycle block draws retain the original path-major RNG order; FHS
innovations and Adaptive restart draws use declared, chunk-invariant time-major
streams. This is a new numerical execution version, never a relabelled240 run.
"""
from array import array
import math
from pathlib import Path
import sys
import hashlib
import numpy as np
from numba import njit

OWN = Path(__file__).resolve().parent
CAMPAIGN = next(p for p in OWN.parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = CAMPAIGN.parent/'french551-full-20261019/code'
for path in (CAMPAIGN, CODE/'src', CODE/'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from expanded_public_smooth_mean_live import module as incumbent
from simfolio_forecasting_methodology.resources.french1926.control_sources.engine_helpers import _politis_white_block_length

ADAPTIVE = 'model_testing_joint_stationary_bootstrap_v1'
REPLACEMENT = 'model_testing_joint_uniform_bounded_block_bootstrap_v1'
CYCLES = 'model_testing_joint_uniform_bounded_cycle_bootstrap_v1'
BEST = incumbent.MODEL_ID
PATHS = 1000
WIDTH = 64
NAMESPACE = 'one_thousand_streaming_horizon_columns_v5'


def block_descriptors(n, horizon, paths, seed, *, cycles=False):
    """Lossless RLE of the original bounded block-index generator's output."""
    if min(n, horizon, paths) < 1:
        raise ValueError('Positive history, horizon and path counts required')
    rng = np.random.default_rng(seed)
    starts, lengths, offsets = array('i'), array('i'), [0]
    maximum = min(n, horizon)
    for _ in range(paths):
        position, remaining, runs = 0, n, [(0, n)]
        while position < horizon:
            if not cycles:
                length = int(rng.integers(1, maximum+1))
                start = int(rng.integers(0, n))
            else:
                if remaining == 0:
                    remaining, runs = n, [(0, n)]
                cap = min(maximum, max(count for _, count in runs))
                length = int(rng.integers(1, cap+1))
                if remaining == n:
                    start = int(rng.integers(0, n))
                    selected, offset, run_start, run_length = 0, 0, start, n
                else:
                    segments = []
                    for index, (run_start, run_length) in enumerate(runs):
                        count = run_length-length+1
                        if count < 1:
                            continue
                        first = min(count, n-run_start)
                        segments.append((run_start, first, index))
                        if first < count:
                            segments.append((0, count-first, index))
                    segments.sort()
                    rank = int(rng.integers(0, sum(count for _, count, _ in segments)))
                    for segment_start, count, selected in segments:
                        if rank < count:
                            start = segment_start+rank
                            break
                        rank -= count
                    run_start, run_length = runs[selected]
                    offset = (start-run_start) % n
            copied = min(length, horizon-position)
            starts.append(start)
            lengths.append(copied)
            if cycles:
                runs.pop(selected)
                if offset:
                    runs.append((run_start, offset))
                after = run_length-offset-copied
                if after:
                    runs.append(((start+copied) % n, after))
                remaining -= copied
            position += copied
        offsets.append(len(starts))
    return (np.asarray(starts, dtype=np.int32), np.asarray(lengths, dtype=np.int32),
            np.asarray(offsets, dtype=np.int64))


@njit(cache=True, fastmath=False)
def emit_blocks(starts, lengths, pointer, within, n, width):
    out = np.empty((len(pointer), width), dtype=np.int64)
    for p in range(len(pointer)):
        for h in range(width):
            out[p, h] = (starts[pointer[p]]+within[p]) % n
            within[p] += 1
            if within[p] == lengths[pointer[p]]:
                pointer[p] += 1
                within[p] = 0
    return out


class BlockStream:
    def __init__(self, n, horizon, paths, seed, *, cycles=False):
        self.n = n
        self.starts, self.lengths, self.offsets = block_descriptors(n, horizon, paths, seed, cycles=cycles)
        self.pointer = self.offsets[:-1].copy()
        self.within = np.zeros(paths, dtype=np.int64)

    def take(self, width):
        return emit_blocks(self.starts, self.lengths, self.pointer, self.within, self.n, width)


class AdaptiveStream:
    def __init__(self, history, horizon, paths, seed):
        self.n, self.position = len(history), 0
        self.block = min(self.n, max(_politis_white_block_length(history),
            _politis_white_block_length((history-history.mean())**2)))
        prior = incumbent.retained.parent.incumbent.prior
        self.last = np.random.default_rng(seed).integers(0, self.n, size=paths, dtype=np.int32)
        self.flags = np.random.default_rng(prior.deterministic_seed(NAMESPACE+'_adaptive_flags', int(seed)))
        self.points = np.random.default_rng(prior.deterministic_seed(NAMESPACE+'_adaptive_points', int(seed)))

    def take(self, width):
        initial = self.position == 0
        out = np.empty((len(self.last), width), dtype=np.int64)
        if initial:
            out[:, 0] = self.last
        count = width-int(initial)
        flags = self.flags.random((count, len(self.last))) < 1./self.block
        points = self.points.integers(0, self.n, size=(count, len(self.last)), dtype=np.int32)
        for h in range(count):
            self.last = np.where(flags[h], points[h], (self.last+1) % self.n)
            out[:, h+int(initial)] = self.last
        self.position += width
        return out


class FhsStream:
    def __init__(self, n, paths, seed):
        self.n, self.paths = n, paths
        self.rng = np.random.default_rng(seed)

    def take(self, width):
        return self.rng.integers(0, self.n, size=(width, self.paths)).T.copy()


@njit(cache=True, fastmath=False)
def native_block(history, source, indices, delta, wealth, previous, cumulative):
    out = np.empty(indices.shape, dtype=np.float64)
    clips = floors = 0
    for p in range(len(wealth)):
        for h in range(indices.shape[1]):
            index = indices[p, h]
            if delta[p] == 0.:
                simple = source[index]
            else:
                raw = math.expm1(history[index]+delta[p])
                simple = min(max(raw, -.999999), np.finfo(np.float64).max/10)
                if raw != simple:
                    clips += 1
            wealth[p] *= 1.+simple
            if wealth[p] < 1e-300:
                floors += 1
            current = math.log(max(wealth[p], 1e-300))
            cumulative[p] += current-previous[p]
            previous[p] = current
            out[p, h] = cumulative[p]
    return out, clips, floors


@njit(cache=True, fastmath=False)
def fhs_block(pool, indices, mean, target, alpha, beta, variance, wealth, previous, cumulative):
    out = np.empty(indices.shape, dtype=np.float64)
    clips = floors = 0
    omega = 1.-alpha-beta
    upper = np.finfo(np.float64).max/10.
    max_log = np.log1p(upper)
    for p in range(len(wealth)):
        for h in range(indices.shape[1]):
            normalized = np.sqrt(variance[p])*pool[indices[p, h]]
            raw = mean[p]+np.sqrt(target)*normalized
            if raw > max_log:
                simple = upper
                clips += 1
            else:
                simple = np.expm1(raw)
                if simple < -.999999:
                    simple = -.999999
                    clips += 1
            wealth[p] *= 1.+simple
            if wealth[p] < 1e-300:
                floors += 1
            current = np.log(max(wealth[p], 1e-300))
            cumulative[p] += current-previous[p]
            previous[p] = current
            out[p, h] = cumulative[p]
            variance[p] = omega+alpha*normalized*normalized+beta*variance[p]
    return out, clips, floors


def terminal_chunks(model_id, raw_history, horizon, paths, seed, *, width=WIDTH):
    """Training-only generator. Held-out observations are never provided here."""
    if model_id not in (ADAPTIVE, REPLACEMENT, CYCLES, BEST) or paths < 1 or horizon < 1 or width < 1:
        raise ValueError('Unknown retained model or invalid execution dimensions')
    simple = np.expm1(np.asarray(raw_history, dtype=np.float64))
    if not np.isfinite(simple).all():
        raise ValueError('Nonfinite prepared training history')
    history = np.log1p(np.clip(simple, -.999999, np.finfo(float).max/10))
    source = np.clip(np.expm1(history), -.999999, np.finfo(float).max/10)
    if model_id == BEST:
        if paths % 4:
            raise ValueError('Equal pool requires even component ensembles')
        prior = incumbent.retained.parent.incumbent.prior
        scale, df, _, _ = incumbent.retained.parent.incumbent.base.batch_mean_scale(history)
        se = incumbent.retained.gaussian_sd(df, scale)
        center = incumbent.regularized_mean(float(history.mean()), se)
        seeds = [prior.deterministic_seed(ns, int(seed)) for ns in incumbent.retained.parent.NAMESPACES]
        deltas = [incumbent.retained.parameter_draws(df, scale, paths//2,
            prior.deterministic_seed(incumbent.retained.parent.incumbent.base.NAMESPACE, int(s)))
            +(center-float(history.mean())) for s in seeds]
        fit = incumbent.retained.FHS.base.fit_variance_target(history)
        streams = [BlockStream(len(history), horizon, paths//2, seeds[0]), FhsStream(len(history), paths//2, seeds[1])]
        variance = np.full(paths//2, fit['next_variance_normalized'])
    else:
        streams = [AdaptiveStream(history, horizon, paths, seed) if model_id == ADAPTIVE else
                   BlockStream(len(history), horizon, paths, seed, cycles=model_id == CYCLES)]
        deltas = [np.zeros(paths)]
    size = len(deltas[0])
    states = [(np.ones(size), np.zeros(size), np.zeros(size)) for _ in streams]
    for start in range(0, horizon, width):
        count = min(width, horizon-start)
        outputs = []
        for i, stream in enumerate(streams):
            indices = stream.take(count)
            if model_id == BEST and i == 1:
                terminal, _, _ = fhs_block(fit['pool'], indices, fit['mean']+deltas[i],
                    fit['target_variance'], fit['alpha'], fit['beta'], variance, *states[i])
            else:
                terminal, _, _ = native_block(history, source, indices, deltas[i], *states[i])
            if not np.isfinite(terminal).all():
                raise ValueError('Nonfinite native terminal forecast')
            outputs.append(terminal)
        yield start, np.concatenate(outputs, axis=0) if len(outputs) == 2 else outputs[0]


def source_identity():
    return {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'paths': PATHS, 'horizon_chunk': WIDTH, 'rng_version': NAMESPACE,
        'mean_variance_volatility_fits_unchanged': True,
        'WR_and_cycles_original_draw_order': True,
        'FHS_and_adaptive_time_major_streams': True,
        'empirical_law_all_paths_no_subsampling': True}
