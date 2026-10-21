"""Research-only 3 lead criteria x 4 scales x 3 request-local learners."""
import hashlib
import itertools
import json
import time
import numpy as np
from candidates.asset_selector_endpoint_flat import candidate as control
from candidates.asset_selector_sequential_learning import candidate as sequential
from candidates.asset_selector_sequential_learning.learning import learn, pack_B, unpack_B
from candidates.asset_selector_fold_scale.canonical_scoring import calendar_log_weights

CRITERIA = ('endpoint', 'uniform_days', 'log_calendar')
SCALES = ('raw', 'train_SD', 'train_SD_sqrt_lead', 'sqrt_lead')
LEARNERS = ('batch', 'boa', 'progressive')
AXES = tuple(itertools.product(CRITERIA, SCALES))
BANK = control.BANK


def configurations(inner_paths=2, schedule='half_doubling'):
    return [{'criterion': c, 'scale': s, 'learner': l,
             'inner_paths': inner_paths, 'schedule': schedule}
            for c, s, l in itertools.product(CRITERIA, SCALES, LEARNERS)]


def key(criterion, scale, learner):
    return ':'.join((criterion, scale, learner))


def identity(config):
    return 'research_score_alignment_v1_'+hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def lead_masses(criterion, origin_date, dates):
    dates = np.asarray(dates, dtype='datetime64[D]')
    # Reuse canonical date validation even for the two non-calendar criteria.
    log_weights = calendar_log_weights(origin_date, dates)
    if criterion == 'log_calendar': return log_weights
    if criterion == 'uniform_days': return np.ones(len(dates))
    if criterion == 'endpoint':
        out = np.zeros(len(dates)); out[-1] = 1.; return out
    raise ValueError('Unknown lead criterion')


def denominator(scale, training_sd, native_lead):
    if scale not in SCALES or native_lead < 1:
        raise ValueError('Known scale and positive native lead required')
    if scale in ('train_SD', 'train_SD_sqrt_lead') and (not np.isfinite(training_sd) or training_sd <= 0):
        raise ArithmeticError('Positive finite fold-prefix portfolio SD(ddof=1) required')
    value = training_sd if scale in ('train_SD', 'train_SD_sqrt_lead') else 1.
    if scale in ('sqrt_lead', 'train_SD_sqrt_lead'): value *= np.sqrt(native_lead)
    if not np.isfinite(value) or value <= 0:
        raise ArithmeticError('Unrepresentable score denominator')
    return value


def replay(cache):
    """Fit all36 weight sets from compact per-fold streams; no paths or outcomes."""
    valid = cache['valid'].tolist(); count = int(cache['represented'].sum())
    k = len(valid); learned = {}; fits = {}; coefficients = {}; trajectories = {}
    for stream, (criterion, scale) in enumerate(AXES):
        A = cache['A'][stream]; B = unpack_B(cache['B_upper'][stream], k)
        total_A = np.zeros(k); total_B = np.zeros((k, k))
        for a, b, mass in zip(A, B, cache['represented']):
            total_A += mass*a; total_B += mass*b
        total_A /= count; total_B /= count
        fit = control.energy.fit(total_A, total_B); online = learn(A, B, cache['represented'])
        coefficients[(criterion, scale)] = {'A': total_A, 'B': total_B, 'valid': valid}
        fits[(criterion, scale)] = fit; trajectories[(criterion, scale)] = online
        for learner, values in [('batch', fit['weights']), ('boa', online['weights']),
                                 ('progressive', online['progressive_weights'])]:
            full = np.zeros(len(BANK)); full[valid] = values
            learned[key(criterion, scale, learner)] = full
    return learned, fits, coefficients, trajectories


def prepare(history, weights, horizon, training_dates, *, inner_paths,
            schedule_rule='half_doubling', seed_replication=0, width=64):
    started = time.perf_counter(); x = np.asarray(history, dtype=float)
    w = np.asarray(weights, dtype=float); dates = np.asarray(training_dates, dtype='datetime64[D]')
    if (x.ndim != 2 or len(x) != len(dates) or x.shape[1] != len(w)
            or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isclose(w.sum(), 1., rtol=0, atol=1e-14) or np.isnat(dates).any()
            or np.any(np.diff(dates) <= np.timedelta64(0, 'D')) or inner_paths < 2
            or inner_paths % 2 or width < 1 or not isinstance(seed_replication, int) or seed_replication < 0):
        raise ValueError('Finite dated joint history, normalized weights and even inner paths required')
    eligible, folds, full_count = control.schedule(len(x), horizon, schedule_rule)
    eligible = [i for i in eligible if i != 0]+([166, 167] if 0 in eligible else [])
    invalid = {}; trace = []; terms_A = []; terms_B = []; stable_valid = None
    for stop, h, represented in folds:
        past = x[:stop]
        prepared, failures = control.prefix_fits(past, [i for i in eligible if i not in invalid], w)
        for i, error in failures.items(): invalid[i] = {'training_rows': stop, **error}
        valid = [i for i in eligible if i not in invalid]
        if not valid or (stable_valid is not None and valid != stable_valid):
            raise ArithmeticError('All score streams require a stable nonempty recipe universe across dated folds')
        stable_valid = valid
        returns = control.portfolio_returns(x[stop:stop+h], w)
        observed = np.cumsum(returns)
        # Match the frozen endpoint sum reduction exactly, not cumsum rounding.
        observed[-1] = float(returns.sum())
        sd = float(np.std(control.portfolio_returns(past, w), ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ArithmeticError('Nonpositive fold-prefix portfolio SD; scale experiment fails closed')
        masses = {c: lead_masses(c, dates[stop-1], dates[stop:stop+h]) for c in CRITERIA}
        suffix = '' if seed_replication == 0 else f':replication={seed_replication}'
        seed = int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{control.signature(past)}:{h}:{inner_paths}{suffix}'.encode()).digest()[:4], 'little')
        A = np.zeros((len(AXES), len(valid))); B = np.zeros((len(AXES), len(valid), len(valid)))
        cursor = 0
        for start, cube in control.all_chunks(past, w, h, inner_paths, seed, valid, prepared, width=width):
            if start != cursor or cube.shape[:2] != (len(valid), inner_paths) or not np.isfinite(cube).all():
                raise ArithmeticError('Incomplete/nonfinite joint historical path cube')
            for offset in range(cube.shape[2]):
                lead = start+offset+1
                a, b = control.energy.coefficients(np.ascontiguousarray(cube[:, :, offset]), observed[lead-1])
                for stream, (criterion, scale) in enumerate(AXES):
                    mass = masses[criterion][lead-1]
                    if mass:
                        # Divide first: exact endpoint/sqrt_lead arithmetic control.
                        divisor = denominator(scale, sd, lead)
                        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
                            try:
                                A[stream] += mass*(a/divisor); B[stream] += mass*(b/divisor)
                            except FloatingPointError as error:
                                raise ArithmeticError('Unrepresentable scaled score terms; no clipping/fallback') from error
            cursor += cube.shape[2]
        if cursor != h: raise ArithmeticError('Missing native validation leads')
        if not np.isfinite(A).all() or not np.isfinite(B).all():
            raise ArithmeticError('Unrepresentable score coefficients; no exclusions')
        terms_A.append(A); terms_B.append(np.asarray([pack_B(b[None, :, :])[0] for b in B]))
        trace.append({'training_rows': stop, 'validation_horizon': h, 'validation_stop': stop+h,
                      'represented_origins': represented, 'training_sha256': control.signature(past),
                      'training_last_date': str(dates[stop-1]), 'validation_last_known_date': str(dates[stop+h-1]),
                      'seed': seed, 'eligible_this_fold': valid, 'PPW_dependence_fit': prepared.get('PPW'),
                      'training_portfolio_SD_ddof1': sd})
    cache = {'A': np.stack(terms_A, axis=1), 'B_upper': np.stack(terms_B, axis=1),
             'valid': np.asarray(stable_valid, dtype=np.int64),
             'represented': np.asarray([f[2] for f in folds], dtype=np.int64),
             'training_rows': np.asarray([f[0] for f in folds]),
             'validation_horizon': np.asarray([f[1] for f in folds]),
             'training_SD': np.asarray([t['training_portfolio_SD_ddof1'] for t in trace])}
    if int(cache['represented'].sum()) != full_count: raise ArithmeticError('Origin mass differs')
    learned, fits, coefficients, online = replay(cache)
    needed = sorted({i for values in learned.values() for i in np.flatnonzero(values > 0)})
    full, failures = control.prefix_fits(x, needed, w)
    if failures: raise ArithmeticError(f'Selected full-history fit failed: {failures}')
    return {**full, 'history_sha256': control.signature(x),
            'training_dates_sha256': hashlib.sha256(dates.tobytes()).hexdigest(),
            'weights': w.copy(), 'horizon': horizon, 'inner_paths': inner_paths,
            'schedule_rule': schedule_rule, 'seed_replication': seed_replication,
            'weights_by_mode': learned, 'fit': fits, 'coefficients': coefficients,
            'sequential': online, 'fold_coefficients': cache, 'trace': trace,
            'invalid_internal_options': invalid,
            'diagnostics': {'selection_and_refit_seconds': time.perf_counter()-started,
                            'inner_origins': len(folds), 'available_internal_origins': full_count,
                            'no_outer_future_dates_used': True, 'automatic_model_fallback': False,
                            'maximum_origin_count': None, 'score_variants': len(learned),
                            'cache_bytes': sum(a.nbytes for a in cache.values()),
                            'cache_scope': 'one exact request; no hidden persistent cache'}}


class ScorePool:
    def __init__(self, config, training_dates, bundle=None):
        if config not in configurations(config.get('inner_paths', 0), config.get('schedule', '')):
            raise ValueError('Known criterion/scale/learner configuration required')
        if config['schedule'] not in control.RULES or config['inner_paths'] < 2 or config['inner_paths'] % 2:
            raise ValueError('Known schedule and even inner paths required')
        self.config = dict(config); self.dates = np.asarray(training_dates, dtype='datetime64[D]')
        self.bundle = bundle; self.model_id = identity(config)

    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        cfg = self.config
        bundle = self.bundle if self.bundle is not None else prepare(history, weights, horizon, self.dates,
            inner_paths=cfg['inner_paths'], schedule_rule=cfg['schedule'])
        # Reuse frozen path accounting and all its dated request-binding checks.
        adapter = object.__new__(sequential.MenuPool)
        adapter.config = {'menu': key(cfg['criterion'], cfg['scale'], cfg['learner']),
                          'inner_paths': cfg['inner_paths'], 'schedule': cfg['schedule']}
        adapter.dates = self.dates; adapter.bundle = bundle
        yield from adapter.terminal_chunks(history, weights, horizon, paths, context_seed, width=width)
        self.last_selection = {**adapter.last_selection, 'score_config': cfg}


def save_cache(bundle, path):
    """Lossless caller-owned single-request cache; paths are never retained."""
    metadata = {k: bundle[k] for k in ('history_sha256', 'training_dates_sha256',
                'horizon', 'inner_paths', 'schedule_rule', 'seed_replication', 'trace')}
    metadata.update(weights=bundle['weights'].tolist(), schema='score_alignment_v1',
                    axes=AXES, recipe_bank_sha256=hashlib.sha256(json.dumps(BANK, sort_keys=True).encode()).hexdigest())
    np.savez_compressed(path, **bundle['fold_coefficients'], metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_cache(path, history, weights, horizon, training_dates, *, inner_paths,
               schedule_rule='half_doubling', seed_replication=0):
    cache, metadata = sequential.load_fold_coefficients(path, history, weights, horizon,
        training_dates, inner_paths=inner_paths, schedule_rule=schedule_rule,
        seed_replication=seed_replication)
    if metadata.get('schema') != 'score_alignment_v1' or metadata.get('axes') != [list(a) for a in AXES]:
        raise ValueError('Score-alignment cache schema or axes differ')
    return cache, metadata
