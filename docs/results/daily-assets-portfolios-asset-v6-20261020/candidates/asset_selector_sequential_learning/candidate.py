"""Frozen flat-bank control and request-local sequential CDF-mixture learners."""
import hashlib, json, time
import numpy as np
from candidates.asset_selector_universe_certified import energy
from candidates.asset_selector_universe_precision.candidate import signature, portfolio_returns
from candidates.asset_selector_information_origins.validation import schedule, RULES
from candidates.asset_selector_menu_components.bank import BANK, prefix_fits, all_chunks, model
MENUS={'batch':list(range(1,168))}
MODES=('batch','boa','progressive')
from .learning import learn, pack_B, unpack_B


def identity(config):
    return 'research_asset_selector_sequential_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]


def prepare(history, weights, horizon, training_dates, *, inner_paths, schedule_rule, seed_replication=0, width=64):
    started = time.perf_counter(); x = np.asarray(history, dtype=float); w = np.asarray(weights, dtype=float); dates = np.asarray(training_dates, dtype='datetime64[D]')
    if (x.ndim != 2 or len(x) != len(dates) or x.shape[1] != len(w) or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-14) or np.isnat(dates).any()
            or np.any(np.diff(dates) <= np.timedelta64(0, 'D')) or inner_paths < 2 or inner_paths % 2
            or not isinstance(seed_replication, int) or seed_replication < 0 or width < 1):
        raise ValueError('Finite dated joint history, unit weights and even simulations required')
    eligible, folds, full_count = schedule(len(x), horizon, schedule_rule)
    eligible = [i for i in eligible if i!=0]+([166,167]if 0 in eligible else[])
    fold_terms = []; invalid = {}; trace = []; total_A = np.zeros(len(BANK)); total_B = np.zeros((len(BANK), len(BANK)))
    for stop, h, represented in folds:
        past = x[:stop]; prepared, failures = prefix_fits(past, [i for i in eligible if i not in invalid], w)
        for i, error in failures.items():
            invalid[i] = {'training_rows': stop, **error}
        valid = [i for i in eligible if i not in invalid]
        actual = float(portfolio_returns(x[stop:stop+h], w).sum())
        suffix = '' if seed_replication == 0 else f':replication={seed_replication}'
        seed = int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(past)}:{h}:{inner_paths}{suffix}'.encode()).digest()[:4], 'little')
        last=None;cursor=0
        for start,cube in all_chunks(past,w,h,inner_paths,seed,valid,prepared,width=width):
            if start!=cursor:raise ArithmeticError('Incomplete historical path cursor')
            cursor+=cube.shape[2];last=np.ascontiguousarray(cube[:,:,-1])
        if cursor!=h or last is None:raise ArithmeticError('Missing historical endpoint')
        A,B=energy.coefficients(last,actual);A/=np.sqrt(h);B/=np.sqrt(h)
        total_A[valid]+=represented*A;total_B[np.ix_(valid,valid)]+=represented*B
        fold_terms.append((valid.copy(), A.copy(), B.copy(), represented))
        trace.append({'training_rows': stop, 'validation_horizon': h, 'validation_stop': stop+h, 'represented_origins':represented,
                      'training_sha256': signature(past), 'training_last_date': str(dates[stop-1]),
                      'validation_last_known_date': str(dates[stop+h-1]), 'seed': seed,
                      'eligible_this_fold': valid, 'PPW_dependence_fit': prepared.get('PPW')})
    learned = {}; fitted = {}; coefficients = {}
    for name, ids in MENUS.items():
        valid = [i for i in eligible if i in ids and i not in invalid]
        if not valid:
            raise ArithmeticError('No valid historical option; no fixed-model fallback')
        A = total_A[valid]/full_count; B = total_B[np.ix_(valid, valid)]/full_count
        fitted[name] = energy.fit(A, B); learned[name] = np.zeros(len(BANK)); learned[name][valid] = fitted[name]['weights']
        coefficients[name] = {'A': A, 'B': B, 'valid': valid}
    # Fixed expert identities are required throughout the sequential experiment.
    # Reject changing validity rather than retroactively use future fold failures.
    valid = coefficients['batch']['valid']
    if any(ids != valid for ids, a, b, m in fold_terms):
        raise ArithmeticError('Sequential comparison requires identical valid recipes on every dated fold')
    fold_A = np.asarray([a for ids, a, b, m in fold_terms])
    fold_B = np.asarray([b for ids, a, b, m in fold_terms])
    representation = np.asarray([m for ids, a, b, m in fold_terms], dtype=np.int64)
    sequential = learn(fold_A, fold_B, representation)
    for name, values in [('boa', sequential['weights']), ('progressive', sequential['progressive_weights'])]:
        learned[name] = np.zeros(len(BANK)); learned[name][valid] = values
    compact_folds = {'A': fold_A, 'B_upper': pack_B(fold_B), 'valid': np.asarray(valid),
                     'represented': representation,
                     'training_rows': np.asarray([t['training_rows'] for t in trace]),
                     'validation_stop': np.asarray([t['validation_stop'] for t in trace]),
                     'seed': np.asarray([t['seed'] for t in trace], dtype=np.uint32)}
    needed = sorted({i for values in learned.values() for i in np.flatnonzero(values > 0)})
    full, failures = prefix_fits(x, needed, w)
    if failures:
        raise ArithmeticError(f'Selected full-history fit failed: {failures}')
    return {**full, 'history_sha256': signature(x), 'training_dates_sha256': hashlib.sha256(dates.tobytes()).hexdigest(),
            'weights': w.copy(), 'horizon': horizon, 'inner_paths': inner_paths, 'schedule_rule':schedule_rule, 'seed_replication':seed_replication, 'weights_by_mode': learned,
            'fit': fitted, 'coefficients': coefficients, 'trace': trace, 'fold_coefficients': compact_folds, 'sequential': sequential, 'invalid_internal_options': invalid,
            'diagnostics': {'selection_and_refit_seconds': time.perf_counter()-started, 'inner_origins': len(folds),
                            'inner_horizon': folds[0][1], 'requested_horizon': horizon, 'available_internal_origins':full_count, 'origin_representation_counts':[f[2]for f in folds], 'internal_criterion':'terminal_CRPS', 'research_learning_modes':list(MODES), 'intermediate_lead_pair_matrices_computed':False,
                            'maximum_origin_count': None, 'automatic_model_fallback': False,
                            'no_outer_future_dates_used': True}}


class MenuPool:
    def __init__(self, config, training_dates, bundle=None):
        if set(config) != {'menu', 'inner_paths','schedule'} or config['schedule']not in RULES or config['menu'] not in MODES or config['inner_paths'] < 2 or config['inner_paths'] % 2:
            raise ValueError('Known menu and even internal simulation count required')
        self.config = dict(config); self.dates = np.asarray(training_dates, dtype='datetime64[D]'); self.bundle = bundle; self.model_id = identity(config)
    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        cfg = self.config
        b = self.bundle if self.bundle is not None else prepare(history, weights, horizon, self.dates, inner_paths=cfg['inner_paths'],schedule_rule=cfg['schedule'])
        if (b['history_sha256'] != signature(history) or b['horizon'] != horizon or not np.array_equal(b['weights'], weights)
                or b['inner_paths'] != cfg['inner_paths'] or b['schedule_rule']!=cfg['schedule'] or b['training_dates_sha256'] != hashlib.sha256(self.dates.tobytes()).hexdigest()):
            raise ValueError('Forecast differs from its dated historical learner')
        learned = b['weights_by_mode'][cfg['menu']]; counts = energy.even_allocation(learned, paths)
        generators = [(i, model(i, b).terminal_chunks(history, weights, horizon, int(n), context_seed, width=width)) for i, n in enumerate(counts) if n]
        self.last_selection = {'config': cfg, 'ideal_weights': learned.tolist(), 'path_counts': counts.tolist(), 'invalid_internal_options': b['invalid_internal_options'], **b['diagnostics']}
        for start in range(0, horizon, width):
            pieces = []
            for i, generator in generators:
                position, values = next(generator)
                if position != start or values.shape[0] != counts[i]:
                    raise ArithmeticError('Incomplete selected path')
                pieces.append(values)
            values = np.concatenate(pieces)
            if not np.isfinite(values).all():
                raise ArithmeticError('Nonfinite forecast; no exclusions')
            yield start, values


def save_fold_coefficients(bundle, path):
    """Explicit caller-owned compact persistence; never write a hidden global cache."""
    metadata = {key: bundle[key] for key in ('history_sha256', 'training_dates_sha256',
                'horizon', 'inner_paths', 'schedule_rule', 'seed_replication', 'trace')}
    metadata['weights'] = bundle['weights'].tolist()
    metadata['recipe_bank_sha256'] = hashlib.sha256(json.dumps(BANK, sort_keys=True).encode()).hexdigest()
    np.savez_compressed(path, **bundle['fold_coefficients'],
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_fold_coefficients(path, history, weights, horizon, training_dates, *, inner_paths,
                           schedule_rule, seed_replication=0):
    """Allow replay only against the exact dated request that generated the folds."""
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored['metadata']))
        expected = {'history_sha256': signature(history),
                    'training_dates_sha256': hashlib.sha256(np.asarray(training_dates, dtype='datetime64[D]').tobytes()).hexdigest(),
                    'weights': np.asarray(weights, dtype=float).tolist(), 'horizon': horizon,
                    'inner_paths': inner_paths, 'schedule_rule': schedule_rule,
                    'seed_replication': seed_replication,
                    'recipe_bank_sha256': hashlib.sha256(json.dumps(BANK, sort_keys=True).encode()).hexdigest()}
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError('Compact fold cache differs from the exact dated request or recipe bank')
        return {key: stored[key].copy() for key in stored.files if key != 'metadata'}, metadata
