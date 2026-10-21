"""Relearn menu alternatives on exactly the same history-only CRPS evidence."""
import hashlib, json, time
import numpy as np
from candidates.asset_selector_universe_certified import energy
from candidates.asset_selector_universe_precision.candidate import signature, portfolio_returns, schedule
from candidates.asset_selector_path_objective.coefficients import lead_masses
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate
from candidates.asset_selector_fast_iid.candidate import identity as original_identity
from .bank import BANK, MENUS, prefix_fits, all_chunks, model


def identity(config):
    if config['menu'] == 'original':
        return original_identity({'criterion': 'path', 'inner_paths': config['inner_paths']})
    return 'model_testing_asset_component_menu_CRPS_v1_' + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def prepare(history, weights, horizon, training_dates, *, inner_paths, seed_replication=0, width=64):
    started = time.perf_counter(); x = np.asarray(history, dtype=float); w = np.asarray(weights, dtype=float); dates = np.asarray(training_dates, dtype='datetime64[D]')
    if (x.ndim != 2 or len(x) != len(dates) or x.shape[1] != len(w) or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-14) or np.isnat(dates).any()
            or np.any(np.diff(dates) <= np.timedelta64(0, 'D')) or inner_paths < 2 or inner_paths % 2
            or not isinstance(seed_replication, int) or seed_replication < 0 or width < 1):
        raise ValueError('Finite dated joint history, unit weights and even simulations required')
    eligible, folds = schedule(len(x), horizon)
    eligible += ([166, 167] if 0 in eligible else []) + [168, 169]
    invalid = {}; trace = []; total_A = np.zeros(len(BANK)); total_B = np.zeros((len(BANK), len(BANK)))
    for stop, h in folds:
        past = x[:stop]; prepared, failures = prefix_fits(past, [i for i in eligible if i not in invalid], w)
        for i, error in failures.items():
            invalid[i] = {'training_rows': stop, **error}
        valid = [i for i in eligible if i not in invalid]
        actual = np.cumsum(portfolio_returns(x[stop:stop+h], w)); days = (dates[stop:stop+h]-dates[stop-1]).astype('timedelta64[D]').astype(int)
        masses = lead_masses(days); suffix = '' if seed_replication == 0 else f':replication={seed_replication}'
        seed = int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(past)}:{h}:{inner_paths}{suffix}'.encode()).digest()[:4], 'little')
        # Sorted pair reductions are independent of other options. Generate and
        # reduce each path once, then reuse the same coefficients in every menu.
        A = np.zeros(len(valid)); B = np.zeros((len(valid), len(valid)))
        for start, cube in all_chunks(past, w, h, inner_paths, seed, valid, prepared, width=width):
            count = cube.shape[2]
            accumulate(cube, actual[start:start+count], masses[start:start+count], A, B)
        total_A[valid] += A; total_B[np.ix_(valid, valid)] += B
        trace.append({'training_rows': stop, 'validation_horizon': h, 'validation_stop': stop+h,
                      'training_sha256': signature(past), 'training_last_date': str(dates[stop-1]),
                      'validation_last_known_date': str(dates[stop+h-1]), 'seed': seed,
                      'eligible_this_fold': valid, 'PPW_dependence_fit': prepared.get('PPW')})
    learned = {}; fitted = {}; coefficients = {}
    for name, ids in MENUS.items():
        valid = [i for i in eligible if i in ids and i not in invalid]
        if not valid:
            raise ArithmeticError('No valid historical option; no fixed-model fallback')
        A = total_A[valid]/len(folds); B = total_B[np.ix_(valid, valid)]/len(folds)
        fitted[name] = energy.fit(A, B); learned[name] = np.zeros(len(BANK)); learned[name][valid] = fitted[name]['weights']
        coefficients[name] = {'A': A, 'B': B, 'valid': valid}
    needed = sorted({i for values in learned.values() for i in np.flatnonzero(values > 0)})
    full, failures = prefix_fits(x, needed, w)
    if failures:
        raise ArithmeticError(f'Selected full-history fit failed: {failures}')
    return {**full, 'history_sha256': signature(x), 'training_dates_sha256': hashlib.sha256(dates.tobytes()).hexdigest(),
            'weights': w.copy(), 'horizon': horizon, 'inner_paths': inner_paths, 'weights_by_mode': learned,
            'fit': fitted, 'coefficients': coefficients, 'trace': trace, 'invalid_internal_options': invalid,
            'diagnostics': {'selection_and_refit_seconds': time.perf_counter()-started, 'inner_origins': len(folds),
                            'inner_horizon': folds[0][1], 'requested_horizon': horizon,
                            'maximum_origin_count': None, 'automatic_model_fallback': False,
                            'no_outer_future_dates_used': True}}


class MenuPool:
    def __init__(self, config, training_dates, bundle=None):
        if set(config) != {'menu', 'inner_paths'} or config['menu'] not in MENUS or config['inner_paths'] < 2 or config['inner_paths'] % 2:
            raise ValueError('Known menu and even internal simulation count required')
        self.config = dict(config); self.dates = np.asarray(training_dates, dtype='datetime64[D]'); self.bundle = bundle; self.model_id = identity(config)
    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        cfg = self.config
        b = self.bundle if self.bundle is not None else prepare(history, weights, horizon, self.dates, inner_paths=cfg['inner_paths'])
        if (b['history_sha256'] != signature(history) or b['horizon'] != horizon or not np.array_equal(b['weights'], weights)
                or b['inner_paths'] != cfg['inner_paths'] or b['training_dates_sha256'] != hashlib.sha256(self.dates.tobytes()).hexdigest()):
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
