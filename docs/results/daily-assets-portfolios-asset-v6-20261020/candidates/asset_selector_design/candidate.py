"""Joint-path learner contrasts: validation design, precision and hard selection."""
from pathlib import Path
import hashlib, json, time
import numpy as np
import common
from candidates.asset_selector_universe.bank import BANK, prefix_fits
from candidates.asset_selector_universe_certified import energy
from candidates.asset_selector_universe_precision.candidate import portfolio_returns, signature
from candidates.asset_selector_path_objective.coefficients import lead_masses
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate
from candidates.asset_selector_fast_iid.paths import all_chunks, model
from candidates.asset_selector_fast_iid.candidate import identity as established_identity
from .validation import schedule, RULES


def identity(config):
    if config['schedule'] == 'established' and config['pool'] == 'stack':
        return established_identity({'criterion': 'path', 'inner_paths': config['inner_paths']})
    return 'model_testing_asset_validation_design_CRPS_v1_' + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def prepare(history, weights, horizon, training_dates, *, inner_paths, schedule_rule, seed_replication=0, width=64):
    started = time.perf_counter(); x = np.asarray(history, dtype=float); w = np.asarray(weights, dtype=float); dates = np.asarray(training_dates, dtype='datetime64[D]')
    if (x.ndim != 2 or len(x) != len(dates) or x.shape[1] != len(w) or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-14) or np.isnat(dates).any()
            or np.any(np.diff(dates) <= np.timedelta64(0, 'D')) or inner_paths < 2 or inner_paths % 2
            or not isinstance(seed_replication, int) or seed_replication < 0 or width < 1):
        raise ValueError('Finite dated joint history, unit weights, even simulations and valid repetition required')
    eligible, folds = schedule(len(x), horizon, schedule_rule)
    invalid = {}; trace = []; total_A = np.zeros(len(BANK)); total_B = np.zeros((len(BANK), len(BANK)))
    for stop, h in folds:
        past = x[:stop]; prepared, failures = prefix_fits(past, [i for i in eligible if i not in invalid])
        for i, error in failures.items():
            invalid[i] = {'training_rows': stop, **error}
        valid = [i for i in eligible if i not in invalid]
        actual = np.cumsum(portfolio_returns(x[stop:stop+h], w))
        days = (dates[stop:stop+h]-dates[stop-1]).astype('timedelta64[D]').astype(int); masses = lead_masses(days)
        suffix = '' if seed_replication == 0 else f':replication={seed_replication}'
        seed = int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(past)}:{h}:{inner_paths}{suffix}'.encode()).digest()[:4], 'little')
        A = np.zeros(len(valid)); B = np.zeros((len(valid), len(valid)))
        for start, cube in all_chunks(past, w, h, inner_paths, seed, valid, prepared, width=width):
            count = cube.shape[2]; accumulate(cube, actual[start:start+count], masses[start:start+count], A, B)
        total_A[valid] += A; total_B[np.ix_(valid, valid)] += B
        trace.append({'training_rows': stop, 'validation_horizon': h, 'validation_stop': stop+h,
                      'training_sha256': signature(past), 'training_last_date': str(dates[stop-1]),
                      'validation_last_known_date': str(dates[stop+h-1]), 'seed': seed, 'eligible_this_fold': valid})
    valid = [i for i in eligible if i not in invalid]
    if not valid:
        raise ArithmeticError('No valid historical option; no fixed-model fallback')
    A = total_A[valid]/len(folds); B = total_B[np.ix_(valid, valid)]/len(folds)
    fitted = energy.fit(A, B); stack = np.zeros(len(BANK)); stack[valid] = fitted['weights']
    select = np.zeros(len(BANK)); select[valid[int(np.argmin(A-.5*np.diag(B)))]] = 1.
    learned = {'stack': stack, 'select': select}
    needed = sorted({i for values in learned.values() for i in np.flatnonzero(values > 0)})
    full, failures = prefix_fits(x, needed)
    if failures:
        raise ArithmeticError(f'Selected full-history fit failed: {failures}')
    return {**full, 'history_sha256': signature(x), 'training_dates_sha256': hashlib.sha256(dates.tobytes()).hexdigest(),
            'weights': w.copy(), 'horizon': horizon, 'inner_paths': inner_paths, 'seed_replication': seed_replication,
            'schedule_rule': schedule_rule, 'weights_by_mode': learned, 'fit': fitted, 'coefficients': {'A': A, 'B': B},
            'trace': trace, 'invalid_internal_options': invalid,
            'diagnostics': {'selection_and_refit_seconds': time.perf_counter()-started, 'inner_origins': len(folds),
                            'inner_horizon': folds[0][1], 'requested_horizon': horizon,
                            'maximum_origin_count': None, 'automatic_model_fallback': False,
                            'no_outer_future_dates_used': True}}


class DesignPool:
    def __init__(self, config, training_dates, bundle=None):
        if (set(config) != {'schedule', 'inner_paths', 'pool'} or config['schedule'] not in RULES
                or config['pool'] not in ('stack', 'select') or config['inner_paths'] < 2 or config['inner_paths'] % 2):
            raise ValueError('Valid historical design, selector and even simulation count required')
        self.config = dict(config); self.dates = np.asarray(training_dates, dtype='datetime64[D]'); self.bundle = bundle; self.model_id = identity(config)
    @property
    def dependency_identity(self):
        return {'method': self.model_id, 'config': self.config, 'forecast_level': 'joint_assets', 'bank': BANK,
                'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self, history, weights, horizon, paths, context_seed, *, width=64):
        cfg = self.config
        b = self.bundle if self.bundle is not None else prepare(history, weights, horizon, self.dates, inner_paths=cfg['inner_paths'], schedule_rule=cfg['schedule'])
        if (b['history_sha256'] != signature(history) or b['horizon'] != horizon or not np.array_equal(b['weights'], weights)
                or b['inner_paths'] != cfg['inner_paths'] or b['schedule_rule'] != cfg['schedule']
                or b['training_dates_sha256'] != hashlib.sha256(self.dates.tobytes()).hexdigest()):
            raise ValueError('Forecast differs from its dated historical learner')
        learned = b['weights_by_mode'][cfg['pool']]; counts = energy.even_allocation(learned, paths)
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
