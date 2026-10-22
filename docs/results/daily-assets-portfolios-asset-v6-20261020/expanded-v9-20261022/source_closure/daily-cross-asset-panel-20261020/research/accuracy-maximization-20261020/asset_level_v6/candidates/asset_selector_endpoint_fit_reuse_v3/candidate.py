"""Endpoint exact fitter and draw reuse; unchanged financial identity/kernels."""
import hashlib,time
import numpy as np
from candidates.asset_selector_endpoint_group_reuse_v1 import candidate as reference
from .fits_cache import GroupWork
from .paths import all_chunks
energy=reference.energy;schedule=reference.schedule;signature=reference.signature
portfolio_returns=reference.portfolio_returns;BANK=reference.BANK;MENUS=reference.MENUS
compiled_coefficients=reference.compiled_coefficients;original_prefix_fits=reference.original_prefix_fits
IMPLEMENTATION_ID='endpoint_group_reuse_compiled_pairs_fit_reuse_fast_draws_v3'
identity=reference.identity
MenuPool=reference.MenuPool

def _prepare(history, weights, horizon, training_dates, *, inner_paths, schedule_rule, seed_replication=0, width=64, fit_cache=None, solver=None):
    started = time.perf_counter(); x = np.asarray(history, dtype=float); w = np.asarray(weights, dtype=float); dates = np.asarray(training_dates, dtype='datetime64[D]')
    if (x.ndim != 2 or len(x) != len(dates) or x.shape[1] != len(w) or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-14) or np.isnat(dates).any()
            or np.any(np.diff(dates) <= np.timedelta64(0, 'D')) or inner_paths < 2 or inner_paths % 2
            or not isinstance(seed_replication, int) or seed_replication < 0 or width < 1):
        raise ValueError('Finite dated joint history, unit weights and even simulations required')
    if fit_cache is not None:fit_cache.validate_request(x,w,horizon,dates,schedule_rule)
    prefix_fits=original_prefix_fits if fit_cache is None else fit_cache.prefix_fits
    fit=energy.fit if solver is None else solver
    if not callable(fit):raise ValueError("Callable simplex solver required")
    eligible, folds, full_count = schedule(len(x), horizon, schedule_rule)
    eligible = [i for i in eligible if i!=0]+([166,167]if 0 in eligible else[])
    invalid = {}; trace = []; total_A = np.zeros(len(BANK)); total_B = np.zeros((len(BANK), len(BANK)))
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
        A,B=compiled_coefficients(last,actual);A/=np.sqrt(h);B/=np.sqrt(h)
        total_A[valid]+=represented*A;total_B[np.ix_(valid,valid)]+=represented*B
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
        fitted[name] = fit(A, B); learned[name] = np.zeros(len(BANK)); learned[name][valid] = fitted[name]['weights']
        coefficients[name] = {'A': A, 'B': B, 'valid': valid}
    needed = sorted({i for values in learned.values() for i in np.flatnonzero(values > 0)})
    full, failures = prefix_fits(x, needed, w)
    if failures:
        raise ArithmeticError(f'Selected full-history fit failed: {failures}')
    return {**full, 'history_sha256': signature(x), 'training_dates_sha256': hashlib.sha256(dates.tobytes()).hexdigest(),
            'weights': w.copy(), 'horizon': horizon, 'inner_paths': inner_paths, 'schedule_rule':schedule_rule, 'seed_replication':seed_replication, 'weights_by_mode': learned,
            'fit': fitted, 'coefficients': coefficients, 'trace': trace, 'invalid_internal_options': invalid,
            'diagnostics': {'selection_and_refit_seconds': time.perf_counter()-started, 'inner_origins': len(folds),
                            'inner_horizon': folds[0][1], 'requested_horizon': horizon, 'available_internal_origins':full_count, 'origin_representation_counts':[f[2]for f in folds], 'internal_criterion':'terminal_CRPS', 'intermediate_lead_pair_matrices_computed':False,
                            'maximum_origin_count': None, 'automatic_model_fallback': False,
                            'no_outer_future_dates_used': True}}

def prepare(history,weights,horizon,training_dates,*,inner_paths,schedule_rule,seed_replication=0,width=64,fit_cache=None,solver=None,skip_unused_parent_calibration=False):
    kwargs=dict(inner_paths=inner_paths,schedule_rule=schedule_rule,seed_replication=seed_replication,width=width,solver=solver)
    if fit_cache is None:
        with GroupWork(history,weights,training_dates,horizons=(horizon,),schedule_rule=schedule_rule,skip_unused_parent_calibration=skip_unused_parent_calibration) as work:
            result=_prepare(history,weights,horizon,training_dates,fit_cache=work,**kwargs)
    else:
        if bool(skip_unused_parent_calibration)!=fit_cache.skip_unused_parent_calibration:raise ValueError('Explicit parent calibration mode must match the group cache')
        result=_prepare(history,weights,horizon,training_dates,fit_cache=fit_cache,**kwargs)
    result['diagnostics'].update({'skip_unused_parent_calibration':bool(skip_unused_parent_calibration),'legacy_unused_calibration_failure_guard_removed':bool(skip_unused_parent_calibration),'universal_old_invalid_mask_parity_claim':False})
    return result
