"""Freeze original assets plus nonredundant, explicitly constructed portfolios."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import numpy as np
import common
import construction as pc
import panel_inputs as source_inputs
import scoring as legacy_score
import sweep as api

OWN = common.OWN
BASE = common.CAMPAIGN/'inputs'
OUT = OWN/'inputs'
MAX_INPUT_BYTES = 512*1024**2
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: api.original._atomic_json(p, d)


def duplicate_of(dates, values, retained):
    """Complete shorter-history equivalence, with eight exact prefilter probes.

    Probe mismatch only avoids a full comparison; exclusion always requires the
    ENTIRE aligned shorter history to match within1e-12 absolute roundoff.
    Correlated but economically different returns are never excluded here.
    """
    for other_dates, group in retained.values():
        first, second = (dates, other_dates) if len(dates) <= len(other_dates) else (other_dates, dates)
        positions = np.searchsorted(second, first)
        if np.any(positions >= len(second)) or not np.array_equal(second[positions], first):
            continue
        probes = np.unique(np.linspace(0, len(first)-1, 8, dtype=int))
        for other_values, sid in group:
            left = values if len(dates) <= len(other_dates) else values[positions]
            right = other_values[positions] if len(dates) <= len(other_dates) else other_values
            if np.any(abs(left[probes]-right[probes]) > 1e-12):
                continue
            if np.all(abs(left-right) <= 1e-12):
                return sid
    return None


def main():
    if OUT.exists():
        raise FileExistsError('Use a new version; frozen/staging input directory already exists')
    source_hash = sha(BASE/'panel.json')
    parent = api.load_panel(BASE)
    quarter_support = set()
    for row in parent['series']:
        quarter_support.update(legacy_score.quarter_key(o['origin_date']) for o in api.schedule(BASE, row))
    definitions, structural_exclusions = pc.catalogue(parent['series'])
    OUT.mkdir()
    shutil.copytree(BASE/'series', OUT/'series')
    source_rows = {r['series_id']: r for r in parent['series']}
    rows = [{**row, 'panel_branch': 'assets'} for row in parent['series']]
    retained = {}
    assets = {}
    for row in parent['series']:
        calendar_hash = row['files'][row['dates_file']]
        if calendar_hash not in retained:
            retained[calendar_hash] = (np.load(BASE/row['dates_file'], allow_pickle=False), [])
        dates = retained[calendar_hash][0]
        values = np.load(BASE/row['returns_file'], allow_pickle=False)
        retained[calendar_hash][1].append((values, row['series_id']))
        assets[row['series_id']] = (dates, values)
    exclusions = []
    origin_exclusions = []
    accepted = []
    for i, definition in enumerate(definitions):
        sid = definition['series_id']
        weights = {key: Fraction(n, d) for key, n, d in definition['rational_weights']}
        dates, values, anchor = pc.portfolio_returns(assets, weights)
        source_inputs.validate_arrays(dates, values)
        duplicate = duplicate_of(dates, values, retained)
        if duplicate:
            exclusions.append({**definition, 'duplicate_of': duplicate,
                'reason': 'entire_aligned_shorter_return_history_equal_within1e-12_roundoff'})
            continue
        original_schedule = source_inputs._schedule(dates)
        origins = [o for o in original_schedule if legacy_score.quarter_key(o['origin_date']) in quarter_support]
        origin_exclusions.extend({'series_id': sid, 'origin': o,
            'reason': 'outside_preknown_individual_asset_evaluation_quarters'} for o in original_schedule if o not in origins)
        if not origins:
            raise ValueError('Portfolio has no supported evaluation origin: '+sid)
        folder = OUT/'series'/sid
        folder.mkdir()
        np.save(folder/'dates.npy', dates, allow_pickle=False)
        np.save(folder/'log_returns.npy', values, allow_pickle=False)
        intervals = np.r_[int((dates[0]-np.datetime64(anchor, 'D')).astype(int)), np.diff(dates).astype(int)].astype('<i4')
        np.save(folder/'interval_days.npy', intervals, allow_pickle=False)
        write(folder/'origins.json', origins)
        write(folder/'origin_calendar.json', {o['origin_label']: {
            'training_first_date': str(dates[0]), 'training_last_date': str(dates[o['position']]),
            'future_first_date': str(dates[o['position']+1]),
            'future_last_date': str(dates[o['position']+o['max_horizon']])} for o in origins})
        files = {str(p.relative_to(OUT)): sha(p) for p in sorted(folder.iterdir())}
        row = {**definition, 'panel_branch': 'portfolios', 'asset_class': 'mixed_portfolios',
            'region': 'mixed', 'observations': len(dates), 'first_date': str(dates[0]), 'last_date': str(dates[-1]),
            'dates_file': str((folder/'dates.npy').relative_to(OUT)),
            'returns_file': str((folder/'log_returns.npy').relative_to(OUT)),
            'origins_file': str((folder/'origins.json').relative_to(OUT)),
            'origin_calendar_file': str((folder/'origin_calendar.json').relative_to(OUT)),
            'interval_days_file': str((folder/'interval_days.npy').relative_to(OUT)),
            'interval_days_kind': 'actual_common_observed_valuation_intervals_all_intermediate_component_returns_compounded',
            'source_frequency': 'daily_source_common_observed_endpoints_no_fill_no_interpolation',
            'origins_count': len(origins), 'max_horizon': max(o['max_horizon'] for o in origins),
            'native_dated_returns_sha256': source_inputs.dated_digest(dates, values), 'files': files,
            'provenance': {'parent_panel_fingerprint': parent['panel_fingerprint'],
                'valuation_anchor': anchor, 'component_dated_return_hashes': {
                    key: source_rows[key]['native_dated_returns_sha256'] for key in weights},
                'source_return_construction_reaudited': False, 'accepted_frozen_parent_histories': True}}
        rows.append(row)
        accepted.append(definition)
        key = hashlib.sha256(dates.astype('<i8').tobytes()).hexdigest()
        if key not in retained:
            retained[key] = (dates, [])
        retained[key][1].append((values, sid))
        if (i+1) % 50 == 0:
            print(json.dumps({'definitions_reviewed': i+1, 'definitions_expected': len(definitions),
                'accepted_portfolios': len(accepted), 'exact_return_equivalents_excluded': len(exclusions)}), flush=True)
    write(OUT/'portfolio_catalogue.json', {'coverage_rule': 'all_broad_cross_class_pairs_all_major_class_subsets_directional_concentration_strategic_templates_all_five_broad_sectors',
        'no_arbitrary_count_cap': True, 'proposed_unique_definitions': len(definitions), 'accepted': accepted,
        'structural_duplicates': structural_exclusions, 'return_equivalents': exclusions})
    write(OUT/'portfolio_origin_exclusions.json', origin_exclusions)
    write(OUT/'evaluation_quarters.json', sorted(quarter_support))
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    total_bytes = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if total_bytes > MAX_INPUT_BYTES:
        raise ValueError('Owned input budget exceeded; preserve staging and revise storage budget before any forecasts')
    panel = {'schema_version': 2, 'panel_id': 'daily-assets-and-portfolios-v4-20261020',
        'status': 'frozen_unscored', 'simulations_per_origin': 240, 'series': rows, 'series_count': len(rows),
        'asset_series_preserved': len(parent['series']), 'portfolio_series_added': len(accepted),
        'expected_tasks': sum(r['origins_count'] for r in rows), 'expected_cells': sum(r['max_horizon'] for r in rows),
        'asset_class_counts': dict(sorted(Counter(r['asset_class'] for r in rows).items())),
        'branch_series_counts': dict(Counter(r['panel_branch'] for r in rows)),
        'branch_task_counts': {branch: sum(r['origins_count'] for r in rows if r['panel_branch'] == branch)
                               for branch in ['assets', 'portfolios']},
        'origin_policy': 'original_native_schedule_portfolio_origins_restricted_to_preknown_asset_quarters',
        'calendar_policy': 'actual_native_assets_and_common_observed_portfolio_valuations_no_fill_all_intermediate_returns_compounded',
        'return_type': 'native_observation_portfolio_log_returns', 'input_bytes': total_bytes, 'files': files,
        'lineage': {'parent_panel_fingerprint': parent['panel_fingerprint'], 'parent_panel_file_sha256': source_hash,
            'construction_sources': {p.name: sha(p) for p in OWN.glob('*.py')},
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'legacy_input_builder_sha256': sha(source_inputs.__file__),
            'single_main_score_document_sha256': sha(OWN/'SCORING.md')},
        'score_protocol_id': 'daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch'}
    panel['panel_fingerprint'] = api.original.stable_digest(panel)
    write(OUT/'panel.json', panel)
    validated = api.load_panel(OUT)
    assert sha(BASE/'panel.json') == source_hash
    assert all(sha(OUT/name) == digest for row in parent['series'] for name, digest in row['files'].items())
    write(OWN/'input_verification.json', {'complete': True, 'panel_fingerprint': validated['panel_fingerprint'],
        'original_assets_bitwise_unchanged': 722, 'portfolio_series': len(accepted),
        'expected_tasks_per_model': panel['expected_tasks'], 'input_bytes': total_bytes,
        'structural_duplicates_excluded': len(structural_exclusions), 'return_equivalents_excluded': len(exclusions),
        'calendar_only_origins_not_admitted': len(origin_exclusions), 'portfolio_quarters_subset_of_asset_quarters': True,
        'model_forecasts_started': False, 'single_main_score': panel['score_protocol_id']})
    print(json.dumps(readable := json.loads((OWN/'input_verification.json').read_text())), flush=True)


if __name__ == '__main__':
    main()
