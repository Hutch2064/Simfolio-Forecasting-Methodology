"""Independently rescore the batched inner-precision study; never promote it."""
from pathlib import Path
import datetime, hashlib, json, statistics, sys
import numpy as np

O = Path(__file__).resolve().parent
V = O / 'asset_level_v6'
D = V / 'candidates/asset_selector_inner_precision_batched/study1000'
sys.path.insert(0, str(V)); sys.path.insert(0, str(O))
import common, sweep_v6 as api, score_main
from inputs import AssetInputs
from candidates.asset_selector_universe_precision.candidate import signature
from candidates.asset_selector_universe_certified import energy
from candidates.asset_selector_inner_precision_batched.candidate import identity
from score_percentage import percent_better

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def allocated_counts(selection):
    weights = np.asarray(selection['ideal_weights'], dtype=float)
    counts = np.asarray(selection['path_counts'])
    assert weights.shape == counts.shape == (166,)
    assert np.isfinite(weights).all() and np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
    assert np.array_equal(counts, energy.even_allocation(weights, 1000))
    assert counts.sum() == 1000
    return counts


def standalone_cost(record, paired_terminal):
    """Charge the shared prepare to either criterion, even when reused on disk."""
    assert record['case'] == paired_terminal['case']
    a, b = record['configuration'], paired_terminal['configuration']
    assert b['criterion'] == 'terminal'
    assert (a['inner_paths'], a['seed_replication']) == (b['inner_paths'], b['seed_replication'])
    output_cpu = record['cpu_seconds'] - record['prepare_cpu_seconds']
    assert output_cpu >= 0 and paired_terminal['prepare_cpu_seconds'] >= 0
    return paired_terminal['prepare_cpu_seconds'] + output_cpu


def main():
    plan = read(D / 'plan.json'); summary = read(D / 'summary.json')
    assert summary['complete_execution'] and summary['qualification_valid']
    assert not summary['errors'] and summary['outer_paths'] == plan['outer_paths'] == 1000
    assert not plan['official'] and plan['full_panel_catalogue_increment'] == 0
    assert all(sha(O / p) == h for p, h in plan['source_hashes'].items())
    assert sha(D / 'references.json') == plan['references_sha256']
    cases, configs = plan['cases'], plan['configs']
    expected = {(i, j) for i in range(len(cases)) for j in range(len(configs))}
    records = summary['records']
    assert len(records) == plan['expected_new_forecasts'] == len(expected)
    assert {(r['case'], r['config_index']) for r in records} == expected
    panel = api.load_panel(V / 'inputs', verify_files=False)
    rows = {r['series_id']: r for r in panel['series']}
    refs = read(D / 'references.json')['cases']
    labels = ('fixed_reference', 'precision1000_terminal', 'precision1000_path')
    reference_rows = {label: [] for label in labels}
    prefixes = {}; recalculated = {}; verified = []
    source_digest = api.original.stable_digest(plan['source_hashes'])
    for i, case in enumerate(cases):
        row = rows[case['series_id']]; origin = case['origin']
        dates, values, origins = api.load_series(V / 'inputs', row, verify_files=False)
        assert origin in origins
        stop, horizon = origin['position'] + 1, origin['max_horizon']
        x, w, ids = AssetInputs(V / 'inputs').training(row, stop)
        prefixes[i] = dates, values, stop, horizon, x, ids
        for label in labels:
            r = refs[str(i)][label]; cp = Path(r['checkpoint'])
            assert r['series_id'] == case['series_id'] and r['origin_label'] == origin['origin_label']
            assert r['paths'] == 1000 and sha(cp) == r['checkpoint_sha256']
            with np.load(cp, allow_pickle=False) as data:
                loss = data['losses']
                assert loss.shape == (horizon,) and np.isfinite(loss).all()
                assert hashlib.sha256(loss.tobytes()).hexdigest() == r['loss_sha256']
            q = score_main.origin_score(loss, values[:stop], origin['origin_date'], dates[stop:stop+horizon])
            assert q == r['score']
            reference_rows[label].append({**r, 'score': q})
    for r in records:
        i, j = r['case'], r['config_index']; case = cases[i]; origin = case['origin']
        dates, values, stop, horizon, x, ids = prefixes[i]
        cfg = configs[j]; n, rep, criterion = cfg['inner_paths'], cfg['seed_replication'], cfg['criterion']
        assert read(D / 'records' / f'{j:03d}-{i:03d}.json') == r
        assert r['configuration'] == cfg and r['outer_paths'] == 1000
        assert r['inner_paths'] == n and r['inner_seed_replication'] == rep
        assert r['series_id'] == case['series_id'] and r['origin_label'] == origin['origin_label']
        assert tuple(r['constituent_ids']) == ids and r['source_closure_digest'] == source_digest
        assert r['final_seed_shared_across_inner_counts'] == api.original.coherent_daily_seed(origin['origin_date'], horizon, 1000)
        recipe = {'criterion': criterion, 'inner_paths': n}
        assert r['model_id'] == identity(recipe) and r['selection']['config'] == recipe
        allocated_counts(r['selection'])
        for fold in r['validation_trace']:
            fstop, fh = fold['training_rows'], fold['validation_horizon']
            assert fstop + fh == fold['validation_stop'] and fstop + fh <= stop
            assert fold['training_sha256'] == signature(x[:fstop])
            assert fold['training_last_date'] == str(dates[fstop-1])
            assert fold['validation_last_known_date'] == str(dates[fstop+fh-1])
            suffix = '' if rep == 0 else f':replication={rep}'
            seed = int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(x[:fstop])}:{fh}:{n}{suffix}'.encode()).digest()[:4], 'little')
            assert seed == fold['seed']
        cp = D / 'losses' / f'{j:03d}-{i:03d}.npz'
        assert sha(cp) == r['checkpoint_sha256']
        with np.load(cp, allow_pickle=False) as data:
            loss = data['losses']
            assert loss.shape == (horizon,) and np.isfinite(loss).all()
            assert hashlib.sha256(loss.tobytes()).hexdigest() == r['loss_sha256']
        q = score_main.origin_score(loss, values[:stop], origin['origin_date'], dates[stop:stop+horizon])
        assert q == r['score']; recalculated[i, j] = {**r, 'score': q}
        verified.append({'case': i, 'configuration': j, 'checkpoint_sha256': r['checkpoint_sha256'], 'loss_sha256': r['loss_sha256']})
    baseline = score_main.aggregate_records(reference_rows['fixed_reference'], rows)['score']
    high = {criterion: score_main.aggregate_records(reference_rows['precision1000_' + criterion], rows)['score'] for criterion in ('terminal', 'path')}
    assert baseline == summary['main_reference_Q']
    assert high == summary['preserved1000inner_reference_Q_by_criterion']
    models = []
    for j, cfg in enumerate(configs):
        # Preserve the report's reduction order; every scalar was rebuilt above.
        selected = [recalculated[r['case'], j] for r in records if r['config_index'] == j]
        q = score_main.aggregate_records(selected, rows)['score']
        old = next(m for m in summary['models'] if m['config_index'] == j)
        assert q == old['main_Q'] and percent_better(q, baseline) == old['percent_better_vs_fixed_reference']
        assert q - high[cfg['criterion']] == old['main_Q_difference_from_preserved1000inner_reference']
        paired = next(k for k, c in enumerate(configs) if c == {**cfg, 'criterion': 'terminal'})
        costs = [standalone_cost(r, recalculated[r['case'], paired]) for r in selected]
        differences = [np.abs(allocated_counts(r['selection']) - np.asarray(refs[str(r['case'])]['precision1000_' + cfg['criterion']]['selection']['path_counts'])).sum() / 2000 for r in selected]
        models.append({'config_index': j, 'configuration': cfg, 'main_Q': q,
                       'percent_better_vs_fixed_reference': percent_better(q, baseline),
                       'main_Q_difference_from_preserved1000inner_reference': q - high[cfg['criterion']],
                       'standalone_attributed_cpu_seconds_total': sum(costs),
                       'standalone_attributed_cpu_seconds_median': statistics.median(costs),
                       'mean_final_allocation_total_variation_vs1000inner': statistics.mean(differences)})
    repetitions = []
    for criterion in ('terminal', 'path'):
        for n in sorted({c['inner_paths'] for c in configs}):
            qs = [m['main_Q'] for m in models if m['configuration']['criterion'] == criterion and m['configuration']['inner_paths'] == n]
            repetitions.append({'criterion': criterion, 'inner_paths': n, 'seed_repetitions': len(qs),
                                'mean_main_Q': statistics.mean(qs), 'median_main_Q': statistics.median(qs),
                                'minimum_main_Q': min(qs), 'maximum_main_Q': max(qs),
                                'sample_standard_deviation_of_main_Q': statistics.stdev(qs),
                                'interpretation': 'Descriptive Monte Carlo variation on reused development cases; no independence or stability certificate'})
    out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'verified': True,
           'summary_sha256': sha(D / 'summary.json'), 'plan_sha256': sha(D / 'plan.json'),
           'saved_candidate_forecasts_verified': len(verified), 'reference_forecasts_verified': len(labels) * len(cases),
           'all_candidate_and_reference_scores_recomputed': True, 'all_internal_seeds_and_causal_prefixes_verified': True,
           'outer_paths': 1000, 'models': models, 'seed_variation': repetitions, 'main_reference_Q': baseline,
           'preserved1000inner_reference_Q_by_criterion': high,
           'high_precision_reference_seed_variation_not_established': True,
           'no_full_panel_promotion_or_production_precision_choice': True,
           'no_independence_or_significance_claim': True,
           'runtime_scope': 'Process CPU per request with shared two-criterion preparation charged to either standalone output. Includes both objective optimizations and union refits, excludes setup/IO, includes warmup; not a standalone production latency or a causal speed ratio against older 1000-inner kernels.',
           'verified_checkpoint_identities': verified, 'verifier_sha256': sha(Path(__file__))}
    api.original._atomic_json(D / 'verification_receipt.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'verified_checkpoint_identities'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
