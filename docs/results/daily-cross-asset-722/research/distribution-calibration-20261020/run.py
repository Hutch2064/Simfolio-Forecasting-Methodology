"""Exact current/Public replays; distribution diagnostics do not affect CRPS."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from types import SimpleNamespace
import argparse
import datetime
import hashlib
import json
import sys
import time
import numpy as np

OWN = Path(__file__).resolve().parent
ROOT = next(p for p in OWN.parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
from expanded_public_smooth_mean_live import build_smooth_mean_pool
from simfolio_forecasting_methodology.french_panel.selected import build_public
import sweep as api
import scoring as sc
import metrics
FACTORIES = {'Public': build_public, 'Incumbent': build_smooth_mean_pool}
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
write = lambda p, d: api.original._atomic_json(p, d)


def prepare():
    assert not (OWN/'plan.json').exists()
    parent = OWN.parent/'multiscale_volatility/plan.json'
    old = read(parent)
    reference = OWN.parent/'mean_shrinkage/development_summary.json'
    p = {'schema': 'descriptive_distribution_calibration_v1', 'utc': now(),
        'parent_plan_sha256': sha(parent), 'panel_fingerprint': old['panel_fingerprint'],
        'input_root': str(ROOT/'inputs'), 'cases': old['phases']['development'],
        'source_hashes': {x.name: sha(x) for x in OWN.glob('*.py')},
        'parent_component_hashes': old['parent_component_hashes'], 'shared_hashes': old['shared_hashes'],
        'native_source_closure_digest': api.original.stable_digest(api.original._source_closure()),
        'reference': str(reference), 'reference_sha256': sha(reference),
        'models': {a: api.original.dependency_identity(f()) for a, f in FACTORIES.items()},
        'expected_origins': 245, 'expected_forecasts': 490, 'paths': 240,
        'diagnostics_change_scores_or_model_selection': False, 'candidate_admission': False,
        'CDF': 'finite_ensemble_midpoint_less_plus_half_ties_divided_by240',
        'quantiles': list(metrics.QUANTILES), 'quantile_convention': 'empirical_CDF_inverse_ceil_Mp',
        'bands': list(metrics.BANDS), 'band_weights': 'canonical_positive_calendar_log_weights_normalized_within_available_band',
        'aggregation': 'unchanged_origin_series_quarter_family_availableclass_hierarchy_for_descriptive_indicators',
        'finite_dependent_stratified_ensemble_not_exact_iid_rank_uniformity_test': True,
        'overlapping_origins_horizons_not_independent': True, 'adaptive_reused_development_evidence': True,
        'score_fit_or_calibration_correction': False, 'owned_output_budget_bytes': 256*1024**2}
    write(OWN/'plan.json', p)
    write(OWN/'plan_receipt.json', {'utc': now(), 'plan_sha256': sha(OWN/'plan.json'),
        'before_first_forecast_replay': True, 'tests_passed': 3})
    print(json.dumps({'prepared': True, 'origins': 245, 'forecast_replays': 490}), flush=True)


def guard():
    p = read(OWN/'plan.json')
    assert sha(OWN/'plan.json') == read(OWN/'plan_receipt.json')['plan_sha256']
    assert all(sha(OWN/k) == v for k, v in p['source_hashes'].items())
    assert all(sha(ROOT/k) == v for k, v in p['shared_hashes'].items())
    assert all(sha(Path(k)) == v for k, v in p['parent_component_hashes'].items())
    assert api.original.stable_digest(api.original._source_closure()) == p['native_source_closure_digest']
    assert sha(Path(p['reference'])) == p['reference_sha256']
    assert sum(x.stat().st_size for x in OWN.rglob('*') if x.is_file()) <= p['owned_output_budget_bytes']
    return p


class Capture:
    def __init__(self, model):
        self.model = model
        self.model_id = model.model_id

    def simulate_daily_log_returns(self, training, context):
        self.daily = self.model.simulate_daily_log_returns(training, context)
        return self.daily


def run():
    p = guard()
    reference = read(Path(p['reference']))
    rows = {a: {(r['series_id'], r['origin_label']): r for r in reference['records']['Candidate' if a == 'Incumbent' else a]}
            for a in FACTORIES}
    models = {a: Capture(f()) for a, f in FACTORIES.items()}
    metadata = {c['series']['series_id']: c['series'] for c in p['cases']}
    records = {a: [] for a in models}
    count = hashes = 0
    samples = []
    write(OWN/'active_run.json', {'utc': now(), 'status': 'running', 'pid': os.getpid(),
        'expected_origins': 245, 'expected_forecasts': 490})
    for case in p['cases']:
        row = case['series']
        sid = row['series_id']
        dates, values, _ = api.load_series(p['input_root'], row)
        for origin in case['origins']:
            started = time.perf_counter()
            n = origin['position']+1
            h = origin['max_horizon']
            future = dates[n:n+h]
            observed = np.cumsum(values[n:n+h])
            w = sc.calendar_log_weights(origin['origin_date'], future)
            elapsed = (future-np.datetime64(origin['origin_date'], 'D')).astype(int)
            seed = api.original.coherent_daily_seed(origin['origin_date'], h, 240)
            for arm in list(models)[::1 if count % 2 else -1]:
                model = models[arm]
                loss, _ = api.original._forecast(model, sid, values, dates, SimpleNamespace(**origin), seed)
                digest = hashlib.sha256(loss.tobytes()).hexdigest()
                expected = rows[arm][sid, origin['origin_label']]
                assert digest == expected['loss_sha256'], 'Canonical loss replay drift'
                primary, sensitivity = sc.normalized_origin_scores(loss, values[:n], origin['origin_date'], future)
                assert max(abs(primary-expected['primary_score']), abs(sensitivity-expected['sensitivity_score'])) < 1e-12
                diagnostic = metrics.weighted_summary(metrics.diagnostics(np.cumsum(model.daily, axis=1), observed), elapsed, w)
                del model.daily
                records[arm].append({'series_id': sid, 'origin_date': origin['origin_date'],
                    'origin_label': origin['origin_label'], 'evaluation_split': origin['evaluation_split'],
                    'loss_sha256': digest, 'primary_score': primary, 'sensitivity_score': sensitivity,
                    'diagnostics': diagnostic})
                hashes += 1
            count += 1
            if count > 1:
                samples.append(time.perf_counter()-started)
            status = {'utc': now(), 'state': 'running', 'pid': os.getpid(), 'completed_origins': count,
                'expected_origins': 245, 'completed_forecasts': hashes, 'expected_forecasts': 490,
                'active_workers': 1, 'exact_historical_loss_hashes_replayed': hashes,
                'native_failures': 0, 'scoring_integrity_errors': 0,
                'mean_remaining_seconds_projection': float(np.mean(samples)*(245-count)) if samples else None,
                'median_remaining_seconds_projection': float(np.median(samples)*(245-count)) if samples else None,
                'ETA_scope': 'Measured forecast-and-diagnostic wall service excluding setup; contention nonstationary, not a scientific speed benchmark.'}
            write(OWN/'status.json', status)
            if count % 25 == 0:
                guard()
                print(json.dumps(status), flush=True)
    guard()
    def aggregate(rr, field):
        return sc.aggregate_records(rr, metadata, score_key=field)['score']
    score_check = {a: {f: aggregate(rr, f) for f in ['primary_score', 'sensitivity_score']} for a, rr in records.items()}
    summaries = {}
    for arm, rr in records.items():
        groups = {'all': {sid for sid in metadata}}
        groups.update({'class:'+c: {sid for sid in metadata if metadata[sid]['asset_class'] == c}
                       for c in sorted({r['asset_class'] for r in metadata.values()})})
        groups.update({'family:'+f: {sid for sid in metadata if metadata[sid]['family'] == f}
                       for f in sorted({r['family'] for r in metadata.values()})})
        summaries[arm] = {}
        for group, ids in groups.items():
            by_band = {}
            for band, _, _ in metrics.BANDS:
                subset = [{**r, **r['diagnostics'][band]} for r in rr if r['series_id'] in ids and band in r['diagnostics']]
                if subset:
                    names = list(subset[0]['diagnostics'][band])
                    by_band[band] = {'origins': len(subset), 'metrics': {name: aggregate(subset, name) for name in names}}
            summaries[arm][group] = by_band
    result = {'utc': now(), 'complete': True, 'origins_per_model': count, 'forecast_replays': hashes,
        'historical_loss_hashes_exact': hashes, 'native_failures': 0, 'scoring_integrity_errors': 0,
        'unchanged_canonical_scores': score_check, 'summaries': summaries,
        'diagnostics_change_scores_or_model_selection': False, 'records': records,
        'adaptive_reused_development_evidence': True, 'independent_holdout': False,
        'no_independent_sample_or_significance_claim': True}
    write(OWN/'summary.json', result)
    status.update(state='complete', active_workers=0)
    write(OWN/'status.json', status)
    write(OWN/'active_run.json', {'utc': now(), 'pid': os.getpid(), 'status': 'complete'})
    print(json.dumps({k: v for k, v in result.items() if k not in ['records', 'summaries']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    action = parser.parse_args().action
    prepare() if action == 'prepare' else run()
