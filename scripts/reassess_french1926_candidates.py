"""Recompute the saved original-shortlist comparison without changing approved selection."""
import argparse
import csv
import heapq
import json
import statistics
from pathlib import Path


def calculate(data):
    baseline = data['reference52_crps']
    timed = [r for r in data['rows'] if r.get('summed_worker_seconds_not_controlled_benchmark') is not None]
    results = []
    for source in data['rows'] + data.get('additional_family_candidates', []):
        row = dict(source)
        score = row.get('CRPS')
        seconds = row.get('summed_worker_seconds_not_controlled_benchmark')
        row['worker_minutes'] = seconds / 60 if seconds is not None else None
        row['percent_better_production52'] = 100 * (1 - score / data['production52_crps']) if score is not None else None
        gain = 100 * (1 - score / baseline) if score is not None else None
        row['percent_better_fixed_M001'] = gain
        row['positive_gain_efficiency_per_worker_minute'] = gain / row['worker_minutes'] if gain is not None and gain > 0 and seconds is not None and seconds > 0 else None
        row['efficiency_status'] = 'unknown_comparable_cost_or_score' if seconds is None or score is None else ('positive_gain' if gain > 0 else 'reference_or_no_gain_not_ranked')
        dominates = [x for x in timed if x['CRPS'] <= score and x['summed_worker_seconds_not_controlled_benchmark'] <= seconds and (x['CRPS'] < score or x['summed_worker_seconds_not_controlled_benchmark'] < seconds)] if seconds is not None and score is not None else []
        row['on_original_measured_score_work_frontier'] = not dominates if seconds is not None and score is not None else None
        row['dominance_witness_model_id'] = min(dominates, key=lambda x: x['summed_worker_seconds_not_controlled_benchmark'])['model_id'] if dominates else None
        results.append(row)
    selected = [r for r in results if r['proposal_include']]
    costs = [r['summed_worker_seconds_not_controlled_benchmark'] for r in selected if r.get('summed_worker_seconds_not_controlled_benchmark') is not None]
    new_families = sum(bool(r.get('new_family_candidate')) for r in selected)
    missing = len(selected) - len(costs) - new_families
    factor = (552 / 52) * (26317 / 11687)
    estimates = {}
    for name, function in [('mean', statistics.mean), ('median', statistics.median)]:
        imputed = function(costs)
        jobs = sorted(costs + [imputed] * missing, reverse=True)
        workers = [0.0] * 12
        for seconds in jobs:
            heapq.heapreplace(workers, workers[0] + seconds * factor / 3600)
        estimates[name] = {'imputed_old52_worker_seconds': imputed, 'total_worker_hours': sum(jobs) * factor / 3600, 'scheduled_wall_hours_at_12_workers': max(workers)}
    approved_costs = [r['summed_worker_seconds_not_controlled_benchmark'] for r in results if r['model_id'] in data['historical19_comparison_ids'] and r.get('summed_worker_seconds_not_controlled_benchmark') is not None]
    return {'original_candidates': 52, 'additional_family_candidates': new_families, 'reviewed_rows_including_required_benchmarks': len(results), 'proposed_methods': len(selected), 'comparable_timing_records_original': len(timed), 'comparable_timing_records_proposed': len(costs), 'unknown_timing_records_proposed': missing + new_families, 'known_worker_seconds_reduction_vs_approved19_percent': 100 * (1 - sum(costs) / sum(approved_costs)), 'runtime_estimate': {'status': 'historical14_subset_only_full_proposal_not_estimated_until_EWMA_profiled', 'unestimated_additional_family_count': new_families, 'workers': 12, 'scaling_factor': factor, 'scheduling': 'longest estimated model jobs first; one worker per model', 'assumption': 'linear series-count and history-length scaling; five missing costs imputed, not measured; contention and nonlinear costs unresolved', 'planning_range_hours': [5, 8], 'estimates': estimates}, 'rows': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / 'src/simfolio_forecasting_methodology/resources/french1926'
    path = folder / 'efficiency_reassessment.json'
    data = json.loads(path.read_text())
    computed = calculate(data)
    assert computed['reviewed_rows_including_required_benchmarks'] == 54 + len(data.get('additional_family_candidates', []))
    assert computed['proposed_methods'] == data['proposed_method_count']
    assert computed['comparable_timing_records_original'] == 49
    assert len(data['historical19_comparison_ids']) == 19
    if args.check:
        assert data['analysis'] == computed
    else:
        data['analysis'] = computed
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        fields = ['label', 'model_id', 'CRPS', 'historical80_crps', 'worker_minutes', 'percent_better_production52', 'percent_better_fixed_M001', 'positive_gain_efficiency_per_worker_minute', 'efficiency_status', 'on_original_measured_score_work_frontier', 'proposal_include', 'proposal_reason']
        with (folder / 'efficiency_reassessment.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            writer.writerows(computed['rows'])
    print(json.dumps({key: value for key, value in computed.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
