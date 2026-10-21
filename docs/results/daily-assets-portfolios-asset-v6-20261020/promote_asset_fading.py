"""Promote only the complete independently verified asset-level full candidate."""
from pathlib import Path
import datetime
import hashlib
import json
import os
O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
C = V/'candidates/asset_fading_vol'
F = C/'full'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    tmp = p.with_suffix(p.suffix+'.new')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(tmp, p)


def main():
    result = read(F/'verification_receipt.json')
    status = read(F/'results/status.json')
    report = read(F/'scoreboard.json')
    baseline = read(V/'full_pair_independent_audit.json')
    assert result['complete'] and result['successful_tasks'] == 64049 and result['series'] == 1264
    assert result['native_failures'] == result['scoring_integrity_errors'] == 0
    assert status['completed_active_tasks'] == status['expected_active_tasks'] == 64049
    assert status['active_workers'] == status['failed_active_tasks'] == status['ready_queue'] == 0
    assert report['scoring_integrity_errors'] == report['native_failures'] == 0
    mid = result['model_id']
    pair = report['paired_comparisons'][mid]
    assert pair['matched_successful_tasks'] == 64049 and not pair['preliminary']
    assert abs(pair['main_score']-result['main_score']) < 2e-12
    ledger = read(O/'goal_progress.json')
    previous_id, previous_score = ledger['current_best_model_id'], ledger['current_best_main_score']
    assert result['main_score'] < previous_score, 'No promotion without improving current verified best'
    reference = baseline['worse_reference_model_id']
    denominator = baseline['models'][reference]['main_score']
    rows = {m: {'main_score': data['main_score'], 'worker_compute_seconds': data['compute_seconds'],
                 'successful_tasks': 64049, 'native_failures': 0, 'scoring_integrity_errors': 0}
            for m, data in baseline['models'].items()}
    rows[mid] = {'main_score': result['main_score'], 'worker_compute_seconds': result['worker_compute_seconds'],
                 'successful_tasks': 64049, 'native_failures': 0, 'scoring_integrity_errors': 0}
    assert len(rows) == 3
    for row in rows.values():
        row['percent_better_vs_fixed_reference'] = 100*(1-row['main_score']/denominator)
    official = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'official_retained_set_only': True, 'series': 1264, 'paths': 1000,
                'same_successful_tasks_per_model': 64049,
                'score_protocol_id': report['score_protocol_id'],
                'sole_reference_model_id': reference, 'current_best_model_id': mid,
                'models': rows, 'independent_validation_or_significance_claim': False,
                'runtime_scope': 'Recorded successful worker compute including input preparation, fitting, simulation and CRPS; no controlled speedup claim',
                'full_candidate_verification_receipt_sha256': sha(F/'verification_receipt.json'),
                'complete_baseline_pair_audit_sha256': sha(V/'full_pair_independent_audit.json'),
                'goal_complete': False}
    write(V/'official_scoreboard.json', official)
    promotion = {'utc': official['utc'], 'previous_best_id': previous_id,
                 'previous_best_main_Q': previous_score, 'new_best_id': mid,
                 'new_best_main_Q': result['main_score'],
                 'percent_better_vs_fixed_worse_reference': rows[mid]['percent_better_vs_fixed_reference'],
                 'full_identical_coverage': 64049, 'native_failures': 0,
                 'scoring_integrity_errors': 0, 'official_scoreboard_sha256': sha(V/'official_scoreboard.json'),
                 'further_credible_tests_remain': True, 'goal_complete': False}
    write(C/'promotion_receipt.json', promotion)
    ledger.update(current_best_model_id=mid, current_best_main_score=result['main_score'],
                  current_best_percent_better_vs_worse_reference=rows[mid]['percent_better_vs_fixed_reference'],
                  official_scoreboard_path=str(V/'official_scoreboard.json'),
                  current_best_promotion_receipt=str(C/'promotion_receipt.json'),
                  actual_goal_tool_status='active', previous_goal_turn_classification='progress')
    write(O/'goal_progress.json', ledger)
    print(json.dumps(promotion), flush=True)


if __name__ == '__main__':
    main()
