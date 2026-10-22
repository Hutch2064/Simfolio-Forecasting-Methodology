"""One completed V9-panel reconciliation; no IO, publication or scientific work."""
from copy import deepcopy
import catalogue

PROTOCOL = 'daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters'
ENDPOINT = 'model_testing_asset_information_endpoint_components_v1_1a5c92ef4802021d'
WR = 'model_testing_asset_joint_blocks_replacement_v6'
WOR = 'model_testing_asset_joint_blocks_history_partition_v6'
PATH_COUNTS = {'1000': 59906, '240': 260111}
REPORT = '../daily-assets-portfolios-asset-v6-20261020/expanded-v9-20261022/README.md'


def reconcile(ledger, records, snapshot, recovery_receipt):
    """Return a new ledger only after exact coverage and recovery bindings pass.

    Records must already contain the owner's actual archive/manifest hashes.
    The recovery receipt uses catalogue.retirement_allowed's explicit verified
    archive schema; no upload-only or unbound boolean grants admission.
    """
    if (not snapshot['complete'] or not snapshot['same_three_model_coverage']
            or snapshot['score_protocol_id'] != PROTOCOL
            or snapshot['paired_requests_per_model'] != 320017
            or snapshot['requests_by_final_paths'] != PATH_COUNTS
            or snapshot['fixed_reference'] != WR):
        raise ValueError('Exact immutable completed V9 snapshot required')
    expected = {ENDPOINT: snapshot['final_scores']['endpoint'],
                WR: snapshot['final_scores'][WR], WOR: snapshot['final_scores'][WOR]}
    if len(records) != 3 or {r['identity']['model_id'] for r in records} != set(expected):
        raise ValueError('Exactly endpoint and the two existing baseline executions required')
    panels = set()
    for record in records:
        catalogue.validate(record)
        coverage = record['coverage']; panels.add(coverage['panel_fingerprint'])
        if (record['score_protocol_id'] != PROTOCOL or coverage['successful_tasks'] != 320017
                or coverage['native_failed_tasks'] or coverage['requests_by_final_paths'] != PATH_COUNTS
                or record['scores']['main'] != expected[record['identity']['model_id']]
                or record['comparison']['public_scores']['main'] != expected[WR]
                or record['comparison']['percent_better']['main'] != 100*(1-record['scores']['main']/expected[WR])):
            raise ValueError('Record differs from complete matched V9 evidence')
        if not catalogue.retirement_allowed(record, recovery_receipt):
            raise ValueError('Owner archive/manifest hashes and full downloaded disk recovery required')
    if len(panels) != 1:
        raise ValueError('All three executions must bind the same expanded benchmark')
    updated = deepcopy(ledger)
    # Earlier current metadata becomes dated history; original executions never change.
    panel = next(iter(panels))
    if (updated.get('current_score_protocol_id'), updated.get('current_panel_fingerprint')) != (PROTOCOL, panel):
        updated.setdefault('historical_status_snapshots', []).append({
            'recorded_context': 'Superseded benchmark/current-best metadata; original results retained',
            'status': {k: deepcopy(ledger.get(k)) for k in (
                'current_score_protocol_id', 'current_panel_fingerprint', 'current_best_model_id',
                'current_best_report', 'active_model_ids', 'current_benchmark_status', 'counting_rule')},
            'superseded_by': REPORT})
    for record in records:
        catalogue.append(updated, deepcopy(record))
    if (updated['unique_full_panel_models'], updated['full_panel_executions'],
            updated['failure_free_models']) != (24, 31, 22):
        raise ValueError('Expected exact23/28 to24/31 logical-method reconciliation')
    updated.update(current_score_protocol_id=PROTOCOL, current_panel_fingerprint=panel,
        current_best_model_id=ENDPOINT, current_best_report=REPORT,
        active_model_ids=[WOR, WR, ENDPOINT], updated_utc=snapshot['utc'],
        scope='Every completed unique full-panel financial method remains preserved, including losers and earlier benchmarks. Current completed expanded V9 evidence is adaptive research; no production behavior changes.')
    updated['counting_rule'] = ('Canonical original model ID plus canonical source-bound financial-method version. Identical reruns, rescoring and verified optimization aliases do not create another unique method; actual execution source digests and numerical-qualification differences remain preserved. Completed expanded executions append evidence without rewriting historical results.')
    updated['current_benchmark_status'] = {
        'status': 'complete_independently_verified_and_private_recovery_verified',
        'requests_per_model': 320017, 'model_executions': 3, 'logical_forecast_requests': 960051,
        'requests_by_final_paths_per_model': PATH_COUNTS.copy(), 'same_count_within_every_pair': True,
        'fixed_reference_model_id': WR, 'best_completed_model_id': ENDPOINT,
        'main_Q_by_model': expected, 'percent_better_than_fixed_WR': 100*(1-expected[ENDPOINT]/expected[WR]),
        'score_protocol_id': PROTOCOL, 'posthoc_score_selection': True,
        'independent_validation_or_significance_claim': False,
        'quarter_weights': False, 'research_development_candidates_counted': False,
        'source_report': REPORT}
    return updated
