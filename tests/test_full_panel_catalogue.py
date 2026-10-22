"""The full-panel inventory must not inflate counts or certify missing recovery."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_panel_catalogue', ROOT / 'tools/full_panel_catalogue/catalogue.py')
catalogue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalogue)
LEDGER = ROOT / 'docs/results/daily-cross-asset-722/catalogue.json'


def example():
    return copy.deepcopy(next(iter(json.loads(LEDGER.read_text())['models'].values()))['executions'][0])


def test_identical_rerun_does_not_increment_model_count():
    ledger = {}
    record = example()
    assert catalogue.append(ledger, record)
    assert not catalogue.append(ledger, copy.deepcopy(record))
    rerun = copy.deepcopy(record)
    rerun['execution_fingerprint'] = 'fresh-execution-seed'
    assert catalogue.append(ledger, rerun)
    assert ledger['unique_full_panel_models'] == 1
    assert ledger['full_panel_executions'] == 2


def test_new_implementation_under_same_name_counts_as_new_version():
    ledger = {}
    record = example()
    catalogue.append(ledger, record)
    record = copy.deepcopy(record)
    record['identity']['implementation_digest'] = 'a' * 64
    record['execution_fingerprint'] = 'new-implementation'
    catalogue.append(ledger, record)
    assert ledger['unique_full_panel_models'] == 2


def test_pilot_or_unattempted_tasks_cannot_enter_full_panel_count():
    record = example()
    record['coverage']['successful_tasks'] -= 1
    with pytest.raises(ValueError, match='entire panel'):
        catalogue.append({}, record)


def test_failed_full_attempt_and_removed_model_are_preserved():
    record = example()
    record['retained_in_active_set'] = False
    record['coverage']['successful_tasks'] -= 17
    record['coverage']['native_failed_tasks'] += 17
    record['comparison']['matched_successful_tasks'] -= 17
    ledger = {}
    catalogue.append(ledger, record)
    assert ledger['unique_full_panel_models'] == 1
    assert ledger['failure_free_models'] == 0


def test_mismatched_comparison_or_changed_execution_is_rejected():
    record = example()
    ledger = {}
    catalogue.append(ledger, record)
    changed = copy.deepcopy(record)
    changed['scores']['primary'] *= 0.9
    with pytest.raises(ValueError, match='evidence changed'):
        catalogue.append(ledger, changed)
    changed['comparison']['matched_successful_tasks'] -= 1
    with pytest.raises(ValueError, match='identical successful'):
        catalogue.append({}, changed)


def test_upload_or_server_digest_is_not_full_recovery():
    record = example()
    receipt = {'archives': [{'archive_sha256': a['archive_sha256'],
        'manifest_sha256': a['manifest_sha256'], 'archive_server_digest_verified': True}
        for a in record['archives']]}
    assert not catalogue.retirement_allowed(record, receipt)
    for a in receipt['archives']:
        a['remote_download_sha256_verified'] = True
        a['all_files_disk_restored_and_sha256_verified'] = True
    assert catalogue.retirement_allowed(record, receipt)
    receipt['archives'][0]['manifest_sha256'] = '0' * 64
    assert not catalogue.retirement_allowed(record, receipt)


def test_backfill_counts_native_coverage_and_both_exact_profiles():
    ledger = json.loads(LEDGER.read_text())
    assert ledger['unique_full_panel_models'] == len(ledger['models']) >= 13
    assert ledger['failure_free_models'] >= 12
    historical = set()
    for model in ledger['models'].values():
        for record in model['executions']:
            catalogue.validate(record)
            if record.get('score_protocol_id') in catalogue.SINGLE_SCORE_PROTOCOLS:
                continue
            historical.add(catalogue.version_key(record['identity']))
            assert record['coverage']['expected_tasks'] == 36813
            for profile in ('primary', 'sensitivity'):
                assert record['comparison']['percent_better'][profile] == 100 * (
                    1 - record['scores'][profile] / record['comparison']['public_scores'][profile])
            for filename in ('native_summary', 'numerical_contract'):
                assert (LEDGER.parent / record['reproducibility'][filename]).is_file()
            reproducibility = record['reproducibility']
            for filename, digest_key in [('native_summary', 'native_summary_sha256'),
                    ('exact_score_source_snapshot', 'exact_score_source_sha256')]:
                assert hashlib.sha256((LEDGER.parent / reproducibility[filename]).read_bytes()).hexdigest() == reproducibility[digest_key]
    assert len(historical) == 13


def test_frozen_scoring_and_remote_archives_are_bound_to_catalogue():
    ledger = json.loads(LEDGER.read_text())
    receipt = json.loads((LEDGER.parent / 'remote-verification.json').read_text())
    remote = {a['archive_sha256']: a for a in receipt['archives']}
    scorer_hash = hashlib.sha256((LEDGER.parent / 'harness/scoring.py').read_bytes()).hexdigest()
    for model in ledger['models'].values():
        for record in model['executions']:
            if record.get('score_protocol_id') in catalogue.SINGLE_SCORE_PROTOCOLS:
                continue
            assert record['reproducibility']['scorer_sha256'] == scorer_hash
            for archive in record['archives']:
                assert remote[archive['archive_sha256']]['manifest_sha256'] == archive['manifest_sha256']
                assert remote[archive['archive_sha256']]['archive_server_digest_verified']
                assert remote[archive['archive_sha256']]['manifest_fresh_download_sha256_verified']
                assert archive['prior_files_disk_restored_and_sha256_verified'] > 0


def test_incumbent_replacement_preserves_previous_full_execution():
    ledger = json.loads(LEDGER.read_text())
    active = set(ledger['active_model_ids'])
    required = {'model_testing_asset_joint_blocks_replacement_v6',
                'model_testing_asset_joint_blocks_history_partition_v6'}
    assert required <= active and len(active) in (2, 3)
    if ledger['current_best_model_id'] is None:
        assert active == required  # New references await their first full result.
    else:
        assert ledger['current_best_model_id'] in active
    previous = 'model_testing_equal_bootstrap_fhs_stratified_matched_Gaussian_mean_pool_v1'
    assert previous not in active
    assert any(m['identity']['model_id'] == previous for m in ledger['models'].values())
    fast = 'R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION'
    assert fast not in active
    assert any(m['identity']['model_id'] == fast and m['executions'] for m in ledger['models'].values())


def test_new_benchmark_has_one_main_score_and_does_not_inflate_unique_model_count():
    record = example()
    ledger = {}
    catalogue.append(ledger, record)
    updated = copy.deepcopy(record)
    updated['score_protocol_id'] = 'daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch'
    updated['execution_fingerprint'] = 'new-panel-execution'
    updated['coverage']['panel_fingerprint'] = 'new-panel'
    updated['scores'] = {'main': .5}
    updated['comparison']['public_scores'] = {'main': .6}
    updated['comparison']['percent_better'] = {'main': 100*(1-.5/.6)}
    assert catalogue.append(ledger, updated)
    assert ledger['unique_full_panel_models'] == 1 and ledger['full_panel_executions'] == 2
    ledger['current_score_protocol_id'] = updated['score_protocol_id']
    ledger['current_panel_fingerprint'] = 'new-panel'
    rendered = catalogue.render(ledger)
    assert 'Main Q' in rendered and 'Sensitivity Q' not in rendered and 'Primary Q' not in rendered
    updated['scores']['sensitivity'] = .5
    with pytest.raises(ValueError, match='exactly one main'):
        catalogue.validate(updated)


def v9_example():
    record = example()
    record['score_protocol_id'] = 'daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters'
    record['scores'] = {'main': .3732134377688732}
    record['comparison'].update(reference_model_id='model_testing_asset_joint_blocks_replacement_v6',
        public_scores={'main': .5008534683031683}, matched_successful_tasks=320017)
    record['coverage'].update(expected_tasks=320017, successful_tasks=320017,
        native_failed_tasks=0, requests_by_final_paths={'1000': 59906, '240': 260111},
        same_count_within_every_pair=True)
    record['runtime'] = {'forecast_and_score_worker_seconds': None,
                         'scope': 'Not reconstructed across mixed generations; no speed comparison.'}
    record['reproducibility']['posthoc_score_selection'] = True
    return record


def test_v9_complete_mixed_paths_fixed_reference_and_disclosure():
    record = v9_example()
    catalogue.validate(record)
    for field, value, message in [('same_count_within_every_pair', False, 'matched mixed-path'),
                                  ('requests_by_final_paths', {'1000': 59906, '240': 1}, 'mixed-path')]:
        bad = copy.deepcopy(record)
        bad['coverage'][field] = value
        with pytest.raises(ValueError, match=message):
            catalogue.validate(bad)
    bad = copy.deepcopy(record)
    bad['comparison']['reference_model_id'] = 'model_testing_asset_joint_blocks_history_partition_v6'
    with pytest.raises(ValueError, match='fixed WR'):
        catalogue.validate(bad)
    bad = copy.deepcopy(record)
    bad['reproducibility']['posthoc_score_selection'] = False
    with pytest.raises(ValueError, match='retrospective'):
        catalogue.validate(bad)


def test_verified_optimization_alias_keeps_logical_identity_execution_changes():
    original = v9_example()
    ledger = {}
    catalogue.append(ledger, original)
    optimized = copy.deepcopy(original)
    optimized['execution_fingerprint'] = 'optimized-fresh-panel'
    optimized['reproducibility']['actual_implementation_digest'] = 'e' * 64
    optimized['reproducibility']['equivalence_receipt_sha256'] = 'f' * 64
    catalogue.append(ledger, optimized)
    assert ledger['unique_full_panel_models'] == 1 and ledger['full_panel_executions'] == 2
    before = copy.deepcopy(ledger)
    assert not catalogue.append(ledger, optimized)
    assert ledger == before


def test_v9_render_uses_current_panel_and_marks_unmeasured_runtime():
    record = v9_example()
    ledger = {}
    catalogue.append(ledger, record)
    ledger.update(current_score_protocol_id=record['score_protocol_id'],
                  current_panel_fingerprint=record['coverage']['panel_fingerprint'])
    rendered = catalogue.render(ledger)
    assert 'Main Q' in rendered and 'Sensitivity Q' not in rendered
    assert 'not reconstructed' in rendered
    assert '0.373213437769' in rendered


def test_prepared_expanded_records_cannot_enter_without_archive_evidence():
    root = ROOT / 'docs/results/daily-assets-portfolios-asset-v6-20261020/expanded-v9-20261022'
    records = json.loads((root / 'execution_records.pending.json').read_text())
    assert len(records) == 3
    assert all(r['coverage']['expected_tasks'] == 320017 for r in records)
    assert len({r['identity']['model_id'] for r in records}) == 3
    for record in records:
        with pytest.raises(ValueError, match='complete evidence'):
            catalogue.append({}, record)


def historical_base_ledger():
    """Test the same transition before and after the real ledger is admitted."""
    ledger = json.loads(LEDGER.read_text())
    protocol = 'daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters'
    for key, model in list(ledger['models'].items()):
        model['executions'] = [e for e in model['executions'] if e.get('score_protocol_id') != protocol]
        if not model['executions']:
            del ledger['models'][key]
    ledger['unique_full_panel_models'] = len(ledger['models'])
    ledger['full_panel_executions'] = sum(len(m['executions']) for m in ledger['models'].values())
    ledger['failure_free_models'] = sum(any(e['coverage']['native_failed_tasks'] == 0
        for e in m['executions']) for m in ledger['models'].values())
    if ledger.get('current_score_protocol_id') == protocol:
        snapshots = ledger['historical_status_snapshots']
        previous = next(x for x in reversed(snapshots)
                        if x.get('status', {}).get('current_score_protocol_id'))
        ledger.update(copy.deepcopy(previous['status']))
        snapshots.remove(previous)
    return ledger


def test_expanded_admission_would_add_one_logical_method_three_executions():
    root = ROOT / 'docs/results/daily-assets-portfolios-asset-v6-20261020/expanded-v9-20261022'
    ledger = historical_base_ledger()
    before = copy.deepcopy(ledger['models'])
    records = json.loads((root / 'execution_records.pending.json').read_text())
    for record in records:
        # Synthetic archive fixture only exercises catalogue counting; this is
        # never written into the real ledger or presented as remote recovery.
        record['archives'] = copy.deepcopy(example()['archives'])
        catalogue.append(ledger, record)
    assert ledger['unique_full_panel_models'] == 24
    assert ledger['full_panel_executions'] == 31
    assert ledger['failure_free_models'] == 22
    for key, model in before.items():
        assert ledger['models'][key]['identity'] == model['identity']
        assert ledger['models'][key]['executions'][:len(model['executions'])] == model['executions']


def expanded_reconciliation_fixture():
    import sys
    sys.path.insert(0, str(ROOT / 'tools/full_panel_catalogue'))
    import expanded_v9
    root = ROOT / 'docs/results/daily-assets-portfolios-asset-v6-20261020/expanded-v9-20261022'
    records = json.loads((root / 'execution_records.pending.json').read_text())
    # This synthetic recovery fixture is never materialized as real admission.
    archive = {'archive_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64}
    for record in records:
        record['archives'] = [copy.deepcopy(archive)]
        record.pop('publication_admission', None)
    recovery = {'archives': [{**archive, 'remote_download_sha256_verified': True,
                             'all_files_disk_restored_and_sha256_verified': True}]}
    snapshot = json.loads((root / 'series_buckets_v9_live/final_scoreboard_snapshot_v9.json').read_text())
    return expanded_v9, records, recovery, snapshot


def test_expanded_reconciliation_preserves_history_corrects_identity_and_is_idempotent():
    module, records, recovery, snapshot = expanded_reconciliation_fixture()
    ledger = historical_base_ledger()
    original = copy.deepcopy(ledger)
    result = module.reconcile(ledger, records, snapshot, recovery)
    assert ledger == original
    assert result['unique_full_panel_models'] == 24 and result['full_panel_executions'] == 31
    assert result['current_best_model_id'] == module.ENDPOINT
    assert result['current_benchmark_status']['fixed_reference_model_id'] == module.WR
    assert result['current_benchmark_status']['percent_better_than_fixed_WR'] == 25.484505671234402
    assert len(result['active_model_ids']) == 3
    assert all('PPW' not in model for model in result['active_model_ids'])
    assert module.reconcile(result, records, snapshot, recovery) == result
    for key, model in original['models'].items():
        assert result['models'][key]['executions'][:len(model['executions'])] == model['executions']


def test_expanded_reconciliation_rejects_upload_only_wrong_manifest_and_subset_score():
    module, records, recovery, snapshot = expanded_reconciliation_fixture()
    ledger = historical_base_ledger()
    for key, value in [('remote_download_sha256_verified', False),
                       ('all_files_disk_restored_and_sha256_verified', False),
                       ('manifest_sha256', 'c' * 64)]:
        bad = copy.deepcopy(recovery); bad['archives'][0][key] = value
        with pytest.raises(ValueError, match='Owner archive'):
            module.reconcile(ledger, records, snapshot, bad)
    bad = copy.deepcopy(records); bad[0]['scores']['main'] += .01
    with pytest.raises(ValueError, match='differs'):
        module.reconcile(ledger, bad, snapshot, recovery)
    assert ledger['unique_full_panel_models'] == 23 and ledger['full_panel_executions'] == 28


def test_remote_archive_metadata_reads_assets_after_the_first_hundred(monkeypatch):
    spec = importlib.util.spec_from_file_location('catalogue_remote', ROOT / 'tools/full_panel_catalogue/verify_remote.py')
    remote = importlib.util.module_from_spec(spec); spec.loader.exec_module(remote)
    payload = b'{"entries": []}'
    archive = {'repository': 'owner/private', 'release_tag': 'tag', 'archive_name': 'target.tar.gz',
               'archive_sha256': 'a' * 64, 'manifest_asset': 'target-manifest.json',
               'manifest_sha256': hashlib.sha256(payload).hexdigest(),
               'prior_files_disk_restored_and_sha256_verified': 1}
    calls = []
    def fetch(url, token, **kwargs):
        calls.append(url)
        if '/assets?per_page=100&page=1' in url:
            return json.dumps([{'name': str(i), 'id': i} for i in range(100)]).encode()
        if '/assets?per_page=100&page=2' in url:
            return json.dumps([{'name': 'target.tar.gz', 'id': 101, 'digest': 'sha256:'+'a'*64},
                               {'name': 'target-manifest.json', 'id': 102, 'url': 'https://manifest'}]).encode()
        if url.startswith('https://manifest'):
            return payload
        return json.dumps([{'id': 1, 'tag_name': 'tag', 'html_url': 'https://release'}]).encode()
    monkeypatch.setattr(remote, 'fetch', fetch)
    ledger = {'models': {'method': {'executions': [{'archives': [archive]}]}}}
    result = remote.verify(ledger, 'synthetic-token')
    assert result['archives'][0]['archive_asset_id'] == 101
    assert any('assets?per_page=100&page=2' in url for url in calls)


def test_admitted_v9_archive_references_keep_remote_verifier_contract():
    root = ROOT / 'docs/results/daily-assets-portfolios-asset-v6-20261020/expanded-v9-20261022'
    path = root / 'execution_records.json'
    if not path.exists():
        pytest.skip('Real owner archive admission has not been applied')
    for record in json.loads(path.read_text()):
        assert len(record['archives']) == 132
        for archive in record['archives']:
            assert all(k in archive for k in ('repository', 'release_tag', 'archive_name',
                'archive_sha256', 'manifest_asset', 'manifest_sha256',
                'prior_files_disk_restored_and_sha256_verified'))
