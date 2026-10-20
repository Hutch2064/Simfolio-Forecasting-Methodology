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
            if record.get('score_protocol_id') == 'daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch':
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
            if record.get('score_protocol_id') == 'daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch':
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
    assert len(active) == 5 and ledger['current_best_model_id'] in active
    previous = 'model_testing_equal_bootstrap_fhs_stratified_matched_Gaussian_mean_pool_v1'
    assert previous not in active
    assert any(m['identity']['model_id'] == previous for m in ledger['models'].values())
    assert 'R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION' in active


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
