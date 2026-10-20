from types import SimpleNamespace
import os
import sqlite3
import numpy as np
import pytest
import common
import sweep as real_api
import native_index as index
import score_main
import reporter


def fixture(tmp_path):
    dates = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-01-07'))
    values = np.array([.01, -.01, .02, -.02, .03, -.01])
    origins = [dict(position=2, max_horizon=3, origin_date='2020-01-03', origin_label='first'),
               dict(position=3, max_horizon=2, origin_date='2020-01-04', origin_label='second')]
    row = {'series_id': 'S', 'asset_class': 'equities', 'family': 'broad', 'panel_branch': 'assets'}
    panel = {'panel_fingerprint': 'toy', 'series': [row]}
    spec = {'factory': 'tests:toy', 'implementation_digest': 'fixed'}
    root = tmp_path/'model'
    identity = {'model_id': 'M', 'panel_fingerprint': 'toy', **spec}
    fp = real_api.original._manifest(root, identity)
    api = SimpleNamespace(original=real_api.original, task_paths=real_api.task_paths,
        task_identity=real_api.task_identity, schedule=lambda root, row: origins,
        load_series=lambda root, row, verify_files=False: (dates, values, origins))
    cp, _ = api.task_paths(root, (0, 0))
    api.original._write_checkpoint(cp, api.task_identity(fp, 'M', 'S', origins[0]), np.array([.1, .2, .3]), {'forecast_and_score_seconds': 0.})
    _, err = api.task_paths(root, (0, 1))
    err.parent.mkdir(parents=True)
    err.write_text('{}')
    return api, panel, root, spec, origins, cp, fp


def test_real_checkpoint_cache_one_score_native_failures_and_byte_drift(tmp_path):
    api, panel, root, spec, origins, cp, fp = fixture(tmp_path)
    cache = tmp_path/'index.sqlite3'
    summary, records = index.collect(tmp_path, panel, root, 'M', cache, api, spec)
    assert summary['completed_tasks'] == 1
    assert summary['native_failures'] == 1 and summary['scoring_integrity_errors'] == 0
    actual = next(iter(records.values()))['score']
    expected = score_main.origin_score([.1, .2, .3], [.01, -.01, .02], '2020-01-03',
        np.array(['2020-01-04', '2020-01-05', '2020-01-06'], dtype='datetime64[D]'))
    assert actual == expected
    again, _ = index.collect(tmp_path, panel, root, 'M', cache, api, spec)
    assert again['processed'] == 0 and again['native_failures'] == 1
    with sqlite3.connect(cache) as con:
        columns = {r[1] for r in con.execute('PRAGMA table_info(task_scores_main)')}
    assert 'main_score' in columns and 'primary_score' not in columns and 'sensitivity_score' not in columns
    stat = cp.stat()
    api.original._write_checkpoint(cp, api.task_identity(fp, 'M', 'S', origins[0]), np.array([.1, .2, .4]), {'forecast_and_score_seconds': 0.})
    os.utime(cp, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    drift, records = index.collect(tmp_path, panel, root, 'M', cache, api, spec)
    assert drift['native_failures'] == 1 and drift['scoring_integrity_errors'] == 1 and not records


def test_manifest_identity_cannot_be_relabelled(tmp_path):
    api, panel, root, spec, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='factory/digest'):
        index.collect(tmp_path, panel, root, 'M', tmp_path/'cache.sqlite3', api,
                      {**spec, 'implementation_digest': 'changed'})


def test_worst_public_reference_and_incomplete_FAST_are_not_mixed_in_leaderboard():
    ids = reporter.PUBLIC_IDS
    metadata = {'S': {'panel_branch': 'assets', 'asset_class': 'equities', 'family': 'broad'}}
    def rows(value):
        return {key: {'series_id': 'S', 'origin_date': date, 'score': value}
                for key, date in [('a', '2000-01-02'), ('b', '2000-04-02')]}
    records = {ids[0]: rows(3.), ids[1]: rows(2.), ids[2]: rows(1.), 'winner': rows(.5),
               'FAST': {'a': rows(2.)['a']}}
    result = reporter.reduce_records(records, metadata, {'a', 'b'}, {mid: mid != 'FAST' for mid in records})
    assert result['reference_model_id'] == ids[0]
    assert not result['reference_selection_preliminary']
    assert result['paired_comparisons']['winner']['percent_better'] == pytest.approx(100*(1-.5/3))
    assert result['paired_comparisons']['FAST']['preliminary']
    assert result['paired_comparisons']['FAST']['matched_successful_tasks'] == 1
    assert 'FAST' not in {r['model_id'] for r in result['complete_coverage_leaderboard']}
