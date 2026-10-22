"""Synthetic declared-weight tests only; no production formula is selected."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

OWN_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('FRENCH551_TEST_HELPERS_ROOT', OWN_ROOT))
sys.path.insert(0, str(OWN_ROOT))
sys.path.insert(0, str(SOURCE / 'tests'))
from test_french_panel_runner import ZeroOracle
from test_french_panel_runner import panel as panel  # noqa: PLC0414 -- pytest fixture re-export

from french_score_sidecar import (
    PUBLIC_REFERENCE,
    SCHEMA,
    ScoreSidecar,
    ScoringPlan,
    training_standard_deviations,
)
from simfolio_forecasting_methodology.french_panel import (
    runner,
    selected,
)
from simfolio_forecasting_methodology.french_panel.sweep import model_directory


@pytest.fixture(autouse=True)
def synthetic_native_identity(monkeypatch):
    monkeypatch.setattr(selected, "source_identity", lambda: {"synthetic": "frozen"})
    monkeypatch.setattr(selected, "implementation_digest", lambda mid, sources=None: "fixture-v1")


def declared_plan(panel, **changes):
    # These intentionally arbitrary weights are fixtures, not a French proposal.
    declaration = {"schema": SCHEMA, "approval_status": "approved", "score_version": "synthetic-v1",
                   "methodology": "synthetic declared normalized weighted loss",
                   "panel_fingerprint": panel[5]["panel_fingerprint"]}
    arrays = {"training_scale": np.array([[1., 2.], [3., 4.]]),
              "horizon_scale": np.array([1., 2., 3., 4.]),
              "horizon_weight": np.array([1., 2., 3., 4.]),
              "series_weight": np.array([2., 1.])}
    arrays.update(changes)
    return ScoringPlan(declaration, arrays, panel[4], panel[5])


def generate(panel, source, mid, max_tasks=None):
    model = ZeroOracle()
    model.model_id = mid
    root = model_directory(source, mid)
    return runner.run_panel(model, panel[0], root, implementation_digest="fixture-v1", max_tasks=max_tasks)


def test_declared_linear_score_matches_independent_cell_first_calculation(panel, tmp_path):
    source = tmp_path / "source"
    raw = generate(panel, source, PUBLIC_REFERENCE)
    root = model_directory(source, PUBLIC_REFERENCE)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan = declared_plan(panel)
    reporter = ScoreSidecar(panel[0], source, tmp_path / "reports", plan)
    try:
        status = reporter.poll()
        row = status["models"][PUBLIC_REFERENCE]
        cells = []
        for i in range(2):
            sums = np.zeros(4)
            counts = np.zeros(4)
            for j, origin in enumerate(panel[4]):
                with np.load(root / "tasks" / f"{i:06d}/{j:03d}.npz") as checkpoint:
                    loss = checkpoint['losses']
                sums[:len(loss)] += loss / plan.arrays['training_scale'][i,j] / plan.arrays['horizon_scale'][:len(loss)]
                counts[:len(loss)] += 1
            cells.append(float(np.average(sums/counts, weights=plan.arrays['horizon_weight'])))
        expected = np.average(cells, weights=plan.arrays['series_weight'])
        assert row['status'] == 'complete'
        assert row['final_score'] == pytest.approx(expected, rel=1e-15)
        assert row['original_raw_crps']['final_score'] == raw['final_score']
        assert status['forecast_runs'] == 0
        assert status['indexed_new_or_changed_origins_this_poll'] == 4
        assert reporter.poll()['indexed_new_or_changed_origins_this_poll'] == 0
        assert all(p.read_bytes() == old for p,old in before.items())
    finally:
        reporter.close()


def test_live_incremental_partial_gating_matching_public_and_restart(panel, tmp_path):
    source, reports = tmp_path / 'source', tmp_path / 'reports'
    generate(panel, source, PUBLIC_REFERENCE, max_tasks=2)
    generate(panel, source, 'candidate', max_tasks=1)
    plan = declared_plan(panel)
    reporter = ScoreSidecar(panel[0], source, reports, plan)
    first = reporter.poll()
    assert first['models']['candidate']['partial_complete_series_score'] is None
    assert first['models']['candidate']['public_reference_comparison']['matched_complete_series_ids'] == []
    generate(panel, source, 'candidate', max_tasks=1)
    second = reporter.poll()
    row = second['models']['candidate']
    assert row['complete_series_ids'] == ['s0']
    assert row['final_score'] is None
    assert row['public_reference_comparison']['matched_complete_series_ids'] == ['s0']
    assert row['public_reference_comparison']['percent_better'] == 0
    assert second['indexed_new_or_changed_origins_this_poll'] == 1
    reporter.close()
    restarted = ScoreSidecar(panel[0], source, reports, plan)
    try:
        assert restarted.poll()['indexed_new_or_changed_origins_this_poll'] == 0
        assert restarted.db.execute('SELECT COUNT(*) FROM origins').fetchone()[0] == 4
        generate(panel, source, 'candidate')
        third = restarted.poll()
        assert third['models']['candidate']['status'] == 'complete'
        assert third['models']['candidate']['public_reference_comparison']['matched_complete_series_ids'] == ['s0']
    finally:
        restarted.close()


def test_plan_scale_invariance_and_no_arbitrary_zero_variance_floor(panel):
    plan = declared_plan(panel)
    losses = np.array([.1,.2,.3,.4])
    changed = declared_plan(panel, training_scale=plan.arrays['training_scale']*7.5)
    assert changed.contribution(0,0,losses*7.5) == pytest.approx(plan.contribution(0,0,losses), rel=1e-15)
    with pytest.raises(ValueError, match='strictly positive'):
        declared_plan(panel, training_scale=np.zeros((2,2)))
    with pytest.raises(ValueError, match='no arbitrary epsilon'):
        training_standard_deviations(np.ones((2,8)), panel[4], ddof=0)
    scales = training_standard_deviations(panel[2], panel[4], ddof=1)
    changed_values = panel[2].copy()
    changed_values[:,2:] *= 1000
    changed_scales = training_standard_deviations(changed_values, panel[4], ddof=1)
    np.testing.assert_array_equal(scales[:,0], changed_scales[:,0])
    assert not np.array_equal(scales[:,1], changed_scales[:,1])


def test_version_cache_and_array_artifact_identities_fail_closed(panel, tmp_path):
    source = tmp_path / 'source'
    generate(panel, source, 'candidate', max_tasks=1)
    plan = declared_plan(panel)
    reporter = ScoreSidecar(panel[0], source, tmp_path / 'reports', plan)
    reporter.poll()
    reporter.close()
    changed = declared_plan(panel, horizon_weight=np.ones(4))
    with pytest.raises(ValueError, match='cache identity differs'):
        ScoreSidecar(panel[0], source, tmp_path / 'reports', changed)
    declaration = dict(plan.declaration, approval_status='pending')
    with pytest.raises(ValueError, match='explicitly approved'):
        ScoringPlan(declaration, plan.arrays, panel[4], panel[5])
    arrays_path = tmp_path / 'arrays.npz'
    np.savez(arrays_path, **plan.arrays)
    declaration = {**plan.declaration, 'arrays_file':'arrays.npz',
                   'arrays_file_sha256':hashlib.sha256(arrays_path.read_bytes()).hexdigest()}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(declaration))
    loaded = ScoringPlan.load(path, panel[4], panel[5])
    assert loaded.arrays.keys() == plan.arrays.keys()
    arrays_path.write_bytes(arrays_path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='arrays checksum'):
        ScoringPlan.load(path, panel[4], panel[5])


def test_checkpoint_mutation_source_drift_and_deleted_series_fail_closed(panel, tmp_path):
    source = tmp_path / 'source'
    generate(panel, source, 'candidate')
    reporter = ScoreSidecar(panel[0], source, tmp_path / 'reports', declared_plan(panel), audit_seconds=0)
    root = model_directory(source, 'candidate')
    try:
        assert reporter.poll()['models']['candidate']['final_score'] is not None
        checkpoint = root / 'tasks/000000/000.npz'
        checkpoint.write_bytes(checkpoint.read_bytes()[:-3])
        status = reporter.poll()
        row = status['models']['candidate']
        assert row['status'] == 'failed_integrity'
        assert row['final_score'] is None
        assert row['partial_complete_series_score'] is None
        assert 'checkpoint changed' in row['error']
    finally:
        reporter.close()
    # A fresh reporter must validate the original per-file checksum too.
    clean_source = tmp_path / 'clean_source'
    generate(panel, clean_source, 'second')
    clean = ScoreSidecar(panel[0], clean_source, tmp_path / 'clean_reports', declared_plan(panel))
    try:
        clean.poll()
        directory = model_directory(clean_source,'second') / 'tasks/000000'
        for p in directory.iterdir(): p.unlink()
        directory.rmdir()
        status = clean.poll()
        assert status['models']['second']['status'] == 'failed_integrity'
        assert status['models']['second']['final_score'] is None
    finally:
        clean.close()


def test_invalid_origin_never_enters_complete_series(panel, tmp_path):
    source = tmp_path/'source'
    generate(panel, source, 'candidate')
    root = model_directory(source,'candidate')
    checkpoint = root/'tasks/000000/001.npz'
    checkpoint.write_bytes(b'invalid')
    reporter = ScoreSidecar(panel[0], source, tmp_path/'reports', declared_plan(panel))
    try:
        result = reporter.poll()
        assert result['models']['candidate']['status'] == 'failed_integrity'
        assert result['models']['candidate']['final_score'] is None
    finally:
        reporter.close()


def test_approved_plan_builder_and_origin_specific_reference_formula(panel, tmp_path):
    from build_scoring_plans import PRIMARY, SENSITIVITY, build
    source = tmp_path/'source'
    generate(panel, source, PUBLIC_REFERENCE)
    manifests = build(panel[0], tmp_path/'plans')
    assert set(manifests) == {PRIMARY, SENSITIVITY}
    primary = ScoringPlan.load(manifests[PRIMARY]['plan_path'], panel[4], panel[5])
    sensitivity = ScoringPlan.load(manifests[SENSITIVITY]['plan_path'], panel[4], panel[5])
    assert primary.declaration['formula']['origin_loss'] == 'rawCRPS_ioh / (s_io * sqrt(h + h*h/n_io))'
    assert primary.declaration['scale_information_set'] == 'training_observations_through_origin_only'
    h = np.arange(1,5,dtype=float)
    a = primary.arrays['origin_horizon_scale']
    np.testing.assert_array_equal(a[0],np.sqrt(h+h*h/2))
    np.testing.assert_array_equal(a[1],np.sqrt(h+h*h/6))
    # At the same horizon and daily sigma, actual n changes the primary scale.
    assert a[0,0] != a[1,0]
    np.testing.assert_array_equal(primary.arrays['training_scale'], training_standard_deviations(panel[2], panel[4], ddof=1))
    np.testing.assert_array_equal(primary.arrays['horizon_weight'], np.log1p(1/h))
    assert primary.horizon_total == np.log1p(4)
    for plan in (primary,sensitivity):
        reporter = ScoreSidecar(panel[0],source,tmp_path/plan.declaration['score_version'],plan)
        try:
            row = reporter.poll()['models'][PUBLIC_REFERENCE]
            per_series=[]
            for i in range(2):
                sums=np.zeros(4); counts=np.zeros(4,dtype=np.int64)
                for j,o in enumerate(panel[4]):
                    with np.load(model_directory(source,PUBLIC_REFERENCE)/'tasks'/f'{i:06d}/{j:03d}.npz') as saved:
                        loss=saved['losses']
                    horizon=np.arange(1,len(loss)+1,dtype=float)
                    sigma=np.std(panel[2][i,:o.position+1],ddof=1)
                    if plan is primary:
                        divisor=sigma*np.sqrt(horizon+horizon*horizon/(o.position+1))
                    else:
                        divisor=sigma*np.sqrt(horizon)
                    sums[:len(loss)] += loss/divisor
                    counts[:len(loss)] += 1
                per_series.append(float(np.sum((sums/counts)*np.log1p(1/h)))/np.log1p(4))
            assert row['final_score'] == sum(per_series)/2
            assert row['derived_score_fingerprint']
            assert row['original_raw_crps']['final_score'] is not None
        finally:
            reporter.close()
    with pytest.raises(ValueError,match='already exist'):
        build(panel[0],tmp_path/'plans')


def test_native_source_implementation_digest_drift_fails_closed(panel,tmp_path):
    source=tmp_path/'source'
    generate(panel,source,'candidate')
    root=model_directory(source,'candidate')
    manifest=json.loads((root/'execution.json').read_text())
    manifest['identity']['implementation_digest']='different-native-source'
    manifest['fingerprint']=runner.stable_digest(manifest['identity'])
    (root/'execution.json').write_text(json.dumps(manifest))
    reporter=ScoreSidecar(panel[0],source,tmp_path/'reports',declared_plan(panel))
    try:
        result=reporter.poll()
        assert any('implementation digest differs' in error for error in result['integrity_errors'].values())
        assert not result['models']
    finally:
        reporter.close()
