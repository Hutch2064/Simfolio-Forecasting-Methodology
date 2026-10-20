"""External admission/report scope tests; frozen scoring mathematics unchanged."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import french_score_sidecar as frozen
from french_score_sidecar_r2 import (
    REGISTRY_SCHEMA,
    ReportingSidecarR2,
    load_external_registry,
    load_model_catalogue,
    run_reporting,
)
from simfolio_forecasting_methodology.french_panel import selected
from simfolio_forecasting_methodology.french_panel.sweep import exclusive_lock
from test_french_score_sidecar import (
    PUBLIC_REFERENCE,
    ZeroOracle,
    declared_plan,
    model_directory,
    runner,
)
from test_french_score_sidecar import panel as panel  # noqa: PLC0414 -- pytest fixture re-export
from test_french_score_sidecar import (
    synthetic_native_identity as synthetic_native_identity,  # noqa: PLC0414 -- pytest fixture re-export
)

EXTERNAL='external-synthetic-comparator'
DISABLED='asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced'
LEGACY_DIGEST='a'*64

@pytest.fixture(autouse=True)
def strict_control_identity(monkeypatch,synthetic_native_identity):
    monkeypatch.setattr(selected,'implementation_digest',lambda mid,sources=None:LEGACY_DIGEST)

def generate(panel,source,mid):
    model=ZeroOracle();model.model_id=mid
    return runner.run_panel(model,panel[0],model_directory(source,mid),implementation_digest=LEGACY_DIGEST)


def registry(tmp_path,mid=EXTERNAL):
    code=tmp_path/'code'
    (code/'scripts').mkdir(parents=True,exist_ok=True)
    module=code/'scripts/external_test.py'
    module.write_text('raise RuntimeError("reporter must never import external factory")\n')
    identity={'schema':'french-external-model-implementation-v1','model_id':mid,
              'factory':'external_test:build_model',
              'source_files':{'scripts/external_test.py':frozen.sha256(module)},
              'numerical_contract':{'method':mid,'source_sha256':frozen.sha256(module),'law':{'replacement':True}}}
    record={'factory':identity['factory'],'implementation_digest':runner.stable_digest(identity),'identity':identity}
    path=tmp_path/'registry.json'
    path.write_text(json.dumps({'schema':REGISTRY_SCHEMA,'source_root':str(code),'models':{mid:record}}))
    return path,record,module


def write_control(path,external=None,disabled=False,unknown=False):
    specs={PUBLIC_REFERENCE:{'enabled':True,'factory':selected.FACTORIES[PUBLIC_REFERENCE],'implementation_digest':LEGACY_DIGEST}}
    if external:
        specs[EXTERNAL]={'enabled':True,'factory':external['factory'],'implementation_digest':external['implementation_digest']}
    if disabled:
        specs[DISABLED]={'enabled':False,'factory':selected.FACTORIES[DISABLED],'implementation_digest':LEGACY_DIGEST}
    if unknown:
        specs['unapproved-model']={'enabled':True,'factory':'unapproved:build','implementation_digest':'a'*64}
    path.write_text(json.dumps({'workers':12,'stop':False,'models':specs}))


def generate_external(panel,source,record,max_tasks=None):
    model=ZeroOracle();model.model_id=EXTERNAL
    model.dependency_identity=record['identity']['numerical_contract']
    previous=runner._dependency_identity
    runner._dependency_identity=lambda model,*args:{'runtime':'synthetic',**getattr(model,'dependency_identity',{})}
    try:
        return runner.run_panel(model,panel[0],model_directory(source,EXTERNAL),
            implementation_digest=record['implementation_digest'],max_tasks=max_tasks)
    finally:runner._dependency_identity=previous


def test_external_allowlist_preserves_old_identity_and_scores_new_model_from_zero(panel,tmp_path):
    source=tmp_path/'source';reports=tmp_path/'reports';control=tmp_path/'control.json'
    path,record,_module=registry(tmp_path)
    raw=generate(panel,source,PUBLIC_REFERENCE)
    original={p:p.read_bytes() for p in model_directory(source,PUBLIC_REFERENCE).rglob('*') if p.is_file()}
    write_control(control,record)
    reporter=ReportingSidecarR2(panel[0],source,reports,declared_plan(panel),control_path=control,registry_path=path)
    try:
        first=reporter.poll()
        assert first['models'][EXTERNAL]['status']=='waiting_for_execution_manifest'
        assert first['models'][EXTERNAL]['completed_origins']==0
        assert first['enabled_model_count']==2 and first['expected_active_origins']==8
        assert first['models'][PUBLIC_REFERENCE]['original_raw_crps']['final_score']==raw['final_score']
        generate_external(panel,source,record,max_tasks=1)
        second=reporter.poll()
        assert second['models'][EXTERNAL]['completed_origins']==1
        assert second['models'][EXTERNAL]['final_score'] is None
        generate_external(panel,source,record)
        final=reporter.poll()
        assert final['models'][EXTERNAL]['final_score']==final['models'][PUBLIC_REFERENCE]['final_score']
        assert final['models'][EXTERNAL]['execution_fingerprint']!=final['models'][PUBLIC_REFERENCE]['execution_fingerprint']
        assert final['models'][EXTERNAL]['derived_score_fingerprint']!=final['models'][PUBLIC_REFERENCE]['derived_score_fingerprint']
        assert final['models'][EXTERNAL]['public_reference_comparison']['percent_better']==0
        assert final['plan_fingerprint']==reporter.plan.fingerprint
        assert all(p.read_bytes()==before for p,before in original.items())
    finally:reporter.close()


def test_registry_unknown_and_source_drift_fail_closed_without_harming_old_scores(panel,tmp_path):
    source=tmp_path/'source';control=tmp_path/'control.json'
    path,record,module=registry(tmp_path)
    generate(panel,source,PUBLIC_REFERENCE)
    generate_external(panel,source,record)
    write_control(control,record,unknown=True)
    reporter=ReportingSidecarR2(panel[0],source,tmp_path/'reports',declared_plan(panel),control_path=control,registry_path=path)
    try:
        first=reporter.poll()
        assert first['models']['unapproved-model']['status']=='failed_integrity'
        assert first['models'][EXTERNAL]['final_score'] is not None
        public_score=first['models'][PUBLIC_REFERENCE]['final_score']
        module.write_text(module.read_text()+'# source changed\n')
        second=reporter.poll()
        assert second['external_registry_error']
        assert second['models'][EXTERNAL]['status']=='failed_integrity'
        assert second['models'][EXTERNAL]['final_score'] is None
        assert second['models'][PUBLIC_REFERENCE]['final_score']==public_score
    finally:reporter.close()


def test_disabled_history_is_retained_but_not_ingested_or_in_denominator(panel,tmp_path,monkeypatch):
    source=tmp_path/'source';reports=tmp_path/'reports';control=tmp_path/'control.json'
    path,_record,_module=registry(tmp_path)
    generate(panel,source,PUBLIC_REFERENCE);generate(panel,source,DISABLED)
    old=frozen.ScoreSidecar(panel[0],source,reports,declared_plan(panel))
    prior=old.poll();old.close()
    disabled_files={p:p.read_bytes() for p in model_directory(source,DISABLED).rglob('*') if p.is_file()}
    write_control(control,disabled=True)
    reporter=ReportingSidecarR2(panel[0],source,reports,declared_plan(panel),control_path=control,registry_path=path)
    scanned=[];original_scan=reporter.scan_model
    reporter.scan_model=lambda mid,*args:(scanned.append(mid),original_scan(mid,*args))[1]
    try:
        status=reporter.poll()
        assert DISABLED not in status['models'] and DISABLED not in scanned
        assert status['excluded_history'][DISABLED]['final_score']==prior['models'][DISABLED]['final_score']
        assert status['enabled_model_count']==1 and status['expected_active_origins']==4
        assert status['completed_active_origins']==4
        assert (reports/'score_status_before_r2.json').exists()
        assert all(p.read_bytes()==before for p,before in disabled_files.items())
    finally:reporter.close()


def test_registry_digest_factory_and_selected_override_rejected(tmp_path):
    path,record,_module=registry(tmp_path)
    accepted,_=load_external_registry(path)
    assert accepted[EXTERNAL]['implementation_digest']==record['implementation_digest']
    payload=json.loads(path.read_text());payload['models'][EXTERNAL]['implementation_digest']='0'*64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='canonical identity'):load_external_registry(path)
    path,_record,_module=registry(tmp_path,mid=PUBLIC_REFERENCE)
    with pytest.raises(ValueError,match='cannot override'):load_external_registry(path)


def test_second_reporter_owner_is_rejected_before_opening_databases(panel,tmp_path):
    reports=tmp_path/'reports'
    with exclusive_lock(reports/'.reporter.lock'), pytest.raises(RuntimeError,match='another process'):
        # Empty plans cannot bypass the original one-owner lock. Use a valid
        # loader fixture so the lock itself is the failing gate.
        plan=declared_plan(panel)
        import french_score_sidecar_r2 as r2
        old_load=r2.frozen.ScoringPlan.load
        r2.frozen.ScoringPlan.load=lambda *args:plan
        try:
            run_reporting(panel[0],tmp_path/'source',reports,[tmp_path/'plan.json'],
                          tmp_path/'control.json',tmp_path/'registry.json',once=True)
        finally:r2.frozen.ScoringPlan.load=old_load


def test_unreadable_control_never_republishes_stale_current_scores(panel,tmp_path):
    source=tmp_path/'source';control=tmp_path/'control.json'
    path,_record,_module=registry(tmp_path)
    generate(panel,source,PUBLIC_REFERENCE)
    write_control(control)
    reporter=ReportingSidecarR2(panel[0],source,tmp_path/'fresh_reports',declared_plan(panel),
                               control_path=control,registry_path=path)
    try:
        first=reporter.poll()
        assert first['cohort_valid'] and first['models'][PUBLIC_REFERENCE]['final_score'] is not None
        control.write_text('{"models":')
        invalid=reporter.poll()
        assert invalid['control_error'] and not invalid['cohort_valid']
        assert invalid['models']=={}
        assert invalid['enabled_model_count'] is None and invalid['expected_active_origins'] is None
        assert invalid['completed_active_origins']==0
        assert reporter.active_specs=={} and reporter.models=={}
        write_control(control)
        recovered=reporter.poll()
        assert recovered['cohort_valid']
        assert recovered['models'][PUBLIC_REFERENCE]['final_score']==first['models'][PUBLIC_REFERENCE]['final_score']
        assert recovered['indexed_new_or_changed_origins_this_poll']==0
    finally:reporter.close()


@pytest.mark.parametrize('field,value',[('source_sha256','0'*64),('law',{'replacement':False})])
def test_saved_runtime_drift_rejected_even_when_current_source_is_restored(panel,tmp_path,field,value):
    source=tmp_path/'source';control=tmp_path/'control.json'
    path,record,_module=registry(tmp_path)
    write_control(control,record)
    changed=json.loads(json.dumps(record))
    changed['identity']['numerical_contract'][field]=value
    # Caller supplies the approved digest while the saved dependency contract
    # records drift. Current allowlisted files have never changed.
    generate_external(panel,source,changed)
    reporter=ReportingSidecarR2(panel[0],source,tmp_path/'reports',declared_plan(panel),control_path=control,registry_path=path)
    try:
        status=reporter.poll()
        row=status['models'][EXTERNAL]
        assert row['status']=='failed_integrity' and row['final_score'] is None
        assert row['completed_origins']==0
        assert 'saved runtime numerical contract' in row['error']
        assert load_external_registry(path)[0][EXTERNAL]==record
    finally:reporter.close()


def test_external_numerical_contract_must_be_declared(tmp_path):
    path,_record,_module=registry(tmp_path)
    payload=json.loads(path.read_text())
    identity=payload['models'][EXTERNAL]['identity']
    del identity['numerical_contract']
    payload['models'][EXTERNAL]['implementation_digest']=runner.stable_digest(identity)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='numerical contract'):load_external_registry(path)


def test_catalogue_display_names_do_not_change_scoring_identity(panel,tmp_path):
    source=tmp_path/'source';control=tmp_path/'control.json'
    path,_record,_module=registry(tmp_path)
    generate(panel,source,PUBLIC_REFERENCE);write_control(control)
    reporter=ReportingSidecarR2(panel[0],source,tmp_path/'reports',declared_plan(panel),control_path=control,registry_path=path)
    try:
        first=reporter.poll()['models'][PUBLIC_REFERENCE]
        assert first['display_name']==PUBLIC_REFERENCE
        catalogue=control.with_name('model_catalogue.json')
        catalogue.write_text(json.dumps({'schema':'french-public-comparator-names-v1',
            'reference_model_id':PUBLIC_REFERENCE,'comparators':[{'model_id':PUBLIC_REFERENCE,'display_name':'Public Adaptive'}]}))
        second=reporter.poll()
        row=second['models'][PUBLIC_REFERENCE]
        assert row['display_name']=='Public Adaptive'
        assert row['public_reference_comparison']['reference_display_name']=='Public Adaptive'
        assert row['derived_score_fingerprint']==first['derived_score_fingerprint']
        assert row['final_score']==first['final_score']
        assert second['model_catalogue_sha256']==frozen.sha256(catalogue)
    finally:reporter.close()


@pytest.mark.parametrize('change',[{'model_id':PUBLIC_REFERENCE,'display_name':'Other'},
    {'model_id':'other','display_name':'Public'}, {'model_id':'other','display_name':' '}])
def test_catalogue_rejects_duplicate_ids_names_and_empty_names(tmp_path,change):
    path=tmp_path/'model_catalogue.json'
    path.write_text(json.dumps({'schema':'french-public-comparator-names-v1','reference_model_id':PUBLIC_REFERENCE,
        'comparators':[{'model_id':PUBLIC_REFERENCE,'display_name':'Public'},change]}))
    with pytest.raises(ValueError):load_model_catalogue(path)
