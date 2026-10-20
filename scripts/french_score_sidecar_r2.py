"""Reporting-only r2: explicit external model closures and enabled-cohort views.

The frozen v2 module owns every normalization and numerical reduction. This
wrapper adds identity admission and report scope, without changing score plans,
original execution manifests, checkpoints, or the forecast scheduler.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import french_score_sidecar as frozen
from simfolio_forecasting_methodology.french_panel import selected
from simfolio_forecasting_methodology.french_panel.sweep import (
    exclusive_lock,
    model_directory,
    read_control,
)

REGISTRY_SCHEMA = 'french-external-model-registry-v1'


def load_external_registry(path):
    registry = json.loads(Path(path).read_text())
    if registry.get('schema') != REGISTRY_SCHEMA or not isinstance(registry.get('models'), dict):
        raise ValueError('invalid external model registry schema')
    source_root = Path(registry['source_root']).resolve()
    records = {}
    for mid, record in registry['models'].items():
        if mid in selected.FACTORIES:
            raise ValueError('external registry cannot override a selected model identity')
        identity = record['identity']
        if identity.get('model_id') != mid or identity.get('factory') != record['factory']:
            raise ValueError('external registry model/factory identity differs')
        if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*',record['factory']):
            raise ValueError('invalid external factory declaration')
        if frozen.original.stable_digest(identity) != record['implementation_digest']:
            raise ValueError('external implementation digest differs from its canonical identity')
        if not isinstance(identity.get('numerical_contract'),dict) or not identity['numerical_contract']:
            raise ValueError('external numerical contract must be explicit and nonempty')
        files = identity.get('source_files')
        if not isinstance(files,dict) or not files:
            raise ValueError('external source closure must be explicit and nonempty')
        for relative, expected in files.items():
            target = (source_root / relative).resolve()
            if Path(relative).is_absolute() or source_root not in target.parents:
                raise ValueError('external closure path must stay inside its source root')
            if not re.fullmatch(r'[0-9a-f]{64}',expected) or frozen.sha256(target) != expected:
                raise ValueError(f'external source closure hash differs: {relative}')
        records[mid] = record
    return records, frozen.sha256(path)


def load_model_catalogue(path):
    path=Path(path)
    if not path.exists():
        return {},None
    value=json.loads(path.read_text())
    if value.get('schema')!='french-public-comparator-names-v1' or value.get('reference_model_id')!=frozen.PUBLIC_REFERENCE:
        raise ValueError('model catalogue schema/reference differs')
    names={}
    for row in value['comparators']:
        mid,name=row['model_id'],row['display_name']
        if not isinstance(mid,str) or not mid.strip() or mid in names:
            raise ValueError('model catalogue IDs must be nonempty and unique')
        if not isinstance(name,str) or not name.strip() or name in names.values():
            raise ValueError('model catalogue display names must be nonempty and unique')
        names[mid]=name
    if frozen.PUBLIC_REFERENCE not in names:
        raise ValueError('model catalogue reference is missing')
    return names,frozen.sha256(path)


class ReportingSidecarR2(frozen.ScoreSidecar):
    def __init__(self,*args,control_path,registry_path,**kwargs):
        report_root = Path(args[2])
        prior_path = report_root / 'score_status.json'
        self.history = {}
        if prior_path.exists():
            previous = json.loads(prior_path.read_text())
            self.history.update(previous.get('excluded_history',{}))
            self.history.update(previous.get('models',{}))
            archive = report_root / 'score_status_before_r2.json'
            if not archive.exists():
                frozen.original._atomic_json(archive,previous)
        super().__init__(*args,**kwargs)
        self.control_path = Path(control_path)
        self.registry_path = Path(registry_path)
        self.active_specs = {}
        self.registry_hash = None
        self.registry_error = None
        self.control_error = None
        self.external = {}
        self.display_names = {}
        self.catalogue_hash = None
        self.catalogue_error = None
        self._runtime_provenance()

    def _runtime_provenance(self):
        frozen.original._atomic_json(self.report_root/'reporting_runtime_provenance_r2.json',{
            'reporting_revision':'r2_external_registry_and_enabled_cohort',
            'wrapper_sha256':frozen.sha256(__file__),
            'frozen_scoring_engine_sha256':frozen.sha256(frozen.__file__),
            'plan_fingerprint':self.plan.fingerprint,
            'external_registry_path':str(self.registry_path),
            'external_registry_sha256':self.registry_hash,
            'read_only_control_path':str(self.control_path),
            'model_catalogue_sha256':self.catalogue_hash,
            'numerical_scoring_change':False,'forecast_runs':0,
            'cache_binding':'unchanged original execution fingerprints + unchanged v2 plan fingerprint',
        })

    def validate_spec(self,mid,spec):
        if mid in selected.FACTORIES:
            if mid not in self.implementation_digests:
                self.implementation_digests[mid]=selected.implementation_digest(mid,self.implementation_sources)
            expected=self.implementation_digests[mid]
            factory = selected.FACTORIES[mid]
        else:
            if mid not in self.external:
                raise ValueError('external model is not in the verified allowlist')
            record = self.external[mid]
            expected,factory = record['implementation_digest'],record['factory']
        if spec['factory'] != factory or spec['implementation_digest'] != expected:
            raise ValueError('control factory/implementation identity differs from approved registry/source')
        self.implementation_digests[mid] = expected

    def manifest(self,root):
        # Populate only an already verified external digest; the frozen manifest
        # reader continues to enforce every original panel/source/task contract.
        value = json.loads((root/'execution.json').read_text())
        mid = value['identity']['model_id']
        if mid not in self.active_specs:
            raise ValueError('model is outside the enabled reporting cohort')
        self.validate_spec(mid,self.active_specs[mid])
        if mid in self.external:
            contract=self.external[mid]['identity']['numerical_contract']
            runtime=value['identity'].get('runtime',{})
            if not isinstance(runtime,dict) or any(key not in runtime or runtime[key]!=expected for key,expected in contract.items()):
                raise ValueError('external saved runtime numerical contract differs from approved registry')
        return super().manifest(root)

    def poll(self):
        self.control_error = None
        self.registry_error = None
        self.catalogue_error = None
        try:
            self.display_names,self.catalogue_hash=load_model_catalogue(self.control_path.with_name('model_catalogue.json'))
        except (OSError,ValueError,TypeError,KeyError) as exc:
            self.display_names={}
            self.catalogue_hash=None
            self.catalogue_error=f'{type(exc).__name__}: {exc}'
        try:
            control = read_control(self.control_path)
            self.active_specs = {mid:spec for mid,spec in control['models'].items() if spec['enabled']}
        except (OSError,ValueError,TypeError,KeyError) as exc:
            self.control_error = f'{type(exc).__name__}: {exc}'
            self.active_specs={}
            self.models={}
            return self.publish(0)
        try:
            self.external,self.registry_hash = load_external_registry(self.registry_path)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            self.external = {}
            self.registry_error = f'{type(exc).__name__}: {exc}'
        for mid in list(self.models):
            if mid not in self.active_specs:
                self.models.pop(mid)
        indexed = 0
        for mid,spec in self.active_specs.items():
            root = model_directory(self.source_root,mid)
            try:
                self.validate_spec(mid,spec)
                if self.db.execute('SELECT 1 FROM errors WHERE model=?',(mid,)).fetchone():
                    continue
                if not (root/'execution.json').exists():
                    if self.db.execute('SELECT 1 FROM origins WHERE model=? LIMIT 1',(mid,)).fetchone():
                        raise ValueError('previously indexed execution manifest disappeared')
                    continue
                _,fingerprint = self.manifest(root)
                indexed += self.scan_model(mid,fingerprint,root)
                self.aggregate_complete_series(mid,fingerprint,root)
            except Exception as exc:  # noqa: BLE001 -- isolate a corrupt model from other reports
                self.db.execute('INSERT OR REPLACE INTO errors VALUES (?,?)',(mid,f'{type(exc).__name__}: {exc}'))
        self.db.commit()
        self._runtime_provenance()
        return self.publish(indexed)

    def publish(self,indexed_this_poll):
        # self.models contains enabled manifests only; excluded checkpoints are
        # neither scanned nor reaggregated. The unchanged engine computes scores.
        status = super().publish(indexed_this_poll)
        expected = len(self.panel['series'])*len(self.origins)
        for mid in self.active_specs:
            if mid not in status['models']:
                count = self.db.execute('SELECT COUNT(*) FROM origins WHERE model=?',(mid,)).fetchone()[0]
                error = status['integrity_errors'].get(mid)
                status['models'][mid] = {
                    'model_id':mid,'status':'failed_integrity' if error else 'waiting_for_execution_manifest',
                    'completed_origins':count,'expected_origins':expected,
                    'complete_series_ids':[],'final_score':None,'partial_complete_series_score':None,
                    'error':error,'public_reference_comparison':{
                        'reference_model_id':frozen.PUBLIC_REFERENCE,'matched_complete_series_ids':[],
                        'model_score':None,'reference_score':None,'percent_better':None},
                }
        for mid,row in status['models'].items():
            row['display_name']=self.display_names.get(mid,mid)
            row['public_reference_comparison']['reference_display_name']=self.display_names.get(frozen.PUBLIC_REFERENCE,frozen.PUBLIC_REFERENCE)
        self.history.update(status['models'])
        status['excluded_history'] = {mid:row for mid,row in self.history.items() if mid not in self.active_specs}
        status['integrity_errors'] = {mid:error for mid,error in status['integrity_errors'].items() if mid in self.active_specs}
        status.update({
            'reporting_revision':'r2_external_registry_and_enabled_cohort',
            'cohort_valid':self.control_error is None,
            'enabled_model_count':len(self.active_specs) if self.control_error is None else None,
            'expected_active_origins':len(self.active_specs)*expected if self.control_error is None else None,
            'completed_active_origins':sum(row['completed_origins'] for row in status['models'].values()),
            'active_model_ids':list(self.active_specs),
            'excluded_model_ids':list(status['excluded_history']),
            'excluded_history_scope':'preserved prior derived reports; excluded checkpoints are not actively ingested/scored',
            'external_registry_sha256':self.registry_hash,
            'external_registry_error':self.registry_error,'control_error':self.control_error,
            'score_formula_changed':False,
            'model_display_names':{mid:self.display_names.get(mid,mid) for mid in self.active_specs},
            'model_catalogue_sha256':self.catalogue_hash,'model_catalogue_error':self.catalogue_error,
        })
        frozen.original._atomic_json(self.report_root/'score_status.json',status)
        return status


def run_reporting(panel_root,source_root,report_root,plan_paths,control_path,registry_path,*,poll_seconds=120,once=False):
    panel = frozen.original.load_panel(panel_root)
    plans = [frozen.ScoringPlan.load(path,tuple(frozen.original.schedule(panel_root)),panel) for path in plan_paths]
    versions = [p.declaration['score_version'] for p in plans]
    if not versions or len(set(versions)) != len(versions):
        raise ValueError('score versions must be unique')
    reporters=[]
    report_root=Path(report_root)
    with exclusive_lock(report_root/'.reporter.lock'):
        try:
            for version,plan in zip(versions,plans,strict=True):
                reporters.append(ReportingSidecarR2(panel_root,source_root,report_root/version,plan,
                         control_path=control_path,registry_path=registry_path))
            while True:
                snapshots={version:reporter.poll() for version,reporter in zip(versions,reporters,strict=True)}
                first=snapshots[versions[0]]
                frozen.original._atomic_json(report_root/'scoring_master.json',{
                    'canonical_score_version':versions[0],
                    'score_versions':{version:{'role':'primary' if index==0 else 'predeclared_sensitivity',
                        'plan_fingerprint':snapshots[version]['plan_fingerprint'],
                        'status_path':str(report_root/version/'score_status.json')}
                        for index,version in enumerate(versions)},
                    'public_reference_model_id':frozen.PUBLIC_REFERENCE,
                    'original_raw_score':'preserved separately in original model summaries',
                    'reporting_revision':'r2_external_registry_and_enabled_cohort',
                    'cohort_valid':first['cohort_valid'],'control_error':first['control_error'],
                    'enabled_model_count':first['enabled_model_count'],
                    'expected_active_origins':first['expected_active_origins'],
                    'external_registry_sha256':first['external_registry_sha256'],
                    'model_display_names':first['model_display_names'],
                    'model_catalogue_sha256':first['model_catalogue_sha256'],
                    'model_catalogue_error':first['model_catalogue_error'],
                    'forecast_runs':0,'updated_utc':datetime.now(UTC).isoformat(),
                })
                if once:
                    return snapshots
                time.sleep(poll_seconds)
        finally:
            for reporter in reporters:
                reporter.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--report-root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,action='append',required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--external-registry',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=float,default=120)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.poll_seconds<1:
        parser.error('poll seconds must be at least one')
    run_reporting(args.panel_root,args.source_root,args.report_root,args.plan,args.control,args.external_registry,
                  poll_seconds=args.poll_seconds,once=args.once)


if __name__=='__main__':
    main()
