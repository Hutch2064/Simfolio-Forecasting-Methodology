from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates/asset_component_weights';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(C/'screen120/summary.json');assert s['complete_execution'] and not s['errors'] and s['paths']==120 and s['anchor_loss_vectors_bitwise_equal_each_case']
anchor=dict(family='SV_RMS',innovations='Gaussian',raw_fade=True,mean_shrinkage=True,mean_uncertainty=True,raw_fraction=.5)
selected=[r for r in s['complete_identical_coverage_screen_ranking']if r['main_Q']<s['reference_main_Q']['current_best'] or r['config']==anchor];assert len(selected)==5
P=C/'confirmation_selection.json';assert not P.exists();P.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'screen_summary_sha256':sha(C/'screen120/summary.json'),'selection':'All four improving120-path configurations plus exact working incumbent; no full-panel escalation.','selected':selected,'paths':1000,'case_count':355,'official':False,'independent_validation_claim':False},indent=2)+'\n')
# Mutable research bookkeeping is preserved as an immutable per-stage admission snapshot.
(C/'screen120/working_incumbent_snapshot.json').write_bytes((O/'working_incumbent.json').read_bytes())
x=(O/'run_asset_feature_confirmation.py').read_text().replace("C=V/'candidates/asset_feature_batch';D=C/'confirmation1000'","C=V/'candidates/asset_component_weights';D=C/'development1000'")
x=x.replace('from candidates.asset_feature_batch.candidate import FeaturePool,identity','from candidates.asset_component_weights.candidate import ComponentPool as FeaturePool,identity\nFAMILIES=(\'SV_RMS\',\'SV_path\',\'constant\')')
x=x.replace("ANCHOR={'family':'GARCH','innovations':'empirical','raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True}","ANCHOR={'family':'SV_RMS','innovations':'Gaussian','raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True,'raw_fraction':.5}")
x=x.replace('for family in fits.FAMILIES','for family in FAMILIES').replace('len(fits.FAMILIES)','len(FAMILIES)')
a=x.index("    source=V/'candidates/asset_mean_uncertainty/fading_vol_screen100'");b=x.index('    hashes=',a)
x=x[:a]+'''    source=V/'candidates/asset_feature_batch/confirmation1000';old=read(source/'summary.json');previous_plan=read(source/'plan.json');assert old['complete_execution'] and not old['errors'] and old['paths']==1000
    assert all(sha(O/p)==h for p,h in previous_plan['source_hashes'].items())
    working=read(O/'working_incumbent.json');assert working['config']=={k:v for k,v in ANCHOR.items()if k!='raw_fraction'}
    assert read(O/'research_workflow_policy.json')['full_panel_launch_requires_stopping_review']
    panel=api.load_panel(V/'inputs',verify_files=False);metadata={r['series_id']:r for r in panel['series']};cases=previous_plan['cases'];assert len(cases)==355
    previous_records={r['case']:r for r in old['records']if r['model_id']==working['model_id']};assert len(previous_records)==355
    old_references=read(source/'references.json');references={};ref_records={'current_best':[],'fixed_reference':[]}
    for number,case in enumerate(cases):
        parent=previous_records[number];path=source/'losses'/f"{parent['config_index']:03d}-{number:03d}.npz";assert sha(path)==parent['checkpoint_sha256']
        with np.load(path,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==parent['loss_sha256']
        fixed=old_references['cases'][str(number)]['fixed_reference'];assert sha(Path(fixed['checkpoint']))==fixed['checkpoint_sha256']
        references[str(number)]={'current_best':{**parent,'checkpoint':str(path)},'fixed_reference':fixed}
        ref_records['current_best'].append(parent);ref_records['fixed_reference'].append(fixed)
''' +x[b:]
x=x.replace("'--expected-gib','1.2'","'--expected-gib','.3'").replace("'--budget-gib','12'","'--budget-gib','16'")
x=x.replace('All355 previously frozen development cases; original72-case screen is an adaptively reused subset, not independent validation.','All355 previously frozen development cases. The72-case120-path screen is an adaptively reused subset, not independent validation; no full-panel escalation.')
x=x.replace('same100path','same1000path')
x=x.replace("    D.mkdir(parents=True,exist_ok=True)","    supplement=read(V/'candidates/asset_feature_batch/transitive_source_receipt.json')\n    for name,digest in supplement['source_files_relative_to_v6'].items():\n        assert sha(V/name)==digest;hashes['asset_level_v6/'+name]=digest\n    hashes[str((C/'screen120/working_incumbent_snapshot.json').relative_to(O))]=sha(C/'screen120/working_incumbent_snapshot.json')\n    D.mkdir(parents=True,exist_ok=True)")
(O/'run_asset_component_confirmation.py').write_text(x)
print('Prepared five-configuration1000-path development confirmation; no full sweep admission')
