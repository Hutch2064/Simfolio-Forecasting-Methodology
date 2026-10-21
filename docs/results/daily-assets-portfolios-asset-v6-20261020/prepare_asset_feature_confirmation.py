"""Freeze breadth-preserving1000-path confirmation after the216-arm exploratory screen."""
from pathlib import Path
import hashlib,json
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates/asset_feature_batch'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(C/'screen100/summary.json');assert s['complete_execution'] and not s['errors']
r=s['complete_identical_coverage_screen_ranking'];selected=[]
for family in sorted({x['config']['family'] for x in r}):selected.append(next(x for x in r if x['config']['family']==family))
for config in [dict(family='GARCH',innovations='empirical',raw_fade=True,mean_shrinkage=True,mean_uncertainty=True),dict(family='GARCH',innovations='Gaussian',raw_fade=True,mean_shrinkage=True,mean_uncertainty=True),dict(family='SV_RMS',innovations='empirical',raw_fade=True,mean_shrinkage=True,mean_uncertainty=True)]:
 x=next(x for x in r if x['config']==config)
 if x not in selected:selected.append(x)
assert len(selected)==12
p=C/'confirmation_selection.json';assert not p.exists()
p.write_text(json.dumps({'screen_summary_sha256':sha(C/'screen100/summary.json'),'selection':'One best100-path configuration per filter family, plus exact incumbent and two single-factor innovation contrasts. Frozen before1000-path forecasts; no horizon or class routing.','selected':selected,'official':False,'paths':1000,'development_cases':355,'later_reserved_required':True,'independent_validation_claim':False},indent=2,sort_keys=True)+'\n')
source=(O/'run_asset_feature_batch.py').read_text()
source=source.replace('"""Pipelined shared-fit216-configuration,100-path exploratory screen."""','"""Shared-fit12-configuration1000-path confirmation; original financial source remains frozen."""')
source=source.replace("D=C/'screen100'","D=C/'confirmation1000'")
source=source.replace("from candidates.asset_feature_batch.candidate import FeaturePool,CONFIGS,identity","from candidates.asset_feature_batch.candidate import FeaturePool,identity\nCONFIGS=[x['config'] for x in json.loads((C/'confirmation_selection.json').read_text())['selected']]")
# Path count changes are solely the predeclared Monte Carlo precision, not financial parameters.
source=source.replace('H,100)','H,1000)').replace('horizon,half','horizon,half')
source=source.replace('H,100,seed','H,1000,seed').replace('paths.shape[0]==100','paths.shape[0]==1000').replace("'paths':100,","'paths':1000,").replace('100*H*len(ids)','1000*H*len(ids)').replace('case_work=[100*','case_work=[1000*')
start=source.index('    selected=[]\n');end=source.index('    hashes=',start)
source=source[:start]+'''    cases=previous_plan['cases'];assert len(cases)==355
    panel_lookup={r['series_id']:(i,r) for i,r in enumerate(panel['series'])}
    references={};ref_records={'current_best':[],'fixed_reference':[]}
    for i,case in enumerate(cases):
        references[str(i)]={};sid=case['series_id'];row_index,row=panel_lookup[sid]
        dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);origin=case['origin'];origin_index=origins.index(origin);stop=origin['position']+1
        for label,root,mid in [('current_best',V/'candidates/asset_fading_vol/full/results',official['current_best_model_id']),('fixed_reference',V/'results',official['reference_model_id'])]:
            model=api.model_directory(root,mid);execution=read(model/'execution.json');cp,_=api.task_paths(model,(row_index,origin_index))
            loss,_=api.original._read_checkpoint(cp,api.task_identity(execution['fingerprint'],mid,sid,origin))
            record={'series_id':sid,'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+origin['max_horizon']]),'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'paths':1000}
            references[str(i)][label]=record;ref_records[label].append(record)
''' +source[end:]
source=source.replace("str(Path(__file__).relative_to(O)):sha(Path(__file__)),","str(Path(__file__).relative_to(O)):sha(Path(__file__)),str((C/'confirmation_selection.json').relative_to(O)):sha(C/'confirmation_selection.json'),")
source=source.replace("'case_selection':'Training-position median per existing family, with the minimum predeclared asset-quarter support needed by the unchanged canonical reducer. No loss values enter selection.'","'case_selection':'All355 previously frozen development cases; original72-case screen is an adaptively reused subset, not independent validation.'")
# All nine families have24 factorial configurations in the exploratory run; confirmation has unequal counts.
source=source.replace("remaining_units={family:24*sum(case_work) for family in fits.FAMILIES}","remaining_units={family:sum(c['family']==family for c in CONFIGS)*sum(case_work) for family in fits.FAMILIES}")
source=source.replace("'--expected-gib','.8'","'--expected-gib','1.2'")
source=source.replace("'exact_incumbent_anchor_required_each_case':True","'exact_incumbent_anchor_required_each_case':True,'selection_sha256':sha(C/'confirmation_selection.json')")
(O/'run_asset_feature_confirmation.py').write_text(source)
print('frozen configs',[(x['config_index'],x['config']) for x in selected])
