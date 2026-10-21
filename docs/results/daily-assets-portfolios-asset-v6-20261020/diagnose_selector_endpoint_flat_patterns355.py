"""Exact additive main-score diagnostic, no new selection or scoring objective."""
from pathlib import Path
from collections import defaultdict
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_endpoint_flat/broader1000';sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import common,sweep_v6 as api,score_main
from diagnose_asset_v6_pair import origin_weights
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(D/'verification_receipt.json');s=read(D/'summary.json');p=read(D/'plan.json');refs=read(D/'references.json')['cases'];assert a['verified']and a['summary_sha256']==sha(D/'summary.json')and s['qualification_valid']and not s['errors'];assert len(p['cases'])==355
meta={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};n=2;selected=[r for r in s['records']if r['inner_paths']==n];assert len(selected)==1065
by_case=defaultdict(list)
for r in selected:by_case[r['case']].append(r)
roster=[(i,meta[c['series_id']],c['origin']['origin_date'])for i,c in enumerate(p['cases'])];weights=origin_weights(roster);data={};pieces={k:defaultdict(lambda:{'baseline':0.,'candidate':0.,'score_mass':0.,'cases':set(),'series':set()})for k in('branch','class','family','series','portfolio_exposure','portfolio_purpose','horizon')};verified=0;bands=['at_most_31_calendar_days','32_to_366_calendar_days','367_to_1826_calendar_days','1827_to_7305_calendar_days','over_7305_calendar_days']

def load_loss(path,r,h):
 global verified
 path=Path(path);assert sha(path)==r['checkpoint_sha256']
 with np.load(path,allow_pickle=False)as cp:loss=cp['losses'].copy()
 assert loss.shape==(h,)and np.isfinite(loss).all()and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256'];verified+=1;return loss

def add(axis,label,i,sid,b,c,m):
 cell=pieces[axis][label];cell['baseline']+=float(b);cell['candidate']+=float(c);cell['score_mass']+=float(m);cell['cases'].add(i);cell['series'].add(sid)

for i,case in enumerate(p['cases']):
 row=meta[case['series_id']];sid=row['series_id'];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
 if sid not in data:data[sid]=api.load_series(V/'inputs',row,verify_files=False)
 dates,values,origins=data[sid];assert origin in origins
 records=by_case[i];assert len(records)==3 and {r['inner_seed_replication']for r in records}=={0,1,2}
 base=refs[str(i)]['fixed_reference'];b=load_loss(base['checkpoint'],base,H);c=np.mean([load_loss(D/'losses'/f"{r['config_index']:03d}-{i:03d}.npz",r,H)for r in records],axis=0)
 days=(dates[stop:stop+H].astype('datetime64[D]')-np.datetime64(origin['origin_date'],'D')).astype(int);masses=score_main.legacy.calendar_log_weights(origin['origin_date'],dates[stop:stop+H]);coeff=weights[i]*masses/(np.std(values[:stop],ddof=1)*np.sqrt(np.arange(1,H+1)));bc=b*coeff;cc=c*coeff
 assert np.isclose(bc.sum(),weights[i]*base['score'],rtol=0,atol=1e-13);assert np.isclose(cc.sum(),weights[i]*np.mean([r['score']for r in records]),rtol=0,atol=1e-13)
 labels={'branch':row['panel_branch'],'class':row['panel_branch']+'/'+row['asset_class'],'family':'/'.join(row[k]for k in('panel_branch','asset_class','family')),'series':sid,'portfolio_exposure':'+'.join(sorted({meta[k]['asset_class']for k in row.get('source_ids',[sid])}))if row['panel_branch']=='portfolios'else'individual_assets','portfolio_purpose':row.get('purpose','individual_assets')}
 for axis,label in labels.items():add(axis,label,i,sid,bc.sum(),cc.sum(),weights[i])
 assignment=np.searchsorted([31,366,1826,7305],days,side='left')
 for j,label in enumerate(bands):
  mask=assignment==j
  if mask.any():add('horizon',label,i,sid,bc[mask].sum(),cc[mask].sum(),weights[i]*masses[mask].sum())
expected=np.mean([m['main_Q']for m in s['models']if m['configuration']['inner_paths']==n]);baseline=s['main_reference_Q'];out={}
for axis,groups in pieces.items():
 assert np.isclose(sum(g['baseline']for g in groups.values()),baseline,rtol=0,atol=2e-12)and np.isclose(sum(g['candidate']for g in groups.values()),expected,rtol=0,atol=2e-12)
 assert np.isclose(sum(g['score_mass']for g in groups.values()),1,rtol=0,atol=1e-12)
 out[axis]=[]
 for label,g in groups.items():
  out[axis].append({'group':label,'display_name':meta[label]['display_name']if axis=='series'else label,'requests':len(g['cases']),'distinct_series':len(g['series']),'baseline_Q_contribution':g['baseline'],'candidate_Q_contribution':g['candidate'],'canonical_score_weight':g['score_mass'],'within_group_percent_better':100*(1-g['candidate']/g['baseline'])if g['baseline']>0 else None,'overall_improvement_percentage_points':100*(g['baseline']-g['candidate'])/baseline})
 out[axis].sort(key=lambda g:-g['overall_improvement_percentage_points'])
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':'Standalone flattened endpoint selector,2internal paths,mean over3predeclared internal seeds','requests':355,'series':len(data),'candidate_forecasts':1065,'baseline_forecasts':355,'saved_loss_vectors_hash_verified':verified,'summary_sha256':sha(D/'summary.json'),'verification_receipt_sha256':sha(D/'verification_receipt.json'),'diagnostic_source_sha256':sha(Path(__file__)),'mean_candidate_main_Q':float(expected),'baseline_main_Q':baseline,'percent_better_vs_fixed_baseline':100*(1-expected/baseline),'canonical_Q_reconciled_each_dimension':True,'diagnostic_only':True,'zero_score_or_selection_weight':True,'no_independence_or_significance_claim':True,'interpretation':'Every dimension separately sums to the same main score and total improvement; contributions are not alternative scores. Within-group percentages describe fixed matched subset losses using their original canonical weights. Data are355reused development requests; portfolios share constituents, origins overlap and3seeds are not independent market observations. Never select a recipe using these outer outcomes.','groups':out}
(O/'selector_endpoint_flat_patterns355.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='groups'},indent=2))
for axis in('branch','class','portfolio_exposure','horizon'):
 print(axis,json.dumps(out[axis],indent=2))
