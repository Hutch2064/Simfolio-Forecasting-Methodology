"""Independent saved-loss, identity, causal-prefix and matched-menu audit."""
from pathlib import Path
import datetime, hashlib, json, sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_menu_components/study1000'
sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_menu_components.candidate import identity
from candidates.asset_selector_menu_components.bank import MENUS,BANK
from candidates.asset_selector_universe_certified import energy
from candidates.asset_selector_universe_precision.candidate import signature
from score_percentage import percent_better
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 plan=read(D/'plan.json');summary=read(D/'summary.json');refs=read(D/'references.json')['cases']
 assert summary['complete_execution'] and summary['qualification_valid'] and not summary['errors']
 assert plan['outer_paths']==summary['outer_paths']==1000 and not plan['official'] and plan['full_panel_catalogue_increment']==0
 assert all(sha(O/p)==h for p,h in plan['source_hashes'].items());assert sha(D/'references.json')==plan['references_sha256']
 cases,configs=plan['cases'],plan['configs'];records=summary['records'];expected={(i,j)for i in range(len(cases))for j in range(len(configs))}
 assert len(records)==plan['expected_new_forecasts']==len(expected) and {(r['case'],r['config_index'])for r in records}==expected
 rows={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};cached={};reference_rows={k:[]for k in ('fixed_reference','original_10','original_30')}
 for i,case in enumerate(cases):
  row=rows[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
  dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
  x,w,ids=AssetInputs(V/'inputs').training(row,stop);cached[i]=(dates,values,x,w,ids,stop,H)
  for label in reference_rows:
   r=refs[str(i)][label];assert r['series_id']==case['series_id'] and r['origin_label']==origin['origin_label'] and r['paths']==1000
   assert sha(Path(r['checkpoint']))==r['checkpoint_sha256']
   with np.load(r['checkpoint'],allow_pickle=False)as data:loss=data['losses'];assert loss.shape==(H,) and np.isfinite(loss).all()
   assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
   q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==r['score']
   reference_rows[label].append({**r,'score':q})
 verified=[];recalculated={};source_digest=api.original.stable_digest(plan['source_hashes'])
 for r in records:
  i,j=r['case'],r['config_index'];case=cases[i];origin=case['origin'];dates,values,x,w,ids,stop,H=cached[i];cfg=configs[j];n=cfg['inner_paths'];rep=cfg['seed_replication'];name=cfg['menu'];recipe={'menu':name,'inner_paths':n}
  assert read(D/'records'/f'{j:03d}-{i:03d}.json')==r and r['configuration']==cfg and r['model_id']==identity(recipe)
  assert r['series_id']==case['series_id'] and r['origin_label']==origin['origin_label'] and tuple(r['constituent_ids'])==ids
  assert r['outer_paths']==1000 and r['inner_paths']==n and r['inner_seed_replication']==rep==0
  assert r['source_closure_digest']==source_digest and r['selection']['config']==recipe
  assert r['final_seed_shared_across_inner_counts']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
  weights=np.asarray(r['selection']['ideal_weights']);counts=np.asarray(r['selection']['path_counts'])
  assert weights.shape==counts.shape==(len(BANK),) and np.isfinite(weights).all() and np.all(weights>=0) and np.isclose(weights.sum(),1,rtol=0,atol=1e-12)
  assert np.array_equal(counts,energy.even_allocation(weights,1000)) and counts.sum()==1000
  assert not any(weights[k] or counts[k]for k in range(len(BANK))if k not in MENUS[name])
  for fold in r['validation_trace']:
   fstop,fh=fold['training_rows'],fold['validation_horizon'];assert fstop+fh==fold['validation_stop'] and fstop+fh<=stop
   assert fold['training_sha256']==signature(x[:fstop]) and fold['training_last_date']==str(dates[fstop-1]) and fold['validation_last_known_date']==str(dates[fstop+fh-1])
   seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(x[:fstop])}:{fh}:{n}'.encode()).digest()[:4],'little');assert fold['seed']==seed
   ppw=fold['PPW_dependence_fit']
   if ppw is not None:assert ppw['history_sha256']==signature(x[:fstop]) and ppw['weights_sha256']==signature(w) and ppw['training_rows']==fstop
  cp=D/'losses'/f'{j:03d}-{i:03d}.npz';assert sha(cp)==r['checkpoint_sha256']
  with np.load(cp,allow_pickle=False)as data:loss=data['losses'];assert loss.shape==(H,) and np.isfinite(loss).all()
  assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
  if name=='original':
   with np.load(refs[str(i)]['original_'+str(n)]['checkpoint'],allow_pickle=False)as data:assert np.array_equal(loss,data['losses'])
  q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==r['score'];recalculated[i,j]={**r,'score':q}
  verified.append({'case':i,'configuration':j,'checkpoint_sha256':r['checkpoint_sha256'],'loss_sha256':r['loss_sha256']})
 baseline=score_main.aggregate_records(reference_rows['fixed_reference'],rows)['score'];assert baseline==summary['main_reference_Q']
 originals={str(n):score_main.aggregate_records(reference_rows['original_'+str(n)],rows)['score']for n in (10,30)};assert originals==summary['matched_original_Q_by_inner_paths']
 models=[]
 for j,cfg in enumerate(configs):
  selected=[recalculated[r['case'],j]for r in records if r['config_index']==j];q=score_main.aggregate_records(selected,rows)['score'];m=next(m for m in summary['models']if m['config_index']==j)
  assert q==m['main_Q'] and percent_better(q,baseline)==m['percent_better_vs_fixed_reference'] and q-originals[str(cfg['inner_paths'])]==m['main_Q_difference_from_matched_original']
  models.append({**m,'average_final_path_share':{label:float(np.mean([sum(r['selection']['path_counts'][k]for k in indices)/1000 for r in selected]))for label,indices in {'nested_parent':[0],'standalone_raw':[166],'standalone_filtered':[167],'PPW_stationary':[168],'PPW_circular':[169]}.items()}})
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'summary_sha256':sha(D/'summary.json'),'plan_sha256':sha(D/'plan.json'),'verifier_sha256':sha(Path(__file__)),'candidate_forecasts_verified':len(verified),'reference_forecasts_verified':len(cases)*len(reference_rows),'original_loss_vectors_byte_exact':2*len(cases),'outer_paths':1000,'main_reference_Q':baseline,'models':models,'no_automatic_promotion_or_production_precision_choice':True,'no_independence_or_significance_claim':True,'verified_checkpoint_identities':verified}
 api.original._atomic_json(D/'verification_receipt.json',receipt)
 compact={k:v for k,v in receipt.items()if k!='verified_checkpoint_identities'};api.original._atomic_json(O/'selector_menu_components_verification.json',compact)
 print(json.dumps(compact,indent=2),flush=True)

if __name__=='__main__':main()
