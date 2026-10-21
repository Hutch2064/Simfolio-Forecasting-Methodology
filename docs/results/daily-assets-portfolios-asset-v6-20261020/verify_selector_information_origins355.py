"""Rebuild saved scores and audit actual historical-design execution identities."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_information_origins/study1000'
sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_information_origins.candidate import identity
from candidates.asset_selector_information_origins.validation import schedule
from candidates.asset_selector_universe_certified import energy
from candidates.asset_selector_universe_precision.candidate import signature
from score_percentage import percent_better
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 plan=read(D/'plan.json');summary=read(D/'summary.json');refs=read(D/'references.json')['cases']
 assert summary['complete_execution'] and summary['qualification_valid'] and not summary['errors']
 assert plan['outer_paths']==summary['outer_paths']==1000 and not plan['official'] and plan['full_panel_catalogue_increment']==0
 assert all(sha(O/p)==h for p,h in plan['source_hashes'].items()) and sha(D/'references.json')==plan['references_sha256']
 cases,configs=plan['cases'],plan['configs'];records=summary['records'];expected={(i,j)for i in range(len(cases))for j in range(len(configs))}
 assert len(records)==plan['expected_new_forecasts']==len(expected) and {(r['case'],r['config_index'])for r in records}==expected
 rows={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};cached={};reference_rows={k:[]for k in('fixed_reference',)}
 for i,case in enumerate(cases):
  row=rows[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
  dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
  x,w,ids=AssetInputs(V/'inputs').training(row,stop);cached[i]=(dates,values,x,w,ids,stop,H)
  for label in reference_rows:
   r=refs[str(i)][label];assert r['series_id']==case['series_id'] and r['origin_label']==origin['origin_label'] and r['paths']==1000
   assert sha(Path(r['checkpoint']))==r['checkpoint_sha256']
   with np.load(r['checkpoint'],allow_pickle=False)as data:loss=data['losses'];assert loss.shape==(H,) and np.isfinite(loss).all()
   assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
   q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==r['score'];reference_rows[label].append({**r,'score':q})
 verified=[];recalculated={};digest=api.original.stable_digest(plan['source_hashes']);original_replays=0
 for r in records:
  i,j=r['case'],r['config_index'];case=cases[i];origin=case['origin'];dates,values,x,w,ids,stop,H=cached[i];cfg=configs[j];n=cfg['inner_paths'];rep=cfg['seed_replication'];recipe={k:cfg[k]for k in('schedule','inner_paths','pool')}
  assert read(D/'records'/f'{j:03d}-{i:03d}.json')==r and r['configuration']==cfg and r['model_id']==identity(recipe)
  assert r['series_id']==case['series_id'] and r['origin_label']==origin['origin_label'] and tuple(r['constituent_ids'])==ids
  assert r['outer_paths']==1000 and r['inner_paths']==n and r['inner_seed_replication']==rep
  assert r['source_closure_digest']==digest and r['selection']['config']==recipe
  assert r['final_seed_shared_across_inner_counts']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
  weights=np.asarray(r['selection']['ideal_weights']);counts=np.asarray(r['selection']['path_counts'])
  assert weights.shape==counts.shape==(166,) and np.isfinite(weights).all() and np.all(weights>=0) and np.isclose(weights.sum(),1,rtol=0,atol=1e-12)
  assert np.array_equal(counts,energy.even_allocation(weights,1000)) and counts.sum()==1000
  if cfg['pool']=='select':assert np.count_nonzero(weights)==np.count_nonzero(counts)==1 and counts.max()==1000
  _,folds,available=schedule(len(x),H,cfg['schedule']);assert [(f['training_rows'],f['validation_horizon'],f['represented_origins'])for f in r['validation_trace']]==folds
  assert sum(f['represented_origins']for f in r['validation_trace'])==available==r['selection']['available_internal_origins']
  for fold in r['validation_trace']:
   fstop,fh=fold['training_rows'],fold['validation_horizon'];assert fstop+fh==fold['validation_stop'] and fstop+fh<=stop
   assert fold['training_sha256']==signature(x[:fstop]) and fold['training_last_date']==str(dates[fstop-1]) and fold['validation_last_known_date']==str(dates[fstop+fh-1])
   suffix=''if rep==0 else f':replication={rep}'
   assert fold['seed']==int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(x[:fstop])}:{fh}:{n}{suffix}'.encode()).digest()[:4],'little')
  cp=D/'losses'/f'{j:03d}-{i:03d}.npz';assert sha(cp)==r['checkpoint_sha256']
  with np.load(cp,allow_pickle=False)as data:loss=data['losses'];assert loss.shape==(H,) and np.isfinite(loss).all()
  assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
  if 'original_'+str(n) in refs[str(i)] and folds==schedule(len(x),H,'all')[1]:
   ref=refs[str(i)]['original_'+str(n)];assert sha(Path(ref['checkpoint']))==ref['checkpoint_sha256']
   with np.load(ref['checkpoint'],allow_pickle=False)as data:assert np.array_equal(loss,data['losses'])
   original_replays+=1
  diagnostic=r['selection'];assert diagnostic['prefix_fit_calls']==len(r['validation_trace'])+1
  assert diagnostic['prefix_fit_rows_processed']==sum(f['training_rows']for f in r['validation_trace'])+len(x)
  assert 0<=diagnostic['prefix_fit_cpu_seconds']<=r['prepare_cpu_seconds']+1e-9
  q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==r['score'];recalculated[i,j]={**r,'score':q}
  verified.append({'case':i,'configuration':j,'checkpoint_sha256':r['checkpoint_sha256'],'loss_sha256':r['loss_sha256']})
 baseline=score_main.aggregate_records(reference_rows['fixed_reference'],rows)['score'];assert baseline==summary['main_reference_Q']
 assert original_replays>0
 models=[]
 for j,cfg in enumerate(configs):
  selected=[recalculated[r['case'],j]for r in records if r['config_index']==j];q=score_main.aggregate_records(selected,rows)['score'];m=next(m for m in summary['models']if m['config_index']==j)
  assert q==m['main_Q'] and percent_better(q,baseline)==m['percent_better_vs_fixed_reference']
  paired=next(k for k,c in enumerate(configs)if c=={**cfg,'pool':'stack'})
  costs=[recalculated[r['case'],paired]['prepare_cpu_seconds']+r['cpu_seconds']-r['prepare_cpu_seconds']for r in selected]
  assert min(costs)>=0
  models.append({**m,'standalone_attributed_cpu_median':statistics.median(costs),'standalone_attributed_cpu_max':max(costs)})
 repetitions=[]
 for rule,n,pool in sorted({(m['configuration']['schedule'],m['configuration']['inner_paths'],m['configuration']['pool'])for m in models}):
  ms=[m for m in models if (m['configuration']['schedule'],m['configuration']['inner_paths'],m['configuration']['pool'])==(rule,n,pool)];qs=[m['main_Q']for m in ms]
  repetitions.append({'schedule':rule,'inner_paths':n,'pool':pool,'seed_repetitions':len(ms),'mean_main_Q':statistics.mean(qs),'sample_sd_main_Q':statistics.stdev(qs)if len(qs)>1 else None,'mean_percent_better_vs_baseline':statistics.mean(m['percent_better_vs_fixed_reference']for m in ms)})
 paired=[]
 for n in(4,30):
  cohort=[i for i in range(len(cases))if 'all_origins_'+str(n)in refs[str(i)]]
  if not cohort:continue
  old=[];base=[]
  for i in cohort:
   dates,values,x,w,ids,stop,H=cached[i];origin=cases[i]['origin'];reference=refs[str(i)]['all_origins_'+str(n)];assert sha(Path(reference['checkpoint']))==reference['checkpoint_sha256']
   with np.load(reference['checkpoint'],allow_pickle=False)as data:loss=data['losses'];assert loss.shape==(H,)and np.isfinite(loss).all()
   assert hashlib.sha256(loss.tobytes()).hexdigest()==reference['loss_sha256']
   q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==reference['score'];old.append({**reference,'score':q});base.append(reference_rows['fixed_reference'][i])
  bq=score_main.aggregate_records(base,rows)['score'];oq=score_main.aggregate_records(old,rows)['score']
  for j,cfg in enumerate(configs):
   if cfg['inner_paths']!=n:continue
   new=[recalculated[i,j]for i in cohort];q=score_main.aggregate_records(new,rows)['score']
   paired.append({'configuration':cfg,'matched_cases':len(cohort),'scope':'Only preserved expensive all-origin controls; distinct from full355coverage, no combined leaderboard','reference_Q':bq,'all_origin_control_Q':oq,'all_origin_control_percent_better_vs_baseline':percent_better(oq,bq),'candidate_Q':q,'candidate_percent_better_vs_baseline':percent_better(q,bq),'candidate_CPU_median':statistics.median(r['cpu_seconds']for r in new),'all_origin_CPU_median':statistics.median(r['cpu_seconds']for r in old),'runtime_scope':'Research process CPU including preparation, final simulation, scoring and compression; different validation learners, not numerical parity or production latency'})
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'summary_sha256':sha(D/'summary.json'),'plan_sha256':sha(D/'plan.json'),'verifier_sha256':sha(Path(__file__)),'candidate_forecasts_verified':len(verified),'reference_forecasts_verified':len(cases)*len(reference_rows)+original_replays,'original_control_loss_vectors_byte_exact':original_replays,'outer_paths':1000,'main_reference_Q':baseline,'models':models,'seed_aggregates':repetitions,'paired_expensive_origin_controls':paired,'no_automatic_promotion_or_production_precision_choice':True,'no_independence_or_significance_claim':True,'runtime_scope':'Process CPU of shared preparation charged to either standalone output; includes both pooling fits, union refit, final1000path forecast, score, compression and hashing. Excludes blocked IO waiting and final record writing; no production latency or isolated whole-request speedup claim.','verified_checkpoint_identities':verified}
 api.original._atomic_json(D/'verification_receipt.json',receipt);compact={k:v for k,v in receipt.items()if k!='verified_checkpoint_identities'};api.original._atomic_json(O/'selector_information_origins_verification.json',compact);print(json.dumps(compact,indent=2),flush=True)

if __name__=='__main__':main()
