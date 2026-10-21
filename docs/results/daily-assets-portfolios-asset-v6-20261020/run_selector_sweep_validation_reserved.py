"""Qualify exact coefficient acceleration on355requests; test355reserved cases."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_coefficient_sweep';D=C/'validation_and_reserved';PRE=V/'candidates/asset_selector_path_objective/development1000';RES=V/'candidates/asset_selector_batched/validation_and_reserved';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_coefficient_sweep.candidate import prepare,PathCriterionPool,identity
from score_percentage import percent_better,normalize_model_percentages
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
CONFIGS=[{'criterion':m,'inner_paths':1000}for m in('terminal','path')]
def initialize():
 global ROWS,PLAN,PRIOR,REFS,COUNT
 ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};PLAN=read(D/'plan.json');PRIOR={(r['case'],r['config_index']):r for r in read(PRE/'summary.json')['records']};REFS=read(D/'references.json')['cases'];COUNT=0
def state(phase,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'phase':phase,'utc':now(),**extra})
def evaluate(kind,number):
 global COUNT
 COUNT+=1;case=PLAN[kind+'_cases'][number];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];begin=time.perf_counter();cpu=time.process_time();state('prepare',kind=kind,case=number)
 try:
  x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
  bundle=prepare(x,w,H,dates[:stop],inner_paths=1000);coefficient_checks=0
  if kind=='parity':
   for index,config in enumerate(CONFIGS):
    old=PRIOR[number,index];assert bundle['trace']==old['validation_trace'],'Historical component ensemble trace differs';np.testing.assert_array_equal(bundle['weights_by_mode'][config['criterion']],old['selection']['ideal_weights']);assert bundle['fit'][config['criterion']]['objective']==old['fit_certificate']['objective']
    if old['coefficient_checkpoint']:
     cp=Path(old['coefficient_checkpoint']);assert sha(cp)==old['coefficient_checkpoint_sha256']
     with np.load(cp,allow_pickle=False)as z:
      for key in('A','B'):np.testing.assert_array_equal(bundle['coefficients'][config['criterion']][key],z[key+'_'+config['criterion']]);coefficient_checks+=1
  else:
   old=REFS[str(number)]['terminal_pool'];assert len(bundle['trace'])==len(old['validation_trace'])
   aliases={'ensemble_sha256':'terminal_ensemble_sha256','terminal_observation':'terminal_y_sum'}
   for new,prior in zip(bundle['trace'],old['validation_trace']):
    for key in prior:assert new[aliases.get(key,key)]==prior[key],f'Reserved endpoint control trace mismatch:{key}'
  actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000)
  for index,config in enumerate(CONFIGS):
   state('forecast',kind=kind,case=number,config_index=index);m=PathCriterionPool(config,dates[:stop],bundle);loss=np.empty(H);cursor=0
   for start,paths in m.terminal_chunks(x,w,H,1000,seed):
    assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
   assert cursor==H and np.isfinite(loss).all();digest=hashlib.sha256(loss.tobytes()).hexdigest()
   if kind=='parity':
    old=PRIOR[number,index];p=PRE/'losses'/f'{index:03d}-{number:03d}.npz';assert sha(p)==old['checkpoint_sha256']and digest==old['loss_sha256'],'Complete outer loss vector differs'
    record={'case':number,'config_index':index,'model_id':m.model_id,'source_checkpoint':str(p),'source_checkpoint_sha256':sha(p),'loss_sha256':digest,'ideal_weights_sha256':hashlib.sha256(bundle['weights_by_mode'][config['criterion']].tobytes()).hexdigest(),'coefficient_sha256':{k:hashlib.sha256(bundle['coefficients'][config['criterion']][k].tobytes()).hexdigest()for k in('A','B')},'complete_outer_loss_vector_exact':True}
   else:
    if index==0:assert digest==REFS[str(number)]['terminal_pool']['loss_sha256'],'Reserved endpoint control loss differs'
    p=D/'reserved/losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(p,losses=loss)
    record={'case':number,'config_index':index,'model_id':m.model_id,'series_id':case['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'paths':1000,'inner_paths':1000,'seed':seed,'constituent_ids':ids,'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'checkpoint_sha256':sha(p),'loss_sha256':digest,'selection':m.last_selection,'validation_trace':bundle['trace'],'fit_certificate':{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in bundle['fit'][config['criterion']].items()}}
   records.append(record)
  result={'kind':kind,'case':number,'records':records,'errors':[],'coefficient_arrays_byte_verified':coefficient_checks,'inner_component_ensembles_verified':sum(len(t['eligible_this_fold'])for t in bundle['trace'])if kind=='parity'else 0,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-begin,'work_units':1000*len(x)*166*(166+len(ids)),'warmup_first_request_for_pid':COUNT==1}
 except Exception as e:result={'kind':kind,'case':number,'records':records,'errors':[{'kind':kind,'case':number,'stage':'integrity'if isinstance(e,AssertionError)else'forecast_generation','type':type(e).__name__,'message':str(e)}]}
 write(D/('parity'if kind=='parity'else'reserved/case_records')/f'{number:03d}.json',result);state('idle');return result

def main():
 with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not D.exists(),'Inspect existing execution; never overwrite'
  assert read(O/'goal_progress.json')['goal_status']=='active'
  prior=read(PRE/'summary.json');audit=read(PRE/'verification_receipt.json');assert audit['verified']and audit['summary_sha256']==sha(PRE/'summary.json')and prior['qualification_valid']and not prior['errors']
  path=prior['models'][identity(CONFIGS[1])];assert path['beats_matched_terminal_pool'],'New path method did not pass development; do not launch reserved stage'
  qualified=read(C/'qualification_receipt.json');assert qualified['tests_passed']==6 and all(sha(O/p)==h for p,h in qualified['source_hashes'].items())
  recovery=read(O/'selector_coefficient_sweep_public_recovery.json');assert recovery['remote_main_verified']and recovery['recovered_tests_passed']==6
  for p,h in qualified['source_hashes'].items():assert recovery['verified_source_hashes'][p]==h
  subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.6','--artifact-root',str(O),'--budget-gib','16'],check=True)
  source_plan=read(PRE/'plan.json');reserved=read(RES/'summary.json');reserved_plan=read(RES/'plan.json');assert reserved['qualification_valid']and not reserved['errors']and read(RES/'verification_receipt.json')['verified']
  cases=source_plan['cases'];new_cases=reserved_plan['reserved_cases'];assert len(cases)==len(new_cases)==355
  key=lambda c:(c['series_id'],json.dumps(c['origin'],sort_keys=True));assert not(set(map(key,cases))&set(map(key,new_cases)))
  old={r['case']:r for r in reserved['records']};refs=read(RES/'references.json')['cases'];references={};ref_rows={k:[]for k in('current_best','fixed_reference','terminal_pool')}
  for i in range(355):
   references[str(i)]={**refs[str(i)],'terminal_pool':{**old[i],'checkpoint':str(RES/'reserved/losses'/f'001-{i:03d}.npz')}}
   for label,r in references[str(i)].items():assert r['paths']==1000 and sha(Path(r['checkpoint']))==r['checkpoint_sha256'];ref_rows[label].append(r)
  closure={**source_plan['source_hashes'],**reserved_plan['source_hashes'],**qualified['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__))};assert all(sha(O/p)==h for p,h in closure.items())
  D.mkdir()
  for folder in('workers','parity','reserved/losses','reserved/case_records'):(D/folder).mkdir(parents=True,exist_ok=True)
  write(D/'references.json',{'cases':references});write(D/'plan.json',{'utc':now(),'parity_cases':cases,'reserved_cases':new_cases,'configs':CONFIGS,'source_hashes':closure,'paths':1000,'inner_paths':1000,'shared_worker_limit':12,'expected_parity_cases':355,'expected_reserved_cases':355,'expected_parity_outer_replays':710,'expected_reserved_forecasts':710,'expected_exact_coefficient_arrays':1132,'no_reserved_acceptance_without_all_parity':True,'no_official_promotion_from_this_stage':True,'official':False,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False})
  write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12});initialize();ref_Q={k:score_main.aggregate_records(v,ROWS)['score']for k,v in ref_rows.items()};assert all(q is not None for q in ref_Q.values())
  g=read(O/'goal_progress.json');g['stage']='selector_exact_sweep_replay_and_path_reserved';g['updated_at']=now();g['active_pilot']={'name':g['stage'],'folder':str(D),'active_path':str(D/'active.json'),'status_path':str(D/'progress.json'),'summary_path':str(D/'summary.json'),'paths':1000,'inner_paths':1000,'expected_parity_forecasts':710,'expected_reserved_forecasts':710,'shared_worker_limit':12,'official':False};write(O/'goal_progress.json',g)
  jobs=[(kind,i)for kind in('parity','reserved')for i in range(355)]
  def units(job):
   kind,i=job;c=(cases if kind=='parity'else new_cases)[i];r=ROWS[c['series_id']];return 1000*(c['origin']['position']+1)*166*(166+(len(r['rational_weights'])if r['panel_branch']=='portfolios'else 1))
  results=[];remaining={j:units(j)for j in jobs};last=0.
  with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
   pending={pool.submit(evaluate,*job):job for job in sorted(jobs,key=units,reverse=True)}
   while pending:
    done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
    for future in done:
     job=pending.pop(future);remaining.pop(job)
     try:results.append(future.result())
     except Exception as e:results.append({'kind':job[0],'case':job[1],'records':[],'errors':[{'kind':job[0],'case':job[1],'stage':'worker_failure','type':type(e).__name__,'message':str(e)}]})
    if time.monotonic()-last>5 or not pending:
     workers=[read(p)for p in(D/'workers').glob('*.json')];workers=[w for w in workers if w['phase']!='idle'];errors=[e for r in results for e in r['errors']];rates=[r['cpu_seconds']/r['work_units']for r in results if not r['errors']and not r.get('warmup_first_request_for_pid',True)];eta=None
     if rates:eta={'utc':now(),'mean_seconds_lower_bound':sum(remaining.values())*float(np.mean(rates))/12,'median_seconds_lower_bound':sum(remaining.values())*float(np.median(rates))/12,'samples':len(rates),'basis':'Measured processCPU per1000*training_rows*options*(options+assets), ideal12core lower bound; warmupfirstrequestexcluded; includes full inflight cost; excludes setup/IO/contention/fitrepair; not calibrated wall ETA'}
     write(D/'progress.json',{'utc':now(),'completed_cases':len(results),'expected_cases':710,'parity_outer_replays':sum(len(r['records'])for r in results if r['kind']=='parity'),'expected_parity_outer_replays':710,'new_reserved_forecasts':sum(len(r['records'])for r in results if r['kind']=='reserved'),'expected_reserved_forecasts':710,'active_workers':len(workers),'workers':workers,'native_failures':sum(e['stage']!='integrity'for e in errors),'scoring_or_parity_integrity_errors':sum(e['stage']=='integrity'for e in errors),'measured_workload_projection':eta,'preliminary':True});write(D/'errors.json',errors);last=time.monotonic()
  assert all(sha(O/p)==h for p,h in closure.items());errors=[e for r in results for e in r['errors']];parity=[r for r in results if r['kind']=='parity'];records=[v for r in results if r['kind']=='reserved'for v in r['records']];valid=len(parity)==355 and all(not r['errors']and len(r['records'])==2 for r in parity)and not errors and len(records)==710;models={}
  for index,config in enumerate(CONFIGS):
   rows=[r for r in records if r['config_index']==index];complete=len(rows)==355 and len({r['case']for r in rows})==355;q=score_main.aggregate_records(rows,ROWS)['score']if complete else None;models[identity(config)]={'config':config,'main_Q':q,'successful_cases':len(rows),'complete':complete,'percent_better_vs_fixed_reference':percent_better(q,ref_Q['fixed_reference'])if q is not None else None,'beats_matched_terminal_pool':q is not None and q<ref_Q['terminal_pool'],'beats_matched_old_working':q is not None and q<ref_Q['current_best']}
  summary={'utc':now(),'complete_execution':True,'paths':1000,'inner_paths':1000,'component_options':166,'qualification_valid':valid,'errors':errors,'expected_parity_outer_replays':710,'parity_outer_replays':sum(len(r['records'])for r in parity),'inner_component_ensembles_verified':sum(r.get('inner_component_ensembles_verified',0)for r in parity),'coefficient_arrays_byte_verified':sum(r.get('coefficient_arrays_byte_verified',0)for r in parity),'expected_new_reserved_forecasts':710,'new_reserved_forecasts':len(records),'models':models,'reference_main_Q':ref_Q,'official':False,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False,'records':records,'parity_receipts':parity,'case_timings':[{k:v for k,v in r.items()if k not in('records','errors')}for r in results]};assert not normalize_model_percentages(summary)[1];write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items()if k not in('records','parity_receipts','case_timings')},indent=2),flush=True)
if __name__=='__main__':main()
