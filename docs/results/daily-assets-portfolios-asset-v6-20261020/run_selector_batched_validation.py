"""One12-worker pool: complete development parity plus distinct reserved forecasts."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_batched';D=C/'validation_and_reserved';PRE=V/'candidates/asset_selector_universe_certified/development1000';R=V/'candidates/asset_trained_pool/reserved1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_batched.candidate import prepare,HorizonPool,CONFIGS,identity
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def initialize():
 global ROWS,PLAN,PRIOR,COUNT
 ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};PLAN=read(D/'plan.json');PRIOR={(r['case'],r['config_index']):r for r in read(PRE/'summary.json')['records']};COUNT=0

def state(phase,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'phase':phase,'utc':now(),**extra})

def evaluate(kind,number):
 global COUNT
 COUNT+=1;case=PLAN[kind+'_cases'][number];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];begin=time.perf_counter();state('prepare',job_kind=kind,case=number)
 try:
  x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
  actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000);cpu=time.process_time();bundle=prepare(x,w,H);fit_cpu=time.process_time()-cpu
  if kind=='parity':
   reference=PRIOR[number,0];assert bundle['trace']==reference['validation_trace'],'Historical component ensemble identity/hash mismatch'
   assert bundle['trace']==PRIOR[number,1]['validation_trace']
   for mode,index in(('select',0),('stack',1)):
    expected=np.asarray(PRIOR[number,index]['selection']['ideal_weights'],dtype=float);assert hashlib.sha256(bundle['weights_by_mode'][mode].tobytes()).hexdigest()==hashlib.sha256(expected.tobytes()).hexdigest(),'Learned weights differ'
  configs=list(enumerate(CONFIGS))if kind=='parity'else[(1,CONFIGS[1])];simulation_cpu=0.
  for index,config in configs:
   state('forecast',job_kind=kind,case=number,config_index=index);cpu=time.process_time();m=HorizonPool(config,bundle);loss=np.empty(H);cursor=0
   for start,paths in m.terminal_chunks(x,w,H,1000,seed):
    assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
   assert cursor==H and np.isfinite(loss).all();simulation_cpu+=time.process_time()-cpu;digest=hashlib.sha256(loss.tobytes()).hexdigest()
   if kind=='parity':
    old=PRIOR[number,index];p=PRE/'losses'/f'{index:03d}-{number:03d}.npz';assert sha(p)==old['checkpoint_sha256'];assert digest==old['loss_sha256'],'Complete outer CRPS loss vector differs'
    record={'case':number,'config_index':index,'model_id':m.model_id,'loss_sha256':digest,'source_checkpoint':str(p),'source_checkpoint_sha256':sha(p),'exact_outer_loss_replay':True}
   else:
    p=D/'reserved/losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(p,losses=loss)
    record={'case':number,'config_index':index,'model_id':m.model_id,'series_id':case['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'paths':1000,'seed':seed,'constituent_ids':ids,'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'loss_sha256':digest,'checkpoint_sha256':sha(p),'selection':m.last_selection,'validation_trace':bundle['trace'],'fit_certificate':{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in bundle['fit'].items()}}
   records.append(record)
  result={'job_kind':kind,'case':number,'records':records,'cpu_seconds':fit_cpu+simulation_cpu,'request_wall_seconds':time.perf_counter()-begin,'warmup_first_request_for_pid':COUNT==1,'pid':os.getpid(),'work_units':1000*len(x)*len(ids)*166,'inner_origins_verified':len(bundle['trace'])if kind=='parity'else 0,'inner_component_ensembles_verified':sum(len(t['eligible_this_fold'])for t in bundle['trace'])if kind=='parity'else 0,'errors':[]}
 except Exception as e:result={'job_kind':kind,'case':number,'records':records,'errors':[{'job_kind':kind,'case':number,'type':type(e).__name__,'message':str(e),'stage':'parity_integrity'if kind=='parity'else'forecast_generation'}]}
 folder='parity_receipts'if kind=='parity'else'reserved/case_records';write(D/folder/f'{number:03d}.json',result);state('idle');return result

def main():
 slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
 assert not D.exists(),'Inspect previous execution instead of restarting'
 subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.4','--artifact-root',str(O),'--budget-gib','16'],check=True)
 assert read(O/'goal_progress.json')['goal_status']=='active';prior=read(PRE/'summary.json');prior_plan=read(PRE/'plan.json');reserved=read(R/'summary.json');reserved_plan=read(R/'plan.json');working=read(O/'working_incumbent.json')
 assert prior['complete_execution']and not prior['errors']and prior['models'][identity(CONFIGS[1])]['beats_same1000path_current_best']
 assert reserved['complete_execution']and not reserved['errors']and reserved['paths']==1000
 assert all(sha(O/p)==h for p,h in prior_plan['source_hashes'].items());assert all(sha(O/p)==h for p,h in reserved_plan['source_hashes'].items())
 q=read(C/'qualification_receipt.json');assert q['tests_passed']==4 and all(sha(C/p)==h for p,h in q['source_hashes'].items())and sha(O/'test_asset_selector_batched.py')==q['test_source_sha256']
 assert read(C/'paired_inner_benchmark/summary.json')['all_component_ensembles_and_joint_terminal_matrix_bitwise_equal']
 parity_cases=prior_plan['cases'];cases=reserved_plan['cases'];assert len(parity_cases)==len(cases)==355
 signature=lambda c:(c['series_id'],json.dumps(c['origin'],sort_keys=True));assert not(set(map(signature,parity_cases))&set(map(signature,cases)))
 index=reserved_plan['configs'].index(working['config']);old={(r['case'],r['config_index']):r for r in reserved['records']};fixed=read(R/'references.json')['cases'];references={};ref_rows={'current_best':[],'fixed_reference':[]}
 for number in range(355):
  r={**old[number,index],'checkpoint':str(R/'losses'/f'{index:03d}-{number:03d}.npz')};f=fixed[str(number)]['fixed_reference']
  for record in(r,f):
   assert record['paths']==1000 and sha(Path(record['checkpoint']))==record['checkpoint_sha256']
   with np.load(record['checkpoint'],allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==record['loss_sha256']
  references[str(number)]={'current_best':r,'fixed_reference':f};ref_rows['current_best'].append(r);ref_rows['fixed_reference'].append(f)
 D.mkdir();
 for folder in('workers','parity_receipts','reserved/losses','reserved/case_records'):(D/folder).mkdir(parents=True,exist_ok=True)
 write(D/'references.json',{'cases':references});write(D/'working_incumbent_snapshot.json',working)
 closure={**prior_plan['source_hashes'],**reserved_plan['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__)),'test_asset_selector_batched.py':sha(O/'test_asset_selector_batched.py')}
 for p in C.glob('*.py'):closure[str(p.relative_to(O))]=sha(p)
 write(D/'plan.json',{'utc':now(),'parity_cases':parity_cases,'reserved_cases':cases,'source_hashes':closure,'paths':1000,'inner_paths':1000,'shared_worker_limit':12,'expected_parity_requests':355,'expected_parity_outer_replays':710,'expected_new_reserved_forecasts':355,'retained_config':CONFIGS[1],'official':False,'full_panel_launch_requires_stopping_review':True,'full_panel_catalogue_increment':0,'case_order':'Longest estimated simulation work first based on training_rows*assets; no financial score selection or cap','source_compatibility_claim_pending_all355_parity':True,'reserved_slice_not_independent_validation':True})
 write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12});initialize();refs_Q={k:score_main.aggregate_records(v,ROWS)['score']for k,v in ref_rows.items()}
 jobs=[(kind,i)for kind in('parity','reserved')for i in range(355)]
 def units(job):
  kind,i=job;c=(parity_cases if kind=='parity'else cases)[i];r=ROWS[c['series_id']];return 1000*(c['origin']['position']+1)*(len(r['rational_weights'])if r['panel_branch']=='portfolios'else 1)*166
 # Native smoke uses the previously measured cheapest parity request; it does
 # not select a model or financial test case based on observed forecast skill.
 first=min((j for j in jobs if j[0]=='parity'),key=units);results=[evaluate(*first)];assert not results[0]['errors'],results[0]['errors'];jobs.remove(first);jobs.sort(key=units,reverse=True);remaining={j:units(j)for j in jobs};last=0.
 with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
  pending={pool.submit(evaluate,*j):j for j in jobs}
  while pending:
   done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:
    job=pending.pop(future);remaining.pop(job)
    try:results.append(future.result())
    except Exception as e:results.append({'job_kind':job[0],'case':job[1],'records':[],'errors':[{'job_kind':job[0],'case':job[1],'stage':'worker_failure','type':type(e).__name__,'message':str(e)}]})
   if time.monotonic()-last>5 or not pending:
    workers=[read(p)for p in(D/'workers').glob('*.json')];workers=[w for w in workers if w['phase']!='idle'];errors=[e for r in results for e in r['errors']];eta={}
    for kind in('parity','reserved'):
     rates=[r['cpu_seconds']/r['work_units']for r in results if r['job_kind']==kind and not r['errors']and not r.get('warmup_first_request_for_pid',True)];left=sum(u for j,u in remaining.items()if j[0]==kind)
     eta[kind]={'mean_seconds_lower_bound':left*float(np.mean(rates))/12,'median_seconds_lower_bound':left*float(np.median(rates))/12,'samples':len(rates)}if rates else None
    write(D/'progress.json',{'utc':now(),'new_reserved_forecasts':sum(len(r['records'])for r in results if r['job_kind']=='reserved'),'expected_new_reserved_forecasts':355,'parity_outer_replays':sum(len(r['records'])for r in results if r['job_kind']=='parity'),'expected_parity_outer_replays':710,'completed_requests':len(results),'expected_requests':710,'active_workers':len(workers),'workers':workers,'native_failures':sum(e['stage']!='parity_integrity'for e in errors),'scoring_or_parity_integrity_errors':sum(e['stage']=='parity_integrity'for e in errors),'measured_workload_projection':{'utc':now(),'basis':'Observed processCPU per1000*training_rows*assets*166, by task kind; first request per process excluded. Ideal12-core compute lower bound; excludes setup/IO/contention and failed-fit repair; full remaining in-flight cost included, not calibrated wall ETA.','by_kind':eta},'preliminary':True});write(D/'errors.json',errors);last=time.monotonic()
 assert all(sha(O/p)==h for p,h in closure.items())
 errors=[e for r in results for e in r['errors']];parity=[r for r in results if r['job_kind']=='parity'];records=[record for r in results if r['job_kind']=='reserved'for record in r['records']];all_parity=len(parity)==355 and all(not r['errors']and len(r['records'])==2 for r in parity);complete=len(records)==355 and len({r['case']for r in records})==355;q=score_main.aggregate_records(records,ROWS)['score']if complete else None
 summary={'utc':now(),'complete_execution':True,'paths':1000,'inner_paths':1000,'component_options':166,'all355_development_requests_exactly_replayed':all_parity,'parity_outer_loss_vectors_verified':sum(len(r['records'])for r in parity),'inner_component_terminal_ensembles_verified':sum(r.get('inner_component_ensembles_verified',0)for r in parity),'new_reserved_forecasts':len(records),'expected_reserved_forecasts':355,'reserved_main_Q':q,'reference_main_Q':refs_Q,'reserved_percent_better_vs_fixed_reference':100*(1-q/refs_Q['fixed_reference'])if complete else None,'reserved_beats_current_working':complete and q<refs_Q['current_best'],'qualification_valid':all_parity and not errors,'errors':errors,'full_panel_catalogue_increment':0,'official':False,'independence_or_significance_claim':False,'records':records,'validation_receipts':parity}
 write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items()if k not in('records','validation_receipts')},indent=2),flush=True)
if __name__=='__main__':main()
