"""Bounded real-data qualification of all36 internal scoring/learning configurations."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_score_alignment';D=C/'smoke1000';P=V/'candidates/asset_selector_endpoint_flat/broader1000'
sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_score_alignment import candidate as core
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def initialize():
 global PLAN,ROWS
 PLAN=read(D/'plan.json');ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}

def prepare_plan():
 assert not D.exists(),'Never overwrite a prepared or running stage'
 parent=read(P/'plan.json');summary=read(P/'summary.json');audit=read(P/'verification_receipt.json');assert audit['verified'] and audit['summary_sha256']==sha(P/'summary.json') and not summary['errors']
 q=read(C/'verification_receipt.json');assert q['tests']['passed']==7 and all(sha(Path(p))==h for p,h in q['source_sha256'].items())
 rows={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};groups={}
 for i,c in enumerate(parent['cases']):
  r=rows[c['series_id']];key=('asset',r['asset_class'])if r['panel_branch']=='assets'else('portfolio',r['family']);groups.setdefault(key,[]).append(i)
 keys=sorted(k for k in groups if k[0]=='asset');portfolio_keys=sorted(k for k in groups if k[0]=='portfolio');keys+=portfolio_keys[:6]
 assert len(keys)==12 and len({k[1]for k in keys if k[0]=='asset'})==6
 selected=[]
 for key in keys:
  ids=sorted(groups[key],key=lambda i:(parent['cases'][i]['origin']['position'],parent['cases'][i]['origin']['max_horizon'],parent['cases'][i]['series_id']))
  selected.append(ids[len(ids)//2])
 assert len(set(selected))==12
 configs=core.configurations(2);records={(r['case'],r['config_index']):r for r in summary['records']};j=next(i for i,c in enumerate(parent['configs'])if c['inner_paths']==2 and c['seed_replication']==0)
 references={};source=read(P/'references.json')['cases']
 for i,pi in enumerate(selected):
  r=records[pi,j];cp=P/'losses'/f'{j:03d}-{pi:03d}.npz';assert sha(cp)==r['checkpoint_sha256']
  references[str(i)]={'parent_flat':{**r,'checkpoint':str(cp)},'fixed_reference':source[str(pi)]['fixed_reference']}
 closure={**parent['source_hashes'],**{str(Path(p).relative_to(O)):h for p,h in q['source_sha256'].items()},str(Path(__file__).relative_to(O)):sha(Path(__file__))}
 # Include the numerical sequential adapter and learner dependencies.
 for name in ['candidate.py','learning.py']:
  path=V/'candidates/asset_selector_sequential_learning'/name;closure[str(path.relative_to(O))]=sha(path)
 assert all(sha(O/p)==h for p,h in closure.items())
 D.mkdir();write(D/'references.json',{'cases':references});write(D/'plan.json',{'utc':now(),'cases':[parent['cases'][i]for i in selected],'parent_case_indices':selected,'configs':configs,'inner_paths':2,'seed_replication':0,'outer_paths':1000,'source_hashes':closure,'references_sha256':sha(D/'references.json'),'expected_forecasts':12*36,'case_selection':'Score-blind median-history request for each6assetclasses and first6sorteddistinctportfoliofamilies from existing355contexts; smoke only, not representative performance ranking','official':False,'full_panel_catalogue_increment':0,'retained_canonical_score_unchanged':True,'qualification_receipt_sha256':sha(C/'verification_receipt.json'),'cache_expiry':'Request-local caches preserved until verified remote archival; no generic hidden cache','worker_limit':12})
 print(json.dumps({'prepared':True,'requests':12,'configurations':36,'forecasts':432,'outer_paths':1000}),flush=True)

def evaluate(i):
 case=PLAN['cases'][i];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];phase='forecast_generation';started=time.process_time()
 def state(phase,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'case':i,'phase':phase,'utc':now(),**extra})
 try:
  state('prepare_shared12scorestreams');x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,returns,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
  bundle=core.prepare(x,w,H,dates[:stop],inner_paths=2,schedule_rule='half_doubling');prepare_cpu=time.process_time()-started
  cache=D/'coefficients'/f'{i:03d}.npz';core.save_cache(bundle,cache);cachehash=sha(cache);actual=np.cumsum(returns[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000)
  refs=read(D/'references.json')['cases'][str(i)];control_replayed=False
  for j,cfg in enumerate(PLAN['configs']):
   state('forecast1000',configuration=j);cpu=time.process_time();model=core.ScorePool(cfg,dates[:stop],bundle);loss=np.empty(H);cursor=0
   for start,paths in model.terminal_chunks(x,w,H,1000,seed):
    assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
   assert cursor==H and np.isfinite(loss).all();phase='scoring_integrity'
   if cfg['criterion']=='endpoint'and cfg['scale']=='sqrt_lead'and cfg['learner']=='batch':
    with np.load(refs['parent_flat']['checkpoint'],allow_pickle=False)as old:assert np.array_equal(loss,old['losses']),'Frozen parent control changed'
    control_replayed=True
   q=score_main.origin_score(loss,returns[:stop],origin['origin_date'],dates[stop:stop+H]);phase='checkpoint_io';cp=D/'losses'/f'{j:03d}-{i:03d}.npz';np.savez_compressed(cp,losses=loss)
   record={'case':i,'config_index':j,'configuration':cfg,'model_id':core.identity(cfg),'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'outer_paths':1000,'inner_paths':2,'seed_replication':0,'final_seed':seed,'constituent_ids':ids,'score':q,'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'checkpoint_sha256':sha(cp),'coefficients_sha256':cachehash,'selection':model.last_selection,'validation_trace':bundle['trace'],'shared_prepare_cpu_seconds':prepare_cpu,'final_cpu_seconds':time.process_time()-cpu,'source_closure_digest':api.original.stable_digest(PLAN['source_hashes'])}
   write(D/'records'/f'{j:03d}-{i:03d}.json',record);records.append(record);phase='forecast_generation'
  assert control_replayed
  result={'case':i,'records':records,'errors':[],'parent_loss_replayed_exact':True,'cpu_seconds':time.process_time()-started}
 except Exception as e:result={'case':i,'records':records,'errors':[{'stage':phase,'type':type(e).__name__,'message':str(e)}],'parent_loss_replayed_exact':False,'cpu_seconds':time.process_time()-started}
 write(D/'cases'/f'{i:03d}.json',result);state('idle');return result

def run():
 plan=read(D/'plan.json');assert not(D/'summary.json').exists()and all(sha(O/p)==h for p,h in plan['source_hashes'].items());assert sha(D/'references.json')==plan['references_sha256']
 with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.3','--artifact-root',str(O),'--budget-gib','16'],check=True)
  for name in('workers','coefficients','losses','records','cases'):(D/name).mkdir()
  initialize();write(D/'active.json',{'pid':os.getpid(),'status':'running','workers':12,'utc':now()});g=read(O/'goal_progress.json');assert g['goal_status']=='active';g['stage']='score_alignment_smoke432';g['active_pilot']={'name':g['stage'],'folder':str(D),'active_path':str(D/'active.json'),'status_path':str(D/'progress.json'),'expected_forecasts':432,'paths':1000,'inner_paths':2,'official':False};write(O/'goal_progress.json',g)
  results=[]
  with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
   pending={pool.submit(evaluate,i):i for i in range(12)}
   while pending:
    done,_=concurrent.futures.wait(pending,timeout=3,return_when=concurrent.futures.FIRST_COMPLETED)
    for f in done:
     i=pending.pop(f)
     try:results.append(f.result())
     except Exception as e:results.append({'case':i,'records':[],'errors':[{'stage':'worker_failure','type':type(e).__name__,'message':str(e)}]})
    errors=[e for r in results for e in r['errors']];workers=[read(p)for p in(D/'workers').glob('*.json')];active=[w for w in workers if w['phase']!='idle']
    write(D/'progress.json',{'utc':now(),'completed_forecasts':len(list((D/'records').glob('*.json'))),'expected_forecasts':432,'completed_requests':len(results),'active_workers':len(active),'workers':active,'native_failures':sum(e['stage']=='forecast_generation'for e in errors),'scoring_integrity_errors':sum(e['stage']=='scoring_integrity'for e in errors),'infrastructure_errors':sum(e['stage']in('checkpoint_io','worker_failure')for e in errors),'preliminary':True,'ETA':None,'ETA_scope':'No calibrated same-workload timing samples yet; not zero remaining work'});write(D/'errors.json',errors)
  errors=[e for r in results for e in r['errors']];records=[r for result in results for r in result['records']];valid=not errors and len(records)==432 and all(r['parent_loss_replayed_exact']for r in results)
  assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
  write(D/'summary.json',{'utc':now(),'complete_execution':True,'qualification_valid':valid,'records':records,'errors':errors,'parent_controls_replayed':sum(r.get('parent_loss_replayed_exact',False)for r in results),'no_promotion_from_smoke':True,'official':False,'full_panel_catalogue_increment':0});write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now(),'qualification_valid':valid});print(json.dumps({'qualification_valid':valid,'forecasts':len(records),'errors':errors}),flush=True)

if __name__=='__main__':run()if '--run'in sys.argv else prepare_plan()
