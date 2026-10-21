"""Matched endpoint/path-objective contrast after the shared forecast pool frees."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_path_objective';D=C/'screen100';PRE=V/'candidates/asset_selector_universe/screen100';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_path_objective.candidate import prepare,PathCriterionPool,identity
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
CONFIGS=[{'criterion':mode,'inner_paths':100}for mode in ('terminal','path')]

def initialize():
    global ROWS,PLAN,PRIOR,COUNT
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};PLAN=read(D/'plan.json')
    PRIOR={r['case']:r for r in read(PRE/'summary.json')['records']if r['config_index']==1};COUNT=0

def state(phase,**extra):
    write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'phase':phase,'utc':now(),**extra})

def evaluate(number):
    global COUNT
    COUNT+=1;case=PLAN['cases'][number];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];errors=[];begin=time.perf_counter();cpu=time.process_time();state('prepare',case=number)
    try:
        x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
        bundle=prepare(x,w,H,dates[:stop],inner_paths=100);fit_cpu=time.process_time()-cpu;actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,100)
        for index,config in enumerate(CONFIGS):
            state('forecast',case=number,config_index=index);m=PathCriterionPool(config,dates[:stop],bundle);loss=np.empty(H);cursor=0
            for start,paths in m.terminal_chunks(x,w,H,100,seed):
                assert start==cursor and paths.shape[0]==100;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            assert cursor==H and np.isfinite(loss).all();digest=hashlib.sha256(loss.tobytes()).hexdigest();p=D/'losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(p,losses=loss)
            records.append({'case':number,'config_index':index,'model_id':identity(config),'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'paths':100,'inner_paths':100,'seed':seed,'constituent_ids':ids,'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'loss_sha256':digest,'checkpoint_sha256':sha(p),'selection':m.last_selection,'validation_trace':bundle['trace'],'fit_certificate':{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in bundle['fit'][config['criterion']].items()}})
            if index==0 and digest!=PRIOR[number]['loss_sha256']:
                errors.append({'case':number,'stage':'endpoint_control_integrity','message':'Endpoint control differs from preserved100-path terminal selector; do not draw an isolated-criterion conclusion'})
        result={'case':number,'records':records,'errors':errors,'cpu_seconds':time.process_time()-cpu,'shared_fit_cpu_seconds':fit_cpu,'wall_seconds':time.perf_counter()-begin,'work_units':100*len(x)*166*(166+len(ids)),'warmup_first_request_for_pid':COUNT==1}
    except Exception as e:
        result={'case':number,'records':records,'errors':errors+[{'case':number,'stage':'forecast_generation','type':type(e).__name__,'message':str(e)}]}
    write(D/'case_records'/f'{number:03d}.json',result);state('idle');return result

def main():
    write(C/'handoff.json',{'pid':os.getpid(),'status':'waiting_for_shared_forecast_slots','utc':now(),'worker_limit':12,'starts_no_extra_forecast_pool_while_lock_owned':True})
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert not D.exists(),'Inspect previous execution instead of overwriting it'
    goal=read(O/'goal_progress.json');assert goal['goal_status']=='active'
    prior=read(PRE/'summary.json');plan=read(PRE/'plan.json');qualification=read(C/'qualification_receipt.json')
    assert prior['complete_execution']and not prior['errors']and prior['paths']==100 and len(plan['cases'])==72
    assert qualification['tests_passed']==7 and all(sha(O/p)==h for p,h in qualification['source_hashes'].items())
    closure=dict(plan['source_hashes']);closure.update(qualification['source_hashes'])
    for p in (V/'candidates/asset_selector_universe_certified').glob('*.py'):closure[str(p.relative_to(O))]=sha(p)
    closure[str(Path(__file__).relative_to(O))]=sha(Path(__file__))
    assert all(sha(O/p)==h for p,h in closure.items())
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.15','--artifact-root',str(O),'--budget-gib','16'],check=True)
    old={r['case']:r for r in prior['records']if r['config_index']==1};assert len(old)==72
    references=read(PRE/'references.json');reference_rows={'fixed_reference':[],'current_best':[],'terminal_pool':[]}
    for number in range(72):
        r=old[number];p=PRE/'losses'/f"001-{number:03d}.npz";assert sha(p)==r['checkpoint_sha256']
        with np.load(p,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
        reference_rows['terminal_pool'].append(r)
        for label in ('fixed_reference','current_best'):
            record=references['cases'][str(number)][label];assert record['paths']==100 and sha(Path(record['checkpoint']))==record['checkpoint_sha256']
            reference_rows[label].append(record)
    D.mkdir()
    for folder in ('workers','losses','case_records'):(D/folder).mkdir()
    write(D/'plan.json',{'utc':now(),'cases':plan['cases'],'configs':CONFIGS,'source_hashes':closure,'paths':100,'inner_paths':100,'worker_limit':12,'expected_forecasts':144,'official':False,'case_selection':'Same frozen72cases/70families; adaptively reused, no independence or significance claim','hypothesis':'Change internal terminal-only CRPS to canonical log-calendar-weighted CRPS through the requested horizon; same166options and chronological folds','causality':'Only training_dates[:stop] and joint training returns enter fitting; no outer future dates/outcomes','single_main_score_unchanged':True,'diagnostics_zero_selection_weight':True,'full_panel_catalogue_increment':0,'full_panel_launch_requires_stopping_review':True,'reference_checkpoints':str(PRE/'references.json')})
    write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12});write(C/'handoff.json',{'pid':os.getpid(),'status':'running','utc':now()})
    goal['stage']='selector_calendar_path_objective_screen100';goal['updated_at']=now();goal['active_pilot']={'name':'selector_calendar_path_objective_screen100','folder':str(D),'active_path':str(D/'active.json'),'status_path':str(D/'progress.json'),'summary_path':str(D/'summary.json'),'paths':100,'inner_paths':100,'expected_forecasts':144,'shared_worker_limit':12,'official':False};write(O/'goal_progress.json',goal)
    initialize();reference_Q={k:score_main.aggregate_records(rows,ROWS)['score']for k,rows in reference_rows.items()};assert all(q is not None and np.isfinite(q)for q in reference_Q.values())
    def units(i):
        c=plan['cases'][i];r=ROWS[c['series_id']];return 100*(c['origin']['position']+1)*166*(166+(len(r['rational_weights'])if r['panel_branch']=='portfolios'else 1))
    ordered=sorted(range(72),key=units,reverse=True);first=evaluate(ordered[-1]);results=[first];assert not first['errors'],first['errors'];remaining={i:units(i)for i in ordered[:-1]};last=0.
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
        pending={pool.submit(evaluate,i):i for i in ordered[:-1]}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                number=pending.pop(future);remaining.pop(number)
                try:results.append(future.result())
                except Exception as e:results.append({'case':number,'records':[],'errors':[{'case':number,'stage':'worker_failure','type':type(e).__name__,'message':str(e)}]})
            if time.monotonic()-last>5 or not pending:
                workers=[read(p)for p in(D/'workers').glob('*.json')];workers=[w for w in workers if w['phase']!='idle'];errors=[e for r in results for e in r['errors']];rates=[r['cpu_seconds']/r['work_units']for r in results if not r['errors']and not r.get('warmup_first_request_for_pid',True)];eta=None
                if rates:eta={'utc':now(),'mean_seconds_lower_bound':sum(remaining.values())*float(np.mean(rates))/12,'median_seconds_lower_bound':sum(remaining.values())*float(np.median(rates))/12,'basis':'Observed processCPU per simulated member-training-row-option-pair work; ideal12-core lower bound, not calibrated wall ETA; full in-flight cost included; excludes setup/IO/contention/fit repair'}
                write(D/'progress.json',{'utc':now(),'completed_cases':len(results),'expected_cases':72,'completed_forecasts':sum(len(r['records'])for r in results),'expected_forecasts':144,'active_workers':len(workers),'workers':workers,'native_failures':sum(e['stage']!='endpoint_control_integrity'for e in errors),'scoring_integrity_errors':sum(e['stage']=='endpoint_control_integrity'for e in errors),'measured_workload_projection':eta,'preliminary_exploratory_only':True});write(D/'errors.json',errors);last=time.monotonic()
    assert all(sha(O/p)==h for p,h in closure.items())
    records=[v for r in results for v in r['records']];errors=[e for r in results for e in r['errors']];models={}
    for index,config in enumerate(CONFIGS):
        rows=[r for r in records if r['config_index']==index];complete=len(rows)==72 and len({r['case']for r in rows})==72;q=score_main.aggregate_records(rows,ROWS)['score']if complete else None
        models[identity(config)]={'config':config,'complete':complete,'successful_cases':len(rows),'main_Q':q,'percent_better_vs_fixed_reference':100*(1-q/reference_Q['fixed_reference'])if q is not None else None,'beats_matched_terminal_pool':q is not None and q<reference_Q['terminal_pool'],'beats_matched_old_working':q is not None and q<reference_Q['current_best']}
    summary={'utc':now(),'complete_execution':True,'paths':100,'inner_paths':100,'expected_forecasts':144,'new_successful_forecasts':len(records),'errors':errors,'reference_main_Q':reference_Q,'models':models,'qualification_valid':not errors and len(records)==144,'official':False,'full_panel_catalogue_increment':0,'no_promotion_from_this_screen':True,'independence_or_significance_claim':False,'records':records,'case_timings':[{k:v for k,v in r.items()if k not in('records','errors')}for r in results]}
    write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});write(C/'handoff.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items()if k not in('records','case_timings')},indent=2),flush=True)
if __name__=='__main__':main()
