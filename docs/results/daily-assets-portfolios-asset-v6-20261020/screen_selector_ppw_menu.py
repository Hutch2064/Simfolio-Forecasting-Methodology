"""Shared historical preparation for the2menus x2criteria matched contrast."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import concurrent.futures,cProfile,datetime,fcntl,hashlib,json,multiprocessing,pstats,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_ppw';D=C/'screen100';PRE=V/'candidates/asset_selector_path_objective/screen100';PRECISION=PRE.parent/'screen1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_ppw.comparison import prepare_pair
from candidates.asset_selector_ppw import candidate as expanded
from candidates.asset_selector_path_objective import candidate as original
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
SPECS=[{'menu':menu,'criterion':criterion,'inner_paths':100}for menu in ('original166','expanded168')for criterion in ('terminal','path')]
def config(spec):return {'criterion':spec['criterion'],'inner_paths':spec['inner_paths']}
def module(spec):return original if spec['menu']=='original166'else expanded
def identity(spec):return module(spec).identity(config(spec))

def initialize():
    global ROWS,PLAN,REFERENCES,COUNT
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};PLAN=read(D/'plan.json');REFERENCES=read(D/'references.json')['cases'];COUNT=0

def state(phase,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'phase':phase,'utc':now(),**extra})

def evaluate(number):
    global COUNT
    COUNT+=1;case=PLAN['cases'][number];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];errors=[];begin=time.perf_counter();cpu=time.process_time();state('prepare',case=number)
    try:
        x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
        if COUNT==1:
            profiler=cProfile.Profile(timer=time.process_time);control,new=profiler.runcall(prepare_pair,x,w,H,dates[:stop],inner_paths=100);profile=D/'profiles'/f'{os.getpid()}-{number:03d}.pstats';profiler.dump_stats(str(profile));stats=pstats.Stats(profiler)
            top=sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:30];write(profile.with_suffix('.json'),{'case':number,'paths':100,'scope':'Inclusive per-call CPU diagnostic on the first request of this process; do not sum overlapping cumulative rows or claim whole-workload savings','profile_sha256':sha(profile),'top_inclusive_functions':[{'file':k[0],'line':k[1],'function':k[2],'primitive_calls':v[0],'calls':v[1],'own_cpu_seconds':v[2],'inclusive_cpu_seconds':v[3]}for k,v in top]})
        else:control,new=prepare_pair(x,w,H,dates[:stop],inner_paths=100)
        fit_cpu=time.process_time()-cpu
        for criterion in ('terminal','path'):assert control['trace']==REFERENCES[str(number)][criterion]['validation_trace'],'Original-control historical trace differs'
        actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,100)
        for index,spec in enumerate(SPECS):
            state('forecast',case=number,config_index=index);bundle=control if spec['menu']=='original166'else new;m=module(spec).PathCriterionPool(config(spec),dates[:stop],bundle);loss=np.empty(H);cursor=0
            for start,paths in m.terminal_chunks(x,w,H,100,seed):
                assert start==cursor and paths.shape[0]==100;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            assert cursor==H and np.isfinite(loss).all();digest=hashlib.sha256(loss.tobytes()).hexdigest();p=D/'losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(p,losses=loss)
            records.append({'case':number,'config_index':index,'specification':spec,'model_id':identity(spec),'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'paths':100,'inner_paths':100,'seed':seed,'constituent_ids':ids,'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'loss_sha256':digest,'checkpoint_sha256':sha(p),'selection':m.last_selection,'validation_trace':bundle['trace'],'fit_certificate':{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in bundle['fit'][spec['criterion']].items()}})
            if spec['menu']=='original166'and digest!=REFERENCES[str(number)][spec['criterion']]['loss_sha256']:errors.append({'case':number,'stage':'control_integrity','message':'Reconstructed original-control loss vector differs','criterion':spec['criterion']})
        result={'case':number,'records':records,'errors':errors,'cpu_seconds':time.process_time()-cpu,'shared_fit_cpu_seconds':fit_cpu,'wall_seconds':time.perf_counter()-begin,'work_units':100*len(x)*168*(168+len(ids)),'warmup_first_request_for_pid':COUNT==1,'profiled':COUNT==1}
    except Exception as e:
        stage='control_integrity'if isinstance(e,AssertionError)and'Original-control'in str(e)else'forecast_generation';result={'case':number,'records':records,'errors':errors+[{'case':number,'stage':stage,'type':type(e).__name__,'message':str(e)}]}
    write(D/'case_records'/f'{number:03d}.json',result);state('idle');return result

def main():
    startup_sources={**read(PRE/'plan.json')['source_hashes'],**read(C/'qualification_receipt.json')['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__))}
    assert all(sha(O/p)==h for p,h in startup_sources.items())
    write(C/'handoff.json',{'pid':os.getpid(),'status':'waiting_for_shared_forecast_slots','utc':now(),'worker_limit':12,'loaded_source_binding_sha256':hashlib.sha256(json.dumps(startup_sources,sort_keys=True).encode()).hexdigest()})
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert not D.exists(),'Inspect previous execution instead of overwriting it'
    assert read(O/'goal_progress.json')['goal_status']=='active'
    assert read(PRECISION/'active.json')['status']=='terminal'and read(PRECISION/'summary.json')['qualification_valid']
    subprocess.run([sys.executable,str(O/'verify_selector_path_stage.py'),'screen1000'],cwd=O,check=True)
    assert read(PRECISION/'verification_receipt.json')['verified']
    prior=read(PRE/'summary.json');plan=read(PRE/'plan.json');q=read(C/'qualification_receipt.json');recovered=read(O/'selector_ppw_public_recovery.json')
    assert prior['qualification_valid']and prior['paths']==100 and len(plan['cases'])==72
    assert q['tests_passed']==9 and recovered['recovered_tests']['tests_passed']==9 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
    closure={**plan['source_hashes'],**q['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__))};assert all(sha(O/p)==h for p,h in closure.items())
    assert closure==startup_sources,'Sources changed while waiting; loaded code must be rebound and requalified before admission'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.2','--artifact-root',str(O),'--budget-gib','16'],check=True)
    old={(r['case'],r['config_index']):r for r in prior['records']};fixed=read(Path(plan['reference_checkpoints']))['cases'];references={};reference_rows={'terminal':[],'path':[],'current_best':[],'fixed_reference':[]}
    for number in range(72):
        references[str(number)]={}
        for index,label in enumerate(('terminal','path')):
            r=old[number,index];p=PRE/'losses'/f'{index:03d}-{number:03d}.npz';assert r['paths']==100 and sha(p)==r['checkpoint_sha256']
            with np.load(p,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
            references[str(number)][label]={**r,'checkpoint':str(p)};reference_rows[label].append(r)
        for label in ('current_best','fixed_reference'):
            r=fixed[str(number)][label];assert r['paths']==100 and sha(Path(r['checkpoint']))==r['checkpoint_sha256'];references[str(number)][label]=r;reference_rows[label].append(r)
    D.mkdir()
    for folder in ('workers','losses','case_records','profiles'):(D/folder).mkdir()
    write(D/'references.json',{'cases':references});write(D/'plan.json',{'utc':now(),'cases':plan['cases'],'specifications':SPECS,'source_hashes':closure,'paths':100,'inner_paths':100,'shared_worker_limit':12,'expected_outer_forecasts':288,'expected_original_control_replays':144,'expected_new_candidate_forecasts':144,'official':False,'single_main_score_unchanged':True,'diagnostics_zero_selection_weight':True,'shared_historical_preparation':'All168options once; exact166principal coefficient block for original controls','independence_or_significance_claim':False,'full_panel_catalogue_increment':0,'first_request_per_process_profiled':True,'full_panel_launch_requires_stopping_review':True})
    write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12});write(C/'handoff.json',{'pid':os.getpid(),'status':'running','utc':now()});g=read(O/'goal_progress.json');g['stage']='selector_PPW_nested_menu_screen100';g['updated_at']=now();g['active_pilot']={'name':g['stage'],'folder':str(D),'active_path':str(D/'active.json'),'status_path':str(D/'progress.json'),'summary_path':str(D/'summary.json'),'paths':100,'inner_paths':100,'expected_forecasts':288,'expected_control_replays':144,'expected_new_candidate_forecasts':144,'shared_worker_limit':12,'official':False};write(O/'goal_progress.json',g)
    initialize();ref_Q={k:score_main.aggregate_records(rows,ROWS)['score']for k,rows in reference_rows.items()};assert all(q is not None and np.isfinite(q)for q in ref_Q.values())
    def units(i):
        c=plan['cases'][i];r=ROWS[c['series_id']];return 100*(c['origin']['position']+1)*168*(168+(len(r['rational_weights'])if r['panel_branch']=='portfolios'else 1))
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
                if rates:eta={'utc':now(),'mean_seconds_lower_bound':sum(remaining.values())*float(np.mean(rates))/12,'median_seconds_lower_bound':sum(remaining.values())*float(np.median(rates))/12,'basis':'Measured processCPU per path-training-row-option-pair work; first profiled request per process excluded; ideal12core lower bound, not calibrated wall ETA; excludes setup/IO/contention/fit repair; full in-flight cost included'}
                records=[v for r in results for v in r['records']];write(D/'progress.json',{'utc':now(),'completed_cases':len(results),'expected_cases':72,'completed_forecasts':len(records),'expected_forecasts':288,'completed_control_replays':sum(r['config_index']<2 for r in records),'completed_new_candidate_forecasts':sum(r['config_index']>=2 for r in records),'active_workers':len(workers),'workers':workers,'native_failures':sum(e['stage']!='control_integrity'for e in errors),'scoring_integrity_errors':sum(e['stage']=='control_integrity'for e in errors),'measured_workload_projection':eta,'preliminary_exploratory_only':True});write(D/'errors.json',errors);last=time.monotonic()
    assert all(sha(O/p)==h for p,h in closure.items())
    records=[v for r in results for v in r['records']];errors=[e for r in results for e in r['errors']];models={}
    for index,spec in enumerate(SPECS):
        rows=[r for r in records if r['config_index']==index];complete=len(rows)==72 and len({r['case']for r in rows})==72;q=score_main.aggregate_records(rows,ROWS)['score']if complete else None
        models[identity(spec)]={'specification':spec,'complete':complete,'successful_cases':len(rows),'main_Q':q,'percent_better_vs_fixed_reference':100*(1-q/ref_Q['fixed_reference'])if q is not None else None,'beats_strongest_original_criterion':q is not None and q<min(ref_Q['terminal'],ref_Q['path'])}
    summary={'utc':now(),'complete_execution':True,'paths':100,'inner_paths':100,'expected_forecasts':288,'successful_outer_forecasts':len(records),'original_control_replays':sum(r['config_index']<2 for r in records),'new_candidate_forecasts':sum(r['config_index']>=2 for r in records),'errors':errors,'reference_main_Q':ref_Q,'models':models,'qualification_valid':not errors and len(records)==288,'official':False,'full_panel_catalogue_increment':0,'no_promotion_from_this_screen':True,'independence_or_significance_claim':False,'records':records,'case_timings':[{k:v for k,v in r.items()if k not in('records','errors')}for r in results]}
    write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});write(C/'handoff.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items()if k not in('records','case_timings')},indent=2),flush=True)
if __name__=='__main__':main()
