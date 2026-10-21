"""Broaden the qualified whole-path objective; reuse72precision cases, simulate283."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,shutil,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_path_objective';D=C/'development1000';WARM=C/'screen1000';BASE=V/'candidates/asset_selector_universe_certified/development1000';PATHS=1000;sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_path_objective.candidate import prepare,PathCriterionPool,identity,schedule
from candidates.asset_selector_stacking.energy import even_allocation
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from score_percentage import percent_better,normalize_model_percentages
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
CONFIGS=[{'criterion':criterion,'inner_paths':PATHS}for criterion in ('terminal','path')]

def initialize():
    global ROWS,PLAN,REFERENCES,COUNT
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};PLAN=read(D/'plan.json');REFERENCES=read(D/'references.json')['cases'];COUNT=0
def state(phase,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'phase':phase,'utc':now(),**extra})

def evaluate(number):
    global COUNT
    COUNT+=1;case=PLAN['cases'][number];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];errors=[];begin=time.perf_counter();cpu=time.process_time();state('prepare',case=number)
    try:
        x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
        bundle=prepare(x,w,H,dates[:stop],inner_paths=PATHS);prior=REFERENCES[str(number)]['terminal_pool'];trace=prior['validation_trace'];assert len(trace)==len(bundle['trace'])
        for old,new in zip(trace,bundle['trace']):
            for key in ('training_rows','training_sha256','validation_horizon','validation_stop','seed','eligible_this_fold'):assert old[key]==new[key],'Original endpoint control training trace differs'
        np.testing.assert_array_equal(even_allocation(bundle['weights_by_mode']['terminal'],PATHS),prior['selection']['path_counts'])
        eligible,_=schedule(len(x),H);valid=np.asarray([i for i in eligible if i not in bundle['invalid_internal_options']],dtype=np.int64)
        coefficient_file=D/'coefficients'/f'{number:03d}.npz';np.savez_compressed(coefficient_file,A_terminal=bundle['coefficients']['terminal']['A'],B_terminal=bundle['coefficients']['terminal']['B'],A_path=bundle['coefficients']['path']['A'],B_path=bundle['coefficients']['path']['B'],valid_component_indices=valid,weights_terminal=bundle['fit']['terminal']['weights'],weights_path=bundle['fit']['path']['weights']);coefficient_sha=sha(coefficient_file)
        actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,PATHS)
        for index,config in enumerate(CONFIGS):
            state('forecast',case=number,config_index=index);m=PathCriterionPool(config,dates[:stop],bundle);loss=np.empty(H);cursor=0
            for start,paths in m.terminal_chunks(x,w,H,PATHS,seed):
                assert start==cursor and paths.shape[0]==PATHS;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            assert cursor==H and np.isfinite(loss).all();digest=hashlib.sha256(loss.tobytes()).hexdigest();cp=D/'losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(cp,losses=loss)
            records.append({'case':number,'config_index':index,'model_id':identity(config),'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'paths':PATHS,'inner_paths':PATHS,'seed':seed,'constituent_ids':ids,'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'loss_sha256':digest,'checkpoint_sha256':sha(cp),'selection':m.last_selection,'validation_trace':bundle['trace'],'fit_certificate':{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in bundle['fit'][config['criterion']].items()},'coefficient_checkpoint':str(coefficient_file),'coefficient_checkpoint_sha256':coefficient_sha,'reused_verified_precision_case':False})
            if index==0:assert digest==prior['loss_sha256'],'Original endpoint control full loss vector differs'
        result={'case':number,'records':records,'errors':[],'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-begin,'work_units':PATHS*len(x)*166*(166+len(ids)),'warmup_first_request_for_pid':COUNT==1,'reused_case':False}
    except Exception as e:
        stage='control_integrity'if isinstance(e,AssertionError)else'forecast_generation';result={'case':number,'records':records,'errors':[{'case':number,'stage':stage,'type':type(e).__name__,'message':str(e)}],'reused_case':False}
    write(D/'case_records'/f'{number:03d}.json',result);state('idle');return result

def main():
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert not D.exists(),'Inspect existing execution instead of overwriting it'
    assert read(O/'goal_progress.json')['goal_status']=='active'
    warm=read(WARM/'summary.corrected.json');warm_plan=read(WARM/'plan.json');audit=read(WARM/'verification_receipt.json')
    assert warm['qualification_valid']and not warm['errors']and warm['paths']==PATHS and audit['verified']and audit['summary_sha256']==sha(WARM/'summary.json')
    assert warm['percentage_metadata_correction']['original_summary_sha256']==sha(WARM/'summary.json')and not normalize_model_percentages(warm)[1]
    assert next(m for m in warm['models'].values()if m['config']['criterion']=='path')['beats_matched_terminal_pool']
    prior=read(BASE/'summary.json');prior_plan=read(BASE/'plan.json');assert prior['complete_execution']and not prior['errors']and read(BASE.parent/'validation_receipt.json')['verified']
    cases=prior_plan['cases'];assert len(cases)==355
    closure={**warm_plan['source_hashes'],**prior_plan['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__)),'score_percentage.py':sha(O/'score_percentage.py')};assert all(sha(O/p)==h for p,h in closure.items())
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.6','--artifact-root',str(O),'--budget-gib','16'],check=True)
    original={r['case']:r for r in prior['records']if r['config_index']==1};fixed=read(BASE/'references.json')['cases'];references={};reference_rows={'terminal_pool':[],'current_best':[],'fixed_reference':[]}
    for i in range(355):
        r=original[i];cp=BASE/'losses'/f'001-{i:03d}.npz';assert sha(cp)==r['checkpoint_sha256'];references[str(i)]={'terminal_pool':{**r,'checkpoint':str(cp)}};reference_rows['terminal_pool'].append(r)
        for label in ('current_best','fixed_reference'):
            r=fixed[str(i)][label];assert r['paths']==PATHS and sha(Path(r['checkpoint']))==r['checkpoint_sha256'];references[str(i)][label]=r;reference_rows[label].append(r)
    key=lambda c:(c['series_id'],json.dumps(c['origin'],sort_keys=True));warm_indices={key(c):i for i,c in enumerate(warm_plan['cases'])};warm_rows={(r['case'],r['config_index']):r for r in warm['records']};reused={i:warm_indices[key(c)]for i,c in enumerate(cases)if key(c)in warm_indices};assert len(reused)==72
    D.mkdir()
    for folder in ('workers','losses','case_records','coefficients'):(D/folder).mkdir()
    write(D/'references.json',{'cases':references});write(D/'plan.json',{'utc':now(),'cases':cases,'configs':CONFIGS,'source_hashes':closure,'paths':PATHS,'inner_paths':PATHS,'shared_worker_limit':12,'expected_outer_forecasts':710,'expected_cases':355,'reused_cases':72,'new_cases':283,'new_outer_forecasts_expected':566,'expected_new_path_forecasts':283,'expected_native_control_replays':283,'new_coefficients_expected':283,'coefficient_scope':'Newly simulated283cases only; matrices enable solver audits and later numerical optimization without repeating historical simulation. Reused72precision cases retain their original evidence and have no new matrices.','official':False,'full_panel_catalogue_increment':0,'no_full_panel_promotion_from_this_development':True,'independence_or_significance_claim':False,'full_panel_launch_requires_stopping_review':True,'reference_precision_summary':str(WARM/'summary.corrected.json'),'reference_precision_summary_sha256':sha(WARM/'summary.corrected.json')})
    results=[]
    for number,source_case in reused.items():
        records=[]
        for index in (0,1):
            old=warm_rows[source_case,index];cp=WARM/'losses'/f'{index:03d}-{source_case:03d}.npz';assert old['paths']==PATHS and sha(cp)==old['checkpoint_sha256']
            with np.load(cp,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==old['loss_sha256']
            if index==0:assert old['loss_sha256']==original[number]['loss_sha256']
            target=D/'losses'/f'{index:03d}-{number:03d}.npz';shutil.copy2(cp,target);assert sha(target)==old['checkpoint_sha256'];records.append({**old,'case':number,'source_case':source_case,'source_checkpoint':str(cp),'source_checkpoint_sha256':sha(cp),'reused_verified_precision_case':True,'coefficient_checkpoint':None,'coefficient_checkpoint_sha256':None})
        result={'case':number,'records':records,'errors':[],'reused_case':True};write(D/'case_records'/f'{number:03d}.json',result);results.append(result)
    write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12});g=read(O/'goal_progress.json');g['stage']='selector_path_objective_development355';g['updated_at']=now();g['active_pilot']={'name':g['stage'],'folder':str(D),'active_path':str(D/'active.json'),'status_path':str(D/'progress.json'),'summary_path':str(D/'summary.json'),'paths':PATHS,'inner_paths':PATHS,'expected_forecasts':710,'reused_forecasts':144,'new_outer_forecasts_expected':566,'shared_worker_limit':12,'official':False};write(O/'goal_progress.json',g)
    initialize();ref_Q={k:score_main.aggregate_records(rows,ROWS)['score']for k,rows in reference_rows.items()};assert all(q is not None and np.isfinite(q)for q in ref_Q.values())
    def units(i):
        c=cases[i];r=ROWS[c['series_id']];return PATHS*(c['origin']['position']+1)*166*(166+(len(r['rational_weights'])if r['panel_branch']=='portfolios'else 1))
    remaining={i:units(i)for i in range(355)if i not in reused};last=0.
    # The unchanged numerical method already passed the complete72case precision
    # stage. Start the shared pool immediately; every new case still replays its
    # preserved endpoint control and saves both complete loss vectors.
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
        pending={pool.submit(evaluate,i):i for i in sorted(remaining,key=units,reverse=True)}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                number=pending.pop(future);remaining.pop(number)
                try:results.append(future.result())
                except Exception as e:results.append({'case':number,'records':[],'errors':[{'case':number,'stage':'worker_failure','type':type(e).__name__,'message':str(e)}],'reused_case':False})
            if time.monotonic()-last>5 or not pending:
                workers=[read(p)for p in(D/'workers').glob('*.json')];workers=[w for w in workers if w['phase']!='idle'];errors=[e for r in results for e in r['errors']];rates=[r['cpu_seconds']/r['work_units']for r in results if not r['errors']and not r['reused_case']and not r.get('warmup_first_request_for_pid',True)];eta=None
                if rates:eta={'utc':now(),'mean_seconds_lower_bound':sum(remaining.values())*float(np.mean(rates))/12,'median_seconds_lower_bound':sum(remaining.values())*float(np.median(rates))/12,'basis':'Measured processCPU per member-training-row-option-pair work, ideal12core lower bound; first request per process excluded; not wall ETA; full in-flight cost included; excludes setup/IO/contention/fit repair'}
                records=[v for r in results for v in r['records']];write(D/'progress.json',{'utc':now(),'completed_cases':len(results),'expected_cases':355,'completed_forecasts':len(records),'expected_forecasts':710,'reused_forecasts':144,'new_forecasts':sum(not r['reused_verified_precision_case']for r in records),'active_workers':len(workers),'workers':workers,'native_failures':sum(e['stage']!='control_integrity'for e in errors),'scoring_integrity_errors':sum(e['stage']=='control_integrity'for e in errors),'measured_workload_projection':eta,'preliminary':True});write(D/'errors.json',errors);last=time.monotonic()
    assert all(sha(O/p)==h for p,h in closure.items())
    records=[v for r in results for v in r['records']];errors=[e for r in results for e in r['errors']];models={}
    for index,config in enumerate(CONFIGS):
        rows=[r for r in records if r['config_index']==index];complete=len(rows)==355 and len({r['case']for r in rows})==355;q=score_main.aggregate_records(rows,ROWS)['score']if complete else None
        models[identity(config)]={'config':config,'complete':complete,'successful_cases':len(rows),'main_Q':q,'percent_better_vs_fixed_reference':percent_better(q,ref_Q['fixed_reference'])if q is not None else None,'beats_matched_terminal_pool':q is not None and q<ref_Q['terminal_pool'],'beats_matched_old_working':q is not None and q<ref_Q['current_best']}
    summary={'utc':now(),'complete_execution':True,'paths':PATHS,'inner_paths':PATHS,'expected_outer_forecasts':710,'successful_outer_forecasts':len(records),'reused_cases':72,'new_cases':283,'errors':errors,'reference_main_Q':ref_Q,'models':models,'qualification_valid':not errors and len(records)==710,'official':False,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False,'no_full_panel_promotion':True,'records':records,'case_timings':[{k:v for k,v in r.items()if k not in('records','errors')}for r in results]};assert not normalize_model_percentages(summary)[1]
    write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items()if k not in('records','case_timings')},indent=2),flush=True)
if __name__=='__main__':main()
