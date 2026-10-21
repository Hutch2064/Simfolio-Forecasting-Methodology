"""Matched endpoint precision/seed study with exact request-local fit reuse.

Default invocation prepares an immutable plan only. --run freezes source identities and obtains the existing exclusive12-worker scientific lock.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_endpoint_flat';D=C/'broader1000';REF=V/'candidates/asset_selector_inner_precision_batched/study1000';sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_endpoint_flat import candidate as core
from candidates.asset_selector_endpoint_flat.candidate import MenuPool,identity
from candidates.asset_selector_endpoint_flat.cache import PreparedCache
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from score_percentage import percent_better
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def initialize():
    global PLAN,ROWS,COUNT
    PLAN=read(D/'plan.json');ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};COUNT=0

def configurations():
    return [{'schedule':'half_doubling','inner_paths':n,'menu':menu,'seed_replication':rep}for n in(2,8)for rep in(0,1,2)for menu in core.MENUS]

def prepare(*args,**kwargs):
    unpatched=core.prefix_fits;original=CURRENT_CACHE.get;calls=[];last=[0.];before_cpu=CURRENT_CACHE.fit_cpu_seconds;before_fits=CURRENT_CACHE.actual_fits
    def measured(history,eligible,weights):
        start=time.process_time();result=original(history,eligible,weights);elapsed=time.process_time()-start
        calls.append({'training_rows':len(history),'cpu_seconds':elapsed,'eligible_options':len(eligible)})
        if time.monotonic()-last[0]>2:
            write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'case':CURRENT_CASE,'config_index':CURRENT_CONFIG,'phase':'prepare_prefix_fits','prefix_fits_completed':len(calls),'last_training_rows':len(history),'prefix_fit_cpu_seconds':sum(c['cpu_seconds']for c in calls),'utc':now()});last[0]=time.monotonic()
        return result
    core.prefix_fits=measured
    try:
        bundle=core.prepare(*args,**kwargs)
    finally:core.prefix_fits=unpatched
    bundle['diagnostics'].update(prefix_fit_requests=len(calls),prefix_access_cpu_seconds=sum(c['cpu_seconds']for c in calls),new_shared_fit_cpu_seconds=CURRENT_CACHE.fit_cpu_seconds-before_cpu,new_shared_fits=CURRENT_CACHE.actual_fits-before_fits,shared_context_fit_cpu_seconds=CURRENT_CACHE.fit_cpu_seconds,shared_fit_cache_entries=len(CURRENT_CACHE.entries),shared_fit_cache_numpy_bytes=CURRENT_CACHE.numpy_bytes(),shared_fit_cache_id=CACHE_ID)
    return bundle

def make_plan():
    assert not(D/'plan.json').exists(),'Existing immutable precision/seed plan'
    parent=V/'candidates/asset_selector_endpoint_precision/study1000';audit=read(parent/'verification_receipt.json');prior=read(parent/'summary.json');pp=read(parent/'plan.json')
    assert audit['verified']and audit['summary_sha256']==sha(parent/'summary.json')and prior['qualification_valid']and not prior['errors']
    pilot=V/'candidates/asset_selector_endpoint_components/study1000';pa=read(pilot/'verification_receipt.json');ps=read(pilot/'summary.json');plan72=read(pilot/'plan.json');assert pa['verified']and pa['summary_sha256']==sha(pilot/'summary.json')and ps['qualification_valid']and not ps['errors']
    q=read(C/'qualification_receipt.json');assert q['tests_passed']==3 and q['worker_tests_passed']==1 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
    cases=pp['cases'];old={(r['case'],r['config_index']):r for r in prior['records']};source=read(parent/'references.json')['cases'];pilot_cases={(c['series_id'],c['origin']['origin_label']):i for i,c in enumerate(plan72['cases'])};pilot_records={(r['case'],r['config_index']):r for r in ps['records']};refs={};costs={};replays=0
    for i,case in enumerate(cases):
        refs[str(i)]={'fixed_reference':source[str(i)]['fixed_reference']};costs[i]=0.
        pi=pilot_cases.get((case['series_id'],case['origin']['origin_label']))
        for n in(2,8):
            for rep in(0,1,2):
                cfg={'schedule':'half_doubling','inner_paths':n,'pool':'stack','seed_replication':rep};j=next(j for j,c in enumerate(pp['configs'])if c==cfg);r=old[i,j];p=parent/'losses'/f'{j:03d}-{i:03d}.npz';assert sha(p)==r['checkpoint_sha256'];refs[str(i)][f'original_{n}_{rep}']={**r,'paths':1000,'checkpoint':str(p)};costs[i]=max(costs[i],r['cpu_seconds'])
                if pi is not None:
                    assert case==plan72['cases'][pi];cfg={'schedule':'half_doubling','inner_paths':n,'menu':'flat','seed_replication':rep};j=next(j for j,c in enumerate(plan72['configs'])if c==cfg);r=pilot_records[pi,j];p=pilot/'losses'/f'{j:03d}-{pi:03d}.npz';assert sha(p)==r['checkpoint_sha256'];refs[str(i)][f'flat_{n}_{rep}']={**r,'paths':1000,'checkpoint':str(p)};replays+=1
    closure={**pp['source_hashes'],**plan72['source_hashes'],**q['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__))};assert all(sha(O/p)==h for p,h in closure.items())
    configs=configurations();rows={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};baseline=score_main.aggregate_records([refs[str(i)]['fixed_reference']for i in range(len(cases))],rows)['score'];assert baseline==prior['main_reference_Q']and len(cases)==355 and len(configs)==6 and replays==432
    D.mkdir();write(D/'references.json',{'cases':refs});write(D/'plan.json',{'utc':now(),'cases':cases,'configs':configs,'source_hashes':closure,'outer_paths':1000,'expected_cases':355,'expected_new_forecasts':2130,'shared_worker_limit':12,'official':False,'full_panel_catalogue_increment':0,'purpose':'Broader confirmation of exact flattened endpoint learner at2/8internal paths and3seeds. Eliminate unused menu/PPW/SAS fitting work; replay all432matched prior-flat losses and compare all355matched original-selector controls','no_automatic_promotion':True,'no_production_precision_choice':True,'references_sha256':sha(D/'references.json'),'parent_summary_sha256':sha(parent/'summary.json'),'pilot_summary_sha256':sha(pilot/'summary.json'),'qualification_sha256':sha(C/'qualification_receipt.json'),'case_order':sorted(costs,key=lambda i:(-costs[i],i)),'case_selection':'Same355reused broader requests, including72development anchors; no independence claim','predeclared_seeds':[0,1,2],'existing_control_replays':432,'main_reference_Q':baseline,'scheduling':'Largest measured parent-caseCPUfirst; no score used','fit_reuse':'Complete historical prefixSHA256and bound portfolio weights, request lifetime only. All financial laws identical to prior flat menu; unused menu fits removed','phase':'prepared_not_launched'})
    print(json.dumps({'prepared':True,'new_forecasts':2130,'cases':355,'outer_paths':1000,'scientific_workers_started':0}),flush=True)

def evaluate(number):
    global COUNT,CURRENT_CASE,CURRENT_CONFIG,CURRENT_CACHE,CACHE_ID
    CURRENT_CASE=number;COUNT+=1;case=PLAN['cases'][number];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];errors=[]
    x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins;actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000)
    eligible,_,_=core.schedule(len(x),H,'half_doubling');eligible=[i for i in eligible if i!=0]+([166,167]if 0 in eligible else[]);CURRENT_CACHE=PreparedCache(eligible,w);CACHE_ID=f'{os.getpid()}:{time.monotonic_ns()}'
    bundle=None;bundle_key=None;shared_prepare_cpu=0.
    for index,configuration in enumerate(PLAN['configs']):
        CURRENT_CONFIG=index
        record_path=D/'records'/f'{index:03d}-{number:03d}.json';loss_path=D/'losses'/f'{index:03d}-{number:03d}.npz'
        if record_path.exists():
            old=read(record_path);assert old['case']==number and old['configuration']==configuration and old['outer_paths']==1000 and old['source_closure_digest']==api.original.stable_digest(PLAN['source_hashes']) and sha(loss_path)==old['checkpoint_sha256']
            with np.load(loss_path,allow_pickle=False)as data:assert hashlib.sha256(data['losses'].tobytes()).hexdigest()==old['loss_sha256']
            records.append(old);continue
        write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'case':number,'config_index':index,'phase':'prepare','utc':now()});started=time.perf_counter();cpu=time.process_time()
        phase='forecast_generation'
        try:
            n=configuration['inner_paths'];rep=configuration['seed_replication']
            if bundle_key!=(configuration['schedule'],n,rep):
                bundle=prepare(x,w,H,dates[:stop],inner_paths=n,seed_replication=rep,schedule_rule=configuration['schedule']);bundle_key=(configuration['schedule'],n,rep);shared_prepare_cpu=time.process_time()-cpu
            prepare_cpu=time.process_time()-cpu
            model=MenuPool({k:configuration[k]for k in('schedule','inner_paths','menu')},dates[:stop],bundle);loss=np.empty(H);cursor=0
            write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'case':number,'config_index':index,'phase':'forecast1000','utc':now()})
            for start,paths in model.terminal_chunks(x,w,H,1000,seed):
                assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            assert cursor==H and np.isfinite(loss).all();phase='scoring_integrity'
            reference=read(D/'references.json')['cases'][str(number)].get(f'flat_{n}_{rep}')
            if reference is not None:
                with np.load(reference['checkpoint'],allow_pickle=False)as saved:assert np.array_equal(loss,saved['losses']),'Shared fit cache changed a complete existing control'
            score=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert np.isfinite(score);phase='checkpoint_io';np.savez_compressed(loss_path,losses=loss)
            record={'case':number,'config_index':index,'configuration':configuration,'model_id':identity({k:configuration[k]for k in('schedule','inner_paths','menu')}),'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'outer_paths':1000,'inner_paths':n,'inner_seed_replication':rep,'final_seed_shared_across_inner_counts':seed,'constituent_ids':ids,'source_closure_digest':api.original.stable_digest(PLAN['source_hashes']),'score':score,'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'checkpoint_sha256':sha(loss_path),'selection':model.last_selection,'validation_trace':bundle['trace'],'cpu_seconds':time.process_time()-cpu,'prepare_cpu_seconds':prepare_cpu,'shared_prepare_cpu_seconds':shared_prepare_cpu,'shared_prepare_id':f'{n}:{rep}','wall_seconds':time.perf_counter()-started,'first_case_for_process':COUNT==1}
            write(record_path,record);records.append(record)
        except Exception as e:errors.append({'case':number,'config_index':index,'stage':phase,'type':type(e).__name__,'message':str(e)})
    result={'case':number,'records':records,'errors':errors};write(D/'case_records'/f'{number:03d}.json',result);write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'phase':'idle','utc':now()});return result

def run():
    assert (D/'plan.json').exists() and not (D/'summary.json').exists(),'Inspect existing prepared/completed stage'
    qualification=read(C/'qualification_receipt.json');assert qualification['tests_passed']==3 and qualification['worker_tests_passed']==1
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    plan=read(D/'plan.json');assert all(sha(O/p)==h for p,h in plan['source_hashes'].items());assert sha(D/'references.json')==plan['references_sha256']
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.4','--artifact-root',str(O),'--budget-gib','16'],check=True)
    for name in ('workers','losses','records','case_records'):(D/name).mkdir(exist_ok=True)
    initialize();write(D/'active.json',{'pid':os.getpid(),'status':'running','workers':12,'utc':now()});g=read(O/'goal_progress.json');assert g['goal_status']=='active';g['stage']='selector_endpoint_flat355';g['active_pilot']={'name':g['stage'],'folder':str(D),'status_path':str(D/'progress.json'),'active_path':str(D/'active.json'),'summary_path':str(D/'summary.json'),'expected_forecasts':plan['expected_new_forecasts'],'paths':1000,'inner_paths':[2,8],'official':False,'shared_worker_limit':12};g['updated_at']=now();write(O/'goal_progress.json',g)
    results=[];last=0.
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
        pending={pool.submit(evaluate,i):i for i in plan['case_order']}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                number=pending.pop(future)
                try:results.append(future.result())
                except Exception as e:results.append({'case':number,'records':[],'errors':[{'case':number,'stage':'worker_failure','type':type(e).__name__,'message':str(e)}]})
            if time.monotonic()-last>5 or not pending:
                workers=[read(p)for p in(D/'workers').glob('*.json')];workers=[w for w in workers if w['phase']!='idle'];errors=[e for r in results for e in r['errors']]
                write(D/'progress.json',{'utc':now(),'completed_cases':len(results),'expected_cases':len(plan['cases']),'completed_forecasts':len(list((D/'records').glob('*.json'))),'expected_forecasts':plan['expected_new_forecasts'],'active_workers':len(workers),'workers':workers,'native_failures':sum(e['stage']=='forecast_generation'for e in errors),'scoring_integrity_errors':sum(e['stage']=='scoring_integrity'for e in errors),'infrastructure_failures':sum(e['stage']in('checkpoint_io','worker_failure')for e in errors),'preliminary':True,'remaining_compute_projection':None,'projection_reason':'No calibrated within-stage workload estimate yet; do not use old timing probes'});write(D/'errors.json',errors);last=time.monotonic()
    assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
    references=read(D/'references.json')['cases'];ref_rows=[references[str(i)]['fixed_reference']for i in range(len(plan['cases']))]
    baseline=score_main.aggregate_records(ref_rows,ROWS)['score']
    control=None
    records=sorted([r for result in results for r in result['records']],key=lambda r:(r['config_index'],r['case']));errors=[e for result in results for e in result['errors']];models=[]
    for index,configuration in enumerate(plan['configs']):
        selected=[r for r in records if r['config_index']==index];complete=len(selected)==len(plan['cases']) and len({r['case']for r in selected})==len(plan['cases'])
        q=score_main.aggregate_records(selected,ROWS)['score']if complete else None
        models.append({'config_index':index,'configuration':configuration,'matched_cases':len(selected),'complete':complete,'main_Q':q,'percent_better_vs_fixed_reference':percent_better(q,baseline)if complete else None,'worker_cpu_seconds':sum(r['cpu_seconds']for r in selected)})
    summary={'utc':now(),'complete_execution':len(results)==len(plan['cases']),'qualification_valid':not errors and len(records)==plan['expected_new_forecasts'],'errors':errors,'outer_paths':1000,'main_reference_Q':baseline,'models':models,'records':records,'official':False,'full_panel_catalogue_increment':0,'automatic_stability_or_promotion_decision':False,'no_independence_or_significance_claim':True,'matched_internal_count_and_seed_with_original_controls':True,'additional_replications_and_reserved_confirmation_may_be_needed':True}
    write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items()if k!='records'},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args();run()if args.run else make_plan()
