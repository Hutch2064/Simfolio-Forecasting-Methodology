"""Broader355request information-spaced historical validation comparisons.

Default invocation prepares an immutable plan only. --run freezes source identities and obtains the existing exclusive12-worker scientific lock.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_information_origins';D=C/'study1000';REF=V/'candidates/asset_selector_inner_precision_batched/study1000';sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_information_origins import candidate as core
from candidates.asset_selector_information_origins.candidate import DesignPool,identity
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from score_percentage import percent_better
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def initialize():
    global PLAN,ROWS,COUNT
    PLAN=read(D/'plan.json');ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};COUNT=0

def configurations():
    return [{'schedule':rule,'inner_paths':n,'pool':'stack','seed_replication':0}for rule in('doubling','half_doubling')for n in(4,30)]

def prepare(*args,**kwargs):
    original=core.prefix_fits;calls=[];last=[0.]
    def measured(history,eligible):
        start=time.process_time();result=original(history,eligible);elapsed=time.process_time()-start
        calls.append({'training_rows':len(history),'cpu_seconds':elapsed,'eligible_options':len(eligible)})
        if time.monotonic()-last[0]>2:
            write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'case':CURRENT_CASE,'config_index':CURRENT_CONFIG,'phase':'prepare_prefix_fits','prefix_fits_completed':len(calls),'last_training_rows':len(history),'prefix_fit_cpu_seconds':sum(c['cpu_seconds']for c in calls),'utc':now()});last[0]=time.monotonic()
        return result
    core.prefix_fits=measured
    try:
        bundle=core.prepare(*args,**kwargs)
    finally:core.prefix_fits=original
    bundle['diagnostics']['prefix_fit_calls']=len(calls);bundle['diagnostics']['prefix_fit_cpu_seconds']=sum(c['cpu_seconds']for c in calls)
    bundle['diagnostics']['prefix_fit_rows_processed']=sum(c['training_rows']for c in calls)
    return bundle

def make_plan():
    assert not(D/'plan.json').exists(),'Existing immutable broader precision study'
    prior=V/'candidates/asset_selector_universe_precision/development1000';prior_plan=read(prior/'plan.json');refs=read(prior/'references.json')['cases']
    design=V/'candidates/asset_selector_design/study1000';audit=read(design/'verification_receipt.json');assert audit['verified'] and audit['summary_sha256']==sha(design/'summary.json')
    dp=read(design/'plan.json');ds=read(design/'summary.json');lookup={(c['series_id'],c['origin']['origin_label']):i for i,c in enumerate(dp['cases'])};old={(r['case'],r['config_index']):r for r in ds['records']}
    anchors=0
    for i,case in enumerate(prior_plan['cases']):
        r=refs[str(i)]['fixed_reference'];assert r['series_id']==case['series_id'] and r['origin_label']==case['origin']['origin_label'] and r['paths']==1000 and sha(Path(r['checkpoint']))==r['checkpoint_sha256']
        key=case['series_id'],case['origin']['origin_label']
        if key in lookup:
            anchors+=1;source=lookup[key];assert dp['cases'][source]==case
            for cfg in configurations():
                j=next(j for j,c in enumerate(dp['configs'])if c=={**cfg,'schedule':'established'});r=old[source,j];p=design/'losses'/f'{j:03d}-{source:03d}.npz';assert sha(p)==r['checkpoint_sha256'];refs[str(i)]['original_'+str(cfg['inner_paths'])]={**r,'paths':1000,'checkpoint':str(p)}
    assert anchors==72 and len(prior_plan['cases'])==355
    q=read(C/'qualification_receipt.json');assert q['tests_passed']==6 and q['worker_tests_passed']==1 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
    closure={**dp['source_hashes'],**q['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__))};assert all(sha(O/p)==h for p,h in closure.items())
    cases=prior_plan['cases'];configs=configurations()
    costs={i:c['origin']['max_horizon'] for i,c in enumerate(cases)}
    old=V/'candidates/asset_selector_precision_broad/study1000';stopped=read(O/'selector_precision_broad_stop_receipt.json')
    critical={r['case']for r in stopped['completed_records']}
    for file in sorted((old/'records').glob('*.json')):
        r=read(file);p=old/'losses'/f"{r['config_index']:03d}-{r['case']:03d}.npz";assert sha(p)==r['checkpoint_sha256'];refs[str(r['case'])]['all_origins_'+str(r['inner_paths'])]={**r,'paths':1000,'checkpoint':str(p)}
    D.mkdir();write(D/'references.json',{'cases':refs})
    write(D/'plan.json',{'utc':now(),'cases':cases,'configs':configs,'source_hashes':closure,'outer_paths':1000,'expected_cases':355,'expected_new_forecasts':1420,'shared_worker_limit':12,'official':False,'full_panel_catalogue_increment':0,'purpose':'Broader information-doubling versus half-doubling origin schedules with count-derived coverage weights;4or30inner paths, unchanged166menu and1000final paths','no_automatic_promotion':True,'no_production_precision_choice':True,'references_sha256':sha(D/'references.json'),'qualification_sha256':sha(C/'qualification_receipt.json'),'case_order':sorted(costs,key=lambda i:(i not in critical,-costs[i],i)),'case_selection':'Exact earlier frozen355case/70family development sample, no outcome-based reselection; adaptively reused evidence, no independence claim','reference_case_selection_plan_sha256':sha(prior/'plan.json'),'known72_case_anchor_references':144,'partial_all_origin_references':len(stopped['completed_records']),'scheduling':'Preserved difficult-request cohort first, then longest native horizons; neither outcomes nor scores used','phase':'prepared_not_launched'})
    print(json.dumps({'prepared':True,'new_forecasts':1420,'cases':355,'outer_paths':1000,'scientific_workers_started':0}),flush=True)

def evaluate(number):
    global COUNT,CURRENT_CASE,CURRENT_CONFIG
    CURRENT_CASE=number;COUNT+=1;case=PLAN['cases'][number];row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];records=[];errors=[]
    x,w,ids=AssetInputs(V/'inputs').training(row,stop);dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins;actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000)
    bundle=None;bundle_key=None
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
            if bundle_key!=(configuration['schedule'],n,rep):bundle=prepare(x,w,H,dates[:stop],inner_paths=n,seed_replication=rep,schedule_rule=configuration['schedule']);bundle_key=(configuration['schedule'],n,rep)
            prepare_cpu=time.process_time()-cpu
            model=DesignPool({k:configuration[k]for k in('schedule','inner_paths','pool')},dates[:stop],bundle);loss=np.empty(H);cursor=0
            write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'case':number,'config_index':index,'phase':'forecast1000','utc':now()})
            for start,paths in model.terminal_chunks(x,w,H,1000,seed):
                assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            assert cursor==H and np.isfinite(loss).all();phase='scoring_integrity'
            if 'original_'+str(n) in read(D/'references.json')['cases'][str(number)] and [(f['training_rows'],f['validation_horizon'],f['represented_origins'])for f in bundle['trace']]==core.schedule(len(x),H,'all')[1]:
                reference=read(D/'references.json')['cases'][str(number)]['original_'+str(n)]
                with np.load(reference['checkpoint'],allow_pickle=False)as saved:assert np.array_equal(loss,saved['losses']),'Established selector loss changed under the qualified fast backend'
            score=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert np.isfinite(score);phase='checkpoint_io';np.savez_compressed(loss_path,losses=loss)
            record={'case':number,'config_index':index,'configuration':configuration,'model_id':identity({k:configuration[k]for k in('schedule','inner_paths','pool')}),'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'outer_paths':1000,'inner_paths':n,'inner_seed_replication':rep,'final_seed_shared_across_inner_counts':seed,'constituent_ids':ids,'source_closure_digest':api.original.stable_digest(PLAN['source_hashes']),'score':score,'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'checkpoint_sha256':sha(loss_path),'selection':model.last_selection,'validation_trace':bundle['trace'],'cpu_seconds':time.process_time()-cpu,'prepare_cpu_seconds':prepare_cpu,'wall_seconds':time.perf_counter()-started,'first_case_for_process':COUNT==1}
            write(record_path,record);records.append(record)
        except Exception as e:errors.append({'case':number,'config_index':index,'stage':phase,'type':type(e).__name__,'message':str(e)})
    result={'case':number,'records':records,'errors':errors};write(D/'case_records'/f'{number:03d}.json',result);write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'phase':'idle','utc':now()});return result

def run():
    assert (D/'plan.json').exists() and not (D/'summary.json').exists(),'Inspect existing prepared/completed stage'
    qualification=read(C/'qualification_receipt.json');assert qualification['tests_passed']==6 and qualification['worker_tests_passed']==1
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    plan=read(D/'plan.json');assert all(sha(O/p)==h for p,h in plan['source_hashes'].items());assert sha(D/'references.json')==plan['references_sha256']
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.4','--artifact-root',str(O),'--budget-gib','16'],check=True)
    for name in ('workers','losses','records','case_records'):(D/name).mkdir(exist_ok=True)
    initialize();write(D/'active.json',{'pid':os.getpid(),'status':'running','workers':12,'utc':now()});g=read(O/'goal_progress.json');assert g['goal_status']=='active';g['stage']='selector_information_origins355';g['active_pilot']={'name':g['stage'],'folder':str(D),'status_path':str(D/'progress.json'),'active_path':str(D/'active.json'),'summary_path':str(D/'summary.json'),'expected_forecasts':plan['expected_new_forecasts'],'paths':1000,'inner_paths':[4,30],'official':False,'shared_worker_limit':12};g['updated_at']=now();write(O/'goal_progress.json',g)
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
