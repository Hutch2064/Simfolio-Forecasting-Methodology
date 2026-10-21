"""R52 Student-copula dependence isolation; frozen 72-case exploratory sample."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_r52_dependence';D=C/'screen100';sys.path.insert(0,str(V))
os.environ['SIMFOLIO_R52_DEPENDENCE_NATIVE_CACHE']=str(O/'v6_r52_native_cache')
import common,sweep_v6 as api,score_main,reporter
from inputs import AssetInputs
from candidates.asset_feature_batch import fits
from candidates.asset_r52_dependence.candidate import R52DependencePool,CONFIGS,BASE_CONFIG,identity,prepare
from candidates.asset_trained_pool.candidate import TrainedPool
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()


def initialize():
    global ROWS,ANCHOR,REFERENCES
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
    ANCHOR=read(D/'working_incumbent_snapshot.json')['config'];REFERENCES=read(D/'references.json')['cases']


def state(kind,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'kind':kind,'utc':now(),**extra})


def evaluate(number,case):
    row=ROWS[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
    state('fit',case=number);begin=time.perf_counter();history,weights,ids=AssetInputs(V/'inputs').training(row,stop)
    dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
    realized=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,100)
    base=fits.base_fit(history);bundles={};prepared={};errors=[];records=[];fit_times={}
    try:
        start=time.perf_counter();bundle=prepare(history,base);fit_times['likelihood_shared']=time.perf_counter()-start
        anchor_bundle=bundle
    except Exception as e:
        state('idle');return {'records':[],'errors':[{'case':number,'stage':'training_fit','type':type(e).__name__,'message':str(e)}],'shared_fit_seconds':time.perf_counter()-begin}
    total_fit=time.perf_counter()-begin
    def losses(model):
        result=np.empty(H);cursor=0
        for start,paths in model.terminal_chunks(history,weights,H,100,seed):
            assert start==cursor and paths.shape[0]==100;cursor+=paths.shape[1]
            result[start:cursor]=empirical_crps_by_horizon(paths,realized[start:cursor])
        assert cursor==H and np.isfinite(result).all();return result
    state('anchor',case=number)
    try:
        replay=losses(TrainedPool(ANCHOR,anchor_bundle));assert hashlib.sha256(replay.tobytes()).hexdigest()==REFERENCES[str(number)]['current_best']['loss_sha256'],'Anchor replay differs'
    except Exception as e:
        state('idle');return {'records':records,'errors':errors+[{'case':number,'stage':'scoring_integrity','type':type(e).__name__,'message':str(e)}],'shared_fit_seconds':total_fit}
    for index,config in enumerate(CONFIGS):
        state('forecast',case=number,config_index=index)
        try:
            begin=time.perf_counter();loss=losses(R52DependencePool(config,bundle));compute=time.perf_counter()-begin
            path=D/'losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(path,losses=loss)
            records.append({'model_id':identity(config),'config_index':index,'case':number,'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'seed':seed,'paths':100,'constituent_ids':ids,'pid':os.getpid(),'simulation_CRPS_seconds_excluding_shared_fit_and_checkpoint_IO':compute,'work_units':100*H*len(ids),'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'checkpoint_sha256':sha(path),'cDCC_a':bundle['dependence']['a'],'cDCC_b':bundle['dependence']['b'],'copula_nu':bundle['dependence']['nu'],'correlation_shrinkage':bundle['dependence']['target_shrinkage'],'fitted_raw_fraction':bundle['pool_fit']['raw_fraction']})
        except Exception as e:errors.append({'case':number,'config_index':index,'model_id':identity(config),'stage':'forecast_generation','type':type(e).__name__,'message':str(e)})
        write(D/'case_progress'/f'{number:03d}.json',{'records':records,'errors':errors,'shared_fit_seconds':total_fit,'filter_dependence_fit_seconds':fit_times})
    state('idle');return {'records':records,'errors':errors,'shared_fit_seconds':total_fit,'anchor_bitwise_exact':True}


def main():
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert not(D/'active.json').exists(),'Inspect previous execution instead of restarting'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.15','--artifact-root',str(O),'--budget-gib','16'],check=True)
    parent=V/'candidates/asset_trained_pool/screen100';prior=read(parent/'summary.json');prior_plan=read(parent/'plan.json');working=read(O/'working_incumbent.json')
    assert prior['complete_execution']and not prior['errors']and prior['paths']==100
    assert working['config']==BASE_CONFIG
    assert read(O/'research_workflow_policy.json')['full_panel_launch_requires_stopping_review']
    assert all(sha(O/p)==h for p,h in prior_plan['source_hashes'].items())
    panel=api.load_panel(V/'inputs',verify_files=False);metadata={r['series_id']:r for r in panel['series']};cases=prior_plan['cases'];assert len(cases)==72
    index=prior_plan['configs'].index(working['config']);old={(r['case'],r['config_index']):r for r in prior['records']};fixed=read(parent/'references.json')['cases']
    references={};reference_records={'current_best':[],'fixed_reference':[]}
    for number in range(len(cases)):
        record=old[number,index];path=parent/'losses'/f'{index:03d}-{number:03d}.npz';assert sha(path)==record['checkpoint_sha256']
        with np.load(path,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==record['loss_sha256']
        ref=fixed[str(number)]['fixed_reference'];assert sha(Path(ref['checkpoint']))==ref['checkpoint_sha256']
        references[str(number)]={'current_best':{**record,'checkpoint':str(path)},'fixed_reference':ref}
        reference_records['current_best'].append(record);reference_records['fixed_reference'].append(ref)
    D.mkdir(parents=True,exist_ok=True)
    for sub in ('losses','workers','case_progress'):(D/sub).mkdir(exist_ok=True)
    write(D/'working_incumbent_snapshot.json',working);write(D/'references.json',{'cases':references})
    hashes={**prior_plan['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__)),**{str(p.relative_to(O)):sha(p)for p in C.iterdir()if p.suffix in('.py','.md','.cpp','.json')}}
    supplement=read(V/'candidates/asset_feature_batch/transitive_source_receipt.json')
    for name,digest in supplement['source_files_relative_to_v6'].items():assert sha(V/name)==digest;hashes['asset_level_v6/'+name]=digest
    hashes['test_asset_r52_dependence.py']=sha(O/'test_asset_r52_dependence.py')
    for p in (V/'candidates/asset_dynamic_correlation').iterdir():
        if p.suffix in ('.py','.md'):hashes[str(p.relative_to(O))]=sha(p)
    hashes[str((D/'working_incumbent_snapshot.json').relative_to(O))]=sha(D/'working_incumbent_snapshot.json')
    write(D/'plan.json',{'utc':now(),'cases':cases,'configs':CONFIGS,'paths':100,'workers':12,'source_hashes':hashes,'expected_forecasts':len(cases)*len(CONFIGS),'anchor_replays_expected':len(cases),'selected_families':70,'official':False,'no_hardcoded_component_mixtures':True,'fixed_reference_model_id':'model_testing_asset_joint_blocks_replacement_v6','working_incumbent_model_id':working['model_id'],'full_panel_launch_requires_stopping_review':True,'diagnostics_zero_selection_weight':True,'full_panel_catalogue_count_unchanged':True,'case_selection':'Same frozen72case/70family sample; adaptively reused, no independence or significance claim'})
    write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12});records=[];errors=[];fit_seconds=0.;completed_cases=0;last=0
    work=[100*c['origin']['max_horizon']*(len(metadata[c['series_id']]['rational_weights'])if metadata[c['series_id']]['panel_branch']=='portfolios'else 1)for c in cases]
    ordered=sorted(range(len(cases)),key=lambda i:work[i],reverse=True)
    initialize();first=evaluate(ordered[0],cases[ordered[0]])
    assert not first['errors'],str(first['errors'])
    records.extend(first['records']);fit_seconds+=first['shared_fit_seconds'];completed_cases+=1
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
        pending={pool.submit(evaluate,i,cases[i]):i for i in ordered[1:]}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                number=pending.pop(future);completed_cases+=1
                try:
                    result=future.result();records.extend(result['records']);errors.extend(result['errors']);fit_seconds+=result['shared_fit_seconds']
                except Exception as e:errors.append({'case':number,'stage':'worker_failure','type':type(e).__name__,'message':str(e)})
            if time.monotonic()-last>5 or not pending:
                partial=[read(p)for p in(D/'case_progress').glob('*.json')];live_records=[r for batch in partial for r in batch['records']];active=[read(p)for p in(D/'workers').glob('*.json')if read(p)['kind']!='idle']
                timings=[r['simulation_CRPS_seconds_excluding_shared_fit_and_checkpoint_IO']/r['work_units']for r in live_records];remaining=max(0,sum(work)*len(CONFIGS)-sum(r['work_units']for r in live_records));eta=None
                if timings:eta={'utc':now(),'kind':'measured_workload_compute_lower_bound_not_wall_ETA','mean_seconds':remaining*float(np.mean(timings))/12,'median_seconds':remaining*float(np.median(timings))/12,'scope':'Remaining simulation/CRPS only, excluding fitting, setup, IO, contention and failure repair; pooled per path-lead-asset work, no calibrated interval'}
                write(D/'progress.json',{'utc':now(),'completed_forecasts':len(live_records),'expected_forecasts':len(cases)*len(CONFIGS),'completed_cases':completed_cases,'expected_cases':len(cases),'active_workers':len(active),'workers':active,'native_failures':sum(e['stage']!='scoring_integrity'for e in errors),'scoring_integrity_errors':sum(e['stage']=='scoring_integrity'for e in errors),'preliminary_exploratory_only':True,'eta':eta});write(D/'errors.json',errors);last=time.monotonic()
    assert all(sha(O/p)==h for p,h in hashes.items())
    reference_Q={label:reporter.scalar(rows,metadata)for label,rows in reference_records.items()};models={};ranked=[]
    for index,config in enumerate(CONFIGS):
        mid=identity(config);rows=[r for r in records if r['model_id']==mid];complete=len(rows)==len(cases)and len({r['case']for r in rows})==len(cases);q=reporter.scalar(rows,metadata)if complete else None
        model={'config':config,'config_index':index,'successful_cases':len(rows),'expected_cases':len(cases),'complete':complete,'main_Q':q,'percent_better_vs_same100path_fixed_reference':100*(1-q/reference_Q['fixed_reference'])if complete else None,'beats_same100path_current_best':complete and q<reference_Q['current_best']};models[mid]=model
        if complete:ranked.append({'model_id':mid,**model})
    ranked.sort(key=lambda r:r['main_Q']);summary={'utc':now(),'complete_execution':True,'paths':100,'cases_per_model':len(cases),'families':70,'configurations':len(CONFIGS),'new_successful_forecasts':len(records),'anchor_replays':len(cases)if not any(e['stage']=='scoring_integrity'for e in errors)else None,'shared_fit_compute_seconds':fit_seconds,'errors':errors,'native_failures':sum(e['stage']!='scoring_integrity'for e in errors),'scoring_integrity_errors':sum(e['stage']=='scoring_integrity'for e in errors),'reference_main_Q':reference_Q,'models':models,'complete_identical_coverage_screen_ranking':ranked,'anchor_loss_vectors_bitwise_equal_each_case':not any(e['stage']=='scoring_integrity'for e in errors),'official':False,'independent_validation_or_significance_claim':False,'full_panel_catalogue_count_unchanged':True,'records':records}
    write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items()if k not in('records','models','complete_identical_coverage_screen_ranking')},default=str),flush=True)
if __name__=='__main__':main()
