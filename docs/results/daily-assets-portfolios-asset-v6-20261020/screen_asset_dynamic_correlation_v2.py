"""Pipelined shared-fit216-configuration,100-path exploratory screen."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_dynamic_correlation';D=C/'screen100';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main,reporter
from inputs import AssetInputs
from candidates.asset_feature_batch import fits
from candidates.asset_dynamic_correlation import correlation
FAMILIES=('GARCH','SV_path','SV_RMS')
from candidates.asset_dynamic_correlation.candidate import CorrelationPool as FeaturePool,CONFIGS,identity
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
ANCHOR=None

def worker_state(kind,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'kind':kind,'utc':now(),**extra})
def initialize():
    global ROWS
    global ANCHOR
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
    ANCHOR=read(D/'references.json')['anchor_config']

@lru_cache(maxsize=4)
def load_bundle(number,family,digest):
    path=D/'fits'/f'{number:03d}-{family}.npz';assert sha(path)==digest
    result={name:{} for name in ('raw','mean','filter','correlation')}
    with np.load(path,allow_pickle=False) as values:
        for key in values.files:
            if '.' in key:
                section,name=key.split('.',1);value=values[key].copy();result[section][name]=value.item() if value.ndim==0 else value
            else:result[key]=values[key].item()
    return result

def fit_case(number,case):
    worker_state('fit',case=number);row=ROWS[case['series_id']];history,_,_=AssetInputs(V/'inputs').training(row,case['origin']['position']+1)
    begin=time.perf_counter();base=fits.base_fit(history);base_seconds=time.perf_counter()-begin;packets=[];errors=[];sv=None;sv_dependence=None
    for family in FAMILIES:
        try:
            start=time.perf_counter()
            if family=='SV_RMS'and sv is not None:bundle={**sv,'family':family};dep=sv_dependence
            else:
                bundle=fits.prepare(history,family,base);dep=correlation.prepare(bundle['filter']['pool'])
                if family=='SV_path':sv=bundle;sv_dependence=dep
            seconds=time.perf_counter()-start;bundle['correlation']={k:v for k,v in dep.items()if k!='fit'}
            arrays={f'{section}.{name}':value for section in ('raw','mean','filter','correlation')for name,value in bundle[section].items()}
            arrays.update(family=family,history_sha256=bundle['history_sha256']);path=D/'fits'/f'{number:03d}-{family}.npz';np.savez_compressed(path,**arrays)
            packets.append({'case':number,'family':family,'sha256':sha(path),'base_fit_compute_seconds':base_seconds,'filter_and_dependence_fit_compute_seconds':seconds,'dependence_fit':dep.get('fit',{'scalar_identity':True})})
        except Exception as e:errors.append({'case':number,'family':family,'type':type(e).__name__,'message':str(e),'stage':'training_fit'})
    worker_state('idle');return {'packets':packets,'errors':errors}

def forecast_family(number,case,packet):
    family=packet['family'];worker_state('forecast',case=number,family=family)
    row=ROWS[case['series_id']];origin=case['origin'];dates,values,origins=api.load_series(V/'inputs',row,verify_files=False)
    assert origin in origins;stop=origin['position']+1;H=origin['max_horizon'];history,weights,ids=AssetInputs(V/'inputs').training(row,stop)
    bundle=load_bundle(number,family,packet['sha256']);assert bundle['history_sha256']==hashlib.sha256(np.ascontiguousarray(history).tobytes()).hexdigest()
    realized=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,100);records=[];errors=[]
    for index,config in enumerate(CONFIGS):
        if config['family']!=family:continue
        mid=identity(config)
        try:
            start=time.perf_counter();loss=np.empty(H);cursor=0
            for first,paths in FeaturePool(config,bundle).terminal_chunks(history,weights,H,100,seed):
                assert first==cursor and paths.shape[0]==100;cursor+=paths.shape[1]
                loss[first:cursor]=empirical_crps_by_horizon(paths,realized[first:cursor])
            assert cursor==H and np.isfinite(loss).all();compute=time.perf_counter()-start
            path=D/'losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(path,losses=loss)
            record={'model_id':mid,'config_index':index,'case':number,'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'seed':seed,'paths':100,'constituent_ids':ids,'pid':os.getpid(),'simulation_CRPS_seconds_excluding_shared_fit_and_checkpoint_IO':compute,'work_units':100*H*len(ids),'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'checkpoint_sha256':sha(path)}
            if config==ANCHOR:
                reference=read(D/'references.json')['cases'][str(number)]['current_best'];assert record['loss_sha256']==reference['loss_sha256'],'Incumbent anchor loss bytes changed'
            records.append(record)
        except Exception as e:errors.append({'case':number,'model_id':mid,'config_index':index,'type':type(e).__name__,'message':str(e),'stage':'scoring_integrity' if 'anchor' in str(e).lower() else 'forecast_generation'})
    worker_state('idle');return {'records':records,'errors':errors}

def main():
    global ANCHOR
    # The active full controller releases forecast slots before audit/promotion. Wait for the
    # owning handoff to finish so this screen's current-best control cannot change mid-admission.
    while True:
        handoff=read(O/'asset_feature_handoff_active.json')
        if handoff['status'].startswith('terminal_'):break
        process=subprocess.run(['ps','-o','command=','-p',str(handoff['pid'])],capture_output=True,text=True)
        assert 'continue_asset_feature_batch.py'in process.stdout,'Current full handoff is missing; inspect, never restart'
        time.sleep(5)
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert not (D/'active.json').exists(),'Inspect an existing screen instead of restarting'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.2','--artifact-root',str(O),'--budget-gib','16'],check=True)
    source=V/'candidates/asset_feature_batch/screen100';old=read(source/'summary.json');previous_plan=read(source/'plan.json');assert old['complete_execution'] and not old['errors'] and old['paths']==100
    assert all(sha(O/p)==h for p,h in previous_plan['source_hashes'].items())
    official=read(V/'official_scoreboard.json');working=read(O/'working_incumbent.json');mid=working['model_id'];base_config=working['config']
    assert read(O/'research_workflow_policy.json')['full_panel_launch_requires_stopping_review']
    for phase in ('confirmation1000','reserved1000','broad1000'):
        result=read(V/'candidates/asset_feature_batch'/phase/'summary.json');assert result['complete_execution'] and not result['errors']
        assert result['models'][mid]['main_Q']<result['reference_main_Q']['current_best']
    ANCHOR={**base_config,'correlation':'sample'};assert ANCHOR in CONFIGS
    # Workers use the immutable references file, not changing official state.
    panel=api.load_panel(V/'inputs',verify_files=False);metadata={r['series_id']:r for r in panel['series']};cases=previous_plan['cases'];assert len(cases)==72
    old_config_index=previous_plan['configs'].index(base_config);old_records={(r['case'],r['config_index']):r for r in old['records']};old_references=read(source/'references.json')
    references={};ref_records={'current_best':[],'fixed_reference':[]}
    for number,case in enumerate(cases):
        record=old_records[number,old_config_index];path=source/'losses'/f'{old_config_index:03d}-{number:03d}.npz';assert sha(path)==record['checkpoint_sha256']
        with np.load(path,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==record['loss_sha256']
        fixed=old_references['cases'][str(number)]['fixed_reference'];assert sha(Path(fixed['checkpoint']))==fixed['checkpoint_sha256']
        references[str(number)]={'current_best':{**record,'checkpoint':str(path)},'fixed_reference':fixed}
        ref_records['current_best'].append(record);ref_records['fixed_reference'].append(fixed)
    hashes={**previous_plan['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__)),**{str(p.relative_to(O)):sha(p) for p in C.iterdir() if p.suffix in ('.py','.md')}}
    supplement=read(V/'candidates/asset_feature_batch/transitive_source_receipt.json')
    for name,digest in supplement['source_files_relative_to_v6'].items():
        assert sha(V/name)==digest;hashes['asset_level_v6/'+name]=digest
    D.mkdir(parents=True,exist_ok=True)
    for sub in ('fits','losses','workers'):(D/sub).mkdir(exist_ok=True)
    write(D/'references.json',{'cases':references,'anchor_config':ANCHOR,'working_incumbent_model_id':mid,'last_full_verified_best_model_id':official['current_best_model_id']});write(D/'plan.json',{'utc':now(),'cases':cases,'configs':CONFIGS,'paths':100,'workers':12,'source_hashes':hashes,'selected_families':70,'case_selection':'The same72cases/70families frozen before the original216-law screen; no new case selection based on observed losses.','expected_forecasts':len(cases)*len(CONFIGS),'reference_models':list(ref_records),'official':False,'full_panel_catalogue_count_unchanged':True,'working_incumbent_model_id':mid,'full_panel_admission_after_stopping_review_only':True,'exact_incumbent_anchor_required_each_case':True,'diagnostics_zero_selection_weight':True})
    write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12});records=[];errors=[];packets=[];last=0
    case_work=[100*c['origin']['max_horizon']*(len(metadata[c['series_id']]['rational_weights']) if metadata[c['series_id']]['panel_branch']=='portfolios' else 1) for c in cases]
    assert all(np.isfinite(reporter.scalar(rows,metadata)) for rows in ref_records.values())
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        pending={pool.submit(fit_case,i,c):('fit',i) for i,c in enumerate(cases)}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                kind,number=pending.pop(future)
                try:
                    result=future.result();errors.extend(result['errors'])
                    if kind=='fit':
                        packets.extend(result['packets'])
                        for packet in result['packets']:pending[pool.submit(forecast_family,number,cases[number],packet)]=('forecast',number)
                    else:records.extend(result['records'])
                except Exception as e:errors.append({'case':number,'stage':kind,'type':type(e).__name__,'message':str(e)})
            if time.monotonic()-last>5 or not pending:
                active=[read(p) for p in (D/'workers').glob('*.json') if read(p)['kind']!='idle']
                timings=np.array([r['simulation_CRPS_seconds_excluding_shared_fit_and_checkpoint_IO']/r['work_units'] for r in records])
                eta=None
                if len(timings):
                    remaining_units={family:sum(c['family']==family for c in CONFIGS)*sum(case_work) for family in FAMILIES};units=defaultdict(list)
                    for r in records:
                        family=CONFIGS[r['config_index']]['family'];remaining_units[family]-=r['work_units'];units[family].append(r['simulation_CRPS_seconds_excluding_shared_fit_and_checkpoint_IO']/r['work_units'])
                    for e in errors:
                        if 'config_index' in e:remaining_units[CONFIGS[e['config_index']]['family']]-=case_work[e['case']]
                    mean_cost=median_cost=0.;fallback=0
                    for family,remaining in remaining_units.items():
                        observed=units[family]
                        if not observed:observed=timings;fallback+=1
                        mean_cost+=max(0,remaining)*np.mean(observed);median_cost+=max(0,remaining)*np.median(observed)
                    eta={'utc':now(),'kind':'measured_filter_stratified_workload_compute_lower_bound_not_wall_ETA','mean_seconds':float(mean_cost/12),'median_seconds':float(median_cost/12),'filter_families_using_pooled_fallback':fallback,'scope':'Remaining simulation/CRPS work stratified by filter and scaled by paths*native horizon*constituents; excludes remaining shared fitting, setup, IO, contention and failure repair; no calibrated uncertainty interval'}
                write(D/'progress.json',{'utc':now(),'completed_forecasts':len(records),'expected_forecasts':len(cases)*len(CONFIGS),'prepared_filter_contexts':len(packets),'expected_filter_contexts':len(cases)*len(FAMILIES),'active_workers':len(active),'workers':active,'native_failures':sum(e['stage'] in ('training_fit','forecast_generation') for e in errors),'scoring_integrity_errors':sum(e['stage'] not in ('training_fit','forecast_generation') for e in errors),'preliminary_exploratory_only':True,'eta':eta});write(D/'errors.json',errors);last=time.monotonic()
    assert all(sha(O/p)==h for p,h in hashes.items())
    reference_Q={label:reporter.scalar(rows,metadata) for label,rows in ref_records.items()};models={};ranked=[]
    for index,config in enumerate(CONFIGS):
        mid=identity(config);rows=[r for r in records if r['model_id']==mid];complete=len(rows)==len(cases) and len({r['case'] for r in rows})==len(cases)
        q=reporter.scalar(rows,metadata) if complete else None
        model={'config':config,'config_index':index,'successful_cases':len(rows),'expected_cases':len(cases),'complete':complete,'main_Q':q,'percent_better_vs_same100path_fixed_reference':100*(1-q/reference_Q['fixed_reference']) if complete else None,'beats_same100path_current_best':complete and q<reference_Q['current_best'],'is_existing_static_sample_control':config['correlation']=='sample','eligible_as_new_dependence_hypothesis':config['correlation']!='sample'}
        models[mid]=model
        if complete:ranked.append({'model_id':mid,**model})
    ranked.sort(key=lambda r:r['main_Q']);anchor=models[identity(ANCHOR)];assert anchor['complete'] and abs(anchor['main_Q']-reference_Q['current_best'])<2e-14
    summary={'utc':now(),'complete_execution':True,'paths':100,'cases_per_model':len(cases),'families':70,'configurations':len(CONFIGS),'new_successful_forecasts':len(records),'errors':errors,'native_failures':sum(e['stage'] in ('training_fit','forecast_generation') for e in errors),'scoring_integrity_errors':sum(e['stage'] not in ('training_fit','forecast_generation') for e in errors),'scoring_protocol_unchanged':True,'reference_main_Q':reference_Q,'models':models,'complete_identical_coverage_screen_ranking':ranked,'anchor_loss_vectors_bitwise_equal_each_case':True,'official':False,'independent_validation_or_significance_claim':False,'full_panel_catalogue_count_unchanged':True,'records':records,'fit_packets':packets}
    write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items() if k not in ('records','fit_packets','models','complete_identical_coverage_screen_ranking')},default=str),flush=True)
if __name__=='__main__':main()
