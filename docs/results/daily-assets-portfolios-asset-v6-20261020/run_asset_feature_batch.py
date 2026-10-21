"""Pipelined shared-fit216-configuration,100-path exploratory screen."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_feature_batch';D=C/'screen100';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main,reporter
from inputs import AssetInputs
from candidates.asset_feature_batch import fits
from candidates.asset_feature_batch.candidate import FeaturePool,CONFIGS,identity
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
ANCHOR={'family':'GARCH','innovations':'empirical','raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True}

def worker_state(kind,**extra):write(D/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'kind':kind,'utc':now(),**extra})
def initialize():
    global ROWS
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}

@lru_cache(maxsize=4)
def load_bundle(number,family,digest):
    path=D/'fits'/f'{number:03d}-{family}.npz';assert sha(path)==digest
    result={name:{} for name in ('raw','mean','filter')}
    with np.load(path,allow_pickle=False) as values:
        for key in values.files:
            if '.' in key:
                section,name=key.split('.',1);value=values[key].copy();result[section][name]=value.item() if value.ndim==0 else value
            else:result[key]=values[key].item()
    return result

def fit_case(number,case):
    worker_state('fit',case=number);row=ROWS[case['series_id']];history,_,_=AssetInputs(V/'inputs').training(row,case['origin']['position']+1)
    begin=time.perf_counter();base=fits.base_fit(history);base_seconds=time.perf_counter()-begin;packets=[];errors=[]
    for family in fits.FAMILIES:
        try:
            start=time.perf_counter();bundle=fits.prepare(history,family,base);seconds=time.perf_counter()-start
            arrays={f'{section}.{name}':value for section in ('raw','mean','filter') for name,value in bundle[section].items()}
            arrays.update(family=family,history_sha256=bundle['history_sha256'])
            path=D/'fits'/f'{number:03d}-{family}.npz';np.savez_compressed(path,**arrays)
            packets.append({'case':number,'family':family,'sha256':sha(path),'base_fit_compute_seconds':base_seconds,'filter_fit_compute_seconds':seconds})
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
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert not (D/'active.json').exists(),'Inspect an existing screen instead of restarting'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.8','--artifact-root',str(O),'--budget-gib','12'],check=True)
    source=V/'candidates/asset_mean_uncertainty/fading_vol_screen100';old=read(source/'summary.json');assert old['complete'] and old['paths']==100 and old['native_failures']==0
    previous_plan=read(source/'plan.json');assert all(sha(O/p)==h for p,h in previous_plan['source_hashes'].items())
    official=read(V/'official_scoreboard.json');assert official['current_best_model_id']=='model_testing_asset_circular_raw_GARCH_fade_FHS_mean_v1'
    panel=api.load_panel(V/'inputs',verify_files=False);metadata={r['series_id']:r for r in panel['series']};groups=defaultdict(list)
    for case in previous_plan['cases']:groups[metadata[case['series_id']]['family']].append(case)
    selected=[]
    for name,cases in sorted(groups.items()):selected.append(sorted(cases,key=lambda c:(c['origin']['position'],c['series_id'],c['origin']['origin_label']))[len(cases)//2])
    quarter=lambda c:(c['origin']['origin_date'][:4],(int(c['origin']['origin_date'][5:7])-1)//3)
    asset_quarters={quarter(c) for c in selected if metadata[c['series_id']]['panel_branch']=='assets'}
    for q in sorted({quarter(c) for c in selected if metadata[c['series_id']]['panel_branch']=='portfolios'}-asset_quarters):
        candidates=[c for c in previous_plan['cases'] if quarter(c)==q and metadata[c['series_id']]['panel_branch']=='assets'];assert candidates
        selected.append(sorted(candidates,key=lambda c:(c['series_id'],c['origin']['origin_label']))[0])
    assert len(groups)==70
    cases=sorted(selected,key=lambda c:(metadata[c['series_id']]['asset_class'],metadata[c['series_id']]['family'],c['series_id'],c['origin']['origin_label']))
    lookup={(r['series_id'],r['origin_label'],r['arm']):r for r in old['records']};references={};ref_records={'current_best':[],'fixed_reference':[]}
    for i,case in enumerate(cases):
        references[str(i)]={}
        for arm,label in [('garch','current_best'),('fixed_reference','fixed_reference')]:
            record=lookup[case['series_id'],case['origin']['origin_label'],arm]
            original_number=previous_plan['cases'].index(case);path=Path(record['verified_original_checkpoint']) if record['reused'] else source/arm/f'{original_number:04}.npz'
            with np.load(path,allow_pickle=False) as z:loss=z['losses']
            assert hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256'];references[str(i)][label]={**record,'checkpoint':str(path),'checkpoint_sha256':sha(path)};ref_records[label].append(record)
    hashes={**previous_plan['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__)),**{str(p.relative_to(O)):sha(p) for p in C.iterdir() if p.suffix in ('.py','.md')}}
    D.mkdir(parents=True,exist_ok=True)
    for sub in ('fits','losses','workers'):(D/sub).mkdir(exist_ok=True)
    write(D/'references.json',{'cases':references});write(D/'plan.json',{'utc':now(),'cases':cases,'configs':CONFIGS,'paths':100,'workers':12,'source_hashes':hashes,'selected_families':70,'case_selection':'Training-position median per existing family, with the minimum predeclared asset-quarter support needed by the unchanged canonical reducer. No loss values enter selection.','expected_forecasts':len(cases)*len(CONFIGS),'reference_models':list(ref_records),'official':False,'full_panel_catalogue_count_unchanged':True,'exact_incumbent_anchor_required_each_case':True,'diagnostics_zero_selection_weight':True})
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
                    remaining_units={family:24*sum(case_work) for family in fits.FAMILIES};units=defaultdict(list)
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
                write(D/'progress.json',{'utc':now(),'completed_forecasts':len(records),'expected_forecasts':len(cases)*len(CONFIGS),'prepared_filter_contexts':len(packets),'expected_filter_contexts':len(cases)*len(fits.FAMILIES),'active_workers':len(active),'workers':active,'native_failures':sum(e['stage'] in ('training_fit','forecast_generation') for e in errors),'scoring_integrity_errors':sum(e['stage'] not in ('training_fit','forecast_generation') for e in errors),'preliminary_exploratory_only':True,'eta':eta});write(D/'errors.json',errors);last=time.monotonic()
    assert all(sha(O/p)==h for p,h in hashes.items())
    reference_Q={label:reporter.scalar(rows,metadata) for label,rows in ref_records.items()};models={};ranked=[]
    for index,config in enumerate(CONFIGS):
        mid=identity(config);rows=[r for r in records if r['model_id']==mid];complete=len(rows)==len(cases) and len({r['case'] for r in rows})==len(cases)
        q=reporter.scalar(rows,metadata) if complete else None
        model={'config':config,'config_index':index,'successful_cases':len(rows),'expected_cases':len(cases),'complete':complete,'main_Q':q,'percent_better_vs_same100path_fixed_reference':100*(1-q/reference_Q['fixed_reference']) if complete else None,'beats_same100path_current_best':complete and q<reference_Q['current_best']}
        models[mid]=model
        if complete:ranked.append({'model_id':mid,**model})
    ranked.sort(key=lambda r:r['main_Q']);anchor=models[identity(ANCHOR)];assert anchor['complete'] and abs(anchor['main_Q']-reference_Q['current_best'])<2e-14
    summary={'utc':now(),'complete_execution':True,'paths':100,'cases_per_model':len(cases),'families':70,'configurations':len(CONFIGS),'new_successful_forecasts':len(records),'errors':errors,'native_failures':sum(e['stage'] in ('training_fit','forecast_generation') for e in errors),'scoring_integrity_errors':sum(e['stage'] not in ('training_fit','forecast_generation') for e in errors),'scoring_protocol_unchanged':True,'reference_main_Q':reference_Q,'models':models,'complete_identical_coverage_screen_ranking':ranked,'anchor_loss_vectors_bitwise_equal_each_case':True,'official':False,'independent_validation_or_significance_claim':False,'full_panel_catalogue_count_unchanged':True,'records':records,'fit_packets':packets}
    write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()});print(json.dumps({k:v for k,v in summary.items() if k not in ('records','fit_packets','models','complete_identical_coverage_screen_ranking')},default=str),flush=True)
if __name__=='__main__':main()
