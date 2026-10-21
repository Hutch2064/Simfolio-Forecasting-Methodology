"""Wait for the live precision writer, then qualify the isolated solver repair."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,fcntl,hashlib,json,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_universe_certified';OLD=V/'candidates/asset_selector_universe_precision/development1000';D=C/'repair_qualification';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_universe_certified.candidate import HorizonPool,CONFIGS,prepare
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(status,**extra):write(O/'selector_solver_handoff_active.json',{'pid':os.getpid(),'status':status,'utc':now(),**extra})

def main():
    state('waiting_for_current_scientific_writer',prior_controller_pid=read(OLD/'active.json')['pid'])
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert read(O/'goal_progress.json')['goal_status']=='active','Do not resume a paused goal'
    prior=read(OLD/'summary.json');plan=read(OLD/'plan.json')
    assert prior['complete_execution']and read(OLD/'active.json')['status']=='terminal'
    assert all(sha(O/p)==h for p,h in plan['source_hashes'].items()),'Prior writer source closure changed'
    state('qualification_tests')
    with (O/'selector_solver_repair_tests.log').open('w')as log:
        subprocess.run([sys.executable,'-m','unittest','test_selector_convex_certificate','test_asset_selector_universe_certified','-v'],cwd=O,stdout=log,stderr=subprocess.STDOUT,check=True)
    recovered=O/'v6_public_recovery/18c4f734da2e'
    with (O/'selector_pool_recovered_tests.log').open('w')as log:
        subprocess.run([sys.executable,'-m','unittest','test_asset_selector_stacking','test_asset_selector_universe','test_asset_selector_universe_precision','-v'],cwd=recovered,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(C/'qualification_receipt.json',{'utc':now(),'tests_passed':10,'recovered_preceding_source_tests':21,'source_hashes':{p.name:sha(p)for p in C.glob('*.py')},'tests_before_actual_request_replay':True})
    panel=api.load_panel(V/'inputs',verify_files=False);rows={r['series_id']:r for r in panel['series']};refs=read(OLD/'references.json')['cases']
    successful={r['case']for r in prior['records']};failures=sorted({e['case']for e in prior['errors']if e['stage']=='training_fit'and 'Iteration limit reached'in e['message']})
    assert failures,'Inspect current error state before claiming a repair'
    replay=min((i for i in successful if rows[plan['cases'][i]['series_id']]['panel_branch']=='portfolios'),key=lambda i:plan['cases'][i]['origin']['max_horizon'])
    selected=failures+[replay];D.mkdir(exist_ok=False);(D/'losses').mkdir();records=[]
    source_hashes={str(p.relative_to(O)):sha(p)for folder in('asset_selector_stacking','asset_selector_universe','asset_selector_universe_certified')for p in(V/'candidates'/folder).glob('*.py')}
    write(D/'plan.json',{'utc':now(),'case_indices':selected,'prior_failed_cases':failures,'successful_multivariate_replay_case':replay,'paths':1000,'inner_paths':1000,'source_hashes':source_hashes,'diagnostic_repair_only':True,'no_panel_score':True})
    for number in selected:
        state('actual_request_repair_or_replay',case=number,remaining_cases=len(selected)-len({r['case']for r in records}))
        case=plan['cases'][number];row=rows[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];x,w,ids=AssetInputs(V/'inputs').training(row,stop)
        dates,values,_=api.load_series(V/'inputs',row,verify_files=False);actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000);begin=time.perf_counter();bundle=prepare(x,w,H);fit_seconds=time.perf_counter()-begin
        coefficient_path=D/f'internal-coefficients-{number:03d}.npz'
        np.savez_compressed(coefficient_path,A=bundle['coefficients']['A'],B=bundle['coefficients']['B'],valid_indices=np.asarray(bundle['valid_bank_indices']),weights=bundle['fit']['weights'])
        for index,config in enumerate(CONFIGS):
            model=HorizonPool(config,bundle);loss=np.empty(H);cursor=0
            for start,paths in model.terminal_chunks(x,w,H,1000,seed):
                assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            assert cursor==H and np.isfinite(loss).all()
            p=D/'losses'/f'{index:03d}-{number:03d}.npz';np.savez_compressed(p,losses=loss)
            if number==replay:
                with np.load(OLD/'losses'/p.name,allow_pickle=False)as z:np.testing.assert_array_equal(loss,z['losses'])
            records.append({'case':number,'config_index':index,'model_id':model.model_id,'series_id':case['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],'paths':1000,'seed':seed,'constituent_ids':ids,'score':score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]),'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'checkpoint_sha256':sha(p),'selection':model.last_selection,'validation_trace':bundle['trace'],'internal_coefficients':str(coefficient_path),'internal_coefficients_sha256':sha(coefficient_path),'fit_certificate':{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in bundle['fit'].items()},'shared_fit_seconds':fit_seconds,'previously_failed':number in failures,'successful_prior_replay_bitwise_equal':number==replay})
        write(D/'progress.json',{'utc':now(),'completed_cases':len({r['case']for r in records}),'expected_cases':len(selected),'records':records})
    assert all(sha(O/p)==h for p,h in source_hashes.items())
    receipt={'utc':now(),'complete':True,'qualified_tests':10,'recovered_preceding_source_tests':21,'repaired_prior_cases':failures,'successful_multivariate_replay_case':replay,'successful_replay_bitwise_equal':True,'new_successful_forecasts':len(records),'native_failures':0,'scoring_integrity_errors':0,'no_canonical_panel_score_or_promotion':True,'records':records,'prior_failed_records_preserved':True,'next_action':'Complete355case comparison using source-compatible/hash-verified preceding successful checkpoints plus repaired failed cases; verify full same-case mainQ before any advancement.'}
    write(D/'summary.json',receipt);state('combining_verified_same_case_completion',receipt=str(D/'summary.json'))
    subprocess.run([sys.executable,str(O/'complete_certified_selector_development.py')],cwd=O,check=True)
    state('terminal_qualified_and_development_complete',receipt=str(D/'summary.json'),development=str(C/'development1000/summary.json'));print(json.dumps({k:v for k,v in receipt.items()if k!='records'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        state('failed',error_type=type(e).__name__,message=str(e));raise
