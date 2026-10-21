"""Exact all-work qualification12 frozen contexts x3 inner seeds; concurrent times diagnostic."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,fcntl,hashlib,itertools,json,multiprocessing,subprocess,sys,time
import numpy as np
import profile_aligned_production as harness
from candidates.asset_selector_aligned_production import candidate
reference=harness.reference;O=harness.O;V=harness.V;D=harness.D;Q=D/'qualification36'
P=V/'candidates/asset_selector_score_alignment/smoke1000/plan.json'
read=lambda p:json.loads(Path(p).read_text());sha=harness.sha;write=harness.api.original._atomic_json

def prepare():
    parent=read(P);source={str((O/n).resolve()):h for n,h in parent['source_hashes'].items()}
    for p in [Path(candidate.__file__),Path(__file__),Path(harness.__file__),D/'test_candidate.py']:
        source[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in source.items())
    profile_records=sorted((D/'profiles').glob('*.json'))
    attestation={'schema':'immutable-aligned-specialization-v1-source-attestation',
        'candidate_file':str(Path(candidate.__file__)),'candidate_sha256':sha(candidate.__file__),
        'reference_file':str(Path(reference.__file__)),'reference_sha256':sha(reference.__file__),
        'profiles':{str(p):sha(p)for p in profile_records},'source_hashes':source,
        'profile_driver_reference_v1_sha256':sha(D/'profile_driver_reference_v1.py')}
    path=D/'source_attestation_v1.json'
    if path.exists():assert read(path)==attestation
    else:write(path,attestation)
    plan={'schema':'aligned-specialization-exact36-v1','cases':parent['cases'],'internal_seed_replications':[0,1,2],
        'outer_paths':1000,'inner_paths':2,'config':candidate.CONFIG,'tasks':[[i,s]for i in range(12)for s in range(3)],
        'source_hashes':source,'source_attestation_sha256':sha(path),'parent_plan_sha256':sha(P),
        'expected_paired_requests':36,'scientific_worker_limit':12,'official':False,
        'no_production_or_full_panel_launch':True,'timings':'Concurrent diagnostic only; serial profile records are speed evidence'}
    plan['fingerprint']=harness.api.original.stable_digest(plan)
    Q.mkdir(exist_ok=True)
    if(Q/'plan.json').exists():assert read(Q/'plan.json')==plan
    else:write(Q/'plan.json',plan)
    return plan

def initialize():
    global PLAN,ROWS
    PLAN=read(Q/'plan.json');assert all(sha(p)==h for p,h in PLAN['source_hashes'].items())
    ROWS={r['series_id']:r for r in harness.api.load_panel(V/'inputs',verify_files=False)['series']}
    os.setpriority(os.PRIO_PROCESS,0,max(15,os.getpriority(os.PRIO_PROCESS,0)))

def evaluate(task):
    i,rep=task;path=Q/f'{i:02d}-{rep}.json';loss_path=Q/f'{i:02d}-{rep}.npz'
    if path.exists():
        old=read(path);assert old['execution_fingerprint']==PLAN['fingerprint']and old['task']==task
        if old.get('verified'):assert sha(loss_path)==old['loss_checkpoint_sha256']
        return old
    started=time.perf_counter();phase='inputs';case=PLAN['cases'][i];origin=case['origin'];h=origin['max_horizon']
    try:
        row=ROWS[case['series_id']];dates,values,origins=harness.api.load_series(V/'inputs',row,verify_files=True)
        assert origin in origins;stop=origin['position']+1
        x,w,ids=harness.AssetInputs(V/'inputs').training(row,stop)
        phase='reference_prepare';r=reference.prepare(x,w,h,dates[:stop],inner_paths=2,schedule_rule='half_doubling',seed_replication=rep)
        phase='candidate_prepare';c=candidate.prepare(x,w,h,dates[:stop],inner_paths=2,schedule_rule='half_doubling',seed_replication=rep)
        phase='coefficient_weight_trace_parity'
        assert r['trace']==c['trace']and r['invalid_internal_options']==c['invalid_internal_options']
        for name in ['A','B']:
            np.testing.assert_array_equal(r['coefficients'][candidate.AXIS][name],c['coefficients'][candidate.AXIS][name])
        np.testing.assert_array_equal(r['weights_by_mode'][candidate.MODE],c['weights_by_mode'][candidate.MODE])
        phase='full1000path_and_loss_parity';seed=harness.api.original.coherent_daily_seed(origin['origin_date'],h,1000)
        left=reference.ScorePool(candidate.CONFIG,dates[:stop],r);right=candidate.ScorePool(candidate.CONFIG,dates[:stop],c)
        actual=np.cumsum(values[stop:stop+h]);loss=np.empty(h);cursor=0;path_hash=hashlib.sha256()
        for a,b in itertools.zip_longest(left.terminal_chunks(x,w,h,1000,seed),right.terminal_chunks(x,w,h,1000,seed)):
            assert a is not None and b is not None;start,p=a;position,q=b
            assert start==position==cursor and p.shape==q.shape and p.shape[0]==1000
            np.testing.assert_array_equal(p,q);cursor+=p.shape[1]
            a_loss=harness.empirical_crps_by_horizon(p,actual[start:cursor]);b_loss=harness.empirical_crps_by_horizon(q,actual[start:cursor])
            np.testing.assert_array_equal(a_loss,b_loss);loss[start:cursor]=a_loss;path_hash.update(p.tobytes())
        assert cursor==h and left.last_selection['path_counts']==right.last_selection['path_counts']
        phase='evidence_io'
        with loss_path.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,losses=loss)
        loss_path.with_suffix('.tmp').replace(loss_path)
        record={'task':task,'execution_fingerprint':PLAN['fingerprint'],'verified':True,'case':case,
            'internal_seed_replication':rep,'actual_horizon':h,'paths':1000,'final_seed':seed,'rows':len(x),'assets':len(w),
            'history_sha256':reference.control.signature(x),'training_dates_sha256':hashlib.sha256(dates[:stop].tobytes()).hexdigest(),
            'A_sha256':hashlib.sha256(c['coefficients'][candidate.AXIS]['A'].tobytes()).hexdigest(),
            'B_sha256':hashlib.sha256(c['coefficients'][candidate.AXIS]['B'].tobytes()).hexdigest(),
            'weights_sha256':hashlib.sha256(c['weights_by_mode'][candidate.MODE].tobytes()).hexdigest(),
            'full_path_sha256':path_hash.hexdigest(),'full_loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),
            'loss_checkpoint_sha256':sha(loss_path),'trace_sha256':harness.api.original.stable_digest(c['trace']),
            'invalid_internal_options':c['invalid_internal_options'],'all_arrays_and_full_paths_exact':True,
            'concurrent_diagnostic_wall_seconds':time.perf_counter()-started}
    except Exception as error:
        record={'task':task,'execution_fingerprint':PLAN['fingerprint'],'verified':False,'phase':phase,'type':type(error).__name__,'message':str(error)}
    write(path,record);return record

def run():
    plan=prepare()
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.03','--artifact-root',str(O),'--budget-gib','16'],check=True)
    with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
            pending={pool.submit(evaluate,task):task for task in plan['tasks']}
            while pending:
                done,_=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    task=pending.pop(future)
                    try:r=future.result()
                    except Exception as error:r={'task':task,'verified':False,'phase':'worker_infrastructure','type':type(error).__name__,'message':str(error)}
                    results.append(r);print(json.dumps({'completed':len(results),'task':task,'verified':r['verified']}),flush=True)
                write(Q/'progress.json',{'completed':len(results),'expected':36,'errors':[r for r in results if not r['verified']]})
        assert all(sha(p)==h for p,h in plan['source_hashes'].items())
        summary={'verified':len(results)==36 and all(r['verified']for r in results),'paired_requests':len(results),
            'expected':36,'outer_paths':1000,'internal_paths':2,'inner_seeds':[0,1,2],'source_attestation_sha256':sha(D/'source_attestation_v1.json'),
            'results':results,'scope':'Exact full-request numerical qualification; concurrent timings diagnostic only; no full-panel or browser CTV claim'}
        write(Q/'verification_receipt.json',summary);print(json.dumps({k:v for k,v in summary.items()if k!='results'}),flush=True)

if __name__=='__main__':run()
