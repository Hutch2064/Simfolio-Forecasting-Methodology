"""100-path sampler-parent contrast; freeze everything except raw replacement."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'
D=V/'candidates/asset_mean_uncertainty/parent_screen100'
PREVIOUS=V/'candidates/asset_mean_uncertainty/components_screen100'
sys.path.insert(0,str(V))
import common,sweep_v6 as api,reporter,score_main
from inputs import AssetInputs
from candidates.asset_prior_best.candidate import AssetPool
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=api.original._atomic_json

def initialize():
    global ROWS
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}

def execute(job):
    number,case=job;started=time.perf_counter()
    row,origin=ROWS[case['series_id']],case['origin']
    dates,values,origins=api.load_series(V/'inputs',row,verify_files=False)
    assert origin in origins
    stop,horizon=origin['position']+1,origin['max_horizon']
    history,weights,ids=AssetInputs(V/'inputs').training(row,stop)
    model=AssetPool(True)
    seed=api.original.coherent_daily_seed(origin['origin_date'],horizon,100)
    losses,realized,cursor=np.empty(horizon),np.cumsum(values[stop:stop+horizon]),0
    for first,paths in model.terminal_chunks(history,weights,horizon,100,seed):
        assert first==cursor and paths.shape[0]==100
        cursor+=paths.shape[1]
        losses[first:cursor]=empirical_crps_by_horizon(paths,realized[first:cursor])
    assert cursor==horizon and np.isfinite(losses).all()
    np.savez_compressed(D/'replacement_hybrid'/f'{number:04}.npz',losses=losses)
    return {'arm':'replacement_hybrid','series_id':row['series_id'],'origin_date':origin['origin_date'],
            'origin_label':origin['origin_label'],'score':score_main.origin_score(losses,values[:stop],origin['origin_date'],dates[stop:stop+horizon]),
            'paths':100,'constituents':ids,'seed':seed,'pid':os.getpid(),
            'worker_compute_seconds':time.perf_counter()-started,'loss_sha256':hashlib.sha256(losses.tobytes()).hexdigest(),'reused':False}

def main():
    lease=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
    assert not (D/'active.json').exists(),'Inspect existing run rather than restarting'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.1','--artifact-root',str(O),'--budget-gib','12'],check=True)
    previous,plan=read(PREVIOUS/'summary.json'),read(PREVIOUS/'plan.json')
    assert previous['complete'] and previous['paths']==100 and previous['native_failures']==0
    assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
    cases=plan['cases'];assert len(cases)==355
    lookup={(c['series_id'],c['origin']['origin_label']):i for i,c in enumerate(cases)}
    rows=[]
    for record in previous['records']:
        if record['arm'] not in ('fixed_reference','hybrid'):continue
        number=lookup[record['series_id'],record['origin_label']]
        path=Path(record['verified_original_checkpoint']) if record['reused'] else PREVIOUS/record['arm']/f'{number:04}.npz'
        with np.load(path,allow_pickle=False) as z:loss=z['losses']
        assert hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256']
        rows.append({**record,'reused':True,'verified_original_checkpoint':str(path),'original_checkpoint_sha256':sha(path)})
    assert len(rows)==710
    hashes={**plan['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__))}
    D.mkdir(parents=True,exist_ok=True);(D/'replacement_hybrid').mkdir(exist_ok=True)
    write(D/'plan.json',{'cases':cases,'paths':100,'workers':12,'source_hashes':hashes,'new_forecasts':355,'reused_forecasts':710,
                        'mechanism':'Only raw block replacement changes; same equal mix, per-asset GARCH, joint shocks, smooth mean and Gaussian mean uncertainty.',
                        'rationale':'Full-pair additive diagnostic: history cycles help supplied assets but hurt portfolio branch; test alternative sampler without outcome-based routes.',
                        'is_official_panel_score':False,'eligible_for_promotion':False})
    write(D/'active.json',{'pid':os.getpid(),'status':'running','paths':100,'workers':12})
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(execute,(i,c)):(i,c) for i,c in enumerate(cases)}
        for future in concurrent.futures.as_completed(futures):
            try:rows.append(future.result())
            except Exception as exc:
                errors.append({'job':futures[future],'exception_type':type(exc).__name__,'message':str(exc)});write(D/'native_errors.json',errors)
            if (len(rows)+len(errors))%50==0:write(D/'progress.json',{'new_completed':sum(not r['reused'] for r in rows),'new_expected':355,'reused':710,'native_failures':len(errors)})
    assert all(sha(O/p)==h for p,h in hashes.items())
    metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
    scores={arm:reporter.scalar([r for r in rows if r['arm']==arm],metadata) if not errors else None for arm in ('fixed_reference','hybrid','replacement_hybrid')}
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':not errors,'new_forecasts':sum(not r['reused'] for r in rows),'reused_forecasts':710,
            'cases_per_arm':355,'paths':100,'native_failures':len(errors),'worker_pids':sorted({r['pid'] for r in rows if not r['reused']}),'exploratory_main_Q':scores,
            'percent_better_vs_same100path_fixed_reference':{arm:100*(1-q/scores['fixed_reference']) if not errors else None for arm,q in scores.items()},
            'warrants1000path_confirmation':bool(not errors and scores['replacement_hybrid']<scores['hybrid']),
            'is_official_panel_score':False,'eligible_for_promotion':False,'independent_validation_claim':False,'records':rows}
    write(D/'summary.json',result);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','paths':100})
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
