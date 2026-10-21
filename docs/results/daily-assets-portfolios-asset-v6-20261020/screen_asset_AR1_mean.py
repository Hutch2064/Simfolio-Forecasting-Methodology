"""100-path equilibrium-first-block contrast after verified GARCH-fade promotion."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'
D=V/'candidates/asset_mean_uncertainty/AR1_mean_screen100'
PREVIOUS=V/'candidates/asset_mean_uncertainty/components_screen100'
sys.path.insert(0,str(V))
import common,sweep_v6 as api,reporter,score_main
from inputs import AssetInputs
from candidates.asset_AR1_mean.candidate import ARPool
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=api.original._atomic_json

def initialize():
    global ROWS
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}

def execute(job):
    number,case,arm=job;started=time.perf_counter()
    row,origin=ROWS[case['series_id']],case['origin']
    dates,values,origins=api.load_series(V/'inputs',row,verify_files=False)
    assert origin in origins
    stop,horizon=origin['position']+1,origin['max_horizon']
    history,weights,ids=AssetInputs(V/'inputs').training(row,stop)
    model=ARPool()
    seed=api.original.coherent_daily_seed(origin['origin_date'],horizon,100)
    losses,realized,cursor=np.empty(horizon),np.cumsum(values[stop:stop+horizon]),0
    for first,paths in model.terminal_chunks(history,weights,horizon,100,seed):
        assert first==cursor and paths.shape[0]==100
        cursor+=paths.shape[1]
        losses[first:cursor]=empirical_crps_by_horizon(paths,realized[first:cursor])
    assert cursor==horizon and np.isfinite(losses).all()
    np.savez_compressed(D/arm/f'{number:04}.npz',losses=losses)
    return {'arm':arm,'series_id':row['series_id'],'origin_date':origin['origin_date'],
            'origin_label':origin['origin_label'],'score':score_main.origin_score(losses,values[:stop],origin['origin_date'],dates[stop:stop+horizon]),
            'paths':100,'constituents':ids,'seed':seed,'pid':os.getpid(),
            'worker_compute_seconds':time.perf_counter()-started,'loss_sha256':hashlib.sha256(losses.tobytes()).hexdigest(),'reused':False}

def main():
    lease=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
    assert not (D/'active.json').exists(),'Inspect existing run rather than restarting'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.1','--artifact-root',str(O),'--budget-gib','12'],check=True)
    source=V/'candidates/asset_mean_uncertainty/fading_vol_screen100'
    previous,plan=read(source/'summary.json'),read(source/'plan.json')
    assert previous['complete'] and previous['paths']==100 and previous['native_failures']==0
    assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
    official=read(V/'official_scoreboard.json')
    assert official['current_best_model_id']=='model_testing_asset_circular_raw_GARCH_fade_FHS_mean_v1'
    proof=read(V/'candidates/asset_fading_vol/full/verification_receipt.json')
    assert proof['complete'] and proof['successful_tasks']==64049 and proof['native_failures']==proof['scoring_integrity_errors']==0
    assert read(O/'archives/asset-v6-fading-1000-20261021/receipt.json')['complete']
    cases=plan['cases'];assert len(cases)==355
    lookup={(c['series_id'],c['origin']['origin_label']):i for i,c in enumerate(cases)}
    rows=[]
    for record in previous['records']:
        if record['arm'] not in ('fixed_reference','garch','circular_hybrid'):continue
        number=lookup[record['series_id'],record['origin_label']]
        path=Path(record['verified_original_checkpoint']) if record['reused'] else source/record['arm']/f'{number:04}.npz'
        with np.load(path,allow_pickle=False) as z:loss=z['losses']
        assert hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256']
        rows.append({**record,'reused':True,'verified_original_checkpoint':str(path),'original_checkpoint_sha256':sha(path)})
    assert len(rows)==1065
    circular_plan={'source_hashes':{}}
    hashes={**plan['source_hashes'],**circular_plan['source_hashes'],str(Path(__file__).relative_to(O)):sha(Path(__file__)),**{str(p.relative_to(O)):sha(p) for p in (V/'candidates/asset_AR1_mean').iterdir() if p.suffix in ('.py','.md')}}
    D.mkdir(parents=True,exist_ok=True)
    for arm in ('AR1',):(D/arm).mkdir(exist_ok=True)
    write(D/'plan.json',{'cases':cases,'paths':100,'workers':12,'source_hashes':hashes,'new_forecasts':355,'reused_forecasts':1065,
                        'mechanism':'Only filtered mean dynamics and its innovation variance fit change. Training-only HAC-shrunk per-asset AR1, separate innovation GARCH, shared joint innovation rows. Raw half and long-run mean uncertainty unchanged.',
                        'rationale':'Training-only546context diagnostic motivates a conditional-mean test, not a performance claim. Earlier scalar MA1 failure is cautionary; this is a different joint-asset AR1 law. No horizon-specific selection.',
                        'is_official_panel_score':False,'eligible_for_promotion':False})
    write(D/'active.json',{'pid':os.getpid(),'status':'running','paths':100,'workers':12})
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(execute,(i,c,arm)):(i,c,arm) for i,c in enumerate(cases) for arm in ('AR1',)}
        for future in concurrent.futures.as_completed(futures):
            try:rows.append(future.result())
            except Exception as exc:
                errors.append({'job':futures[future],'exception_type':type(exc).__name__,'message':str(exc)});write(D/'native_errors.json',errors)
            if (len(rows)+len(errors))%50==0:write(D/'progress.json',{'new_completed':sum(not r['reused'] for r in rows),'new_expected':355,'reused':1065,'native_failures':len(errors)})
    assert all(sha(O/p)==h for p,h in hashes.items())
    metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
    scores={arm:reporter.scalar([r for r in rows if r['arm']==arm],metadata) if not errors else None for arm in ('fixed_reference','garch','circular_hybrid','AR1')}
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':not errors,'new_forecasts':sum(not r['reused'] for r in rows),'reused_forecasts':1065,
            'cases_per_arm':355,'paths':100,'native_failures':len(errors),'worker_pids':sorted({r['pid'] for r in rows if not r['reused']}),'exploratory_main_Q':scores,
            'percent_better_vs_same100path_fixed_reference':{arm:100*(1-q/scores['fixed_reference']) if not errors else None for arm,q in scores.items()},
            'promising_arms_for1000path_confirmation':[arm for arm in ('AR1',) if not errors and scores[arm]<scores['garch']],
            'is_official_panel_score':False,'eligible_for_promotion':False,'independent_validation_claim':False,'records':rows}
    write(D/'summary.json',result);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','paths':100})
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
