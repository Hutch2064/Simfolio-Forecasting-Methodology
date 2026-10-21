"""Training-only fit qualification; no forecast, score, or outcome-based routing."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_gjr';D=C/'qualification'
sys.path.insert(0,str(V))
import common,sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_gjr import fit
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=api.original._atomic_json

def initialize():
    global INPUTS
    INPUTS=AssetInputs(V/'inputs')

def execute(case):
    row,origin=INPUTS.by_id[case['series_id']],case['origin']
    dates,_=INPUTS.source(row);stop=origin['position']+1
    assert str(dates[stop-1])==origin['origin_date'][:10]
    matrix,weights,ids=INPUTS.training(row,stop)
    assert len(matrix)==stop and np.isfinite(matrix).all()
    outputs=[]
    for a,sid in enumerate(ids):
        f=fit.fit(matrix[:,a]);pool=f.pop('pool')
        assert np.isfinite(pool).all() and len(pool)==stop
        assert f['omega_forecast']>0 and f['innovation_persistence']<1
        assert f['objective']<=f['nested_parent_objective']+1e-12
        outputs.append({'asset_id':sid,'fit':f,'innovation_pool_sha256':hashlib.sha256(pool.tobytes()).hexdigest()})
    return {'series_id':row['series_id'],'family':row['family'],'asset_class':row['asset_class'],
            'origin_label':origin['origin_label'],'last_training_date':str(dates[stop-1]),'training_rows':stop,
            'matrix_sha256':hashlib.sha256(matrix.tobytes()).hexdigest(),'pid':os.getpid(),'fits':outputs}

def main():
    with (O/'asset_v6_forecast_slots.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not (D/'active.json').exists(),'Inspect existing qualification rather than restarting'
        subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.05','--artifact-root',str(O),'--budget-gib','12'],check=True)
        plan=read(V/'candidates/asset_prior_best/plan.json')
        cases=plan['phases']['development']+plan['phases']['reserved'];assert len(cases)==710
        files=[Path(__file__),*C.glob('*.py'),C/'DESIGN.md',V/'inputs.py',V/'candidates/asset_prior_best/fit.py']
        hashes={str(p.relative_to(O)):sha(p) for p in files}
        D.mkdir(exist_ok=True)
        write(D/'plan.json',{'cases':cases,'source_hashes':hashes,'before_training_fit_qualification':True,'no_held_out_outcomes_used':True,'panel_fingerprint':read(V/'inputs/panel.json')['panel_fingerprint']})
        write(D/'active.json',{'pid':os.getpid(),'status':'running','workers':12,'type':'training_only_no_forecast'})
        results,errors=[],[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
            jobs={pool.submit(execute,c):c for c in cases}
            for future in concurrent.futures.as_completed(jobs):
                try:results.append(future.result())
                except Exception as exc:errors.append({'case':jobs[future],'error_type':type(exc).__name__,'message':str(exc)})
                if (len(results)+len(errors))%50==0:write(D/'progress.json',{'prefixes_qualified':len(results),'expected':710,'training_fit_failures':len(errors)})
        assert all(sha(O/name)==digest for name,digest in hashes.items())
        summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':not errors,'prefixes':len(results),'expected_prefixes':710,
                 'asset_fits':sum(len(r['fits']) for r in results),'families':len({r['family'] for r in results}),
                 'training_fit_failures':len(errors),'errors':errors,'worker_pids':sorted({r['pid'] for r in results}),
                 'field_forecasts':0,'held_out_values_used_for_fitting':False,'is_accuracy_result':False,'records':results}
        write(D/'summary.json',summary);write(D/'active.json',{'pid':os.getpid(),'status':'terminal','training_fit_failures':len(errors)})
        print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
