"""Bounded end-to-end dated-request kernel profiling; no suite or production edits."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,cProfile,hashlib,json,pstats,sys,time
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_aligned_production'
sys.path[:0]=[str(V),str(O)]
import common,sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_selector_score_alignment import candidate as reference
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_case(name):
    if name=='week0':p=V/'candidates/asset_selector_score_alignment/requested_week1000/plan.json';index=0;short=True
    elif name=='portfolio9':p=V/'candidates/asset_selector_score_alignment/smoke1000/plan.json';index=9;short=False
    elif name=='broad350':p=V/'candidates/asset_selector_endpoint_flat/broader1000/plan.json';index=350;short=False
    else:raise ValueError(name)
    plan=json.loads(p.read_text());case=plan['cases'][index]
    panel=api.load_panel(V/'inputs',verify_files=False);row=next(r for r in panel['series']if r['series_id']==case['series_id'])
    dates,values,origins=api.load_series(V/'inputs',row,verify_files=True);assert case['origin']in origins
    stop=case['origin']['position']+1;h=case['requested_horizon']if short else case['origin']['max_horizon']
    x,w,ids=AssetInputs(V/'inputs').training(row,stop)
    cfg=reference.configurations(2)[18]
    source_files={str((O/n).resolve()):s for n,s in plan['source_hashes'].items()}
    assert all(sha(path)==s for path,s in source_files.items())
    manifest={'name':name,'parent_plan':str(p),'parent_plan_sha256':sha(p),'parent_case_index':index,
              'case':case,'actual_horizon':h,'rows':len(x),'assets':len(w),'constituents':list(ids),
              'history_sha256':reference.control.signature(x),'training_dates_sha256':hashlib.sha256(dates[:stop].tobytes()).hexdigest(),
              'weights_sha256':hashlib.sha256(w.tobytes()).hexdigest(),'config':cfg,'source_hashes':source_files}
    return manifest,x,w,dates[:stop],np.cumsum(values[stop:stop+h])


def entry(module, config, x, w, dates, horizon, seed, width=64):
    """One call starts with dated arrays; no bundle is supplied by the caller."""
    model=module.ScorePool(config,dates)
    chunks=list(model.terminal_chunks(x,w,horizon,1000,seed,width=width))
    return model,chunks


def profile_case(name,kind='reference',repeats=2):
    D.mkdir(exist_ok=True);out=D/'profiles';out.mkdir(exist_ok=True)
    manifest,x,w,dates,actual=build_case(name);h=manifest['actual_horizon'];seed=api.original.coherent_daily_seed(manifest['case']['origin']['origin_date'],h,1000)
    if kind=='candidate':
        from candidates.asset_selector_aligned_production import candidate as module
    else:module=reference
    config=manifest['config'];results=[]
    for repeat in range(repeats):
        prepared=[];original=module.prepare
        def timed(*args,**kwargs):
            c=time.process_time();t=time.perf_counter();bundle=original(*args,**kwargs)
            prepared.append({'cpu_seconds':time.process_time()-c,'wall_seconds':time.perf_counter()-t,'bundle':bundle});return bundle
        module.prepare=timed;profile=cProfile.Profile();cpu=time.process_time();wall=time.perf_counter();profile.enable()
        try:model,chunks=entry(module,config,x,w,dates,h,seed)
        finally:
            profile.disable();elapsed_cpu=time.process_time()-cpu;elapsed_wall=time.perf_counter()-wall;module.prepare=original
        assert len(prepared)==1
        path=out/f'{kind}-{name}-{repeat}.pstats';profile.dump_stats(path)
        loss=np.empty(h);cursor=0;pathhash=hashlib.sha256()
        for start,values in chunks:
            assert start==cursor and values.shape[0]==1000 and np.isfinite(values).all()
            end=start+values.shape[1];loss[start:end]=empirical_crps_by_horizon(values,actual[start:end]);cursor=end
            pathhash.update(values.tobytes())
        assert cursor==h
        stats=pstats.Stats(profile);top=sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:20]
        functions=[{'file':f[0],'line':f[1],'function':f[2],'calls':s[1],'self_seconds':s[2],'cumulative_seconds':s[3]}for f,s in top]
        bundle=prepared[0].pop('bundle');coef=bundle['coefficients'][('uniform_days','train_SD_sqrt_lead')]
        learned=bundle['weights_by_mode']['uniform_days:train_SD_sqrt_lead:batch']
        result={'kind':kind,'case':name,'process_state':'fresh process; existing numerical disk caches'if repeat==0 else 'same process warm; all request fits recomputed',
                'repeat':repeat,'cpu_seconds':elapsed_cpu,'wall_seconds':elapsed_wall,'prepare':prepared[0],
                'full_path_generation_cpu_seconds':elapsed_cpu-prepared[0]['cpu_seconds'],
                'full_path_generation_wall_seconds':elapsed_wall-prepared[0]['wall_seconds'],
                'profile_sha256':sha(path),'profile_overhead_included':True,'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                'manifest':manifest,'seed':seed,'paths':1000,'path_sha256':pathhash.hexdigest(),
                'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'weights_sha256':hashlib.sha256(learned.tobytes()).hexdigest(),
                'A_sha256':hashlib.sha256(coef['A'].tobytes()).hexdigest(),'B_sha256':hashlib.sha256(coef['B'].tobytes()).hexdigest(),
                'top_functions':functions,'driver_sha256':sha(__file__),'website_CTV_claim':False}
        (out/f'{kind}-{name}-{repeat}.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        print(json.dumps({k:result[k]for k in ['kind','case','repeat','cpu_seconds','wall_seconds','prepare','full_path_generation_cpu_seconds','path_sha256']}),flush=True)
        del bundle,chunks,model,loss
    return results


if __name__=='__main__':
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True,choices=['week0','portfolio9','broad350']);parser.add_argument('--kind',default='reference',choices=['reference','candidate']);parser.add_argument('--repeats',default=2,type=int);args=parser.parse_args()
    with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);profile_case(args.case,args.kind,args.repeats)
