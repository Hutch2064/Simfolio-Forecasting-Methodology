"""Local numerical timing of complete nested selection; never website latency."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,fcntl,hashlib,json,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_fast_selector';D=C/'runtime';D.mkdir(exist_ok=True);sys.path.insert(0,str(V))
import common,sweep_v6 as api
from candidates.asset_fast_selector.candidate import FastSelector,CONFIGS,BASE_CONFIG,prepare
from candidates.asset_trained_pool.candidate import TrainedPool,prepare as parent_prepare
from threadpoolctl import threadpool_info
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();write=api.original._atomic_json
slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
assert not(D/'active.json').exists(),'Inspect existing benchmark before restarting'
write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now()})
panel=api.load_panel(V/'inputs',verify_files=False);sources=sorted([r for r in panel['series']if r['panel_branch']=='assets'and r['series_id'].startswith('FR')and r['observations']>=2520],key=lambda r:r['series_id'])[:200];assert len(sources)==200
common_dates=np.load(V/'inputs'/sources[0]['dates_file'],allow_pickle=False)
for row in sources[1:]:common_dates=np.intersect1d(common_dates,np.load(V/'inputs'/row['dates_file'],allow_pickle=False),assume_unique=True)
common_dates=common_dates[-2520:];assert len(common_dates)==2520
columns=[]
for row in sources:
    dates=np.load(V/'inputs'/row['dates_file'],allow_pickle=False);values=np.load(V/'inputs'/row['returns_file'],allow_pickle=False);positions=np.searchsorted(dates,common_dates);assert np.array_equal(dates[positions],common_dates);columns.append(values[positions])
allx=np.column_stack(columns);records=[]
write(D/'plan.json',{'utc':now(),'scope':'Local single-process numerical compute; warmed code, no reusable fit cache; no website/API/network or deployment claim','histories':'Latest2520actual common observations of first200sorted FR assets, used as fixed scaling fixture only, not out-of-sample accuracy evidence','source_ids':[r['series_id']for r in sources],'source_files':{r['series_id']:{k:r['files'][r[k]]for k in('dates_file','returns_file')}for r in sources},'input_matrix_sha256':hashlib.sha256(allx.tobytes()).hexdigest(),'first_date':str(common_dates[0]),'last_date':str(common_dates[-1]),'assets':[1,10,200],'horizons':[21,252],'repetitions':3,'outer_paths':1000,'inner_paths':32,'maximum_inner_origins':3,'threads':threadpool_info(),'paired_interleaving':True})
# Warm numerical dispatchers only, preserve warmup separately from paired samples.
warm=prepare(allx[:600,:1],np.ones(1),21)
list(FastSelector(CONFIGS[-1],warm).terminal_chunks(allx[:600,:1],np.ones(1),21,32,88120))
for assets in(1,10,200):
    x=allx[:,:assets].copy();weights=np.full(assets,1/assets)
    for H in(21,252):
        for repeat in range(3):
            for kind in(('parent','selector')if repeat%2==0 else('selector','parent')):
                started=time.perf_counter();cpu=time.process_time()
                if kind=='parent':
                    fit_start=time.perf_counter();bundle=parent_prepare(x,'SV_RMS','EWMA');fit=time.perf_counter()-fit_start;model=TrainedPool(BASE_CONFIG,bundle);diagnostics={'parent_fit_seconds':fit}
                else:
                    bundle=prepare(x,weights,H);model=FastSelector(CONFIGS[-1],bundle);diagnostics=bundle['diagnostics']
                simulation=time.perf_counter();cursor=0;digest=hashlib.sha256()
                for start,paths in model.terminal_chunks(x,weights,H,1000,88121):
                    assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];assert np.isfinite(paths).all();digest.update(paths.tobytes())
                assert cursor==H
                row={'utc':now(),'kind':kind,'assets':assets,'horizon':H,'repeat':repeat,'whole_request_compute_seconds':time.perf_counter()-started,'process_cpu_seconds':time.process_time()-cpu,'final1000path_simulation_seconds':time.perf_counter()-simulation,'diagnostics':diagnostics,'output_sha256':digest.hexdigest()}
                if kind=='selector':row['selected']=model.last_selection['specification']
                records.append(row);write(D/'records.json',records);print(json.dumps(row),flush=True)
summary=[]
for assets in(1,10,200):
    for H in(21,252):
        data={kind:[r['whole_request_compute_seconds']for r in records if r['kind']==kind and r['assets']==assets and r['horizon']==H]for kind in('parent','selector')}
        summary.append({'assets':assets,'horizon':H,'parent_mean_seconds':float(np.mean(data['parent'])),'parent_median_seconds':float(np.median(data['parent'])),'selector_mean_seconds':float(np.mean(data['selector'])),'selector_median_seconds':float(np.median(data['selector'])),'selector_over_parent_median_ratio':float(np.median(data['selector'])/np.median(data['parent']))})
write(D/'summary.json',{'utc':now(),'complete':True,'cases':6,'paired_requests':36,'scope':'Local whole numerical request only; no web latency or production speedup claim; no CRPS scoring in runtime fixture','summary':summary});write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()})
