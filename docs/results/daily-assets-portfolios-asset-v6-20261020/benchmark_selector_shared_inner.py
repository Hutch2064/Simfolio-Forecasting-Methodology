"""Paired local simulation timings on the same frozen historical inner fold."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,fcntl,hashlib,json,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_batched';D=C/'paired_inner_benchmark';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_selector_universe.bank import model,prefix_fits
from candidates.asset_selector_batched.terminal import all_terminals
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
lock=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
assert not D.exists();D.mkdir();profile=read(O/'selector_component_runtime_profile.json');prior=V/'candidates/asset_selector_universe_precision/development1000';case=read(prior/'plan.json')['cases'][profile['case']]
row=next(r for r in api.load_panel(V/'inputs',verify_files=False)['series']if r['series_id']==case['series_id']);history,w,ids=AssetInputs(V/'inputs').training(row,case['origin']['position']+1);x=history[:profile['history_rows']];H=profile['validation_horizon'];seed=profile['seed'];options=[r['bank_index']for r in profile['records']];prepared,invalid=prefix_fits(x,options);assert not invalid
def reference():
    predictions=[]
    for index in options:
        cursor=0;final=None
        for start,values in model(index,prepared).terminal_chunks(x,w,H,1000,seed):cursor+=values.shape[1];final=values[:,-1].copy()
        assert cursor==H;predictions.append(final)
    return np.stack(predictions)
def shared():return all_terminals(x,w,H,1000,seed,options,prepared)
expected=[r['terminal_sha256']for r in profile['records']];samples=[];matrix_hash=None
for pair,order in [(-1,['reference','shared']),(0,['reference','shared']),(1,['shared','reference']),(2,['reference','shared'])]:
 for variant in order:
    begin=time.perf_counter();cpu=time.process_time();predictions=reference()if variant=='reference'else shared();cpu=time.process_time()-cpu;wall=time.perf_counter()-begin
    hashes=[hashlib.sha256(row.tobytes()).hexdigest()for row in predictions];assert hashes==expected,'Component terminal ensemble differs from original all-work reference'
    digest=hashlib.sha256(predictions.tobytes()).hexdigest()
    if matrix_hash is None:matrix_hash=digest;np.savez_compressed(D/'reference_terminals.npz',terminals=predictions)
    assert digest==matrix_hash;samples.append({'pair':pair,'warmup':pair<0,'variant':variant,'cpu_seconds':cpu,'wall_seconds':wall,'terminal_matrix_sha256':digest})
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'One actual historical inner fold; all166components and1000members, warm-code interleaved paired local numerical simulation only. No full-request/production/browser claim.','case':profile['case'],'assets':len(ids),'history_rows':len(x),'horizon':H,'paths':1000,'source_profile':str(O/'selector_component_runtime_profile.json'),'source_profile_sha256':sha(O/'selector_component_runtime_profile.json'),'sources':{str(p.relative_to(O)):sha(p)for p in C.glob('*.py')},'samples':samples,'all_component_ensembles_and_joint_terminal_matrix_bitwise_equal':True,'matrix_sha256':matrix_hash,'checkpoint_sha256':sha(D/'reference_terminals.npz'),'matrix_reused_by_hash_no_identical_output_discarded':True,'shared_fits_excluded_from_both_simulation_timings':True,'full_development_workload_parity_still_required':True,'raw_unfiltered_methods_unchanged':True}
for metric in('cpu_seconds','wall_seconds'):
 old=[r[metric]for r in samples if not r['warmup']and r['variant']=='reference'];new=[r[metric]for r in samples if not r['warmup']and r['variant']=='shared'];result[metric]={'reference_median':float(np.median(old)),'shared_median':float(np.median(new)),'median_ratio':float(np.median(old)/np.median(new)),'reference_worst':max(old),'shared_worst':max(new)}
write(D/'summary.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('sources','samples')},indent=2),flush=True)
