"""Paired synthetic inner-workload timing; no complete-request speed claim."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V));import common
from candidates.asset_selector_universe.bank import BANK,model,prefix_fits
from candidates.asset_selector_batched_paths.paths import all_chunks
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate
from candidates.asset_selector_path_objective.coefficients import lead_masses

rng=np.random.default_rng(62823);x=rng.normal(0,.01,size=(160,3));w=np.array([.2,.3,.5]);H=128;prepared,failures=prefix_fits(x,list(range(len(BANK))));assert not failures;valid=list(range(len(BANK)));actual=np.cumsum(rng.normal(0,.008,H));mass=lead_masses(np.arange(1,H+1));records=[]
def individual(paths):
    generators=[model(i,prepared).terminal_chunks(x,w,H,paths,84381,width=64)for i in valid]
    for start in range(0,H,64):
        pieces=[]
        for generator in generators:
            position,values=next(generator);assert position==start;pieces.append(values)
        yield start,np.stack(pieces)
def evaluate(kind,paths,score):
    generator=individual(paths)if kind=='individual'else all_chunks(x,w,H,paths,84381,valid,prepared,width=64);cpu=0.;wall=0.;digest=hashlib.sha256();A=np.zeros(len(valid));B=np.zeros((len(valid),len(valid)))
    for start in range(0,H,64):
        begin=time.process_time();t=time.perf_counter();position,cube=next(generator)
        if score:accumulate(cube,actual[start:start+cube.shape[2]],mass[start:start+cube.shape[2]],A,B)
        cpu+=time.process_time()-begin;wall+=time.perf_counter()-t;assert position==start;digest.update(cube.tobytes())
    return {'cpu_seconds':cpu,'wall_seconds':wall,'all_lead_paths_sha256':digest.hexdigest(),'A_sha256':hashlib.sha256(A.tobytes()).hexdigest(),'B_sha256':hashlib.sha256(B.tobytes()).hexdigest()}
for paths in (10,1000):
    for score in (False,True):
        left=evaluate('individual',paths,score);right=evaluate('grouped',paths,score)
        for key in ('all_lead_paths_sha256','A_sha256','B_sha256'):assert left[key]==right[key]
        for repeat in range(3):
            pair={}
            for kind in (('individual','grouped')if repeat%2==0 else('grouped','individual')):pair[kind]=evaluate(kind,paths,score)
            for key in ('all_lead_paths_sha256','A_sha256','B_sha256'):assert pair['individual'][key]==pair['grouped'][key]
            records.append({'paths_per_option':paths,'includes_fast_coefficient_accumulation':score,'repeat':repeat,**pair,'cpu_ratio':pair['individual']['cpu_seconds']/pair['grouped']['cpu_seconds']})
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'synthetic_training_rows':len(x),'assets':3,'options':len(BANK),'native_horizon':H,'pairs':records,'median_cpu_ratios':[{'paths_per_option':paths,'includes_fast_coefficient_accumulation':score,'ratio':statistics.median(r['cpu_ratio']for r in records if r['paths_per_option']==paths and r['includes_fast_coefficient_accumulation']==score)}for paths in(10,1000)for score in(False,True)],'all_compared_native_leads_and_coefficients_byte_exact':True,'scope':'Synthetic inner component simulation, separately with and without the already optimized coefficient kernel; excludes fitting, weight optimization, final1000path forecast, CRPS and IO','digest_cost_excluded_equally_from_timing':True,'warm_code':True,'three_interleaved_pairs_per_workload':True,'concurrent_existing12scientific_workers':True,'wall_timings_have_shared_workload_contention':True,'no_complete_request_or_production_speedup_claim':True,'source_hashes':{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [* (V/'candidates/asset_selector_batched_paths').glob('*.py'),Path(__file__)]}}
(O/'selector_batched_paths_benchmark.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='pairs'},indent=2),flush=True)
