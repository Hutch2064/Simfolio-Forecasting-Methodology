"""Bounded matched CPU benchmark of reducers only, with exact all-work parity."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,sys,time
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_path_objective.coefficients import accumulate as original
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate as sweep
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(function,x,y,m):
 a=np.zeros(x.shape[0]);b=np.zeros((x.shape[0],x.shape[0]));cpu=time.process_time();wall=time.perf_counter();function(x,y,m,a,b)
 return {'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,'A_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'B_sha256':hashlib.sha256(b.tobytes()).hexdigest()}
rng=np.random.default_rng(90434);rows=[]
for members in(100,1000):
 x=rng.normal(size=(166,members,16));y=rng.normal(size=16);m=np.log1p(np.arange(1,17))/np.sqrt(np.arange(1,17));m/=m.sum()
 run(original,x[:2,:,:2],y[:2],m[:2]);run(sweep,x[:2,:,:2],y[:2],m[:2])
 for pair in range(3):
  values={}
  for name,function in((('original',original),('sweep',sweep))if pair%2==0 else(('sweep',sweep),('original',original))):values[name]=run(function,x,y,m)
  assert values['original']['A_sha256']==values['sweep']['A_sha256']and values['original']['B_sha256']==values['sweep']['B_sha256'];rows.append({'members':members,'options':166,'leads':16,'pair':pair,**values})
summary={}
for members in(100,1000):
 selected=[r for r in rows if r['members']==members];summary[str(members)]={'original_median_cpu_seconds':float(np.median([r['original']['cpu_seconds']for r in selected])),'sweep_median_cpu_seconds':float(np.median([r['sweep']['cpu_seconds']for r in selected])),'median_paired_cpu_ratio':float(np.median([r['original']['cpu_seconds']/r['sweep']['cpu_seconds']for r in selected])),'median_paired_wall_ratio':float(np.median([r['original']['wall_seconds']/r['sweep']['wall_seconds']for r in selected]))}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Synthetic166-option/all-pair coefficient reducer,16leads,3interleaved warm-code pairs per member count; twelve concurrent scientific workers are a wall-time confounder','all_coefficients_byte_identical':True,'no_financial_paths_fits_or_forecast_scores_changed':True,'whole_request_or_production_speedup_claim':False,'source_hashes':{str(p.relative_to(O)):sha(p)for p in(Path(__file__),O/'asset_level_v6/candidates/asset_selector_coefficient_sweep/coefficients.py',O/'asset_level_v6/candidates/asset_selector_path_objective/coefficients.py')},'summary':summary,'samples':rows}
(O/'selector_coefficient_sweep_benchmark.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='samples'},indent=2))
