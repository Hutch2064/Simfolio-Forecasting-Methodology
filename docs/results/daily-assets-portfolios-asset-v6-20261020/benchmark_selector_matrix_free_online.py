"""Warm synthetic coefficient-step pairs, explicitly not complete requests."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
from candidates.asset_selector_matrix_free_online.gradient import accumulate
from candidates.asset_selector_matrix_free_online.gradient_numpy import accumulate as vectorized
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate as dense

rng=np.random.default_rng(262466);n=166;members=1000;leads=16
x=rng.normal(size=(n,members,leads));y=rng.normal(size=leads);m=np.arange(1,leads+1,dtype=float);m/=m.sum()
records=[]
for label in ('uniform166','three_positive_options'):
 w=np.ones(n)/n if label=='uniform166' else np.zeros(n)
 if label!='uniform166':w[[1,64,121]]=[.2,.3,.5]
 def evaluate(kind):
  A=np.zeros(n);begin=time.process_time();wall=time.perf_counter()
  if kind=='dense':
   B=np.zeros((n,n));dense(x,y,m,A,B);C=B@w
  elif kind=='matrix_free':
   C=np.zeros(n);accumulate(x,y,m,w,A,C)
  else:
   C=np.zeros(n);vectorized(x,y,m,w,A,C)
  return A,C,time.process_time()-begin,time.perf_counter()-wall
 # Warm each compiled signature; do not count compilation as steady-state work.
 reference=evaluate('dense')
 for kind in ('matrix_free','vectorized'):
  test=evaluate(kind);np.testing.assert_array_equal(reference[0],test[0]);np.testing.assert_allclose(reference[1],test[1],rtol=5e-12,atol=5e-12)
 for repeat in range(3):
  values={}
  for kind in (('dense','matrix_free','vectorized')if repeat%2==0 else('vectorized','matrix_free','dense')):
   a,c,cpu,wall=evaluate(kind);values[kind]={'cpu_seconds':cpu,'wall_seconds':wall}
   np.testing.assert_array_equal(a,reference[0]);np.testing.assert_allclose(c,reference[1],rtol=5e-12,atol=5e-12)
  records.append({'mixture':label,'repeat':repeat,**values,'cpu_ratio':values['dense']['cpu_seconds']/values['matrix_free']['cpu_seconds'],'vectorized_cpu_ratio':values['dense']['cpu_seconds']/values['vectorized']['cpu_seconds']})
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'options':n,'members':members,'leads':leads,'pairs':records,'cpu_median_ratios':{label:statistics.median(r['cpu_ratio']for r in records if r['mixture']==label)for label in ('uniform166','three_positive_options')},'vectorized_cpu_median_ratios':{label:statistics.median(r['vectorized_cpu_ratio']for r in records if r['mixture']==label)for label in ('uniform166','three_positive_options')},'reference':'Already optimized global-sweep dense coefficient reducer followed by B@w','scope':'Synthetic historical coefficient/gradient phase only; no component simulation, fitting, fold choice, final paths or outer score','dense_option_objective_A_byte_exact':True,'weighted_cross_matches_dense_within_declared_float_tolerance':True,'bit_exact_solver_or_forecast_claim':False,'scientific12worker_study_running_concurrently':True,'wall_timings_confounded_by_existing_shared_workload':True,'no_full_request_or_production_speedup_claim':True,'source_hashes':{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [* (V/'candidates/asset_selector_matrix_free_online').glob('*.py'),Path(__file__)]}}
(O/'selector_matrix_free_online_benchmark.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='pairs'},indent=2),flush=True)
