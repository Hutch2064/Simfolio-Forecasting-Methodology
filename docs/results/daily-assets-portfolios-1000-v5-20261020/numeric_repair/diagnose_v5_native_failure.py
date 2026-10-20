"""Read-only exact failed-origin replay; never mutate checkpoints or scores."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,datetime,math
import numpy as np
OWN=Path(__file__).resolve().parent;V5=OWN/'simulation_1000_v5';sys.path.insert(0,str(V5))
import streaming as s
import sweep_v5 as api
errors=sorted((V5/'results/models').glob('*/errors/**/*.json'))
records=[]
for path in errors:
 error=json.loads(path.read_text());identity=error['identity'];o=identity['origin'];mid=identity['model_id']
 if mid!=s.BEST:continue
 panel=api.load_panel(V5/'inputs',verify_files=False);row=next(r for r in panel['series'] if r['series_id']==identity['series_id']);dates,values,_=api.load_series(V5/'inputs',row)
 history=np.log1p(np.clip(np.expm1(values[:o['position']+1]),-.999999,np.finfo(float).max/10))
 fit=s.incumbent.retained.FHS.base.fit_variance_target(history);parent=s.incumbent.retained;prior=parent.parent.incumbent.prior
 seed=prior.deterministic_seed(parent.parent.NAMESPACES[1],int(identity['seed']))
 scale,df,_,_=parent.parent.incumbent.base.batch_mean_scale(history);se=parent.gaussian_sd(df,scale);center=s.incumbent.regularized_mean(float(history.mean()),se)
 delta=parent.parameter_draws(df,scale,500,prior.deterministic_seed(parent.parent.incumbent.base.NAMESPACE,int(seed)))+(center-float(history.mean()))
 stream=s.FhsStream(len(history),500,seed);states=(np.full(500,fit['next_variance_normalized']),np.ones(500),np.zeros(500),np.zeros(500));failure=None
 for start in range(0,o['max_horizon'],s.WIDTH):
  indices=stream.take(min(s.WIDTH,o['max_horizon']-start));before=tuple(x.copy() for x in states)
  terminal,clips,floors=s.fhs_block(fit['pool'],indices,fit['mean']+delta,fit['target_variance'],fit['alpha'],fit['beta'],*states)
  bad=np.argwhere(~np.isfinite(terminal))
  if len(bad):
   pi,hi=map(int,bad[0]);variance,wealth,previous,cumulative=(float(x[pi]) for x in before)
   trace=[]
   for h in range(hi+1):
    normalized=math.sqrt(variance)*float(fit['pool'][indices[pi,h]])
    raw=float(fit['mean']+delta[pi])+math.sqrt(fit['target_variance'])*normalized
    upper=np.finfo(float).max/10;simple=upper if raw>math.log1p(upper) else max(math.expm1(raw),-.999999)
    oldwealth=wealth
    with np.errstate(over='ignore',invalid='ignore'):wealth=float(np.float64(wealth)*(1+simple))
    current=math.log(max(wealth,1e-300));cumulative+=current-previous;previous=current
    trace.append({'horizon':start+h+1,'conditional_variance_normalized':variance,'normalized_shock':normalized,'raw_daily_log_return':raw,'simple_return':simple,'wealth_before':oldwealth if math.isfinite(oldwealth) else None,'wealth_after':wealth if math.isfinite(wealth) else None,'wealth_overflow':not math.isfinite(wealth)})
    variance=1.-fit['alpha']-fit['beta']+fit['alpha']*normalized*normalized+fit['beta']*variance
   failure={'component':'FHS','path_index':pi,'first_nonfinite_horizon':start+hi+1,'chunk_daily_clips':int(clips),'chunk_floor_events':int(floors),'trace':trace[-8:]};break
 assert failure is not None,'Exact failure did not reproduce'
 records.append({'error_record':str(path),'error_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'identity':identity,'fit':{k:v for k,v in fit.items() if k!='pool'},'training_history_sha256':hashlib.sha256(history.tobytes()).hexdigest(),'failure_reproduced':True,'failure':failure,'no_checkpoints_or_scores_modified':True})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'scope':'Exact replay diagnostic only, not a new model or repaired-score claim'}
(OWN/'v5_native_failure_diagnostic.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,allow_nan=False),flush=True)
