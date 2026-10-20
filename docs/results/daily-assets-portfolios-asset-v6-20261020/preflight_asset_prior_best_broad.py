"""Training-only real-input qualification; no OOS forecasts or extra12-slot queue."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,datetime,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
from inputs import AssetInputs
from candidates.asset_prior_best import fit,mean,candidate
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source={str(p.relative_to(V)):sha(p)for p in candidate.OWN.glob('*.py')};source['profile.json']=sha(V/'profile.json');inputs=AssetInputs(V/'inputs');by_id={r['series_id']:i for i,r in enumerate(inputs.panel['series'])}; tasks=[]
 for phase, phase_cases in read(candidate.OWN/'plan.json')['phases'].items():
  for item in phase_cases:
   i=by_id[item['series_id']]; origins=read(V/'inputs'/inputs.panel['series'][i]['origins_file']); tasks.append((i,origins.index(item['origin'])));records=[];failures=[];began=time.perf_counter()
 for index,(i,j)in enumerate(tasks):
  row=inputs.panel['series'][i];origins=read(V/'inputs'/row['origins_file']);origin=origins[j];history,w,ids=inputs.training(row,origin['position']+1)
  entry={'series_id':row['series_id'],'origin_label':origin['origin_label'],'training_last_date':origin['origin_date'],'training_rows':len(history),'constituent_ids':ids,'training_matrix_sha256':hashlib.sha256(history.tobytes()).hexdigest(),'fits':[]}
  try:
   batch=mean.estimate(history);entry['batch']={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in batch.items()}
   for a,sid in enumerate(ids):
    fitted=fit.fit_variance_target(history[:,a]);pool=fitted.pop('pool');assert pool.shape==(len(history),)and np.isfinite(pool).all();assert abs(pool.mean())<1e-12 and abs(np.mean(pool**2)-1)<1e-12
    entry['fits'].append({'asset_id':sid,'fit':fitted,'pool_sha256':hashlib.sha256(pool.tobytes()).hexdigest()})
  except Exception as exc:failures.append({'task':[i,j],'series_id':row['series_id'],'exception':type(exc).__name__,'message':str(exc)})
  records.append(entry)
  if(index+1)%10==0:print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prefixes_checked':index+1,'expected':len(tasks),'failures':len(failures)}),flush=True)
 assert all(sha(V/name)==h for name,h in source.items())
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_id':candidate.MODEL_ID,'training_only':True,'heldout_forecasts_or_scores':0,'representative_prefixes':len(tasks),'successful_asset_fits':sum(len(r['fits'])for r in records),'failures':failures,'source_hashes':source,'serial_training_preflight_reason':'All12scientific forecast slots remain occupied by full baseline pair; this is a low-priority training-only fit check, not a parallel forecast queue or runtime benchmark','records':records,'elapsed_diagnostic_seconds':time.perf_counter()-began,'candidate_admitted':False,'parent_selection_pending_completed_baselines':True}
 (candidate.OWN/'broad_training_preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['records','source_hashes']}),flush=True)
 if failures:raise RuntimeError('Training qualification failures; preserve and diagnose before field admission')
if __name__=='__main__':main()
