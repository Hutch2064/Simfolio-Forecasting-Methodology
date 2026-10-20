"""Read-only snapshot audit of every completed repaired/native overlap."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,datetime
O=Path(__file__).resolve().parent;V=O/'simulation_1000_v5';sys.path.insert(0,str(V))
import sweep_v5 as api
from research_stable_wealth import adapter
read=lambda p:json.loads(Path(p).read_text())
old=adapter.s.BEST;new=adapter.MODEL_ID
roots={m:api.model_directory(V/'results',m) for m in [old,new]}
fps={m:read(p/'execution.json')['fingerprint'] for m,p in roots.items()}
panel=api.load_panel(V/'inputs',verify_files=False)
paths=sorted((roots[new]/'tasks').glob('*/*.npz'));origins={};entries=[];repaired=[];missing=[]
replay=read(O/'v5_stable_failed_origin_replay.json')
for p in paths:
 i,j=int(p.parent.name),int(p.stem)
 row=panel['series'][i]
 if i not in origins:origins[i]=api.schedule(V/'inputs',row)
 origin=origins[i][j]
 losses,d=api.original._read_checkpoint(p,api.task_identity(fps[new],new,row['series_id'],origin))
 assert d['simulations']==1000
 digest=hashlib.sha256(losses.tobytes()).hexdigest()
 q,e=api.task_paths(roots[old],(i,j))
 if q.exists():
  prior,_=api.original._read_checkpoint(q,api.task_identity(fps[old],old,row['series_id'],origin))
  assert hashlib.sha256(prior.tobytes()).hexdigest()==digest,(i,j,row['series_id'],origin)
  entries.append({'series_index':i,'origin_index':j,'loss_sha256':digest})
 elif e.exists():
  error=read(e)
  assert hashlib.sha256(e.read_bytes()).hexdigest()==replay['source_error_sha256']
  assert digest==replay['loss_sha256']
  repaired.append({'series_id':row['series_id'],'origin_label':origin['origin_label'],'loss_sha256':digest})
 else:missing.append({'series_index':i,'origin_index':j})
assert not missing
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All completed repaired checkpoints in initial file snapshot; live run continues. No subset extrapolation.',
 'new_model_id':new,'prior_model_id':old,'snapshot_checkpoints':len(paths),'ordinary_complete_loss_vectors_bitwise_exact':len(entries),'repaired_known_error_checkpoints':repaired,
 'identity_checks':2*len(entries)+len(repaired),'missing':missing,'mismatches':0,'all_paths':1000,'recomputed_forecasts':0,'active_control_or_sources_changed':False,
 'full_panel_audit_complete':len(paths)==64049,'entries':entries}
api.original._atomic_json(O/'v5_stable_parity_audit.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='entries'}),flush=True)
