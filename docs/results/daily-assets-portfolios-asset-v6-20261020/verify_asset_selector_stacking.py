"""Recompute saved score decomposition and replay a fresh multivariate request."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_stacking';D=C/'screen1000';sys.path.insert(0,str(V))
import common,score_main,sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_selector_stacking.candidate import HorizonPool,CONFIGS,prepare,energy
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(D/'summary.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases'];metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
assert s['complete_execution']and not s['errors']and s['new_successful_forecasts']==144
assert read(D/'active.json')['status']=='terminal'
assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
verified=0;reduced={}
for index,config in enumerate(CONFIGS):
 rows=[r for r in s['records']if r['config_index']==index];assert len(rows)==len(plan['cases'])
 for r in rows:
  number=r['case'];case=plan['cases'][number];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
  path=D/'losses'/f'{index:03d}-{number:03d}.npz';assert sha(path)==r['checkpoint_sha256']
  with np.load(path,allow_pickle=False)as z:loss=z['losses'].copy()
  assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']and len(loss)==H and r['paths']==1000
  dates,values,_=api.load_series(V/'inputs',metadata[case['series_id']],verify_files=False)
  assert score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H])==r['score']
  assert sum(r['selection']['path_counts'])==1000
  for trace in r['validation_trace']:assert trace['training_rows']+trace['validation_horizon']<=stop
  verified+=1
 reduced[index]=score_main.aggregate_records(rows,metadata)['score'];assert reduced[index]==s['models'][rows[0]['model_id']]['main_Q']
for label in('current_best','fixed_reference'):
 rows=[]
 for number in range(len(plan['cases'])):
  r=refs[str(number)][label];path=Path(r['checkpoint']);assert sha(path)==r['checkpoint_sha256']
  with np.load(path,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
  rows.append(r)
 assert score_main.aggregate_records(rows,metadata)['score']==s['reference_main_Q'][label]
number=min((i for i,c in enumerate(plan['cases'])if metadata[c['series_id']]['panel_branch']=='portfolios'),key=lambda i:plan['cases'][i]['origin']['max_horizon'])
case=plan['cases'][number];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];x,w,ids=AssetInputs(V/'inputs').training(metadata[case['series_id']],stop);bundle=prepare(x,w,H)
dates,values,_=api.load_series(V/'inputs',metadata[case['series_id']],verify_files=False);actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000)
for index,config in enumerate(CONFIGS):
 loss=np.empty(H)
 for start,paths in HorizonPool(config,bundle).terminal_chunks(x,w,H,1000,seed):loss[start:start+paths.shape[1]]=empirical_crps_by_horizon(paths,actual[start:start+paths.shape[1]])
 with np.load(D/'losses'/f'{index:03d}-{number:03d}.npz',allow_pickle=False)as z:np.testing.assert_array_equal(loss,z['losses'])
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'selected_checkpoints_verified':verified,'reference_checkpoints_verified':144,'cases_per_model':72,'outer_paths':1000,'main_Q_recomputed':reduced,'reference_main_Q':s['reference_main_Q'],'fresh_multivariate_replay':{'case':number,'constituents':ids,'both_configurations_bitwise_equal':True},'source_closure_verified':True,'native_failures':0,'scoring_integrity_errors':0,'no_promotion':True,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False}
(C/'validation_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
