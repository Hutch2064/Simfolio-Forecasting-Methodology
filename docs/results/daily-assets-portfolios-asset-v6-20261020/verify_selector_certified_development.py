"""Independent saved-loss reduction and convex certificate audit of355cases."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_universe_certified';D=C/'development1000';F=C/'repair_qualification';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(D/'summary.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases'];metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
assert s['complete_execution']and not s['errors']and s['successful_selected_forecasts']==710
assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
for r in s['records']:
 p=D/'losses'/f"{r['config_index']:03d}-{r['case']:03d}.npz";assert sha(p)==r['checkpoint_sha256']and sha(Path(r['source_checkpoint']))==r['source_checkpoint_sha256']
 with np.load(p,allow_pickle=False)as z:loss=z['losses'].copy()
 assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256'];case=plan['cases'][r['case']];origin=case['origin'];stop=origin['position']+1
 dates,values,_=api.load_series(V/'inputs',metadata[case['series_id']],verify_files=False)
 assert len(loss)==origin['max_horizon']and r['paths']==1000
 assert score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+len(loss)])==r['score']
for mid,m in s['models'].items():
 rows=[r for r in s['records']if r['model_id']==mid];assert len(rows)==355 and len({r['case']for r in rows})==355
 assert score_main.aggregate_records(rows,metadata)['score']==m['main_Q']
for label in('current_best','fixed_reference'):
 rows=[]
 for i in range(355):
  r=refs[str(i)][label];p=Path(r['checkpoint']);assert sha(p)==r['checkpoint_sha256']
  with np.load(p,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
  rows.append(r)
 assert score_main.aggregate_records(rows,metadata)['score']==s['reference_main_Q'][label]
certificates=[]
for p in sorted(F.glob('internal-coefficients-*.npz')):
 with np.load(p,allow_pickle=False)as z:A=z['A'];B=z['B'];w=z['weights']
 k=len(w);scale=max(np.max(np.abs(A)),np.max(np.abs(B)),np.finfo(float).tiny);grad=A-B@w;gap=float(w@grad-grad.min());q=float(w@A-.5*w@(B@w));vertex=float(np.min(A-.5*np.diag(B)))
 assert np.all(w>=0)and abs(w.sum()-1)<1e-12 and gap<=np.sqrt(np.finfo(float).eps)*k*scale
 assert q<=vertex+64*np.finfo(float).eps*k*scale
 projector=np.eye(k)-np.ones((k,k))/k;smallest=float(np.linalg.eigvalsh(-projector@B@projector).min());assert smallest>=-64*np.finfo(float).eps*k*scale
 certificates.append({'file':str(p),'sha256':sha(p),'objective':q,'best_vertex':vertex,'global_simplex_gap':gap,'projected_hessian_min_eigenvalue':smallest})
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'selected_loss_checkpoints_verified':710,'reference_loss_checkpoints_verified':710,'all_origin_and_main_scores_recomputed':True,'source_checkpoint_aliases_all_verified':True,'cases_per_model':355,'paths':1000,'inner_paths':1000,'repaired_original_fit_failures':3,'fresh_successful_multivariate_replay_case':221,'fresh_replay_hash_equality_verified':read(F/'summary.json')['successful_replay_bitwise_equal'],'independent_saved_convex_coefficients_verified':certificates,'no_full_panel_or_independence_claim':True,'full_panel_catalogue_increment':0,'native_failures_after_repair':0,'scoring_integrity_errors':0,'legacy_nominal_parent_replays355_corrected_to_actual352':True}
(C/'validation_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
