"""Verify a terminal355-case path study; never fit, simulate or promote."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_path_objective/development1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_path_objective.candidate import identity,schedule
from candidates.asset_selector_universe.bank import BANK
from candidates.asset_selector_universe_precision.candidate import signature
from candidates.asset_selector_stacking.energy import even_allocation,objective
from score_percentage import percent_better,normalize_model_percentages
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
 s=read(D/'summary.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases']
 assert read(D/'active.json')['status']=='terminal'and s['complete_execution']and s['qualification_valid']and not s['errors']
 assert s['paths']==s['inner_paths']==plan['paths']==1000 and len(plan['cases'])==355 and len(s['records'])==710
 assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
 assert not normalize_model_percentages(s)[1]
 metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};inputs=AssetInputs(V/'inputs');seen=set();reused=set();new=set();cache_keys={};reference_rows={k:[]for k in('fixed_reference','current_best','terminal_pool')}
 for r in s['records']:
  i=r['case'];index=r['config_index'];assert(i,index)not in seen;seen.add((i,index));case=plan['cases'][i];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];p=D/'losses'/f'{index:03d}-{i:03d}.npz'
  assert sha(p)==r['checkpoint_sha256']and r['paths']==r['inner_paths']==1000 and r['seed']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
  assert r['series_id']==case['series_id']and r['model_id']==identity(plan['configs'][index])and r['origin_date']==origin['origin_date']
  with np.load(p,allow_pickle=False)as z:loss=z['losses'].copy()
  assert len(loss)==H and np.isfinite(loss).all()and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
  dates,values,_=api.load_series(V/'inputs',row,verify_files=False);assert score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H])==r['score']
  x,w,ids=inputs.training(row,stop);assert list(ids)==r['constituent_ids'];eligible,folds=schedule(len(x),H);assert len(folds)==len(r['validation_trace'])
  for (prefix,lead),trace in zip(folds,r['validation_trace']):
   assert trace['training_rows']==prefix and trace['validation_horizon']==lead and trace['validation_stop']==prefix+lead<=stop and trace['training_sha256']==signature(x[:prefix])
   assert trace['training_last_date']==str(dates[prefix-1].astype('datetime64[D]'))and trace['validation_last_known_date']==str(dates[prefix+lead-1].astype('datetime64[D]'))
  selection=r['selection'];weights=np.asarray(selection['ideal_weights']);assert len(weights)==len(BANK)and np.all(weights>=0)and abs(weights.sum()-1)<1e-12
  np.testing.assert_array_equal(even_allocation(weights,1000),selection['path_counts']);assert not selection['automatic_model_fallback']and selection['no_outer_future_dates_used']and selection['maximum_origin_count']is None
  if index==0:assert r['loss_sha256']==refs[str(i)]['terminal_pool']['loss_sha256']
  if r['reused_verified_precision_case']:
   reused.add(i);assert sha(Path(r['source_checkpoint']))==r['source_checkpoint_sha256']==r['checkpoint_sha256']and r['coefficient_checkpoint']is None
  else:
   new.add(i);cp=Path(r['coefficient_checkpoint']);assert sha(cp)==r['coefficient_checkpoint_sha256']
   with np.load(cp,allow_pickle=False)as z:
    mode=plan['configs'][index]['criterion'];A=z['A_'+mode];B=z['B_'+mode];cw=z['weights_'+mode];valid=z['valid_component_indices'];assert set(valid)<=set(eligible)
    assert A.shape==cw.shape==(len(valid),)and B.shape==(len(valid),len(valid))and np.isfinite(A).all()and np.isfinite(B).all();np.testing.assert_array_equal(B,B.T)
    expected=np.zeros(len(BANK));expected[valid]=cw;np.testing.assert_array_equal(weights,expected);np.testing.assert_array_equal(cw,r['fit_certificate']['weights'])
    assert np.all(cw>=0)and abs(cw.sum()-1)<1e-12
    scale=max(float(np.max(abs(A))),float(np.max(abs(B))),np.finfo(float).tiny);a=A/scale;b=B/scale;gradient=a-b@cw;gap=float(cw@gradient-gradient.min());tol=np.sqrt(np.finfo(float).eps)*len(A)
    assert gap<=tol and objective(cw,a,b)<=float(np.min(a-.5*np.diag(b)))+64*np.finfo(float).eps*len(A)
    assert abs(objective(cw,A,B)-r['fit_certificate']['objective'])<=64*np.finfo(float).eps*len(A)*scale
   key={'training_sha256':signature(x),'training_weights_sha256':hashlib.sha256(np.asarray(w,dtype=float).tobytes()).hexdigest(),'training_dates_sha256':hashlib.sha256(dates[:stop].astype('datetime64[D]').tobytes()).hexdigest(),'requested_horizon':H,'inner_paths':1000,'source_hashes':plan['source_hashes'],'validation_trace_sha256':hashlib.sha256(json.dumps(r['validation_trace'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'coefficient_file':str(cp),'coefficient_file_sha256':sha(cp)}
   if i in cache_keys:assert cache_keys[i]==key
   cache_keys[i]=key
 assert len(reused)==72 and len(new)==283 and not reused&new and len(seen)==710
 for label,rows in reference_rows.items():
  for i in range(355):
   r=refs[str(i)][label];p=Path(r['checkpoint']);assert r['paths']==1000 and sha(p)==r['checkpoint_sha256']
   with np.load(p,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
   rows.append(r)
  assert score_main.aggregate_records(rows,metadata)['score']==s['reference_main_Q'][label]
 for mid,m in s['models'].items():
  rows=[r for r in s['records']if r['model_id']==mid];assert len(rows)==355 and len({r['case']for r in rows})==355 and m['complete']
  q=score_main.aggregate_records(rows,metadata)['score'];assert q==m['main_Q']and percent_better(q,s['reference_main_Q']['fixed_reference'])==m['percent_better_vs_fixed_reference']
  assert m['beats_matched_terminal_pool']==(q<s['reference_main_Q']['terminal_pool'])and m['beats_matched_old_working']==(q<s['reference_main_Q']['current_best'])
 api.original._atomic_json(D/'coefficient_reuse_identity_index.json',{'schema':'training-only-coefficient-provenance-v1','cases':cache_keys,'case_count':283,'scope':'New283cases only; recovered72have no stored matrices; reuse additionally requires compatible component/prefix semantics','outer_outcomes_or_scores_in_cache_key':False})
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'candidate_checkpoints_verified':710,'reference_checkpoints_verified':1065,'exact_endpoint_control_loss_replays':355,'reused_case_source_checkpoints_verified':144,'new_coefficient_matrices_verified':283,'scaled_simplex_certificates_verified':566,'historical_prefix_and_calendar_causality_verified':710,'all_main_Q_and_percentage_reductions_recomputed':True,'summary_sha256':sha(D/'summary.json'),'verifier_source_sha256':sha(Path(__file__)),'source_closure_verified':True,'models':s['models'],'reference_main_Q':s['reference_main_Q'],'official':False,'full_panel_promotion':False,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False}
 api.original._atomic_json(D/'verification_receipt.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':verify()
