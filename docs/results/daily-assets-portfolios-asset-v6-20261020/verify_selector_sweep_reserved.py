"""Audit terminal all-development replay and all reserved forecast checkpoints."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_coefficient_sweep/validation_and_reserved';PRE=V/'candidates/asset_selector_path_objective/development1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_coefficient_sweep.candidate import identity,schedule
from candidates.asset_selector_universe.bank import BANK
from candidates.asset_selector_universe_precision.candidate import signature
from candidates.asset_selector_stacking.energy import even_allocation
from score_percentage import percent_better,normalize_model_percentages
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda x:hashlib.sha256(x.tobytes()).hexdigest()
def verify():
 s=read(D/'summary.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases'];prior=read(PRE/'summary.json');old={(r['case'],r['config_index']):r for r in prior['records']}
 assert read(D/'active.json')['status']=='terminal'and s['complete_execution']and s['qualification_valid']and not s['errors']
 assert s['paths']==s['inner_paths']==1000 and len(s['records'])==710 and len(s['parity_receipts'])==355
 assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())and not normalize_model_percentages(s)[1]
 assert len(plan['parity_cases'])==len(plan['reserved_cases'])==355
 key=lambda c:(c['series_id'],json.dumps(c['origin'],sort_keys=True));assert not(set(map(key,plan['parity_cases']))&set(map(key,plan['reserved_cases'])))
 seen=set();arrays=0;parity_forecasts=0
 for receipt in s['parity_receipts']:
  i=receipt['case'];assert i not in seen;seen.add(i);assert not receipt['errors']and len(receipt['records'])==2
  for r in receipt['records']:
   reference=old[i,r['config_index']];assert r['complete_outer_loss_vector_exact']and r['model_id']==reference['model_id']and sha(Path(r['source_checkpoint']))==r['source_checkpoint_sha256']==reference['checkpoint_sha256']and r['loss_sha256']==reference['loss_sha256']
   with np.load(r['source_checkpoint'],allow_pickle=False)as z:assert digest(z['losses'])==r['loss_sha256']
   assert r['ideal_weights_sha256']==digest(np.asarray(reference['selection']['ideal_weights'],dtype=float));parity_forecasts+=1
   if reference['coefficient_checkpoint']:
    cp=Path(reference['coefficient_checkpoint']);assert sha(cp)==reference['coefficient_checkpoint_sha256'];mode=plan['configs'][r['config_index']]['criterion']
    with np.load(cp,allow_pickle=False)as z:
     for k in('A','B'):assert r['coefficient_sha256'][k]==digest(z[k+'_'+mode]);arrays+=1
 assert len(seen)==355 and parity_forecasts==710 and arrays==1132==s['coefficient_arrays_byte_verified']
 metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};inputs=AssetInputs(V/'inputs');seen=set();endpoint_controls=0
 for r in s['records']:
  i=r['case'];index=r['config_index'];assert(i,index)not in seen;seen.add((i,index));case=plan['reserved_cases'][i];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];cp=D/'reserved/losses'/f'{index:03d}-{i:03d}.npz'
  assert sha(cp)==r['checkpoint_sha256']and r['paths']==r['inner_paths']==1000 and r['seed']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
  assert r['series_id']==case['series_id']and r['model_id']==identity(plan['configs'][index])and r['origin_date']==origin['origin_date']
  with np.load(cp,allow_pickle=False)as z:loss=z['losses'].copy()
  assert len(loss)==H and np.isfinite(loss).all()and digest(loss)==r['loss_sha256'];dates,values,_=api.load_series(V/'inputs',row,verify_files=False)
  assert score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H])==r['score'];x,w,ids=inputs.training(row,stop);assert list(ids)==r['constituent_ids'];eligible,folds=schedule(len(x),H);assert len(folds)==len(r['validation_trace'])
  for (prefix,lead),trace in zip(folds,r['validation_trace']):
   assert trace['training_rows']==prefix and trace['validation_horizon']==lead and trace['validation_stop']==prefix+lead<=stop and trace['training_sha256']==signature(x[:prefix])
   assert trace['training_last_date']==str(dates[prefix-1].astype('datetime64[D]'))and trace['validation_last_known_date']==str(dates[prefix+lead-1].astype('datetime64[D]'))
  weights=np.asarray(r['selection']['ideal_weights']);assert len(weights)==len(BANK)and np.all(weights>=0)and abs(weights.sum()-1)<1e-12;np.testing.assert_array_equal(even_allocation(weights,1000),r['selection']['path_counts'])
  invalid={int(i)for i in r['selection']['invalid_internal_options']};valid=[i for i in eligible if i not in invalid];np.testing.assert_array_equal(weights[valid],r['fit_certificate']['weights']);assert np.all(weights[[i for i in range(len(BANK))if i not in valid]]==0)
  assert r['selection']['no_outer_future_dates_used']and not r['selection']['automatic_model_fallback']and r['selection']['maximum_origin_count']is None
  if index==0:assert r['loss_sha256']==refs[str(i)]['terminal_pool']['loss_sha256'];endpoint_controls+=1
 assert len(seen)==710 and endpoint_controls==355
 for label in('fixed_reference','current_best','terminal_pool'):
  rows=[]
  for i in range(355):
   r=refs[str(i)][label];cp=Path(r['checkpoint']);assert r['paths']==1000 and sha(cp)==r['checkpoint_sha256']
   with np.load(cp,allow_pickle=False)as z:assert digest(z['losses'])==r['loss_sha256']
   rows.append(r)
  assert score_main.aggregate_records(rows,metadata)['score']==s['reference_main_Q'][label]
 for mid,m in s['models'].items():
  rows=[r for r in s['records']if r['model_id']==mid];assert len(rows)==355 and len({r['case']for r in rows})==355 and m['complete'];q=score_main.aggregate_records(rows,metadata)['score']
  assert q==m['main_Q']and percent_better(q,s['reference_main_Q']['fixed_reference'])==m['percent_better_vs_fixed_reference']
  assert m['beats_matched_terminal_pool']==(q<s['reference_main_Q']['terminal_pool'])and m['beats_matched_old_working']==(q<s['reference_main_Q']['current_best'])
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'all355_development_replay_receipts_verified':True,'development_forecast_loss_replays_verified':710,'development_coefficient_array_hashes_verified':1132,'reserved_candidate_checkpoints_verified':710,'reserved_reference_checkpoints_verified':1065,'reserved_endpoint_control_replays_verified':355,'all_reserved_origin_and_main_scores_and_percentages_recomputed':True,'historical_prefix_and_calendar_causality_verified':710,'reserved_fit_certificate_scope':'Weights/simplex/path allocation and disclosed causal support verified; raw per-fold reserved coefficient cubes are not stored or independently refit by this audit','source_closure_verified':True,'summary_sha256':sha(D/'summary.json'),'verifier_source_sha256':sha(Path(__file__)),'models':s['models'],'reference_main_Q':s['reference_main_Q'],'official':False,'full_panel_promotion':False,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False};api.original._atomic_json(D/'verification_receipt.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':verify()
