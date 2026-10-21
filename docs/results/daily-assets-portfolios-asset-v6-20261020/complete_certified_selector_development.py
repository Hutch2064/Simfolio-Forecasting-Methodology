"""Complete matched development from verified compatible successes and repairs."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_universe_certified';OLD=V/'candidates/asset_selector_universe_precision/development1000';FIX=C/'repair_qualification';D=C/'development1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from candidates.asset_selector_universe_certified.candidate import CONFIGS,identity
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
old=read(OLD/'summary.json');plan=read(OLD/'plan.json');repair=read(FIX/'summary.json');fix_plan=read(FIX/'plan.json')
assert old['complete_execution']and read(OLD/'active.json')['status']=='terminal'
assert repair['complete']and repair['successful_replay_bitwise_equal']and not repair['native_failures']and not repair['scoring_integrity_errors']
assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
assert all(sha(O/p)==h for p,h in fix_plan['source_hashes'].items())
# The complete forecasting body is text-identical apart from numerical fit import
# and metadata identity. All coefficients, seeds, allocation and path code match.
before=(V/'candidates/asset_selector_universe_precision/candidate.py').read_text()
expected=before.replace('from candidates.asset_selector_stacking import energy','from . import energy').replace('model_testing_asset_broad_horizon_pool_precision_v1_','model_testing_asset_broad_horizon_pool_certified_v1_')
assert(C/'candidate.py').read_text()==expected
assert not D.exists(),'Never overwrite an earlier development execution'
D.mkdir();(D/'losses').mkdir();rows={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};refs=read(OLD/'references.json')['cases'];records=[];reused=0;new=0
old_rows={(r['case'],r['config_index']):r for r in old['records']};fixed_rows={(r['case'],r['config_index']):r for r in repair['records']}
for index,config in enumerate(CONFIGS):
 for number,case in enumerate(plan['cases']):
  key=number,index
  if key in old_rows:source=old_rows[key];folder=OLD;reused+=1
  else:source=fixed_rows[key];folder=FIX;new+=1
  src=folder/'losses'/f'{index:03d}-{number:03d}.npz';assert sha(src)==source['checkpoint_sha256']
  with np.load(src,allow_pickle=False)as z:loss=z['losses'].copy()
  assert hashlib.sha256(loss.tobytes()).hexdigest()==source['loss_sha256']and source['paths']==1000
  assert source['series_id']==case['series_id']and source['origin_date']==case['origin']['origin_date']
  dates,values,_=api.load_series(V/'inputs',rows[case['series_id']],verify_files=False);origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];assert len(loss)==H
  assert source['seed']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
  q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==source['score']
  dst=D/'losses'/src.name;shutil.copy2(src,dst);assert sha(dst)==sha(src)
  records.append({**source,'model_id':identity(config),'source_model_id':source['model_id'],'source_checkpoint':str(src),'source_checkpoint_sha256':sha(src),'reused_compatible_success':folder==OLD})
reference_Q={}
for label in('current_best','fixed_reference'):
 ref_rows=[]
 for number in range(len(plan['cases'])):
  r=refs[str(number)][label];p=Path(r['checkpoint']);assert sha(p)==r['checkpoint_sha256']
  with np.load(p,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
  ref_rows.append(r)
 reference_Q[label]=score_main.aggregate_records(ref_rows,rows)['score'];assert np.isfinite(reference_Q[label])
models={};ranking=[]
for index,config in enumerate(CONFIGS):
 mid=identity(config);selected=[r for r in records if r['model_id']==mid];assert len(selected)==355 and len({r['case']for r in selected})==355
 q=score_main.aggregate_records(selected,rows)['score'];assert np.isfinite(q)
 r={'model_id':mid,'config':config,'config_index':index,'successful_cases':355,'expected_cases':355,'complete':True,'main_Q':q,'percent_better_vs_same1000path_fixed_reference':100*(1-q/reference_Q['fixed_reference']),'beats_same1000path_current_best':q<reference_Q['current_best']}
 models[mid]=r;ranking.append(r)
ranking.sort(key=lambda r:r['main_Q'])
closure={**plan['source_hashes'],**fix_plan['source_hashes'],'complete_certified_selector_development.py':sha(Path(__file__))}
write(D/'plan.json',{**plan,'source_hashes':closure,'configs':CONFIGS,'numerical_repair':'Accept certified global empirical convex optimum regardless of solver success flag; repeat uncertified iteration-limit chunks without total cap','old_execution':str(OLD),'repair_qualification':str(FIX),'reduction_order':'config_index then original frozen case number','source_compatibility':'Same initial numerical solve and identical forecast body; preceding successful fits pass identical certificate and produce identical weights/paths. No claim every reused fit was freshly re-executed. Actual multivariate replay and numeric tests verified.'})
write(D/'references.json',{'cases':refs})
valid_fit_seconds=0.
for number in sorted({r['case']for r in records}):
 if(number,0)in old_rows and(number,1)in old_rows:valid_fit_seconds+=read(OLD/'case_progress'/f'{number:03d}.json')['shared_fit_seconds']
 else:valid_fit_seconds+=next(r['shared_fit_seconds']for r in repair['records']if r['case']==number)
runtime={'scope':'Local numerical experiment timing; shared selection/refit costs are not per-mode standalone timings or website latency','successful_request_shared_fit_seconds':valid_fit_seconds,'initial_execution_shared_fit_seconds_including_failed_fits':old['shared_fit_compute_seconds'],'initial_saved_forecast_simulation_CRPS_seconds':sum(r['simulation_CRPS_seconds_excluding_shared_fit_and_checkpoint_IO']for r in old['records']),'repair_and_successful_replay_wall_seconds_including_IO':(datetime.datetime.fromisoformat(repair['utc'])-datetime.datetime.fromisoformat(fix_plan['utc'])).total_seconds(),'fit_cost_includes_both_policies_shared':True,'full_panel_runtime_not_measured':True}
anchors={'initial_parent_replay_successes_proven_by_forecast_records':len({r['case']for r in old['records']}),'legacy_summary_nominal_anchor_count':old.get('anchor_replays'),'legacy_nominal_count_includes_fit_failures_before_anchor_and_is_not_a_verified_replay_count':True,'reference_loss_checkpoints_hash_verified_this_phase':2*len(plan['cases']),'financial_score_unchanged_by_metadata_correction':True}
summary={'utc':now(),'complete_execution':True,'official':False,'paths':1000,'inner_paths':1000,'cases_per_model':355,'component_options':166,'configurations':2,'successful_selected_forecasts':len(records),'reused_verified_compatible_forecasts':reused,'newly_repaired_forecasts':new,'historical_initial_precision_fit_failures':len(old['errors']),'historical_initial_failures_preserved':True,'native_failures':0,'scoring_integrity_errors':0,'errors':[],'runtime':runtime,'parent_reference_verification':anchors,'reference_main_Q':reference_Q,'models':models,'complete_identical_coverage_development_ranking':ranking,'records':records,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False,'no_official_promotion':True}
assert all(sha(O/p)==h for p,h in closure.items())
write(D/'summary.json',summary);write(D/'active.json',{'pid':0,'status':'terminal','utc':now(),'no_scientific_queue':True})
print(json.dumps({k:v for k,v in summary.items()if k not in('records','models')},indent=2),flush=True)
# This separate optimization prototype cannot invalidate the completed scores.
# The parent handoff still owns the shared worker lock during this unit gate.
with(O/'selector_batched_qualification.log').open('w')as log:
 result=subprocess.run([sys.executable,'-m','unittest','test_asset_selector_batched','-v'],cwd=O,stdout=log,stderr=subprocess.STDOUT)
batch=V/'candidates/asset_selector_batched'
write(batch/'qualification_receipt.json',{'utc':now(),'tests_passed':3 if result.returncode==0 else None,'return_code':result.returncode,'source_hashes':{p.name:sha(p)for p in batch.glob('*.py')},'test_source_sha256':sha(O/'test_asset_selector_batched.py'),'status':'unit_gate_passed_all_work_reference_validation_still_required'if result.returncode==0 else 'unit_gate_failed_no_admission','scientific_model_or_score_changed':False,'speedup_not_yet_measured':True})
with(O/'selector_component_profile.log').open('w')as log:
 profile=subprocess.run([sys.executable,str(O/'profile_selector_inner_components.py')],cwd=O,stdout=log,stderr=subprocess.STDOUT)
write(O/'selector_component_profile_execution.json',{'utc':now(),'return_code':profile.returncode,'diagnostic_only':True,'canonical_development_scores_unchanged':True})
