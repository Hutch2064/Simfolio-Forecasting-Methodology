"""Native complete evaluation of a broad-qualified frozen feature model."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,datetime,hashlib,fcntl,subprocess,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_feature_batch';F=C/'full';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from candidates.asset_feature_batch.candidate import CONFIGS,FeaturePool,identity
from audit_asset_v6_full_pair import independent_origin,independent_reduce
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json;now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    selection=read(C/'full_selection.json');config=selection['config'];mid=identity(config);assert mid==selection['model_id']
    prior=read(V/'official_scoreboard.json');parent=prior['current_best_model_id'];parent_Q=prior['models'][parent]['main_score'];assert parent=='model_testing_asset_circular_raw_GARCH_fade_FHS_mean_v1'
    phases=('confirmation1000','reserved1000','broad1000');source_files={};evidence=[]
    for phase in phases:
        d=C/phase;plan=read(d/'plan.json');summary=read(d/'summary.json');assert summary['paths']==1000 and summary['complete_execution'] and not summary['errors']
        assert summary['anchor_loss_vectors_bitwise_equal_each_case']
        candidate=summary['models'][mid];assert candidate['complete'] and candidate['main_Q']<summary['reference_main_Q']['current_best']
        for name,digest in plan['source_hashes'].items():
            assert sha(O/name)==digest
            if name.startswith('asset_level_v6/'):source_files[name[len('asset_level_v6/'):]]=digest
        evidence.append({'phase':phase,'summary_sha256':sha(d/'summary.json'),'plan_sha256':sha(d/'plan.json')})
    assert not (F/'active.json').exists(),'Inspect an earlier full evaluation instead of restarting'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','1.8','--artifact-root',str(O),'--budget-gib','16'],check=True)
    F.mkdir(exist_ok=True);factory=f'candidates.asset_feature_batch.candidate:build_{CONFIGS.index(config):03d}';model=FeaturePool(config)
    declared={'model_id':mid,'factory':factory,'source_files':source_files,'numerical_contract':api.original.dependency_identity(model)};digest=api.original.stable_digest(declared)
    spec={'enabled':True,'factory':factory,'implementation_digest':digest};write(F/'control.json',{'workers':12,'stop':False,'models':{mid:spec}})
    write(F/'registry.json',{'schema':'french-external-model-registry-v1','source_root':str(V),'models':{mid:{'identity':declared,'factory':factory,'implementation_digest':digest}}})
    target=api.model_directory(F/'results',mid);target.mkdir(parents=True,exist_ok=True)
    model,panel,origins,fingerprint=api.initialize_model(mid,spec,V/'inputs',target,F/'registry.json')
    by_id={r['series_id']:(i,r)for i,r in enumerate(panel['series'])};inputs=__import__('inputs').AssetInputs(V/'inputs');reused=[];native_replays=[]
    # Reuse only complete source-bound1000-path vectors with identical asset/context seeds.
    for phase in phases:
        d=C/phase;plan=read(d/'plan.json');summary=read(d/'summary.json');packets={(r['case'],r['family']):r for r in summary['fit_packets']}
        rows=[r for r in summary['records']if r['model_id']==mid];assert len(rows)==len(plan['cases'])
        for record in rows:
            case=plan['cases'][record['case']];sid=case['series_id'];i,row=by_id[sid];origin=case['origin'];j=list(origins[i]).index(origin);task_identity=api.task_identity(fingerprint,mid,sid,origin)
            assert record['seed']==task_identity['seed'] and record['paths']==1000
            source=d/'losses'/f"{record['config_index']:03d}-{record['case']:03d}.npz";assert sha(source)==record['checkpoint_sha256']
            with np.load(source,allow_pickle=False)as z:loss=z['losses'].copy()
            assert len(loss)==origin['max_horizon'] and hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256']
            ids=[r[0]for r in row['rational_weights']]if row['panel_branch']=='portfolios'else[sid];assert list(record['constituent_ids'])==ids
            # Small direct native adapter replays establish equivalence before importing any phase.
            if len(native_replays)<4 and (row['panel_branch']=='portfolios' or len(native_replays)==0):
                dates,values,_=api.load_series(V/'inputs',row,verify_files=True)
                from types import SimpleNamespace
                regenerated,_=api.forecast_asset(model,V/'inputs',row,values,dates,SimpleNamespace(**origin),task_identity['seed'])
                np.testing.assert_array_equal(regenerated,loss)
                native_replays.append({'phase':phase,'case':record['case'],'loss_sha256':record['loss_sha256'],'branch':row['panel_branch']})
            packet=packets[record['case'],config['family']]
            attributed=record['simulation_CRPS_seconds_excluding_shared_fit_and_checkpoint_IO']+packet['base_fit_compute_seconds']+packet['filter_fit_compute_seconds']
            diagnostics={'forecast_and_score_seconds':attributed,'timing_scope':'Imported pilot simulation/CRPS plus attributable base/filter fitting; excludes pilot input preparation, checkpoint IO, queue; native remaining tasks include input preparation. Not controlled speed comparison.','simulations':1000,'all_paths_scored_at_every_horizon':True,'horizon_days':len(loss),'horizon_chunk':64,'constituent_ids':ids,'joint_asset_paths_simulated':True,'portfolio_weights_applied_inside_paths':True,'forecast_level':'joint_assets','native_training_rows':origin['position']+1,'worker_first_origin_warmup':False,'imported_verified_pilot':True,'source_phase':phase,'source_checkpoint_sha256':sha(source)}
            cp,_=api.task_paths(target,(i,j));assert not cp.exists(),'Frozen phase coverage must be disjoint';api.original._write_checkpoint(cp,task_identity,loss,diagnostics)
            check,_=api.original._read_checkpoint(cp,task_identity);np.testing.assert_array_equal(check,loss);reused.append({'task':[i,j],'phase':phase,'checkpoint_sha256':sha(cp),'original_loss_sha256':record['loss_sha256']})
    assert len(reused)==8128 and len(native_replays)==4
    write(F/'admission.json',{'utc':now(),'model_id':mid,'config':config,'expected_tasks':64049,'paths':1000,'workers':12,'bounded_full_task_count':64049,'panel_fingerprint':panel['panel_fingerprint'],'same_source_bound_pilot_checkpoints_reused':8128,'native_adapter_loss_byte_replays':native_replays,'reused_checkpoint_records':reused,'phase_evidence':evidence,'previous_verified_best_id':parent,'previous_verified_best_main_Q':parent_Q,'runner_sha256':sha(Path(__file__)),'not_promoted_before_complete_full_score':True,'baseline_results_unchanged':True})
    write(F/'active.json',{'pid':os.getpid(),'status':'running','utc':now()})
    final=api.run_controller(V/'inputs',F/'results',F/'control.json',registry_path=F/'registry.json',poll_seconds=.05,max_tasks=64049)
    assert final['completed_active_tasks']==final['expected_active_tasks']==64049 and final['failed_active_tasks']==final['active_workers']==final['ready_queue']==0
    assert all(sha(V/name)==value for name,value in source_files.items());fcntl.flock(slots,fcntl.LOCK_UN)
    records=[];compute=0.;commitment=hashlib.sha256();imported=0
    for i,row in enumerate(panel['series']):
        dates,values,scheduled=api.load_series(V/'inputs',row,verify_files=True);ids=[r[0]for r in row['rational_weights']]if row['panel_branch']=='portfolios'else[row['series_id']]
        for j,origin in enumerate(scheduled):
            cp,_=api.task_paths(target,(i,j));loss,d=api.original._read_checkpoint(cp,api.task_identity(fingerprint,mid,row['series_id'],origin));assert d['simulations']==1000 and d['all_paths_scored_at_every_horizon'] and d['forecast_level']=='joint_assets' and d['constituent_ids']==ids
            n=origin['position']+1;records.append((row['series_id'],origin['origin_date'],independent_origin(loss,values[:n],origin['origin_date'],dates[n:n+len(loss)])));compute+=d['forecast_and_score_seconds'];imported+=bool(d.get('imported_verified_pilot',False));commitment.update(f'{i}/{j}:{sha(cp)}\n'.encode())
    assert imported==8128;q=independent_reduce(records,{r['series_id']:r for r in panel['series']})['main_score'];audit=read(V/'full_pair_independent_audit.json');ref=audit['worse_reference_model_id'];ref_Q=audit['models'][ref]['main_score']
    result={'utc':now(),'complete':True,'model_id':mid,'config':config,'series':1264,'successful_tasks':64049,'simulations':1000,'native_failures':0,'scoring_integrity_errors':0,'main_score':q,'sole_reference_model_id':ref,'reference_main_score':ref_Q,'percent_better_vs_worse_reference':100*(1-q/ref_Q),'previous_verified_best_id':parent,'previous_verified_best_main_Q':parent_Q,'beats_better_parent_on_full_identical_coverage':q<parent_Q,'eligible_for_current_best_promotion':q<parent_Q,'checkpoint_commitment':commitment.hexdigest(),'worker_compute_seconds':compute,'runtime_scope':'55,921 native tasks include input preparation;8,128 imported pilot tasks include simulation/CRPS plus attributed fit cost, excluding shared input preparation. No controlled speed comparison.','independent_validation_or_significance_claim':False,'goal_complete':False}
    write(F/'verification_receipt.json',result);write(F/'active.json',{'utc':now(),'pid':os.getpid(),'status':'terminal'});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
