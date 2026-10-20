"""Independent full joint-asset pair audit; no forecasts, admission or mutation."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,datetime,math,statistics,importlib.util
from collections import defaultdict
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
import common,sweep_v6 as api
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def independent_origin(loss,history,origin_date,future_dates):
    loss=np.asarray(loss,dtype=float);x=np.asarray(history,dtype=float)
    assert x.ndim==loss.ndim==1 and len(x)>1 and len(loss)==len(future_dates)
    assert np.isfinite(x).all() and np.isfinite(loss).all() and np.min(loss)>=0
    sd=float(np.std(x,ddof=1));assert sd>0 and math.isfinite(sd)
    days=(np.asarray(future_dates).astype('datetime64[D]')-np.datetime64(str(origin_date)[:10],'D')).astype(int)
    assert days[0]>0 and np.all(np.diff(days)>0)
    log_edges=np.log1p(days.astype(float))
    masses=np.diff(np.concatenate(([0.],log_edges)))/log_edges[-1]
    return float(np.sum(masses*loss/(sd*np.sqrt(np.arange(1,len(loss)+1,dtype=float)))))

def independent_reduce(records,metadata):
    # Group by branch, quarter, class, family and individual return stream first.
    groups=defaultdict(list)
    for sid,date,value in records:
        m=metadata[sid];q=f'{date[:4]}-Q{(int(date[5:7])-1)//3+1}'
        assert math.isfinite(value) and value>=0
        groups[(m['panel_branch'],q,m['asset_class'],m['family'],sid)].append(value)
    families=defaultdict(list)
    for (branch,q,cls,family,sid),values in groups.items():families[(branch,q,cls,family)].append(statistics.fmean(values))
    classes=defaultdict(list)
    for (branch,q,cls,family),values in families.items():classes[(branch,q,cls)].append(statistics.fmean(values))
    branches=defaultdict(list)
    for (branch,q,cls),values in classes.items():branches[(branch,q)].append(statistics.fmean(values))
    quarters=defaultdict(dict)
    for (branch,q),values in branches.items():quarters[q][branch]=statistics.fmean(values)
    qvalues={}
    for q,bs in quarters.items():
        assert 'assets' in bs,'Portfolio-only quarter must fail closed'
        assert set(bs)<= {'assets','portfolios'}
        qvalues[q]=.5*(bs['assets']+bs['portfolios']) if 'portfolios' in bs else bs['assets']
    return {'main_score':statistics.fmean(qvalues.values()),'quarters':qvalues,
            'portfolio_cap':max(.5 if 'portfolios' in v else 0 for v in quarters.values()),
            'quarter_count':len(qvalues)}

def verify():
 spec=importlib.util.spec_from_file_location('sealed_v6_run_pair_audit',V/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);runner.guard()
 active=read(V/'controller_active.json');assert active['status']=='terminal' and active['stage']=='full'
 terminal=read(V/'controller_terminal_receipt.json')['final_status'];assert terminal['active_workers']==terminal['ready_queue']==terminal['failed_active_tasks']==0;assert terminal['completed_active_tasks']==terminal['expected_active_tasks']==128098
 report=read(V/'results/scoreboard.json');assert report['native_failures']==report['scoring_integrity_errors']==0 and len(report['complete_coverage_leaderboard'])==2 and not report['reference_selection_preliminary']
 control=api.read_control(V/'control.json');assert sha(V/'control.json')==report['control_sha256'];enabled={m:s for m,s in control['models'].items()if s['enabled']};assert len(enabled)==2
 panel=api.load_panel(V/'inputs',verify_files=True);metadata={r['series_id']:r for r in panel['series']};assert len(metadata)==1264 and panel['expected_tasks']==64049
 roots={mid:api.model_directory(V/'results',mid)for mid in enabled};fps={mid:read(root/'execution.json')['fingerprint']for mid,root in roots.items()};summaries={m:{'records':[],'verified':0,'compute_seconds':0.,'checkpoint_commitment':hashlib.sha256()}for m in enabled}
 for mid,specification in enabled.items():
  api.admitted_contract(mid,specification,V/'registry.json');manifest=read(roots[mid]/'execution.json');assert manifest['identity']['runtime']['forecast_level']=='joint_assets'
 for i,row in enumerate(panel['series']):
  dates,values,origins=api.load_series(V/'inputs',row,verify_files=True);ids=[sid for sid,_,_ in row['rational_weights']]if row['panel_branch']=='portfolios'else[row['series_id']]
  for j,origin in enumerate(origins):
   for mid in enabled:
    cp,_=api.task_paths(roots[mid],(i,j));loss,d=api.original._read_checkpoint(cp,api.task_identity(fps[mid],mid,row['series_id'],origin));assert d['simulations']==1000 and d['all_paths_scored_at_every_horizon'] and d['forecast_level']=='joint_assets' and d['joint_asset_paths_simulated'] and d['portfolio_weights_applied_inside_paths'];assert d['constituent_ids']==ids
    n=origin['position']+1;value=independent_origin(loss,values[:n],origin['origin_date'],dates[n:n+origin['max_horizon']]);s=summaries[mid];s['records'].append((row['series_id'],origin['origin_date'],value));s['verified']+=1;s['compute_seconds']+=d['forecast_and_score_seconds'];s['checkpoint_commitment'].update(f'{i}/{j}:{sha(cp)}\n'.encode())
  if(i+1)%200==0:print(json.dumps({'series_audited':i+1,'checkpoints':sum(s['verified']for s in summaries.values())}),flush=True)
 for mid,s in summaries.items():
  reduced=independent_reduce(s.pop('records'),metadata);assert s['verified']==64049 and reduced['portfolio_cap']<=.5;assert abs(reduced['main_score']-report['models'][mid]['main_score_on_own_coverage'])<2e-12;s.update(reduced);s['checkpoint_commitment']=s['checkpoint_commitment'].hexdigest()
 worse=max(summaries,key=lambda m:(summaries[m]['main_score'],m));better=min(summaries,key=lambda m:(summaries[m]['main_score'],m));assert worse==report['reference_model_id']
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_pair_audit':True,'total_checkpoints_verified':128098,'series':1264,'tasks_per_model':64049,'simulations':1000,'native_failures':0,'scoring_integrity_errors':0,'models':summaries,'worse_reference_model_id':worse,'better_starting_model_id':better,'same_successful_coverage':True,'score_and_hierarchical_reduction_independently_verified':True,'actual_joint_path_source_bound_and_constituent_identity_verified':True,'timing_scope':'Recorded worker compute; no controlled speedup claim','goal_complete':False}
 api.original._atomic_json(V/'full_pair_independent_audit.json',result);print(json.dumps({k:v for k,v in result.items()if k!='models'}),flush=True)
if __name__=='__main__':verify()
