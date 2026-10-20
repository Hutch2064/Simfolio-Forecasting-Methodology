"""Independent fixed-score audit after full1,000-path retained execution closes.

Reads every retained checkpoint; does not fit, forecast, mutate sources/controls,
change score weights, restore archives or admit a model.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,datetime,sqlite3,math,statistics
from collections import defaultdict
import numpy as np
O=Path(__file__).resolve().parent;V=O/'simulation_1000_v5';sys.path.insert(0,str(V))
import sweep_v5 as api
import native_index
import run as frozen_run
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


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
    frozen_run.guard()
    a=read(V/'controller_active.json');assert a['status']=='terminal'
    try:os.kill(a['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('Baseline controller still live')
    r=read(V/'results/scoreboard.json');control=api.read_control(V/'control.json')
    enabled={m:s for m,s in control['models'].items() if s['enabled']}
    assert len(enabled)==4 and not any('FAST' in m or 'R52' in m for m in enabled)
    assert r['native_failures']==r['scoring_integrity_errors']==0
    assert not r['reference_selection_preliminary'] and len(r['complete_coverage_leaderboard'])==4
    assert r['control_sha256']==sha(V/'control.json')
    panel=api.load_panel(V/'inputs',verify_files=True)
    assert len(panel['series'])==1264 and panel['expected_tasks']==64049
    metadata={x['series_id']:x for x in panel['series']}
    assert sum(x['panel_branch']=='assets' for x in metadata.values())==722
    assert sum(x['panel_branch']=='portfolios' for x in metadata.values())==542
    cache=sqlite3.connect(f'file:{V / "results/native-loss-index.sqlite3"}?mode=ro',uri=True)
    cache.row_factory=sqlite3.Row;cache.execute('BEGIN')
    assert cache.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    h=native_index.source_digest();summaries={};record_map={};fingerprints={};roots={}
    for mid,spec in enabled.items():
        api.admitted_contract(mid,spec,V/'registry.json')
        root=api.model_directory(V/'results',mid);roots[mid]=root
        manifest=read(root/'execution.json');fingerprints[mid]=manifest['fingerprint']
        assert manifest['identity']['simulations_per_origin']==1000
        rows=cache.execute('SELECT * FROM task_scores_main WHERE panel_fingerprint=? AND scorer_hash=? AND model_id=? AND execution_fingerprint=? AND status=?',
          (panel['panel_fingerprint'],h,mid,manifest['fingerprint'],'ok')).fetchall()
        assert len(rows)==64049
        record_map[mid]={row['task_key']:row for row in rows}
        assert len(record_map[mid])==64049
        summaries[mid]={'records':[],'sum_forecast_score_seconds':0.,'max_independent_origin_difference':0.,'checkpoint_commitment':hashlib.sha256(),'verified':0}
    for i,row in enumerate(panel['series']):
        dates,values,origins=api.load_series(V/'inputs',row,verify_files=True)
        for j,origin in enumerate(origins):
            key=json.dumps([row['series_id'],origin],sort_keys=True,separators=(',',':'))
            for mid in enabled:
                indexed=record_map[mid][key]
                cp,_=api.task_paths(roots[mid],(i,j))
                digest=sha(cp);assert digest==indexed['checkpoint_sha256'] and str(cp)==indexed['checkpoint_path']
                losses,d=api.original._read_checkpoint(cp,api.task_identity(fingerprints[mid],mid,row['series_id'],origin))
                assert d['simulations']==1000 and d['all_paths_scored_at_every_horizon']
                assert hashlib.sha256(losses.tobytes()).hexdigest()==indexed['loss_sha256']
                n=origin['position']+1;end=n+origin['max_horizon']
                value=independent_origin(losses,values[:n],origin['origin_date'],dates[n:end])
                delta=abs(value-indexed['main_score']);assert delta<=2e-12*max(1.,abs(value))
                s=summaries[mid];s['max_independent_origin_difference']=max(delta,s['max_independent_origin_difference'])
                s['records'].append((row['series_id'],origin['origin_date'],value));s['sum_forecast_score_seconds']+=d['forecast_and_score_seconds']
                s['checkpoint_commitment'].update(f'{i}/{j}:{digest}\n'.encode());s['verified']+=1
        if (i+1)%100==0:print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'series_audited':i+1,'checkpoints':sum(s['verified'] for s in summaries.values())}),flush=True)
    cache.close()
    for mid,s in summaries.items():
        reduced=independent_reduce(s.pop('records'),metadata)
        assert s['verified']==64049 and reduced['portfolio_cap']<=.5
        assert abs(reduced['main_score']-r['models'][mid]['main_score_on_own_coverage'])<2e-12
        s.update(reduced);s['checkpoint_commitment']=s['checkpoint_commitment'].hexdigest()
    public=[mid for mid in enabled if mid.startswith('model_testing_joint_')];assert len(public)==3
    worst=max(public,key=lambda m:(summaries[m]['main_score'],m));assert worst==r['reference_model_id']
    for mid,s in summaries.items():s['percent_better_vs_worst_Public']=100*(1-s['main_score']/summaries[worst]['main_score'])
    old_id='model_testing_equal_bootstrap_FHS_smooth_mean_shrinkage_Gaussian_uncertainty_v1'
    old_root=api.model_directory(V/'results',old_id)
    old_errors=list((old_root/'errors').rglob('*.json'));assert len(old_errors)==1
    assert sha(old_errors[0])==read(O/'v5_stable_failed_origin_replay.json')['source_error_sha256']
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_retained_suite_audit':True,'panel_series':1264,'tasks_per_model':64049,'total_checkpoints_verified':256196,'simulations':1000,'native_failures':0,'scoring_integrity_errors':0,'independent_origin_normalization_and_hierarchical_reduction':True,'reference_model_id':worst,'models':summaries,'historical_inactive_generation_failure_not_discarded':1,'timing_scope':'Sum of recorded fit, simulation and exact CRPS seconds; excludes checkpoint I/O. Not a controlled relative speed benchmark.','no_goal_completion_or_new_winner_claim':True}
    api.original._atomic_json(O/'v5_full_suite_independent_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='models'}),flush=True)
if __name__=='__main__':verify()
