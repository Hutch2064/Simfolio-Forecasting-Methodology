"""Describe the closed mean-regime contrast; never alter its frozen score gate."""
from pathlib import Path
import datetime,hashlib,json,sqlite3,sys
import numpy as np
import candidate
import random_effects
import sweep as api
import scoring as identity
OWN=Path(__file__).resolve().parent;V4=OWN.parent/'portfolio_expansion_v4'
sys.path.insert(0,str(V4));import score_main
read=lambda p:json.loads(p.read_text())
plan=read(OWN/'plan.json');summary=read(OWN/'development_summary.json')
assert summary['complete'] and summary['candidate_percent_better_vs_incumbent']<0
panel=api.load_panel(V4/'inputs');meta={r['series_id']:r for r in panel['series']}
bands={'1-30d':(1,30),'31-365d':(31,365),'366-1826d':(366,1826),'1827-7305d':(1827,7305),'7306+d':(7306,np.inf)}
root=OWN/'results/development';fp=read(root/'execution.json')['fingerprint']
mid=plan['initial_parent_model_id'];records={arm:[] for arm in ['Candidate','Incumbent']};training=[];audited=0
with sqlite3.connect('file:'+str(V4/'results/native-loss-index.sqlite3')+'?mode=ro',uri=True) as con:
    for case in plan['phases']['development']:
        sid=case['series_id'];o=case['origin'];row=meta[sid];dates,values,_=api.load_series(V4/'inputs',row)
        key=identity._task_key(sid,o)
        reference=con.execute("SELECT checkpoint_path,loss_sha256,main_score FROM task_scores_main WHERE model_id=? AND task_key=? AND status='ok'",(mid,key)).fetchall();assert len(reference)==1
        base_path,base_sha,base_q=reference[0]
        base_root=api.model_directory(V4/'results',mid);base_fp=read(base_root/'execution.json')['fingerprint']
        candidate_loss,diag=api.original._read_checkpoint(root/sid/(o['origin_label']+'.npz'),api.task_identity(fp,candidate.MODEL_ID,sid,o))
        base_loss,_=api.original._read_checkpoint(Path(base_path),api.task_identity(base_fp,mid,sid,o))
        assert hashlib.sha256(base_loss.tobytes()).hexdigest()==base_sha
        stop=o['position']+1;history=values[:stop];future=dates[stop:stop+o['max_horizon']]
        days=(future-np.datetime64(o['origin_date'])).astype('timedelta64[D]').astype(int)
        weights=identity.calendar_log_weights(o['origin_date'],future)
        denominator=np.std(history,ddof=1)*np.sqrt(np.arange(1,len(future)+1))
        for arm,loss in [('Candidate',candidate_loss),('Incumbent',base_loss)]:
            contrib=loss/denominator*weights
            full=score_main.origin_score(loss,history,o['origin_date'],future)
            assert abs(float(contrib.sum())-full)<1e-12
            if arm=='Incumbent':assert abs(full-base_q)<1e-12
            row_values={band:float(contrib[(days>=lo)&(days<=hi)].sum()) for band,(lo,hi) in bands.items()}
            records[arm].append({'series_id':sid,'origin_date':o['origin_date'],'score':full,**row_values})
            audited+=1
        native=np.log1p(np.clip(np.expm1(history),-.999999,np.finfo(float).max/10))
        groups=np.array_split(native,int(np.sqrt(len(native))));means=np.array([g.mean() for g in groups])
        correlation=float(np.corrcoef(means[:-1],means[1:])[0,1]) if np.std(means[:-1])>0 and np.std(means[1:])>0 else None
        comps=diag['components'];ratio=[c['combined_Gaussian_mean_sd']/c['parent_Gaussian_mean_sd'] if c['parent_Gaussian_mean_sd']>0 else 1 for c in comps]
        training.append({'series_id':sid,'origin_label':o['origin_label'],'panel_branch':row['panel_branch'],
            'asset_class':row['asset_class'],'family':row['family'],'adjacent_batch_mean_correlation':correlation,
            'mean_uncertainty_SD_ratio':float(np.mean(ratio))})
aggregate={arm:score_main.aggregate_records(rows,meta)['score'] for arm,rows in records.items()}
for arm in aggregate:assert abs(aggregate[arm]-summary['scores'][arm]['score'])<1e-12
contributions={band:{arm:score_main.aggregate_records(rows,meta,score_key=band)['score'] for arm,rows in records.items()} for band in bands}
assert abs(sum(b['Candidate'] for b in contributions.values())-aggregate['Candidate'])<1e-12
assert abs(sum(b['Incumbent'] for b in contributions.values())-aggregate['Incumbent'])<1e-12
byclass={}
for group in sorted({r['asset_class'] for r in training}):
    selected=[r for r in training if r['asset_class']==group]
    corr=[r['adjacent_batch_mean_correlation'] for r in selected if r['adjacent_batch_mean_correlation'] is not None]
    byclass[group]={'origins':len(selected),'median_adjacent_batch_mean_correlation':float(np.median(corr)) if corr else None,
        'median_mean_uncertainty_SD_ratio':float(np.median([r['mean_uncertainty_SD_ratio'] for r in selected]))}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'native_loss_vectors_audited':audited,
    'main_Q_reconciled':True,'additive_horizon_contributions':contributions,'training_descriptives_by_class':byclass,
    'training_records':training,'diagnostics_have_zero_score_or_selection_influence':True,
    'adaptive_dependent_prefixes_not_independent_sample':True,
    'decision':'Reject frozen static full-horizon mean-regime variance contrast; no reserved or full evaluation',
    'wider_goal_active':True}
api.original._atomic_json(OWN/'diagnostics.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='training_records'}),flush=True)
