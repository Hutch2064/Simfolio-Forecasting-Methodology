"""Frozen single-score native candidate pilot against immutable full v4 references."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
from types import SimpleNamespace
import argparse,datetime,hashlib,json,sqlite3,sys
import candidate
import numpy as np
import sweep as api
import scoring as identity
OWN=Path(__file__).resolve().parent
V4=OWN.parent/'portfolio_expansion_v4'
sys.path.insert(0,str(V4))
import score_main
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,d:api.original._atomic_json(p,d)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()


def guard():
    plan=read(OWN/'plan.json')
    assert sha(OWN/'plan.json')==read(OWN/'plan_receipt.json')['plan_sha256']
    assert all(sha(OWN/p)==d for p,d in plan['source_hashes'].items())
    assert sha(OWN/'DESIGN.md')==plan['design_sha256']
    assert api.original.stable_digest(api.original._source_closure())==plan['native_source_closure_digest']
    receipt=read(OWN/'runner_receipt.json')
    assert receipt['plan_sha256']==sha(OWN/'plan.json') and receipt['runner_sha256']==sha(Path(__file__))
    assert score_main.SCORE_PROTOCOL_ID==plan['score_protocol_id']
    return plan


def run(phase):
    plan=guard()
    assert (V4/'controller_terminal_receipt.json').exists()
    assert read(V4/'reporter_active.json')['status']=='terminal'
    report=read(V4/'results/scoreboard.json')
    assert not report['scoring_integrity_errors']
    assert report['panel_fingerprint']==plan['panel_fingerprint']
    assert report['complete_coverage_leaderboard'][0]['model_id']==plan['initial_parent_model_id'], 'Pilot must target actual current v4 winner'
    if phase=='reserved':
        prior=read(OWN/'development_summary.json')
        assert prior['complete'] and prior['candidate_percent_better_vs_incumbent']>0
    panel=api.load_panel(V4/'inputs');meta={r['series_id']:r for r in panel['series']}
    cases=plan['phases'][phase]
    model=candidate.build_candidate()
    reference_id=report['reference_model_id']
    ids={'Incumbent':plan['initial_parent_model_id'],'Public':reference_id}
    reference={}
    with sqlite3.connect('file:'+str(V4/'results/native-loss-index.sqlite3')+'?mode=ro',uri=True) as c:
        for arm,mid in ids.items():
            reference[arm]={r[0]:{'series_id':r[1],'origin_date':r[2],'score':r[3],'loss_sha256':r[4]}
                for r in c.execute("SELECT task_key,series_id,origin_date,main_score,loss_sha256 FROM task_scores_main WHERE model_id=? AND status='ok'",(mid,))}
    for case in cases:
        key=identity._task_key(case['series_id'],case['origin'])
        assert all(key in rows for rows in reference.values()), 'Incomplete matched reference'
    execution={'plan_sha256':sha(OWN/'plan.json'),'runner_receipt_sha256':sha(OWN/'runner_receipt.json'),
        'phase':phase,'panel_fingerprint':panel['panel_fingerprint'],'model_id':model.model_id,
        'factory':'persistent_mean_v4.candidate:build_candidate','runtime':api.original.dependency_identity(model),
        'score_protocol_id':score_main.SCORE_PROTOCOL_ID,'main_score_source_sha256':sha(Path(score_main.__file__)),
        'reference_scoreboard_sha256':sha(V4/'results/scoreboard.json'),
        'reference_execution_identities':{mid:read(api.model_directory(V4/'results',mid)/'execution.json')['fingerprint'] for mid in ids.values()}}
    root=OWN/'results'/phase
    fingerprint=api.original._manifest(root,execution)
    write(OWN/'active_run.json',{'utc':now(),'pid':os.getpid(),'phase':phase,'expected_candidate_forecasts':len(cases),'status':'running'})
    records={arm:[] for arm in ['Candidate','Incumbent','Public']}
    loaded={}; native=0;errors=[]
    for i,case in enumerate(cases):
        sid=case['series_id'];origin=case['origin'];row=meta[sid]
        if sid not in loaded:
            loaded={sid:api.load_series(V4/'inputs',row)}
        dates,values,_=loaded[sid]
        key=identity._task_key(sid,origin)
        task=api.task_identity(fingerprint,model.model_id,sid,origin)
        checkpoint=root/sid/(origin['origin_label']+'.npz')
        try:
            if checkpoint.exists():
                losses,diagnostics=api.original._read_checkpoint(checkpoint,task)
            else:
                losses,diagnostics=api.original._forecast(model,sid,values,dates,SimpleNamespace(**origin),task['seed'])
                diagnostics={**diagnostics,**model.last_metadata}
                api.original._write_checkpoint(checkpoint,task,losses,diagnostics)
        except Exception as exc:
            native+=1
            error={'task_key':key,'kind':type(exc).__name__,'message':str(exc)}
            errors.append(error);write(root/sid/(origin['origin_label']+'.error.json'),error)
            continue
        assert np.isfinite(losses).all() and np.all(losses>=0)
        assert not diagnostics.get('daily_clips',0) and not diagnostics.get('wealth_floor_events',0)
        stop=origin['position']+1
        q=score_main.origin_score(losses,values[:stop],origin['origin_date'],dates[stop:stop+origin['max_horizon']])
        records['Candidate'].append({'series_id':sid,'origin_date':origin['origin_date'],'score':q,
            'origin_label':origin['origin_label'],'loss_sha256':hashlib.sha256(losses.tobytes()).hexdigest(),
            'forecast_seconds':diagnostics['forecast_and_score_seconds'],'task_key':key})
        for arm in reference:
            records[arm].append({**reference[arm][key],'origin_label':origin['origin_label'],'task_key':key})
        if i%25==0 or i+1==len(cases):
            write(OWN/'status.json',{'utc':now(),'phase':phase,'completed_candidate_forecasts':len(records['Candidate']),
                'expected_candidate_forecasts':len(cases),'native_failures':native,'scoring_integrity_errors':0,'workers':1})
    complete=native==0 and len(records['Candidate'])==len(cases)
    scores={a:score_main.aggregate_records(rows,meta) for a,rows in records.items()}
    public=scores['Public']['score'];inc=scores['Incumbent']['score'];q=scores['Candidate']['score']
    result={'utc':now(),'phase':phase,'complete':complete,'expected_each_arm':len(cases),'candidate_successes':len(records['Candidate']),
        'native_failures':native,'scoring_integrity_errors':0,'errors':errors,'records':records,
        'scores':scores,'candidate_percent_better_vs_incumbent':100*(1-q/inc),
        'percent_better_vs_worst_Public':{a:100*(1-s['score']/public) for a,s in scores.items()},
        'reference_model_id':reference_id,'main_score_only':True,'diagnostics_in_score':False,
        'adaptive_benchmark_reuse':True,'independent_validation_or_significance_claim':False,
        'forecast_and_score_worker_seconds':sum(r['forecast_seconds'] for r in records['Candidate'])}
    write(OWN/(phase+'_summary.json'),result)
    guard()
    write(OWN/'active_run.json',{'utc':now(),'pid':os.getpid(),'phase':phase,'status':'complete'})
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','scores','errors']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['development','reserved']);args=parser.parse_args();run(args.phase)
