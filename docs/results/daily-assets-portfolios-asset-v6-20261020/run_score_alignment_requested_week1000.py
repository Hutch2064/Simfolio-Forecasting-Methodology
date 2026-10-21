"""Actual calendar-week requested-horizon experiment; prepare only unless --run.

Same12 frozen contexts, all36 freshly short-trained configurations, fresh WR/WOR
baselines. Answers a horizon question; does not alter benchmark or promote models.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_score_alignment/requested_week1000';P=D.parent/'smoke1000'
sys.path[:0]=[str(V),str(O)]
import run_score_alignment_development as io
from candidates.asset_selector_score_alignment import candidate as core
import asset_models,score_main
from inputs import AssetInputs
api=io.api;read=io.read;write=io.write;sha=io.sha;digest=io.digest;now=io.now


def requested_week(dates, origin):
    """Observed native targets after origin and <=origin+7calendar days, inclusive."""
    dates=np.asarray(dates,dtype='datetime64[D]');stop=origin['position']+1
    day=np.datetime64(origin['origin_date'],'D');boundary=day+np.timedelta64(7,'D')
    if stop<2 or stop>len(dates) or dates[stop-1]!=day or np.isnat(dates).any() or np.any(np.diff(dates)<=np.timedelta64(0,'D')):
        raise ValueError('Dated catalogue origin does not bind to native history')
    end=int(np.searchsorted(dates,boundary,side='right'));h=end-stop
    if h<1 or h>origin['max_horizon']:raise ValueError('No complete requested-week targets inside catalogue support')
    return {'requested_horizon':h,'calendar_week_boundary':str(boundary),
            'target_first_date':str(dates[stop]),'target_last_date':str(dates[end-1]),
            'target_dates_sha256':hashlib.sha256(dates[stop:end].tobytes()).hexdigest()}


def prepare_plan():
    if D.exists():raise ValueError('Never overwrite an immutable prepared stage')
    parent=read(P/'plan.json');audit=read(P/'verification_receipt.json')
    assert audit['verified'] and audit['summary_sha256']==sha(P/'summary.json') and read(P/'summary.json')['qualification_valid']
    rows={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
    cases=[]
    for case in parent['cases']:
        dates,returns,origins=api.load_series(V/'inputs',rows[case['series_id']],verify_files=True)
        assert case['origin'] in origins
        cases.append({**case,**requested_week(dates,case['origin'])})
    assert len(cases)==12
    configs=core.configurations(2)
    assert len(configs)==36 and configs[12]['criterion']=='uniform_days' and configs[12]['scale']=='raw' and configs[12]['learner']=='batch'
    closure={str((O/path).resolve()):h for path,h in parent['source_hashes'].items()}
    for path in [Path(__file__),O/'test_score_alignment_requested_week_binding.py',Path(io.__file__),V/'asset_models.py',V/'registry.json']:
        closure[str(path.resolve())]=sha(path)
    assert all(sha(path)==h for path,h in closure.items())
    panel=api.load_panel(V/'inputs',verify_files=False)
    input_files={str((V/'inputs'/relative).resolve()):h for row in panel['series'] for relative,h in row['files'].items()}
    plan={'schema':'actual-requested-calendar-week-v1','utc':now(),'cases':cases,'configs':configs,
          'outer_paths':1000,'inner_paths':2,'seed_replication':0,'expected_forecasts':456,
          'expected_candidate_forecasts':432,'expected_baseline_forecasts':24,'worker_limit':12,
          'fixed_reference':'with_replacement','context_reference':'without_replacement',
          'source_hashes':closure,'input_files':input_files,'panel_fingerprint':panel['panel_fingerprint'],
          'input_panel_sha256':sha(V/'inputs/panel.json'),'parent_plan_sha256':sha(P/'plan.json'),
          'case_selection':'Exact12qualifiedsmoke contexts;6asset classes and6portfolio families; original catalogue origins retained',
          'horizon_contract':'Actual requested H counts native targets after original origin through origin+7calendar days inclusive; freshly learn with this H',
          'external_score':'raw losses.mean() across requested native days, unchanged hierarchy aggregation',
          'focus_config_index':12,'official':False,'full_panel_catalogue_increment':0,
          'no_promotion':True,'benchmark_or_goal_scoring_change':False,'phase':'prepared_not_launched',
          'storage_expected_growth_gib':.09,'storage_budget_gib':16,'cache_expiry':'one request RAM only; no persisted coefficient cache'}
    plan['design_digest']=digest({k:v for k,v in plan.items()if k not in ('utc','phase')})
    D.mkdir();write(D/'plan.json',plan)
    print(json.dumps({'prepared':True,'requests':12,'forecasts':456,'actual_native_horizons':[c['requested_horizon']for c in cases],'workers_started':0}),flush=True)


def initialize():
    global PLAN,ROWS
    PLAN=read(D/'plan.json');io.verify_sources(PLAN)
    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
    # Set absolute nice15 without adding15 to an already nice15 launcher.
    os.setpriority(os.PRIO_PROCESS,0,max(15,os.getpriority(os.PRIO_PROCESS,0)))


def identity(plan,number,index):
    case=plan['cases'][number];original=case['origin'];h=case['requested_horizon']
    cfg=None if index<0 else plan['configs'][index]
    semantic=asset_models.build_replacement().model_id if index==-1 else asset_models.build_without_replacement().model_id if index==-2 else core.identity(cfg)
    return {'execution_fingerprint':plan['design_digest'],'model_id':semantic,'case':number,'config_index':index,
            'configuration':cfg,'series_id':case['series_id'],'catalogue_origin':original,
            'origin':{**original,'max_horizon':h},'requested_horizon':h,
            'calendar_week_boundary':case['calendar_week_boundary'],'target_dates_sha256':case['target_dates_sha256'],
            'outer_paths':1000,'inner_paths':2,'seed_replication':0,
            'source_closure_digest':digest(plan['source_hashes']),
            'seed':api.original.coherent_daily_seed(original['origin_date'],h,1000)}


def load_context(plan,number):
    case=plan['cases'][number];origin=case['origin'];row=ROWS[case['series_id']];stop=origin['position']+1
    ids=[row['series_id']] if row['panel_branch']=='assets' else [r[0]for r in row['rational_weights']]
    for r in ROWS.values():
        if r['series_id'] in set(ids+[row['series_id']]):
            for relative in r['files']:
                path=str((V/'inputs'/relative).resolve())
                if sha(path)!=plan['input_files'][path]:raise io.ResumeMismatch('Frozen input differs')
    dates,values,origins=api.load_series(V/'inputs',row,verify_files=False)
    if origin not in origins or any(case[k]!=v for k,v in requested_week(dates,origin).items()):
        raise io.ResumeMismatch('Original catalogue origin or requested-week binding differs')
    h=case['requested_horizon'];x,w,constituents=AssetInputs(V/'inputs').training(row,stop)
    return {'history':x,'weights':w,'dates':dates[:stop],'target_dates':dates[stop:stop+h],
            'actual':np.cumsum(values[stop:stop+h]),'target_history':values[:stop],'constituents':list(constituents)}


def prepare_short(context, h):
    return core.prepare(context['history'],context['weights'],h,context['dates'],inner_paths=2,
                        schedule_rule='half_doubling',seed_replication=0)


def forecast_short(kind, config, context, h, seed, bundle=None):
    model=(asset_models.build_replacement() if kind=='WR' else asset_models.build_without_replacement()
           if kind=='WOR' else core.ScorePool(config,context['dates'],bundle))
    loss=io.forecast_loss(model,context,h,1000,seed)
    return loss,model


def evaluate(i):
    records=[];errors=[];pending={};case=PLAN['cases'][i];h=case['requested_horizon']
    for j in [-1,-2,*range(36)]:
        binding=identity(PLAN,i,j);record,error=io.existing(D,binding)
        if record:records.append(record)
        elif error:errors.append(error)
        else:pending[j]=binding
    if not pending:return {'case':i,'records':records,'errors':errors}
    context=load_context(PLAN,i);bundle=None;prepare_error=None
    for j,binding in pending.items():
        phase='native';started=time.perf_counter();cpu=time.process_time()
        write(D/'workers'/f'{os.getpid()}.json',{'case':i,'configuration':j,'phase':'baseline'if j<0 else 'short_horizon_forecast','requested_horizon':h,'utc':now()})
        try:
            cfg=None if j<0 else PLAN['configs'][j]
            if j>=0 and bundle is None:
                if prepare_error is not None:raise prepare_error
                try:bundle=prepare_short(context,h)
                except Exception as error:prepare_error=error;raise
            loss,model=forecast_short('WR'if j==-1 else 'WOR'if j==-2 else 'candidate',cfg,context,h,binding['seed'],bundle)
            phase='scoring';score=float(loss.mean())
            if not np.isfinite(score):raise io.ScoringIntegrityError('Nonfinite equal-day requested raw CRPS')
            record={**binding,'score':score,'external_requested_raw_equal_day_score':score,
                    'forecast_and_score_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
                    'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),'target_first_date':case['target_first_date'],
                    'target_last_date':case['target_last_date'],'constituent_ids':context['constituents'],
                    'validation_trace':None if j<0 else bundle['trace'],
                    'coefficients_sha256':None if j<0 else digest(io.coefficient_summary(bundle)),
                    'selection':None if j<0 else model.last_selection,
                    'cache_expiry':'current request only; no persisted coefficient cube','official':False}
            phase='infrastructure';cp,rp,ep=io.paths(D,i,j)
            api.original._write_checkpoint(cp,binding,loss,record);saved={**record,'checkpoint_sha256':sha(cp)}
            write(rp,saved);ep.unlink(missing_ok=True);records.append(saved)
        except io.ResumeMismatch:raise
        except Exception as error:errors.append(io.failure(D,binding,'scoring'if isinstance(error,io.ScoringIntegrityError)else phase,error))
    bundle=None;result={'case':i,'records':records,'errors':errors}
    write(D/'cases'/f'{i:03d}.json',result);write(D/'workers'/f'{os.getpid()}.json',{'phase':'idle','utc':now()})
    return result


def run():
    plan=read(D/'plan.json');io.verify_sources(plan)
    if sha(V/'inputs/panel.json')!=plan['input_panel_sha256']:raise io.ResumeMismatch('Panel manifest differs')
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.09','--artifact-root',str(O),'--budget-gib','16'],check=True)
    with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for folder in ('losses','records','errors','workers','cases'):(D/folder).mkdir(exist_ok=True)
        initialize();write(D/'active.json',{'status':'running','pid':os.getpid(),'workers':12,'utc':now()})
        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)as pool:
            pending={pool.submit(evaluate,i):i for i in range(12)}
            while pending:
                done,_=concurrent.futures.wait(pending,timeout=3,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    number=pending.pop(future)
                    try:results.append(future.result())
                    except Exception as error:results.append({'case':number,'records':[],'errors':[{'stage':'infrastructure','type':type(error).__name__,'message':str(error)}]})
                errors=[e for r in results for e in r['errors']]
                write(D/'progress.json',{'utc':now(),'completed_forecasts':len(list((D/'records').glob('*.json'))),
                    'expected_forecasts':456,'completed_requests':len(results),'errors':errors,
                    'native_failures':sum(e['stage']=='native'for e in errors),'scoring_integrity_errors':sum(e['stage']=='scoring'for e in errors),
                    'infrastructure_errors':sum(e['stage']=='infrastructure'for e in errors),'ETA':None,'requested_horizon_scope':'actual<=7calendar day H; freshly trained short request'})
        io.verify_sources(plan);records=[r for result in results for r in result['records']];errors=[e for result in results for e in result['errors']]
        comparisons=[];baseline_scores={}
        for j in [-1,-2]:
            chosen=[r for r in records if r['config_index']==j]
            baseline_scores[str(j)]=score_main.aggregate_records(chosen,ROWS)['score']if len(chosen)==12 else None
        for j,cfg in enumerate(plan['configs']):
            chosen=[r for r in records if r['config_index']==j]
            comparisons.append({'config_index':j,'configuration':cfg,'requests':len(chosen),
                'requested_raw_equal_day_hierarchy_score':score_main.aggregate_records(chosen,ROWS)['score']if len(chosen)==12 else None})
        valid=len(records)==456 and not errors
        write(D/'summary.json',{'utc':now(),'qualification_valid':valid,'records':records,'errors':errors,
              'models':comparisons,'baseline_hierarchy_scores':baseline_scores,'focus_config_index':12,
              'actual_requested_native_horizons':[c['requested_horizon']for c in plan['cases']],
              'external_score':'loss.mean() per short request; unchanged hierarchy','expected_forecasts':456,
              'outer_paths':1000,'official':False,'full_panel_catalogue_increment':0,'benchmark_or_goal_changed':False})
        write(D/'active.json',{'status':'terminal','qualification_valid':valid,'utc':now()})

if __name__=='__main__':run()if '--run'in sys.argv else prepare_plan()
