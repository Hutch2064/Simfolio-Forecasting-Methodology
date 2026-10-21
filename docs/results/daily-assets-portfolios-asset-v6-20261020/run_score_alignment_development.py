"""Prepare/resume the finite72-request exploratory100-path score-design screen.

Default is plan preparation only. --run requires the root's separate review plus
successful .3GiB storage preflight; no goal/catalogue/production files are touched.
"""
import os
THREAD_KEYS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREAD_KEYS: os.environ[name] = '1'
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
O = Path(__file__).resolve().parent; V = O/'asset_level_v6'
D = V/'candidates/asset_selector_score_alignment/development_screen100'
sys.path[:0] = [str(V),str(O)]
import common
import sweep_v6 as api
import score_main
import scoring
import asset_models
from inputs import AssetInputs
from candidates.asset_selector_score_alignment import candidate as core
from candidates.asset_selector_endpoint_flat import candidate as flat
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon

read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
digest = api.original.stable_digest
write = api.original._atomic_json
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
BASELINE_ID = 'model_testing_asset_joint_blocks_replacement_v6'
PARENT = V/'candidates/asset_selector_endpoint_components/study1000/plan.json'


class ResumeMismatch(RuntimeError): pass


class ScoringIntegrityError(ArithmeticError): pass


def configurations():
    return [dict(c,seed_replication=0) for n in (2,8) for c in core.configurations(n)]


def verify_sources(plan):
    for path, expected in plan['source_hashes'].items():
        if sha(path) != expected: raise ResumeMismatch(f'Frozen source differs: {path}')


def make_plan(root=D):
    root = Path(root); parent = read(PARENT)
    cases = parent['cases']; configs = configurations()
    if len(cases) != 72 or len(configs) != 72: raise ValueError('Unexpected development design')
    source_hashes = {}
    for ancestor in [PARENT,V/'candidates/asset_selector_endpoint_flat/broader1000/plan.json']:
        for relative, expected in read(ancestor)['source_hashes'].items():
            path = (O/relative).resolve()
            if path.suffix == '.py':
                if sha(path) != expected: raise ResumeMismatch(f'Frozen ancestor source differs: {path}')
                source_hashes[str(path)] = expected
    registry = read(V/'registry.json'); baseline = registry['models'][BASELINE_ID]
    for relative, expected in baseline['identity']['source_files'].items():
        path = (Path(registry['source_root'])/relative).resolve()
        if sha(path) != expected: raise ResumeMismatch(f'Baseline source differs: {path}')
        source_hashes[str(path)] = expected
    package_root = Path(api.original.__file__).resolve().parents[1]
    for relative, expected in api.original._source_closure().items():
        source_hashes[str(package_root/relative)] = expected
    extras = [Path(__file__), O/'test_score_alignment_development_worker.py', Path(scoring.__file__),
              V/'registry.json', V/'profile.json', PARENT]
    for folder in ('asset_selector_score_alignment','asset_selector_sequential_learning','asset_selector_fold_scale'):
        extras += list((V/'candidates'/folder).glob('*.py'))
    for path in extras: source_hashes[str(path.resolve())] = sha(path)
    panel = api.load_panel(V/'inputs',verify_files=False)
    rows = {r['series_id']:r for r in panel['series']}
    families = {rows[c['series_id']]['family'] for c in cases}
    if len(families) != 70: raise ValueError('Expected70 development families')
    input_files = {str((V/'inputs'/relative).resolve()): expected
                   for row in panel['series'] for relative,expected in row['files'].items()}
    work = []
    for case in cases:
        stop = case['origin']['position']+1; h = case['origin']['max_horizon']
        _, folds, _ = flat.schedule(stop,h,'half_doubling')
        work.append({'prepare': 167**2*10*sum(f[1] for f in folds), 'forecast':73*100*h})
    plan = {'schema':'score-alignment-development-screen100-v1','utc':now(),
        'phase':'prepared_not_launched','cases':cases,'configs':configs,
        'case_order':parent['case_order'],'expected_cases':72,'families':70,
        'expected_candidate_forecasts':5184,'expected_baseline_forecasts':72,
        'expected_forecasts':5256,'outer_paths':100,'internal_paths':[2,8],
        'predeclared_seeds':[0],'shared_worker_limit':12,'source_hashes':source_hashes,
        'input_files':input_files,'input_panel_sha256':sha(V/'inputs/panel.json'),
        'panel_fingerprint':panel['panel_fingerprint'],'baseline':baseline,
        'baseline_model_id':BASELINE_ID,'baseline_factory':'asset_models:build_replacement',
        'baseline_paths_semantics':'Exact joint replacement law;100-path exploratory draw overrides registry official1000 execution count only',
        'external_score_protocol':score_main.SCORE_PROTOCOL_ID,
        'final_seed_contract':'coherent_daily_seed(origin_date,max_horizon,100), shared across variants and baseline',
        'internal_seed_replication':0,'expected_growth_gib':.3,'artifact_budget_gib':16,
        'cache_policy':'One inner-count bundle in request RAM; expire before next inner count; no persisted fold caches',
        'work_units':work,'official':False,'full_panel_catalogue_increment':0,
        'no_promotion_from_screen':True,'case_selection':'Exact reused72development requests covering70families; dependent exploratory seed0 design',
        'comparisons':'100to100 only; never compare against cached1000path baseline',
        'root_review_required_before_run':True,'scientific_workers_started':0}
    plan['design_digest'] = digest({k:v for k,v in plan.items() if k not in ('utc','phase')})
    root.mkdir(parents=True,exist_ok=True)
    path = root/'plan.json'
    if path.exists():
        old = read(path)
        if old['design_digest'] != plan['design_digest']: raise ResumeMismatch('Existing immutable plan differs')
        return old
    write(path,plan); return plan


def load_context(plan, number):
    case = plan['cases'][number]; origin = case['origin']; stop = origin['position']+1
    panel = api.load_panel(V/'inputs',verify_files=False)
    row = next(r for r in panel['series'] if r['series_id']==case['series_id'])
    # Verify both target and constituent files before any science.
    ids = [row['series_id']] if row['panel_branch']=='assets' else [r[0] for r in row['rational_weights']]
    selected = [r for r in panel['series'] if r['series_id'] in set(ids+[row['series_id']])]
    for r in selected:
        for relative in r['files']:
            path = str((V/'inputs'/relative).resolve())
            if sha(path) != plan['input_files'][path]: raise ResumeMismatch(f'Input differs: {path}')
    dates,values,origins = api.load_series(V/'inputs',row,verify_files=False)
    if origin not in origins: raise ResumeMismatch('Request origin differs from frozen input schedule')
    history,weights,constituents = AssetInputs(V/'inputs').training(row,stop)
    return {'history':history,'weights':weights,'dates':dates[:stop],
            'target_dates':dates[stop:stop+origin['max_horizon']],
            'target_history':values[:stop],'actual':np.cumsum(values[stop:stop+origin['max_horizon']]),
            'constituents':list(constituents)}


def task_identity(plan,number,index):
    case = plan['cases'][number]; cfg = None if index<0 else plan['configs'][index]
    semantic_id = BASELINE_ID if index<0 else core.identity({k:v for k,v in cfg.items() if k!='seed_replication'})
    return {'execution_fingerprint':plan['design_digest'],'model_id':semantic_id,
            'screen_id':'exploratory100:'+semantic_id,'series_id':case['series_id'],
            'origin':case['origin'],'case':number,'config_index':index,'configuration':cfg,
            'outer_paths':plan['outer_paths'],'internal_seed_replication':0,
            'seed':api.original.coherent_daily_seed(case['origin']['origin_date'],case['origin']['max_horizon'],plan['outer_paths']),
            'source_closure_digest':digest(plan['source_hashes']),'input_panel_fingerprint':plan['panel_fingerprint']}


def paths(root,number,index):
    stem = f'{index+1:03d}-{number:03d}'
    return Path(root)/'losses'/f'{stem}.npz',Path(root)/'records'/f'{stem}.json',Path(root)/'errors'/f'{stem}.json'


def existing(root,identity):
    checkpoint,record_path,error_path = paths(root,identity['case'],identity['config_index'])
    if checkpoint.exists():
        try:
            loss,record = api.original._read_checkpoint(checkpoint,identity)
            expected = {**record,'checkpoint_sha256':sha(checkpoint)}
            if record_path.exists() and read(record_path)!=expected: raise ResumeMismatch('Checkpoint/record metadata differs')
            if not record_path.exists(): write(record_path,expected)
            return expected,None
        except Exception as error: raise ResumeMismatch(f'Invalid existing checkpoint: {checkpoint}') from error
    if record_path.exists(): raise ResumeMismatch('Record exists without atomic loss checkpoint')
    if error_path.exists():
        error = read(error_path); checksum = error.pop('checksum')
        if digest(error)!=checksum or error['identity']!=identity: raise ResumeMismatch('Failure checkpoint metadata differs')
        return None,{**error,'checksum':checksum}
    return None,None


def failure(root,identity,stage,error):
    payload = {'identity':identity,'stage':stage,'type':type(error).__name__,'message':str(error),'utc':now()}
    payload['checksum'] = digest(payload); write(paths(root,identity['case'],identity['config_index'])[2],payload)
    return payload


def coefficient_summary(bundle):
    return {'arrays':{k:{'shape':list(a.shape),'sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
                      for k,a in bundle['fold_coefficients'].items()},
            'method':'float64 exact per-lead empirical CRPS;12criterion-scale streams;batch/BOA/progressive',
            'trace_sha256':digest(bundle['trace']),'cache_bytes':bundle['diagnostics']['cache_bytes']}


def forecast_loss(model,context,horizon,paths_count,seed):
    loss = np.empty(horizon); cursor = 0
    for start,values in model.terminal_chunks(context['history'],context['weights'],horizon,paths_count,seed):
        if start!=cursor or values.shape[0]!=paths_count or values.ndim!=2 or not np.isfinite(values).all():
            raise ArithmeticError('Incomplete/nonfinite native joint forecast')
        end = start+values.shape[1]
        if end>horizon: raise ArithmeticError('Forecast exceeds requested native horizon')
        try: loss[start:end] = empirical_crps_by_horizon(values,context['actual'][start:end])
        except Exception as error: raise ScoringIntegrityError('Native-law CRPS evaluation failed') from error
        cursor=end
    if cursor!=horizon: raise ArithmeticError('Incomplete native forecast coverage')
    if not np.isfinite(loss).all() or np.any(loss<0): raise ScoringIntegrityError('Invalid raw loss vector')
    return loss


def verify_flat_control(context,bundle,inner_paths,seed,outer_paths=100):
    h = bundle['horizon']; b = flat.prepare(context['history'],context['weights'],h,context['dates'],
        inner_paths=inner_paths,schedule_rule='half_doubling')
    for axis in ('A','B'):
        if not np.array_equal(b['coefficients']['flat'][axis],bundle['coefficients'][('endpoint','sqrt_lead')][axis]):
            raise ArithmeticError('Direct flat100 control coefficient mismatch')
    if not np.array_equal(b['weights_by_mode']['flat'],bundle['weights_by_mode']['endpoint:sqrt_lead:batch']):
        raise ArithmeticError('Direct flat100 control weights mismatch')
    cfg = {'criterion':'endpoint','scale':'sqrt_lead','learner':'batch','inner_paths':inner_paths,'schedule':'half_doubling'}
    left = forecast_loss(flat.MenuPool({'menu':'flat','inner_paths':inner_paths,'schedule':'half_doubling'},context['dates'],b),context,h,outer_paths,seed)
    right = forecast_loss(core.ScorePool(cfg,context['dates'],bundle),context,h,outer_paths,seed)
    if not np.array_equal(left,right): raise ArithmeticError('Direct flat100 control loss mismatch')
    return {'verified':True,'paths':outer_paths,'loss_sha256':hashlib.sha256(left.tobytes()).hexdigest()}


def evaluate_case(plan,root,number,context=None):
    records=[]; errors=[]; pending={}
    # Validate/resume every existing task before creating any new result.
    for index in [-1,*range(len(plan['configs']))]:
        identity=task_identity(plan,number,index); record,error=existing(root,identity)
        if record: records.append(record)
        elif error: errors.append(error)
        else: pending[index]=identity
    if not pending: return {'case':number,'records':records,'errors':errors}
    try: ctx=load_context(plan,number) if context is None else context
    except ResumeMismatch: raise
    except Exception as error:
        errors.extend(failure(root,identity,'infrastructure',error) for identity in pending.values())
        return {'case':number,'records':records,'errors':errors}
    h=plan['cases'][number]['origin']['max_horizon']; bundle=None; bundle_n=None; prep_failed={}
    for index,identity in pending.items():
        started=time.perf_counter(); cpu=time.process_time(); phase='native'; prepare_cpu=0.; prep_info=None
        write(Path(root)/'workers'/f'{os.getpid()}.json',{'case':number,'config_index':index,'phase':'prepare' if index>=0 else 'baseline100','utc':now()})
        try:
            if index<0: model=asset_models.build_replacement()
            else:
                cfg=plan['configs'][index]; n=cfg['inner_paths']
                if n in prep_failed: raise prep_failed[n]
                if bundle_n!=n:
                    bundle=None  # request cache expires before the next internal count
                    begin=time.process_time()
                    try:
                        bundle=core.prepare(ctx['history'],ctx['weights'],h,ctx['dates'],inner_paths=n,
                            schedule_rule=cfg['schedule'],seed_replication=0)
                    except Exception as error: prep_failed[n]=error; raise
                    prepare_cpu=time.process_time()-begin; bundle_n=n; prep_info=coefficient_summary(bundle)
                    write(Path(root)/'preparations'/f'{n:02d}-{number:03d}.json',
                          {'case':number,'inner_paths':n,'cpu_seconds':prepare_cpu,'summary':prep_info,
                           'work_units':len(bundle['fold_coefficients']['valid'])**2*n*sum(t['validation_horizon'] for t in bundle['trace']),
                           'source_closure_digest':identity['source_closure_digest']})
                model=core.ScorePool({k:v for k,v in cfg.items() if k!='seed_replication'},ctx['dates'],bundle)
            forecast_start=time.process_time(); loss=forecast_loss(model,ctx,h,plan['outer_paths'],identity['seed'])
            forecast_cpu=time.process_time()-forecast_start; phase='scoring'
            score=score_main.origin_score(loss,ctx['target_history'],identity['origin']['origin_date'],ctx['target_dates'])
            if not np.isfinite(score): raise ArithmeticError('Nonfinite unchanged canonical score')
            record={**identity,'score':float(score),'forecast_and_score_seconds':time.perf_counter()-started,
                    'cpu_seconds':time.process_time()-cpu,'forecast_cpu_seconds':forecast_cpu,
                    'prepare_cpu_seconds':prepare_cpu,'forecast_work_units':plan['outer_paths']*h,
                    'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),
                    'constituent_ids':ctx['constituents'],'official':False,'cache_expiry':'current request/current internal count',
                    'coefficient_summary':None if index<0 else coefficient_summary(bundle),
                    'selection':None if index<0 else model.last_selection}
            phase='infrastructure'; checkpoint,record_path,error_path=paths(root,number,index)
            api.original._write_checkpoint(checkpoint,identity,loss,record)
            result={**record,'checkpoint_sha256':sha(checkpoint)}; write(record_path,result)
            error_path.unlink(missing_ok=True); records.append(result)
        except ResumeMismatch: raise
        except Exception as error:
            errors.append(failure(root,identity,'scoring' if isinstance(error,ScoringIntegrityError) else phase,error))
    bundle=None
    result={'case':number,'records':records,'errors':errors}
    write(Path(root)/'case_records'/f'{number:03d}.json',result)
    write(Path(root)/'workers'/f'{os.getpid()}.json',{'phase':'idle','utc':now()})
    return result


def progress(plan,root,results):
    records=[r for result in results for r in result['records']]
    errors=[e for result in results for e in result['errors']]
    complete={result['case'] for result in results}; remaining=[i for i in range(len(plan['cases'])) if i not in complete]
    elapsed_units=sum(r['forecast_work_units'] for r in records)
    rates=[r['forecast_cpu_seconds']/r['forecast_work_units'] for r in records if r['forecast_work_units']>0]
    prepare_rates=[]
    for path in (Path(root)/'preparations').glob('*.json'):
        prep=read(path)
        if prep['work_units']>0: prepare_rates.append(prep['cpu_seconds']/prep['work_units'])
    eta=None
    if remaining and rates and prepare_rates:
        eta=(sum(plan['work_units'][i]['forecast'] for i in remaining)*min(rates)
            +sum(plan['work_units'][i]['prepare'] for i in remaining)*min(prepare_rates))/12
    return {'utc':now(),'expected_forecasts':plan['expected_forecasts'],'completed_forecasts':len(records),
            'completed_cases':len(complete),'expected_cases':len(plan['cases']),
            'candidate_forecasts':sum(r['config_index']>=0 for r in records),
            'baseline_forecasts':sum(r['config_index']<0 for r in records),
            'native_failures':sum(e['stage']=='native' for e in errors),
            'scoring_integrity_errors':sum(e['stage']=='scoring' for e in errors),
            'infrastructure_failures':sum(e['stage']=='infrastructure' for e in errors),
            'errors':errors,'measured_forecast_cpu_seconds':sum(r['forecast_cpu_seconds'] for r in records),
            'measured_forecast_work_units':elapsed_units,'remaining_work_lower_bound_wall_seconds':eta,
            'projection_scope':'Within-stage fastest measured CPU/unit divided by12; optimistic work-rate estimate, no confidence guarantee',
            'projection_reason':'Unresolved failures remain; no zero completion ETA' if errors and not remaining else
                                'Insufficient within-stage phase calibration' if eta is None else 'Measured phase rates; pending case work only',
            'official':False,'full_panel_catalogue_increment':0,'no_promotion_from_screen':True}


def worker_init(plan_path):
    global PLAN
    PLAN=read(plan_path); verify_sources(PLAN)
    try: os.nice(15)
    except OSError: pass


def worker(number): return evaluate_case(PLAN,D,number)


def run(root=D):
    if Path(root)!=D: raise ValueError('Official prepared output root is fixed')
    plan=read(D/'plan.json'); verify_sources(plan)
    if sha(V/'inputs/panel.json')!=plan['input_panel_sha256']: raise ResumeMismatch('Input panel differs')
    # Storage failure stops before any scientific worker starts.
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py',
        '--expected-gib','.3','--artifact-root',str(O),'--budget-gib','16'],check=True)
    with (O/'asset_v6_forecast_slots.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for folder in ('losses','records','errors','workers','preparations','case_records'):(D/folder).mkdir(exist_ok=True)
        write(D/'active.json',{'status':'running','workers':12,'pid':os.getpid(),'utc':now()})
        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),
                initializer=worker_init,initargs=(str(D/'plan.json'),)) as pool:
            pending={pool.submit(worker,i):i for i in plan['case_order']}
            while pending:
                done,_=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    number=pending.pop(future)
                    try: results.append(future.result())
                    except Exception as error:
                        results.append({'case':number,'records':[],'errors':[{'stage':'infrastructure','type':type(error).__name__,'message':str(error),'case':number}]})
                write(D/'progress.json',progress(plan,D,results))
        verify_sources(plan); summary=progress(plan,D,results)
        summary['complete']=summary['completed_forecasts']==plan['expected_forecasts'] and not summary['errors']
        summary['comparisons']='Matched100-path reference only; raw loss checkpoints retained for root12-rule rescore'
        write(D/'summary.json',summary);write(D/'active.json',{'status':'terminal','utc':now()})
        return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    args=parser.parse_args(); result=run() if args.run else make_plan()
    print(json.dumps({'phase':'terminal' if args.run else 'prepared_not_launched',
        'target_forecasts':result.get('expected_forecasts'),'scientific_workers_started':12 if args.run else 0}))
