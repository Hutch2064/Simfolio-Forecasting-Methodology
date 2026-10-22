"""Frozen most-worthy candidate full panel, reusing saved official baselines.

Root selects endpoint or confirmed uniform_shrinkage before plan preparation.
Each selected candidate gets the unchanged64,049 requests, inner2, outer1000.
No pilot checkpoints imported. Preparing does not authorize launching.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
import concurrent.futures as futures
from collections import Counter
import datetime
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import sqlite3
import subprocess
import sys
import time
import numpy as np

O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
D = O/'mostworthy_panel_v7'
sys.path.insert(0, str(V))
import common
import sweep_v6 as api
import score_aligned_v7 as scoring
from inputs import AssetInputs
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon

AVAILABLE_MODELS = {
    'endpoint': {'module':'candidates.asset_selector_endpoint_flat.candidate',
        'class':'MenuPool','config':{'menu':'flat','inner_paths':2,'schedule':'half_doubling'}},
    'uniform_shrinkage': {'module':'candidates.asset_uniform_shrinkage_full.candidate',
        'class':'ShrinkagePool','config':{'family':'uniform_shrinkage','inner_paths':2,'schedule':'half_doubling'}}}
MODELS = {'endpoint': AVAILABLE_MODELS['endpoint']}

GATES = {'optimizations','storage','independent_reviews','baseline_rescore'}
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj,sort_keys=True,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
    os.replace(temporary,path)

def closure(models=None):
    models=MODELS if models is None else models
    # Follow every static local import (including function-local factory imports).
    # Third-party numerical packages are bound separately by environment versions.
    roots = [V, common.CODE/'src', common.CODE/'scripts', common.CAMPAIGN]
    def resolve(module):
        for root in roots:
            base = root.joinpath(*module.split('.'))
            for path in (base.with_suffix('.py'),base/'__init__.py'):
                if path.is_file(): return path.resolve()
    pending = [Path(__file__).resolve(),resolve('score_aligned_v7'),resolve('inputs'),resolve('sweep_v6')]
    pending += [resolve(spec['module']) for spec in models.values()]
    paths=set()
    while pending:
        path=pending.pop()
        if path is None or path in paths: continue
        paths.add(path)
        for root in roots:
            if root.resolve() in path.parents:
                parent=path.parent
                while root.resolve() in parent.parents or parent==root.resolve():
                    init=parent/'__init__.py'
                    if init.is_file() and init not in paths:pending.append(init)
                    if parent==root.resolve():break
                    parent=parent.parent
                break
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=node.module or ''
                if node.level:
                    for root in roots:
                        try: relative=path.relative_to(root.resolve());break
                        except ValueError: continue
                    else: raise ValueError('Relative import outside numerical closure')
                    package=list(relative.parts[:-1])
                    package=package[:len(package)-node.level+1]
                    prefix='.'.join(package+([prefix] if prefix else []))
                names=([prefix] if prefix else [])+[prefix+'.'+a.name if prefix else a.name for a in node.names]
            for name in names:
                dep=resolve(name)
                if dep and dep not in paths: pending.append(dep)
    paths.add(O/'final_scoring_protocol_v7.json')
    return {str(p):sha(p) for p in sorted(paths)}

def environment():
    return {'python':sys.version,'executable':str(Path(sys.executable).resolve()),
        'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}

def cases_for(panel):
    return [{'series_index':i,'series_id':r['series_id'],'origin_index':j,'origin':origin}
            for i,r in enumerate(panel['series']) for j,origin in enumerate(api.schedule(V/'inputs',r))]

def prepare(gates, destination=D, candidates=None, cohort_plan=None):
    selected=list(MODELS) if candidates is None else list(candidates)
    if not selected or len(set(selected))!=len(selected) or any(name not in AVAILABLE_MODELS for name in selected):raise ValueError('Select frozen endpoint or uniform_shrinkage before preparing')
    selected_models={name:AVAILABLE_MODELS[name] for name in selected}
    destination = Path(destination)
    if destination.exists(): raise ValueError('Never overwrite a prepared plan')
    if set(gates) != GATES: raise ValueError('Bind root-selected optimizations/storage/independent_reviews/baseline_rescore receipts')
    panel = api.load_panel(V/'inputs',verify_files=True)
    full_cases=cases_for(panel);cases=full_cases
    if len(panel['series']) != 1264 or len(full_cases) != 64049: raise ValueError('Frozen panel coverage differs')
    coverage = read(O/'full_requested_horizon_coverage.json')
    if coverage['requests'] != len(full_cases) or not coverage['verified']: raise ValueError('Coverage review differs')
    cohort_source=None
    if cohort_plan is not None:
        cohort_path=Path(cohort_plan).resolve();requested=read(cohort_path)['cases']
        by_key={(c['series_id'],digest(c['origin'])):c for c in full_cases}
        cases=[by_key[(c['series_id'],digest(c['origin']))] for c in requested]
        if len(cases)!=355 or len({digest(c) for c in cases})!=355:raise ValueError('Only frozen355 development cohort supported')
        cohort_source={'path':str(cohort_path),'sha256':sha(cohort_path)}
    baseline_receipt=read(gates['baseline_rescore'])
    index_path=Path(baseline_receipt['score_index_path']).resolve()
    if sha(index_path)!=baseline_receipt['score_index_sha256']:raise ValueError('Baseline score index differs')
    models = {name:{**spec,'model_id':importlib.import_module(spec['module']).identity(spec['config'])}
              for name,spec in selected_models.items()}
    plan = {'schema':'aligned-panel-v7-plan','utc':now(),'models':models,'cases':cases,
        'cases_digest':digest(cases),'full_cases_digest':digest(full_cases),'cohort_source':cohort_source,'source_hashes':closure(models),'environment':environment(),'panel_fingerprint':panel['panel_fingerprint'],
        'baseline_score_index':{'path':str(index_path),'sha256':sha(index_path)},
        'fixed_reference_model_id':baseline_receipt['reference_model_id_under_v7'],
        'panel_json_sha256':sha(V/'inputs/panel.json'),'coverage_sha256':sha(O/'full_requested_horizon_coverage.json'),
        'gate_receipts':{name:{'path':str(Path(path).resolve()),'sha256':sha(path)} for name,path in gates.items()},
        'outer_paths':1000,'inner_paths':2,'seed_replications':[0],'worker_limit':12,'nice':15,
        'expected_per_model':len(cases),'expected_tasks':len(cases)*len(models),'expected_growth_gib':3.5*len(models) if cohort_source is None else len(models)*sum(c['origin']['max_horizon']*8+41500 for c in cases)/(1024**3),'budget_gib':16,
        'growth_estimate':{'raw_loss_bytes':len(models)*sum(c['origin']['max_horizon']*8 for c in cases),'metadata_allowance_bytes':len(cases)*len(models)*40000,
            'attempt_journal_allowance_bytes':len(cases)*len(models)*1000,'total_ceiling_gib':3.5*len(models),'assumption':'Selectedmodels rawfloat64 losses plus40KB/request metadata; compressed losses no larger than conservative raw allowance'},
        'evaluation_scope':'development355' if cohort_source else 'full64049','score_protocol':scoring.SCORE_PROTOCOL_ID,'official':False,'automatic_promotion':False,
        'retention_rule':'After independent review only: both saved baseline rescoring results plus full-panel winner; never select by development score',
        'pilot_reuse':{'count':0,'reason':'Regenerate selected candidate(s); no proven full-request exact execution/source identity for development checkpoints'},
        'requested_horizon_warning':'Frozen H28..19835 has no actual day/week request; do not expand grid or claim short-request qualification',
        'cache_policy':'Request-local coefficients only; persist raw loss plus compact metadata, no full forecast paths',
        'goal_tool_mutation':False,'future_model_admissions':False}
    plan['fingerprint'] = digest(plan)
    destination.mkdir()
    write(destination/'plan.json',plan)
    return plan

def verify_plan(plan):
    if digest({k:v for k,v in plan.items() if k!='fingerprint'}) != plan['fingerprint']: raise ValueError('Plan digest differs')
    if not plan['models'] or any(name not in AVAILABLE_MODELS or {k:v for k,v in spec.items() if k!='model_id'} != AVAILABLE_MODELS[name] for name,spec in plan['models'].items()):
        raise ValueError('Model declaration differs')
    if plan['environment'] != environment(): raise ValueError('Numerical environment differs')
    if plan['source_hashes'] != closure(plan['models']): raise ValueError('Source closure differs; re-prepare in a new directory')
    if sha(V/'inputs/panel.json') != plan['panel_json_sha256']: raise ValueError('Panel manifest differs')
    panel = api.load_panel(V/'inputs',verify_files=True)
    full_cases=cases_for(panel)
    if panel['panel_fingerprint'] != plan['panel_fingerprint'] or digest(full_cases)!=plan['full_cases_digest']:raise ValueError('Full input catalogue differs')
    cohort=plan['cohort_source']
    if cohort is None:
        if full_cases!=plan['cases']:raise ValueError('Exact full catalogue differs')
    else:
        if sha(cohort['path'])!=cohort['sha256']:raise ValueError('Cohort source differs')
        by_key={(c['series_id'],digest(c['origin'])):c for c in full_cases}
        expected=[by_key[(c['series_id'],digest(c['origin']))] for c in read(cohort['path'])['cases']]
        if expected!=plan['cases'] or len(expected)!=355:raise ValueError('Exact355 cohort differs')
    if sha(O/'full_requested_horizon_coverage.json') != plan['coverage_sha256']: raise ValueError('Coverage receipt differs')
    if sha(plan['baseline_score_index']['path'])!=plan['baseline_score_index']['sha256']:raise ValueError('Baseline index changed')
    for receipt in plan['gate_receipts'].values():
        if sha(receipt['path']) != receipt['sha256']: raise ValueError('Bound gate receipt changed')
    return panel

def admit(plan, receipt):
    if (receipt.get('schema')!='aligned-panel-v7-root-admission' or receipt.get('plan_fingerprint')!=plan['fingerprint']
            or receipt.get('root_cleared_launch') is not True or receipt.get('passed') is not True
            or set(receipt.get('passed_gates',[])) != GATES
            or receipt.get('gate_receipts') != plan['gate_receipts']):
        raise ValueError('Root must explicitly clear this exact plan and all four bound gates')

def identity(plan, name, case):
    return {'execution_fingerprint':plan['fingerprint'],'model':name,'model_id':plan['models'][name]['model_id'],
            'series_id':case['series_id'],'origin':case['origin'],'outer_paths':1000,'inner_paths':2,'inner_seed_replication':0,
            'outer_seed':api.original.coherent_daily_seed(case['origin']['origin_date'],case['origin']['max_horizon'],1000)}

def paths(destination,name,index):
    folder = Path(destination)/'checkpoints'/name
    return folder/f'{index:05d}.npz', folder/f'{index:05d}.json', folder/f'{index:05d}.failure.json'

def verified_checkpoint(destination,plan,name,index):
    cp,rp,fp = paths(destination,name,index)
    if fp.exists(): raise ValueError(f'Previously failed task requires explicit review in new execution: {fp}')
    if not cp.exists() and not rp.exists():
        if cp.with_suffix('.attempt.json').exists() or cp.with_name(cp.name+'.tmp').exists():
            raise ValueError('Interrupted/uncommitted task; explicit review required before any rerun')
        return None
    if not cp.exists() or not rp.exists(): raise ValueError(f'Orphan checkpoint; no silent rerun: {cp}')
    record = read(rp); case = plan['cases'][index]
    if record.get('record_digest') != digest({k:v for k,v in record.items() if k!='record_digest'}): raise ValueError('Record digest differs')
    if record['identity'] != identity(plan,name,case) or sha(cp) != record['checkpoint_sha256']: raise ValueError('Checkpoint identity/hash differs')
    with np.load(cp,allow_pickle=False) as z:
        if z.files != ['losses']: raise ValueError('Unexpected checkpoint payload')
        losses = z['losses']
    if (losses.dtype != np.dtype('float64') or losses.shape != (case['origin']['max_horizon'],)
            or not np.isfinite(losses).all() or np.any(losses<0) or hashlib.sha256(losses.tobytes()).hexdigest()!=record['loss_sha256']):
        raise ValueError('Raw losses differ')
    if record['source_closure_digest'] != digest(plan['source_hashes']): raise ValueError('Checkpoint source closure differs')
    return record

def initialize(destination):
    global WORK, PLAN, PANEL, INPUTS, FIRST
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0))); WORK = Path(destination); PLAN = read(WORK/'plan.json')
    PANEL = api.load_panel(V/'inputs',verify_files=False); INPUTS=AssetInputs(V/'inputs'); FIRST=True

def evaluate(task):
    global FIRST
    name,index = task;case=PLAN['cases'][index];origin=case['origin'];row=PANEL['series'][case['series_index']]
    H=origin['max_horizon'];stop=origin['position']+1;phase='input_integrity'
    cp,rp,fp=paths(WORK,name,index);started=time.perf_counter(); warmup=FIRST;FIRST=False
    def state(stage):write(WORK/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'task':[name,index],'phase':stage,'utc':now()})
    state('input_preparation')
    try:
        dates,returns,origins=api.load_series(V/'inputs',row,verify_files=False)
        if origins[case['origin_index']] != origin: raise ValueError('Origin changed')
        x,w,ids=INPUTS.training(row,stop); input_digest=digest({'matrix_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
            'weights':w.tolist(),'dates_sha256':hashlib.sha256(dates[:stop].tobytes()).hexdigest(),'constituent_ids':ids})
        spec=PLAN['models'][name];module=importlib.import_module(spec['module']);compute_start=time.process_time()
        state('selection');phase='native_forecast'; bundle=module.prepare(x,w,H,dates[:stop],inner_paths=2,schedule_rule='half_doubling',seed_replication=0)
        model=getattr(module,spec['class'])(spec['config'],dates[:stop],bundle)
        state('forecast_and_CRPS');actual=np.cumsum(returns[stop:stop+H]);loss=np.empty(H);cursor=0;seed=identity(PLAN,name,case)['outer_seed']
        for start,chunk in model.terminal_chunks(x,w,H,1000,seed,width=64):
            phase='scoring_integrity'
            if start!=cursor or chunk.shape!=(1000,min(64,H-cursor)) or not np.isfinite(chunk).all(): raise ArithmeticError('Path chunks incomplete/nonfinite')
            cursor+=chunk.shape[1]; loss[start:cursor]=empirical_crps_by_horizon(chunk,actual[start:cursor]);phase='native_forecast'
        phase='scoring_integrity'
        if cursor!=H or not np.isfinite(loss).all() or np.any(loss<0): raise ArithmeticError('Incomplete CRPS')
        score=scoring.origin_score(loss,returns[:stop],origin['origin_date'],dates[stop:stop+H]);compute_seconds=time.process_time()-compute_start
        record={'identity':identity(PLAN,name,case),'source_closure_digest':digest(PLAN['source_hashes']),
            'input_training_digest':input_digest,'requested_H':H,'training_rows':stop,'constituent_ids':ids,
            'series_id':row['series_id'],'origin_date':origin['origin_date'],'origin_label':origin['origin_label'],
            'score':score,'selection':model.last_selection,'validation_folds':bundle['trace'],
            'timing':{'forecast_selection_score_cpu_seconds':compute_seconds,'worker_first_request':warmup,
                'fit_repair':False,'scope':'Selection/native forecast/CRPS CPU only; excludes IO/setup/contention; not speedup evidence'},
            'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest()}
        phase='checkpoint_io'; cp.parent.mkdir(parents=True,exist_ok=True)
        temp=cp.with_name(cp.name+'.tmp')
        with temp.open('wb') as stream: np.savez_compressed(stream,losses=loss)
        os.replace(temp,cp);record['checkpoint_sha256']=sha(cp)
        record['timing']['total_request_wall_seconds']=time.perf_counter()-started
        record['record_digest']=digest(record)
        if len(json.dumps(record))>40000: raise ValueError('Metadata exceeds declared40KB/request storage allowance')
        write(rp,record)
        state('idle');return {'task':task,'record':compact(record)}
    except Exception as error:
        result={'task':task,'stage':phase,'error_type':type(error).__name__,'message':str(error),'utc':now()}
        write(fp,result);state('idle');return result

class WorkProjection:
    def __init__(self,plan,tasks):
        self.plan=plan;self.remaining=Counter(self.key(*task) for task in tasks);self.samples={}
    def key(self,name,index):
        o=self.plan['cases'][index]['origin']
        return name,int(math.log2(o['position']+1)),int(math.log2(o['max_horizon']))
    def accept(self,name,index,record,completed=False):
        key=self.key(name,index)
        if completed:self.remaining[key]-=1
        if record is None:return
        t=record['timing']
        if t['worker_first_request'] or t['fit_repair']:return
        values=self.samples.setdefault(key,[])
        if len(values)<128:values.append(t['forecast_selection_score_cpu_seconds'])
    def report(self):
        mean=median=0.;covered=uncovered=0
        for key,count in self.remaining.items():
            if count<=0:continue
            if key not in self.samples:uncovered+=count;continue
            values=self.samples[key];covered+=count
            mean+=count*statistics.mean(values);median+=count*statistics.median(values)
        return {'kind':'measured_same_stratum_CPU_work_lower_bound_not_calibrated_wall_ETA',
            'mean_seconds_lower_bound':mean/12 if covered else None,'median_seconds_lower_bound':median/12 if covered else None,
            'covered_remaining_tasks':covered,'uncovered_remaining_tasks':uncovered,
            'scope':'Only observed model/log2N/log2H strata; excludes IO/setup/contention, fit repair and first requests; unseen work omitted, not zero'}

def projection(records,remaining,plan):
    tracker=WorkProjection(plan,remaining)
    for name,index,record in records:tracker.accept(name,index,record)
    return tracker.report()

def compact(record):
    return {key:record[key] for key in ('series_id','origin_date','score','timing')}


class Hierarchy:
    """Incremental unchanged origin/series/family/class/branch/quarter means."""
    def __init__(self,metadata):self.metadata=metadata;self.nodes={};self.leaves={}
    def add(self,record):
        sid=record['series_id'];meta=self.metadata[sid];date=record['origin_date']
        quarter=f"{date[:4]}Q{(int(date[5:7])-1)//3+1}"
        path=(quarter,meta['panel_branch'],meta['asset_class'],meta['family'],sid)
        total,count=self.leaves.get(path,(0.,0));total+=record['score'];count+=1
        self.leaves[path]=(total,count);value=total/count
        for depth in range(len(path)-1,-1,-1):
            prefix=path[:depth];key=path[depth]
            node=self.nodes.setdefault(prefix,{'values':{},'sum':0.})
            node['sum']+=value-node['values'].get(key,0.);node['values'][key]=value
            value=node['sum']/len(node['values'])
            # Only quarters with individual-asset support enter the main score.
            if depth==1 and 'assets' not in node['values']:return
    def score(self):
        node=self.nodes.get(())
        return node['sum']/len(node['values']) if node else None

class LiveComparison:
    BASELINES=('model_testing_asset_joint_blocks_replacement_v6','model_testing_asset_joint_blocks_history_partition_v6')
    def __init__(self,plan,panel):
        self.plan=plan;metadata={row['series_id']:row for row in panel['series']}
        if plan['fixed_reference_model_id'] not in self.BASELINES:raise ValueError('Unknown fixed full-panel reference')
        self.reducers={name:{model:Hierarchy(metadata) for model in ('candidate',*self.BASELINES)} for name in plan['models']}
        self.counts=Counter();self.scores={};expected={(c['series_index'],c['origin_index']):c for c in plan['cases']}
        path=Path(plan['baseline_score_index']['path'])
        with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True) as connection:
            query='SELECT task_key,model_id,series_index,origin_index,series_id,origin_date,native_horizon,training_observations,panel_fingerprint,protocol_id,score_v7 FROM task_scores_v7'
            for task_key,model,i,j,sid,date,H,n,panel_id,protocol,score in connection.execute(query):
                case=expected.get((i,j))
                if case is None:continue
                if (model not in self.BASELINES or case is None or json.loads(task_key)!=[case['series_id'],case['origin']]
                        or sid!=case['series_id'] or date!=case['origin']['origin_date']
                        or H!=case['origin']['max_horizon'] or n!=case['origin']['position']+1 or panel_id!=plan['panel_fingerprint']
                        or protocol!=scoring.SCORE_PROTOCOL_ID or not math.isfinite(score) or score<0):
                    raise ValueError('Baseline score index identity/coverage differs')
                key=(model,i,j)
                if key in self.scores:raise ValueError('Duplicate baseline score key')
                self.scores[key]=score
        if len(self.scores)!=2*len(plan['cases']):raise ValueError('Both baseline full coverage required')
    def add(self,name,index,record):
        self.counts[name]+=1;case=self.plan['cases'][index];reducers=self.reducers[name]
        reducers['candidate'].add(record)
        for model in self.BASELINES:
            reducers[model].add({**record,'score':self.scores[(model,case['series_index'],case['origin_index'])]})
    def report(self):
        output={}
        for name,reducers in self.reducers.items():
            values={model:reducer.score() for model,reducer in reducers.items()}
            candidate=values['candidate'];baselines={model:values[model] for model in self.BASELINES}
            fixed=self.plan['fixed_reference_model_id']
            reference=baselines[fixed] if candidate is not None else None
            output[name]={'successful_matched_tasks':self.counts[name],'candidate_Q':candidate,'baseline_Q_same_tasks':baselines,
                'percent_better_vs_each_baseline':{model:100*(1-candidate/value) if candidate is not None and value else None for model,value in baselines.items()},
                'fixed_full_panel_reference_model_id':fixed,'percent_better_vs_fixed_reference':100*(1-candidate/reference) if reference else None,
                'scope':'Preliminary exact same successful-task intersection; unchanged hierarchy; excludes quarters pending asset support; not full-panel reference/promotion',
                'official':False}
        return output


def worker_states(destination):
    states=[]
    for path in (Path(destination)/'workers').glob('*.json'):
        state=read(path)
        try:os.kill(state['pid'],0)
        except ProcessLookupError:continue
        states.append(state)
    return states

def run(destination,admission_path):
    destination=Path(destination);plan=read(destination/'plan.json');panel=verify_plan(plan);admit(plan,read(admission_path))
    # A separate per-run lock prevents overlapping resume/report writers. Shared
    # scientific slots are held by this parent for the complete executor lifetime.
    with (destination/'controller.lock').open('a') as own, (O/'asset_v6_forecast_slots.lock').open('a') as slots:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(slots,fcntl.LOCK_EX)
        verify_plan(plan)
        live=LiveComparison(plan,panel)
        records=[];todo=[]
        for index in range(len(plan['cases'])):
            for name in plan['models']:
                record=verified_checkpoint(destination,plan,name,index)
                if record:
                    records.append((name,index,compact(record)));live.add(name,index,record)
                else:todo.append((name,index))
        remaining_bytes=sum(plan['cases'][index]['origin']['max_horizon']*8+41500 for _,index in todo)
        growth=plan['expected_growth_gib'] if len(todo)==plan['expected_tasks'] else remaining_bytes/(1024**3)
        subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib',str(growth),
                        '--artifact-root',str(O),'--budget-gib','16'],check=True)
        tracker=WorkProjection(plan,todo)
        counts=Counter(name for name,_,_ in records)
        for name,index,record in records:tracker.accept(name,index,record)
        errors=[];terminal={'status':'failed','pid':os.getpid(),'utc':now(),'expected_tasks':plan['expected_tasks'],'qualification_valid':False}
        write(destination/'active.json',{**terminal,'status':'running','resumed_verified':len(records),'workers':12})
        try:
            # At most24 pending requests; finite frozen task iterator, no new admissions.
            tasks=iter(todo);pending={};remaining=set(todo)
            with futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(str(destination),)) as pool:
                exhausted=False;last_preflight=time.monotonic();last_progress=0.
                while pending or not exhausted:
                    while len(pending)<24 and not exhausted:
                        try:task=next(tasks)
                        except StopIteration:exhausted=True;break
                        write(paths(destination,*task)[0].with_suffix('.attempt.json'),{'identity':identity(plan,task[0],plan['cases'][task[1]]),'submitted_utc':now()})
                        pending[pool.submit(evaluate,task)]=task
                    if not pending: break
                    done,_=futures.wait(pending,timeout=5,return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        task=pending.pop(future);remaining.remove(task)
                        try:result=future.result()
                        except Exception as error:
                            result={'task':task,'stage':'worker_failure','message':str(error)}
                            write(paths(destination,*task)[2],result)
                        if 'record' in result:
                            records.append((*task,result['record']));counts[task[0]]+=1;live.add(*task,result['record'])
                        else: errors.append(result)
                        tracker.accept(*task,result.get('record'),completed=True)
                    if time.monotonic()-last_progress>=5 or not pending and exhausted:
                        workers=worker_states(destination)
                        write(destination/'progress.json',{'utc':now(),'status':'running','successful_tasks':len(records),
                            'expected_tasks':plan['expected_tasks'],'expected_per_model':plan['expected_per_model'],
                            'successful_per_model':dict(counts),'attempted_failures':len(errors),
                            'native_failures':sum(e['stage']=='native_forecast' for e in errors),
                            'scoring_integrity_errors':sum(e['stage'] in ('scoring_integrity','input_integrity') for e in errors),
                            'IO_infrastructure_errors':sum(e['stage'] in ('checkpoint_io','worker_failure') for e in errors),
                            'matched_v7_comparisons':live.report(),'projection':tracker.report(),'active_workers':sum(w['phase']!='idle' for w in workers),'official':False})
                        last_progress=time.monotonic()
                    if time.monotonic()-last_preflight>900:
                        remaining_bytes=sum(plan['cases'][index]['origin']['max_horizon']*8+41500 for _,index in remaining)
                        subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib',str(remaining_bytes/(1024**3)),
                            '--artifact-root',str(O),'--budget-gib','16'],check=True);last_preflight=time.monotonic()
            terminal.update(successful_tasks=len(records),failed_tasks=len(errors),forecast_workers_terminal=True)
            terminal['status']='complete' if not errors and len(records)==plan['expected_tasks'] else 'incomplete'
            verify_plan(plan)
            try:
                metadata={r['series_id']:r for r in panel['series']}
                scores={name:scoring.aggregate_records([r for n,_,r in records if n==name],metadata)
                        for name in plan['models']} if terminal['status']=='complete' else {}
                write(destination/'comparison.json',{'utc':now(),'complete':terminal['status']=='complete','models':scores,
                    'evaluation_scope':plan.get('evaluation_scope'),'expected_per_model':plan['expected_per_model'],'errors':errors,'official':False,'automatic_promotion':False,'requires_independent_full_pair_baseline_review':True})
                terminal['reporter_status']='terminal';terminal['qualification_valid']=terminal['status']=='complete'
            except Exception as error:
                terminal['reporter_status']='failed';terminal['reporter_error']=str(error)
                raise
        except BaseException as error:
            terminal['controller_error']=f'{type(error).__name__}: {error}'
            raise
        finally:
            terminal['utc']=now();terminal['all_processes_drained']=True;terminal['forecast_workers_terminal']=True
            write(destination/'active.json',terminal)
            write(destination/'terminal.json',terminal)
    return terminal

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','verify','run'])
    parser.add_argument('--directory',type=Path,default=D)
    parser.add_argument('--gate',action='append',default=[],help='NAME=/absolute/receipt.json; all four required')
    parser.add_argument('--root-admission',type=Path)
    parser.add_argument('--cohort-plan',type=Path,help='Optional existing355 development plan; nativeH unchanged')
    parser.add_argument('--candidate',action='append',choices=sorted(AVAILABLE_MODELS),help='Freeze atprepare; default endpoint')
    args=parser.parse_args()
    if args.command=='prepare': result=prepare(dict(v.split('=',1) for v in args.gate),args.directory,args.candidate,args.cohort_plan)
    elif args.command=='verify': result={'verified':True,'series':len(verify_plan(read(args.directory/'plan.json'))['series'])}
    else:
        if args.root_admission is None: parser.error('run requires --root-admission')
        result=run(args.directory,args.root_admission)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_hashes')},sort_keys=True))

if __name__=='__main__': main()
