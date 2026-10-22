"""Frozen requested-H1/5/10/21 supplement; existing full forecasts stay intact.

For each request r=(series,origin,H), q_r=mean_{h=1..H}
CRPS_r,h/(training_SD_ddof1*sqrt(h)). Within each series/quarter, average
q_r equally over requests, then preserve family/class/branch/quarter means.
Thus a70-year request and a1-day request each have one request's weight;
there are no horizon-band weights and no global equal-unique-H pooling.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import subprocess
import time
import sys
import zlib
import numpy as np
import run_mostworthy_panel_v7 as base
from asset_models import build_replacement,build_without_replacement
from candidates.asset_selector_endpoint_group_reuse_v1 import candidate as endpoint
from candidates.asset_selector_endpoint_active_qp_v1 import energy as fast_energy
from candidates.asset_selector_endpoint_active_qp_rankdef_v2 import energy as rankdef_energy
from candidates.asset_selector_endpoint_active_qp_unique_v3 import energy as unique_energy
import run_short_horizon_optimized_v2 as parent_runner

O=base.O;V=base.V;D=O/'short_horizon_final240_v3'
FINAL_PATHS=240
HORIZONS=(1,5,10,21)
WR='model_testing_asset_joint_blocks_replacement_v6'
WOR='model_testing_asset_joint_blocks_history_partition_v6'
MODEL_NAMES=('endpoint',WR,WOR)
CONFIG={'menu':'flat','inner_paths':2,'schedule':'half_doubling'}
SOLVERS={'reference':None,'active_qp':fast_energy.fit,'active_qp_rankdef_v2':rankdef_energy.fit,'active_qp_unique_v3':unique_energy.fit}
MAX_GROUP_BYTES=28672
read=base.read;write=base.write;sha=base.sha;digest=base.digest;now=base.now

def source_hashes(adapter_module='candidates.asset_selector_endpoint_group_reuse_v1.candidate',fast_baselines=False):
    modules={'endpoint':{'module':adapter_module},'cached_reference':{'module':'candidates.asset_selector_endpoint_group_reuse_v1.candidate'},
        'solver':{'module':'candidates.asset_selector_endpoint_active_qp_v1.energy'},'rankdef_solver':{'module':'candidates.asset_selector_endpoint_active_qp_rankdef_v2.energy'},'unique_solver':{'module':'candidates.asset_selector_endpoint_active_qp_unique_v3.energy'},'baselines':{'module':'asset_models'}}
    if fast_baselines:modules['exact_baseline_implementation']={'module':'candidates.asset_exact_blocks_v1.model'}
    sources=base.closure(modules)
    for path in (Path(__file__).resolve(),Path(parent_runner.__file__).resolve(),V/'profile.json',V/'control.json',
                 V/'launch_plan.json',O/'expanded_short_horizon_authorization_v1.json'):
        sources[str(path)]=sha(path)
    return sources

def prefix_groups(cases):
    representatives={};aliases={}
    for index,case in enumerate(cases):
        key=(case['series_id'],case['origin']['position'])
        representative=representatives.setdefault(key,index)
        aliases.setdefault(str(representative),[]).append(index)
    return list(representatives.values()),aliases

def prepare(parent_directory,destination=D,authorization=O/'final240_authorization_v3.json'):
    destination=Path(destination);parent_directory=Path(parent_directory).resolve()
    if destination.exists():raise ValueError('Never overwrite an implementation generation')
    authorization=Path(authorization).resolve();authorized=read(authorization)
    if any(authorized.get(k)!=v for k,v in {'final_paths':240,'inner_paths':2,'retain_completed_1000':True,'same_count_pairs':True}.items()):raise ValueError('Explicit mixed-count user authorization required')
    parent=read(parent_directory/'plan.json');parent_runner.verify(parent)
    solver=parent['solver'];adapter_module=parent['adapter_module'];skip_unused_parent_calibration=parent['skip_unused_parent_calibration'];fast_baselines=parent['fast_baselines']
    plan={k:v for k,v in parent.items() if k not in ('fingerprint','utc','source_hashes')}
    adapter=importlib.import_module(adapter_module)
    if skip_unused_parent_calibration and adapter_module not in ('candidates.asset_selector_endpoint_fit_acceleration_v2.candidate','candidates.asset_selector_endpoint_fit_reuse_v3.candidate'):raise ValueError('Explicit removed-guard flag only supported by qualified fit-acceleration adapter')
    if adapter.identity(CONFIG)!=parent['model_ids']['endpoint']:raise ValueError('Adapter financial model identity differs')
    if fast_baselines:
        fast=importlib.import_module('candidates.asset_exact_blocks_v1.model')
        for name,factory in ((WR,fast.build_replacement),(WOR,fast.build_without_replacement)):
            model=factory()
            if model.model_id!=parent['model_ids'][name] or model.dependency_identity!=parent['baseline_factories'][name]:raise ValueError('Fast baseline financial dependency identity differs')
    plan.update(schema='short-horizon-final240-generation-v3',utc=now(),source_hashes=source_hashes(adapter_module,fast_baselines),adapter_module=adapter_module,
        parent_supplement_directory=str(parent_directory),parent_supplement_plan_sha256=sha(parent_directory/'plan.json'),
        parent_supplement_fingerprint=parent['fingerprint'],parent_scientific_source_hashes=parent['source_hashes'],
        solver=solver,skip_unused_parent_calibration=bool(skip_unused_parent_calibration),fast_baselines=bool(fast_baselines),baseline_implementation=parent['baseline_implementation'],implementation_id='endpoint_execution_final240_v3:'+parent['implementation_id'],
        paths=FINAL_PATHS,final_paths=FINAL_PATHS,inner_paths=2,user_authorization={'path':str(authorization),'sha256':sha(authorization)},
        baseline_dependency_identity_scope='Original immutable1000 factory/profile identity retained as financial-law provenance; actual per-request final simulation count is recorded240 for all new outputs and1000 for inherited outputs',
        mixed_count_rule='Preserve every completed1000 request; all new models use240 final paths and2 inner paths. Pair every request at identical pathcount. Saved1000 baseline SQLite scores only pair with saved1000 endpoint requests; new oldH tasks compute all3 models240. No reweighting by pathcount.',
        cache_policy='64MiB capped group-local fit state plus private training copy; explicit close after group; no persistent cache',
        mixed_generation_policy='Inherited checkpoints retain original bytes/identity; each new short and oldH checkpoint binds this new implementation fingerprint',
        environment=base.environment())
    plan['fingerprint']=digest(plan);destination.mkdir();write(destination/'plan.json',plan);return plan

def verify(plan):
    if digest({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']:raise ValueError('Plan digest differs')
    if plan['source_hashes']!=source_hashes(plan['adapter_module'],plan.get('fast_baselines',False)) or plan['environment']!=base.environment():raise ValueError('Implementation source/environment differs')
    if plan.get('final_paths')!=240 or plan.get('inner_paths')!=2 or sha(plan['user_authorization']['path'])!=plan['user_authorization']['sha256']:raise ValueError('Mixed-count authorization/settings differ')
    directory=Path(plan['parent_supplement_directory']);parent=read(directory/'plan.json')
    if sha(directory/'plan.json')!=plan['parent_supplement_plan_sha256'] or parent['fingerprint']!=plan['parent_supplement_fingerprint']:
        raise ValueError('Scientific parent differs')
    panel=parent_runner.verify(parent)
    if any(plan[key]!=parent[key] for key in ('adapter_module','solver','skip_unused_parent_calibration','fast_baselines','model_ids','baseline_factories')):raise ValueError('Numerical methods differ from qualified parent')
    if parent['cases']!=plan['cases'] or parent['group_indices']!=plan['group_indices'] or parent['prefix_aliases']!=plan['prefix_aliases']:
        raise ValueError('Scientific request coverage/alias weighting differs')
    return panel

def admission(plan,receipt,smoke=False):
    if receipt.get('schema')!='short-horizon-final240-v3-root-admission' or receipt.get('plan_fingerprint')!=plan['fingerprint'] or receipt.get('passed') is not True:
        raise ValueError('Root admission must bind this exact supplement')
    if receipt.get('root_cleared_smoke' if smoke else 'root_cleared_launch') is not True:raise ValueError('Explicit root clearance required')
    if not smoke:
        bound=receipt['smoke_receipt'];result=read(bound['path'])
        if sha(bound['path'])!=bound['sha256'] or not result['passed'] or result['plan_fingerprint']!=plan['fingerprint'] or result.get('final_paths')!=240:
            raise ValueError('Passed source-bound actual smoke required')
        qualified=receipt['numerical_qualification']
        if sha(qualified['path'])!=qualified['sha256'] or receipt.get('numerical_qualification_passed') is not True:raise ValueError('Root-approved source-bound numerical qualification required')
        quality=read(qualified['path'])
        if not quality.get('complete'):raise ValueError('Numerical qualification is incomplete')
        if quality.get('module')!=plan['adapter_module'] or bool(quality.get('skip_unused_parent_calibration',False))!=bool(plan.get('skip_unused_parent_calibration')):raise ValueError('Qualified adapter/mode differs')
        expected_solver=SOLVERS[plan['solver']]
        if quality.get('solver')!=(expected_solver.__module__ if expected_solver else None):raise ValueError('Qualified solver differs')
        solver=SOLVERS[plan['solver']]
        adapter=importlib.import_module(plan['adapter_module'])
        for module_path in [Path(adapter.__file__).resolve()]+([Path(sys.modules[solver.__module__].__file__).resolve()] if solver else []):
            if quality['source_hashes'].get(str(module_path))!=sha(module_path):raise ValueError('Qualified implementation source differs')
        if plan.get('fast_baselines'):
            qualified=receipt['baseline_qualification'];quality=read(qualified['path'])
            if sha(qualified['path'])!=qualified['sha256'] or receipt.get('baseline_qualification_passed') is not True or not quality.get('complete') or quality.get('fast_baselines') is not True:raise ValueError('Root-approved exact baseline qualification required')
            for module in ('candidates.asset_exact_blocks_v1.model','candidates.asset_exact_blocks_v1.blocks'):
                path=Path(importlib.import_module(module).__file__).resolve()
                if quality.get('source_hashes',{}).get(str(path))!=plan['source_hashes'].get(str(path)):raise ValueError('Qualified fast baseline source differs')
        if set(result['branches'])!={'assets','portfolios'} or result['forecasts']!=24:raise ValueError('Both branch/four-H/three-model smoke required')

def identity(plan,index):
    return {'plan':plan['fingerprint'],'case_index':index,'case':plan['cases'][index],
            'horizons':list(HORIZONS),'model_ids':plan['model_ids'],'paths':FINAL_PATHS,'inner_paths':2,'replication':0}

def paths(destination,index):
    stem=Path(destination)/'groups'/f'{index:05d}'
    return stem.with_suffix('.npz'),stem.with_suffix('.json'),stem.with_suffix('.attempt.json')

def save_group(destination,plan,index,losses,metadata):
    cp,rp,_=paths(destination,index);cp.parent.mkdir(parents=True,exist_ok=True)
    metadata_bytes=json.dumps(metadata,sort_keys=True,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()).encode()
    payload={**losses,'metadata_zlib':np.frombuffer(zlib.compress(metadata_bytes,9),dtype=np.uint8)}
    temporary=cp.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**payload)
    if temporary.stat().st_size>MAX_GROUP_BYTES:raise ValueError('Group exceeds declared28KiB storage bound; preserve uncommitted file for review')
    os.replace(temporary,cp)
    record={'plan':plan['fingerprint'],'index':index,'checkpoint_sha256':sha(cp),'metadata_sha256':hashlib.sha256(metadata_bytes).hexdigest()}
    if len(json.dumps(record))>512:raise ValueError('Sidecar exceeds storage allowance')
    write(rp,record)

def read_group(destination,plan,index):
    cp,rp,attempt=paths(destination,index)
    if not cp.exists() and not rp.exists():
        if attempt.exists() or cp.with_suffix('.npz.tmp').exists():raise ValueError('Interrupted group; explicit review, no silent rerun')
        return inherited_group(destination,plan,index)
    if not cp.exists() or not rp.exists():raise ValueError('Orphan group; no silent rerun')
    record=read(rp)
    if record['plan']!=plan['fingerprint'] or record['index']!=index or sha(cp)!=record['checkpoint_sha256']:raise ValueError('Group hash/identity differs')
    with np.load(cp,allow_pickle=False) as archive:
        metadata_bytes=zlib.decompress(archive['metadata_zlib'].tobytes());metadata=json.loads(metadata_bytes)
        if hashlib.sha256(metadata_bytes).hexdigest()!=record['metadata_sha256'] or metadata['identity']!=identity(plan,index):raise ValueError('Metadata differs')
        losses={name:archive[name].copy() for name in archive.files if name!='metadata_zlib'}
    for key,loss in losses.items():
        H=int(key.rsplit('_',1)[1])
        if H not in HORIZONS or loss.dtype!=np.dtype('float64') or loss.shape!=(H,) or not np.isfinite(loss).all() or np.any(loss<0):raise ValueError('Raw CRPS vector differs')
    if metadata['errors']:raise ValueError('Previously failed forecasts require explicit review; saved evidence retained')
    if set(losses)!={f'{m}_{h}' for m in MODEL_NAMES for h in HORIZONS}:raise ValueError('Missing model/H losses')
    return losses,metadata

_MIGRATION_CACHE={}
_PARENT_PLAN_CACHE={}
def migration(plan,destination,refresh=False):
    key=(plan['fingerprint'],str(Path(destination).resolve()))
    if not refresh and key in _MIGRATION_CACHE:return _MIGRATION_CACHE[key][1]
    binding=read(Path(destination)/'root_launch_admission.json')['handoff']
    if key in _MIGRATION_CACHE and binding!=_MIGRATION_CACHE[key][0]:raise ValueError('Root migration binding changed during execution')
    if sha(binding['path'])!=binding['sha256']:raise ValueError('Migration receipt changed')
    receipt=read(binding['path'])
    if receipt['plan_fingerprint']!=plan['fingerprint']:raise ValueError('Wrong implementation migration')
    _MIGRATION_CACHE[key]=(binding,receipt);return receipt

def parent_plan(plan):
    key=(plan['parent_supplement_directory'],plan['parent_supplement_plan_sha256'])
    if key not in _PARENT_PLAN_CACHE:
        path=Path(key[0])/'plan.json'
        if sha(path)!=key[1]:raise ValueError('Inherited scientific plan changed')
        _PARENT_PLAN_CACHE[key]=read(path)
    return _PARENT_PLAN_CACHE[key]

def inherited_group(destination,plan,index):
    admission_path=Path(destination)/'root_launch_admission.json'
    if not admission_path.exists():return None
    receipt=migration(plan,destination);bound=receipt['inherited_short_groups'].get(str(index))
    if bound is None:return None
    directory=Path(plan['parent_supplement_directory']);parent=parent_plan(plan);cp,rp=resolve_parent_group(parent_runner,directory,parent,index)
    if str(cp)!=bound['checkpoint_path'] or str(rp)!=bound['record_path'] or sha(cp)!=bound['checkpoint_sha256'] or sha(rp)!=bound['record_sha256']:raise ValueError('Inherited group changed')
    return parent_runner.read_group(directory,parent,index)

def resolve_parent_group(module,directory,plan,index):
    """Resolve physical bytes only after the parent's own reader validates lineage."""
    cp,rp,_=module.paths(directory,index)
    if cp.exists() or rp.exists():return cp.resolve(),rp.resolve()
    if 'parent_supplement_directory' not in plan:raise ValueError('Missing inherited group payload')
    directory=Path(plan['parent_supplement_directory']);return resolve_parent_group(module.parent_runner,directory,module.parent_plan(plan),index)

def pair_binding(cp,rp,kind):
    return {'kind':kind,'checkpoint_path':str(cp.resolve()),'record_path':str(rp.resolve()),'checkpoint_sha256':sha(cp),'record_sha256':sha(rp),'paths':1000}

def parent_old_inventory(plan):
    parent=parent_plan(plan);directory=Path(plan['parent_supplement_directory']);old_directory=Path(plan['original_directory']);old=read(old_directory/'plan.json');saved={}
    for index in range(len(plan['cases'])):
        generated=parent_runner.read_long(directory,parent,index)
        original=base.verified_checkpoint(old_directory,old,'endpoint',index)
        if generated is not None and original is not None:raise ValueError('Duplicate completed originalH implementations')
        if generated is not None:
            if generated[1]['identity']['paths']!=1000:raise ValueError('Unexpected parent originalH pathcount')
            cp,rp,_=parent_runner.long_paths(directory,index);saved[str(index)]=pair_binding(cp,rp,'parent_long')
        elif original is not None:
            cp,rp,_=base.paths(old_directory,'endpoint',index);saved[str(index)]=pair_binding(cp,rp,'original')
    return saved

def long_paths(destination,index):
    stem=Path(destination)/'new_long_groups'/f'{index:05d}'
    return stem.with_suffix('.npz'),stem.with_suffix('.json'),stem.with_suffix('.attempt.json')

def long_identity(plan,index):
    return {'implementation_plan':plan['fingerprint'],'scientific_parent':plan['parent_supplement_fingerprint'],
        'kind':'original_requested_H_paired_models','case_index':index,'case':plan['cases'][index],'model_ids':plan['model_ids'],
        'paths':FINAL_PATHS,'inner_paths':2,'replication':0}

def read_long(destination,plan,index):
    cp,rp,attempt=long_paths(destination,index)
    if not cp.exists() and not rp.exists():
        if attempt.exists() or cp.with_suffix('.npz.tmp').exists():raise ValueError('Interrupted new-generation oldH request; no silent rerun')
        return None
    if not cp.exists() or not rp.exists():raise ValueError('Orphan new-generation oldH request')
    record=read(rp)
    if record.get('record_digest')!=digest({k:v for k,v in record.items() if k!='record_digest'}):raise ValueError('New-generation oldH record digest differs')
    if record['identity']!=long_identity(plan,index) or sha(cp)!=record['checkpoint_sha256']:raise ValueError('New-generation oldH binding differs')
    with np.load(cp,allow_pickle=False) as archive:losses={name:archive[name].copy() for name in archive.files}
    H=plan['cases'][index]['origin']['max_horizon']
    if set(losses)!=set(MODEL_NAMES) or set(record['results'])!=set(MODEL_NAMES):raise ValueError('Missing paired originalH model')
    for name,loss in losses.items():
        if loss.dtype!=np.dtype('float64') or loss.shape!=(H,) or not np.isfinite(loss).all() or np.any(loss<0) or hashlib.sha256(loss.tobytes()).hexdigest()!=record['results'][name]['loss_sha256'] or record['results'][name]['final_paths']!=240:raise ValueError('New-generation oldH losses/count differ')
    return losses,record

def initialize(destination):
    global WORK,PLAN,PANEL,INPUTS,endpoint
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)));WORK=Path(destination);PLAN=read(WORK/'plan.json')
    endpoint=importlib.import_module(PLAN['adapter_module'])
    PANEL=base.api.load_panel(V/'inputs',verify_files=False);INPUTS=base.AssetInputs(V/'inputs')

def diagnostic_json(value):
    if isinstance(value,np.ndarray):return diagnostic_json(value.tolist())
    if isinstance(value,np.generic):return diagnostic_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return {'diagnostic_nonfinite_float':str(value)}
    if isinstance(value,dict):return {key:diagnostic_json(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [diagnostic_json(item) for item in value]
    return value

def forecast_one(name,x,w,H,dates,actual,origin_date,fit_cache=None):
    seed=base.api.original.coherent_daily_seed(origin_date,H,FINAL_PATHS);selection=None;trace=None;solver_fit=None
    if name=='endpoint':
        options={'skip_unused_parent_calibration':True} if PLAN.get('skip_unused_parent_calibration') else {}
        bundle=endpoint.prepare(x,w,H,dates,inner_paths=2,schedule_rule='half_doubling',seed_replication=0,fit_cache=fit_cache,solver=SOLVERS[PLAN['solver']],**options)
        if bundle['horizon']!=H:raise ValueError('Selector did not receive actual requestedH')
        model=endpoint.MenuPool(CONFIG,dates,bundle);trace=bundle['trace'];solver_fit=diagnostic_json({key:value for key,value in bundle.get('fit',{}).get('flat',{}).items() if key!='weights'})
    else:
        factories=importlib.import_module('candidates.asset_exact_blocks_v1.model') if PLAN.get('fast_baselines') else None
        model=(factories.build_replacement() if name==WR else factories.build_without_replacement()) if factories else (build_replacement() if name==WR else build_without_replacement())
    loss=np.empty(H);cursor=0
    for start,chunk in model.terminal_chunks(x,w,H,FINAL_PATHS,seed,width=64):
        if start!=cursor or chunk.shape!=(FINAL_PATHS,min(64,H-cursor)) or not np.isfinite(chunk).all():raise ArithmeticError('Incomplete joint path coverage')
        cursor+=chunk.shape[1];loss[start:cursor]=base.empirical_crps_by_horizon(chunk,actual[start:cursor])
    if cursor!=H or not np.isfinite(loss).all() or np.any(loss<0):raise ArithmeticError('Incomplete CRPS')
    if name=='endpoint':selection=model.last_selection
    return loss,{'model_id':model.model_id,'requested_H_seen':H,'outer_seed':seed,
        'final_paths':FINAL_PATHS,'candidate_inner_replication':0 if name=='endpoint' else None,'baseline_implementation':PLAN.get('baseline_implementation','asset_models') if name!='endpoint' else None,'solver_fit':solver_fit,'selection':selection,'validation_folds':trace}

def evaluate_group(index):
    case=PLAN['cases'][index];row=PANEL['series'][case['series_index']];origin=case['origin'];stop=origin['position']+1
    cp,_,_=paths(WORK,index);losses={};details=[];errors=[];phase='input_integrity';started=time.process_time()
    try:
        dates,returns,origins=base.api.load_series(V/'inputs',row,verify_files=False)
        if origins[case['origin_index']]!=origin:raise ValueError('Origin differs')
        x,w,ids=INPUTS.training(row,stop)
        training={'joint_matrix_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'weights':w.tolist(),'constituent_ids':ids,
            'dates_sha256':hashlib.sha256(dates[:stop].tobytes()).hexdigest(),'training_rows':stop}
        with endpoint.GroupWork(x,w,dates[:stop],horizons=HORIZONS,**({'skip_unused_parent_calibration':True} if PLAN.get('skip_unused_parent_calibration') else {})) as work:
            for H in HORIZONS:
                actual=np.cumsum(returns[stop:stop+H])
                for name in MODEL_NAMES:
                    phase='native_forecast'
                    try:
                        begin=time.process_time();loss,diagnostic=forecast_one(name,x,w,H,dates[:stop],actual,origin['origin_date'],fit_cache=work)
                        phase='scoring_integrity';score=base.scoring.origin_score(loss,returns[:stop],origin['origin_date'],dates[stop:stop+H])
                        losses[f'{name}_{H}']=loss;details.append({'model':name,'H':H,'score':score,'cpu_seconds':time.process_time()-begin,**diagnostic})
                    except Exception as error:errors.append({'model':name,'H':H,'stage':phase,'type':type(error).__name__,'message':str(error)})
            cache_diagnostics=dict(work.diagnostics())
        phase='checkpoint_io';metadata={'identity':identity(PLAN,index),'training':training,'results':details,'errors':errors,'cpu_seconds':time.process_time()-started,'implementation_id':PLAN['implementation_id'],'adapter_module':PLAN['adapter_module'],'solver':PLAN['solver'],'removed_unused_parent_failure_guard':PLAN.get('skip_unused_parent_calibration',False),'cache_diagnostics':cache_diagnostics}
        save_group(WORK,PLAN,index,losses,metadata)
        return {'index':index,'successful_forecasts':len(details),'errors':errors,'cpu_seconds':metadata['cpu_seconds']}
    except Exception as error:
        result={'index':index,'stage':phase,'type':type(error).__name__,'message':str(error)}
        write(cp.with_suffix('.failure.json'),result);return {'index':index,'successful_forecasts':0,'errors':[result]}

def evaluate_long(index):
    case=PLAN['cases'][index];origin=case['origin'];row=PANEL['series'][case['series_index']]
    H=origin['max_horizon'];stop=origin['position']+1;cp,rp,_=long_paths(WORK,index);phase='input_integrity'
    try:
        dates,returns,origins=base.api.load_series(V/'inputs',row,verify_files=False)
        if origins[case['origin_index']]!=origin:raise ValueError('Original origin changed')
        x,w,ids=INPUTS.training(row,stop);start=time.process_time();losses={};results={}
        with endpoint.GroupWork(x,w,dates[:stop],horizons=(H,),**({'skip_unused_parent_calibration':True} if PLAN.get('skip_unused_parent_calibration') else {})) as work:
            for name in MODEL_NAMES:
                phase='native_forecast';loss,diagnostic=forecast_one(name,x,w,H,dates[:stop],np.cumsum(returns[stop:stop+H]),origin['origin_date'],fit_cache=work)
                phase='scoring_integrity';score=base.scoring.origin_score(loss,returns[:stop],origin['origin_date'],dates[stop:stop+H])
                losses[name]=loss;results[name]={'score':score,'loss_sha256':hashlib.sha256(loss.tobytes()).hexdigest(),**diagnostic}
            cache_diagnostics=dict(work.diagnostics())
        record={'identity':long_identity(PLAN,index),'implementation_id':PLAN['implementation_id'],'adapter_module':PLAN['adapter_module'],'solver':PLAN['solver'],'removed_unused_parent_failure_guard':PLAN.get('skip_unused_parent_calibration',False),'results':results,
            'series_id':case['series_id'],'origin_date':origin['origin_date'],'cpu_seconds':time.process_time()-start,
            'constituent_ids':ids,'joint_training_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'cache_diagnostics':cache_diagnostics,
            'final_paths':FINAL_PATHS,'inner_paths':2}
        phase='checkpoint_io';cp.parent.mkdir(parents=True,exist_ok=True);temp=cp.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:np.savez_compressed(stream,**losses)
        os.replace(temp,cp);record['checkpoint_sha256']=sha(cp);record['record_digest']=digest(record)
        if len(json.dumps(record))>40000:raise ValueError('New-generation oldH metadata exceeds40KB allowance')
        write(rp,record)
        return {'kind':'old','index':index,'successful_forecasts':3,'successful_logical_tasks':1,'errors':[]}
    except Exception as error:
        failure={'kind':'old','index':index,'successful_forecasts':0,'errors':[{'stage':phase,'message':str(error)}]}
        write(cp.with_suffix('.failure.json'),failure);return failure

def growth(indices):return len(indices)*(MAX_GROUP_BYTES+1024)/(1024**3)

def make_handoff(destination):
    destination=Path(destination);plan=read(destination/'plan.json');verify(plan)
    parent_directory=Path(plan['parent_supplement_directory']);parent=read(parent_directory/'plan.json')
    with (O/'asset_v6_forecast_slots.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        terminal_path=parent_directory/'terminal.json';terminal=read(terminal_path)
        if not terminal.get('workers_terminal'):raise ValueError('Scientific parent must be controlled-drained')
        pause=read(parent_directory/'operational_pause_receipt.json')
        if pause['controller_pid']!=terminal['pid'] or pause['terminal_sha256']!=sha(terminal_path):raise ValueError('Root controlled-pause receipt differs')
        inherited={};saved=parent_old_inventory(plan)
        for index in plan['group_indices']:
            result=parent_runner.read_group(parent_directory,parent,index)
            if result is not None:
                if result[1]['identity']['paths']!=1000:raise ValueError('Unexpected inherited short pathcount')
                cp,rp=resolve_parent_group(parent_runner,parent_directory,parent,index)
                inherited[str(index)]=pair_binding(cp,rp,'inherited_short')
        result={'schema':'source-bound-final240-migration-v3','utc':now(),'plan_fingerprint':plan['fingerprint'],
            'parent_plan_sha256':plan['parent_supplement_plan_sha256'],'parent_terminal_sha256':sha(terminal_path),
            'parent_pause_receipt_sha256':sha(parent_directory/'operational_pause_receipt.json'),
            'saved_old_slots':saved,'inherited_short_groups':inherited,
            'catchup_group_indices':[i for i in plan['group_indices'] if any(str(alias) in saved for alias in plan['prefix_aliases'][str(i)])],
            'preserved_1000_original_requests':len(saved),'preserved_1000_short_requests_per_model':4*len(inherited),'new_paths':240,'inner_paths':2,
            'all_holes_explicit':True,'no_checkpoint_copies_or_identity_rewrites':True}
        path=destination/'handoff.json'
        if path.exists():raise ValueError('Never overwrite a migration inventory')
        write(path,result);return result

def check_handoff(plan,binding):
    if sha(binding['path'])!=binding['sha256']:raise ValueError('Migration hash differs')
    receipt=read(binding['path']);parent_directory=Path(plan['parent_supplement_directory'])
    if receipt['plan_fingerprint']!=plan['fingerprint'] or sha(parent_directory/'terminal.json')!=receipt['parent_terminal_sha256']:
        raise ValueError('Implementation parent/drain differs')
    if sha(parent_directory/'operational_pause_receipt.json')!=receipt['parent_pause_receipt_sha256']:raise ValueError('Parent pause receipt differs')
    parent=read(parent_directory/'plan.json');actual_short={}
    for index in plan['group_indices']:
        result=parent_runner.read_group(parent_directory,parent,index)
        if result is not None:
            cp,rp=resolve_parent_group(parent_runner,parent_directory,parent,index)
            actual_short[str(index)]=pair_binding(cp,rp,'inherited_short')
    if actual_short!=receipt['inherited_short_groups']:raise ValueError('Parent short checkpoint inventory changed')
    actual=parent_old_inventory(plan)
    if actual!=receipt['saved_old_slots']:raise ValueError('Inherited original-H inventory changed')
    for index in range(len(plan['cases'])):
        if read_long(Path(binding['path']).parent,plan,index) is not None:
            if str(index) in actual:raise ValueError('New240 checkpoint duplicates saved1000 request')
            actual[str(index)]={'implementation_generation':plan['fingerprint'],'paths':240}
    return receipt,actual

def execute_task(task):
    kind,index=task
    state=WORK/'workers'/f'{os.getpid()}.json'
    write(state,{'pid':os.getpid(),'kind':kind,'index':index,'phase':'active','utc':now()})
    if kind=='short':
        result={'kind':kind,**evaluate_group(index)};write(state,{'pid':os.getpid(),'phase':'idle','utc':now()});return result
    result=evaluate_long(index);write(state,{'pid':os.getpid(),'phase':'idle','utc':now()});return result

def run(destination,receipt_path,smoke=False,workers=12):
    if not 1<=workers<=12:raise ValueError('Operational workers must be1..12')
    destination=Path(destination);plan=read(destination/'plan.json');panel=verify(plan);receipt=read(receipt_path);admission(plan,receipt,smoke)
    indices=plan['group_indices']
    if smoke:
        indices=[next(i for i in indices if panel['series'][plan['cases'][i]['series_index']]['panel_branch']==branch) for branch in ('assets','portfolios')]
        destination=destination/'smoke';destination.mkdir(exist_ok=False);write(destination/'plan.json',plan)
    with (destination/'controller.lock').open('a') as own,(O/'asset_v6_forecast_slots.lock').open('a') as slots:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(destination/'active.json',{'pid':os.getpid(),'status':'queued_for_shared_slots','utc':now()})
        fcntl.flock(slots,fcntl.LOCK_EX);verify(plan)
        handoff,saved=({}, {}) if smoke else check_handoff(plan,receipt['handoff'])
        todo=[];completed=0
        for index in indices:
            result=read_group(destination,plan,index)
            if result is None:todo.append(index)
            else:completed+=12
        catchup=set(handoff.get('catchup_group_indices',[]))
        phases=[[('short',i) for i in todo if i in catchup],[]]
        if smoke:phases=[[],[('short',i) for i in todo]]
        else:
            # Pair remaining old slots with their unique short group where possible;
            # Every remaining originalH task forecasts all3 models at240.
            old_tasks=[('old',i) for i in range(len(plan['cases'])) if str(i) not in saved]
            short_tasks=[('short',i) for i in todo if i not in catchup]
            from itertools import zip_longest
            phases[1]=[task for pair in zip_longest(old_tasks,short_tasks) for task in pair if task is not None]
        old_remaining=sum(kind=='old' for phase in phases for kind,_ in phase)
        old_growth=sum(plan['cases'][i]['origin']['max_horizon']*24+43000 for phase in phases for kind,i in phase if kind=='old')/(1024**3)
        subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib',str(growth(todo)+old_growth),
            '--artifact-root',str(O),'--budget-gib','24'],check=True)
        terminal={'status':'failed','pid':os.getpid(),'expected_short_forecasts':12*len(indices),'expected_new_old_forecasts':3*old_remaining,'expected_new_old_logical_tasks':old_remaining,'final_paths_for_new_work':240,'inner_paths':2,
            'plan_fingerprint':plan['fingerprint'],'utc':now()}
        errors=[];old_completed=0;write(destination/'active.json',{**terminal,'status':'running','workers':workers})
        try:
            with futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(destination),)) as pool:
                last_report=0.;last_storage=time.monotonic();remaining={task for phase in phases for task in phase}
                tasks=iter(phases[0]+phases[1]);pending={};exhausted=False;catchup_remaining=set(phases[0])
                while pending or not exhausted:
                    while len(pending)<2*workers and not exhausted:
                        try:task=next(tasks)
                        except StopIteration:exhausted=True;break
                        kind,index=task
                        if kind=='short':
                            cp,_,attempt=paths(destination,index)
                            if cp.with_suffix('.failure.json').exists():raise ValueError('Previously failed group requires review')
                            write(attempt,{'plan':plan['fingerprint'],'index':index,'submitted_utc':now()})
                        else:
                            cp,_,attempt=long_paths(destination,index)
                            write(attempt,{'identity':long_identity(plan,index),'submitted_utc':now()})
                        pending[pool.submit(execute_task,task)]=task
                    if not pending:break
                    done,_=futures.wait(pending,timeout=5,return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        task=pending.pop(future);remaining.remove(task);catchup_remaining.discard(task)
                        try:result=future.result()
                        except Exception as error:
                            result={'index':task[1],'successful_forecasts':0,'errors':[{'stage':'worker_failure','message':str(error)}]}
                            path=paths(destination,task[1])[0] if task[0]=='short' else long_paths(destination,task[1])[0]
                            write(path.with_suffix('.failure.json'),result)
                        if task[0]=='short':completed+=result['successful_forecasts']
                        else:old_completed+=result.get('successful_logical_tasks',0)
                        errors.extend(result['errors'])
                        if any(e.get('stage')=='checkpoint_io' for e in result['errors']):raise RuntimeError('Checkpoint IO/storage error: stop new submissions, drain pending, preserve complete temporary evidence')
                    if time.monotonic()-last_report>=5:
                        write(destination/'progress.json',{'utc':now(),'status':'running','phase':'short_catchup' if catchup_remaining else 'combined_expanded_panel',
                            'completed_short_forecasts':completed,'expected_short_forecasts':12*len(indices),'completed_new_old_forecasts':3*old_completed,'completed_new_old_logical_tasks':old_completed,'completed_new_old_baseline_forecasts':2*old_completed,'final_paths_for_new_work':240,
                            'native_failures':sum(e.get('stage') in ('native_forecast','forecast_generation') for e in errors),
                            'scoring_integrity_errors':sum(e.get('stage') in ('input_integrity','scoring_integrity') for e in errors),
                            'IO_errors':sum(e.get('stage') in ('checkpoint_io','worker_failure') for e in errors),
                            'reused_old_slots':len(saved),'active_workers':sum(w['phase']!='idle' for w in base.worker_states(destination)),'catchup_groups_remaining':len(catchup_remaining),'failed_forecasts':len(errors),'remaining_groups_or_old_requests':len(remaining),'projection':None,
                            'projection_scope':'No calibrated same-workload estimate; unknown work is not zero','official':False})
                        last_report=time.monotonic()
                    if time.monotonic()-last_storage>900:
                        reserve=growth([i for kind,i in remaining if kind=='short'])+sum(plan['cases'][i]['origin']['max_horizon']*24+43000 for kind,i in remaining if kind=='old')/(1024**3)
                        subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib',str(reserve),
                            '--artifact-root',str(O),'--budget-gib','24'],check=True);last_storage=time.monotonic()
            verify(plan)
            if not smoke:migration(plan,destination,refresh=True)
            complete=completed==12*len(indices) and old_completed==old_remaining and not errors
            terminal.update(status='complete' if complete else 'incomplete',successful_short_forecasts=completed,successful_new_old_forecasts=3*old_completed,successful_new_old_logical_tasks=old_completed,successful_new_old_baseline_forecasts=2*old_completed,errors=errors)
            if smoke:write(destination/'verification_receipt.json',{'passed':complete,'plan_fingerprint':plan['fingerprint'],
                'branches':['assets','portfolios'],'case_indices':indices,'horizons':list(HORIZONS),'forecasts':completed,'final_paths':240,'inner_paths':2,'errors':errors,'source_hashes':plan['source_hashes']})
        except BaseException as error:terminal['controller_error']=f'{type(error).__name__}: {error}';raise
        finally:
            terminal.update(utc=now(),workers_terminal=True);write(destination/'active.json',terminal);write(destination/'terminal.json',terminal)
    if not smoke and terminal['status']=='complete':
        try:
            combine(destination);terminal['reporter_status']='terminal'
        except Exception as error:
            terminal['reporter_status']='failed';terminal['reporter_error']=str(error)
            write(destination/'terminal.json',terminal);write(destination/'active.json',terminal);raise
        write(destination/'terminal.json',terminal);write(destination/'active.json',terminal)
    return terminal


def supported_quarters(plan,metadata):
    quarters={'assets':set(),'portfolios':set()}
    for case in plan['cases']:
        date=case['origin']['origin_date'];quarter=f"{date[:4]}Q{(int(date[5:7])-1)//3+1}"
        quarters[metadata[case['series_id']]['panel_branch']].add(quarter)
    unsupported=quarters['portfolios']-quarters['assets']
    if unsupported:raise ValueError(f'Portfolio quarters lack asset support: {sorted(unsupported)}')
    if not all(quarters.values()):raise ValueError('Full combined score requires both panel branches')
    return {key:sorted(value) for key,value in quarters.items()}

def canonical_score(reducer,metadata):
    rows=[]
    for (quarter,branch,asset_class,family,sid),(total,count) in reducer.leaves.items():
        month=(int(quarter[-1])-1)*3+1
        rows.append({'series_id':sid,'origin_date':f'{quarter[:4]}-{month:02d}-01','score':total/count})
    canonical=base.scoring.aggregate_records(rows,metadata)['score'];incremental=reducer.score()
    if not np.isclose(canonical,incremental,rtol=1e-12,atol=1e-12):raise ValueError('Canonical and incremental full hierarchy differ')
    return canonical,abs(canonical-incremental)

def combine(destination):
    destination=Path(destination);plan=read(destination/'plan.json');panel=verify(plan)
    terminal=read(destination/'terminal.json');old_directory=Path(plan['original_directory']);old_plan=read(old_directory/'plan.json')
    if terminal['status']!='complete':raise ValueError('Combined takeover must finish; old parent terminal status may be drained/incomplete')
    metadata={row['series_id']:row for row in panel['series']};quarter_support=supported_quarters(plan,metadata);reducers={name:base.Hierarchy(metadata) for name in MODEL_NAMES};seen=set();commit=hashlib.sha256()
    baseline={};group_indices=set(plan['group_indices']);preserved=migration(plan,destination)['saved_old_slots'];path_counts={name:{'240':0,'1000':0} for name in MODEL_NAMES}
    with sqlite3.connect(Path(plan['baseline_score_index']['path']).as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        for model,i,j,task,protocol,panel_id,score in db.execute('SELECT model_id,series_index,origin_index,task_key,protocol_id,panel_fingerprint,score_v7 FROM task_scores_v7'):
            if model not in (WR,WOR) or protocol!=base.scoring.SCORE_PROTOCOL_ID or panel_id!=plan['panel_fingerprint']:raise ValueError('Baseline identity differs')
            baseline[model,i,j]=(json.loads(task),score)
    if len(baseline)!=2*len(plan['cases']):raise ValueError('Original baseline coverage differs')
    def add(name,case,H,score,count):
        key=(name,case['series_id'],case['origin']['position'],H,case['origin']['origin_label'] if H not in HORIZONS else 'unique_short')
        if key in seen:raise ValueError('Duplicate requestedH task')
        if not np.isfinite(score) or score<0:raise ValueError('Invalid request score')
        if count not in (240,1000):raise ValueError('Undeclared final pathcount')
        path_counts[name][str(count)]+=1
        seen.add(key);reducers[name].add({'series_id':case['series_id'],'origin_date':case['origin']['origin_date'],'score':score})
    for index,case in enumerate(plan['cases']):
        generated=read_long(destination,plan,index)
        if generated is not None:
            if str(index) in preserved:raise ValueError('Saved1000 request was recomputed')
            losses,record=generated;row=panel['series'][case['series_index']]
            dates,returns,_=base.api.load_series(V/'inputs',row,verify_files=False);stop=case['origin']['position']+1;H=case['origin']['max_horizon']
            for name in MODEL_NAMES:
                score=base.scoring.origin_score(losses[name],returns[:stop],case['origin']['origin_date'],dates[stop:stop+H])
                if score!=record['results'][name]['score']:raise ValueError('New-generation paired oldH score replay differs')
                add(name,case,H,score,240)
        else:
            bound=preserved.get(str(index))
            if bound is None or bound['paths']!=1000 or sha(bound['checkpoint_path'])!=bound['checkpoint_sha256'] or sha(bound['record_path'])!=bound['record_sha256']:raise ValueError('Missing/changed inherited1000 originalH request')
            if bound['kind']=='parent_long':
                original=parent_runner.read_long(plan['parent_supplement_directory'],parent_plan(plan),index)
                if original is None:raise ValueError('Missing parent originalH request')
                score=original[1]['score']
            elif bound['kind']=='original':
                original=base.verified_checkpoint(old_directory,old_plan,'endpoint',index)
                if original is None:raise ValueError('Missing original endpoint request')
                score=original['score']
            else:raise ValueError('Unknown inherited originalH implementation')
            add('endpoint',case,case['origin']['max_horizon'],score,1000)
            for name in (WR,WOR):
                task,score=baseline[name,case['series_index'],case['origin_index']]
                if task!=[case['series_id'],case['origin']]:raise ValueError('Original baseline task differs')
                add(name,case,case['origin']['max_horizon'],score,1000)
        if index not in group_indices:continue
        loss,group=read_group(destination,plan,index)
        count=group['identity']['paths']
        row=panel['series'][case['series_index']];dates,returns,_=base.api.load_series(V/'inputs',row,verify_files=False);stop=case['origin']['position']+1
        for item in group['results']:
            name,H=item['model'],item['H'];score=base.scoring.origin_score(loss[f'{name}_{H}'],returns[:stop],case['origin']['origin_date'],dates[stop:stop+H])
            if score!=item['score']:raise ValueError('Supplement score replay differs')
            if item.get('final_paths',count)!=count:raise ValueError('Short model pathcount differs from paired request')
            add(name,case,H,score,count)
        cp=paths(destination,index)[0]
        if not cp.exists():cp,_=resolve_parent_group(parent_runner,plan['parent_supplement_directory'],parent_plan(plan),index)
        commit.update(f'{index}:{sha(cp)}\n'.encode())
    if len(seen)!=3*plan['combined_requests_per_model']:raise ValueError('Combined distinct request coverage differs')
    if any(sum(key[0]==name for key in seen)!=plan['combined_requests_per_model'] for name in MODEL_NAMES):raise ValueError('Per-model combined request coverage differs')
    if any(counts!=path_counts['endpoint'] for counts in path_counts.values()):raise ValueError('Paired pathcount coverage differs')
    reconciled={name:canonical_score(reducer,metadata) for name,reducer in reducers.items()}
    scores={name:value[0] for name,value in reconciled.items()}
    result={'complete':True,'requests_per_model':plan['combined_requests_per_model'],'distinct_tasks':len(seen),'models':scores,
        'fixed_reference_model_id':plan['fixed_reference_model_id'],'score_protocol':base.scoring.SCORE_PROTOCOL_ID,'paired_final_path_counts':path_counts,'mixed_count_rule':plan['mixed_count_rule'],'all_requests_same_count_paired':True,
        'implementation_generation':plan['implementation_id'],'scientific_parent_fingerprint':plan['parent_supplement_fingerprint'],'supplement_checkpoint_commitment':commit.hexdigest(),'quarter_support':quarter_support,'strict_supported_quarters':True,'official':False,'requires_independent_review':True,
        'canonical_hierarchy_reconciliation':{'absolute_differences':{name:value[1] for name,value in reconciled.items()},'rtol':1e-12,'atol':1e-12,'unit':'one mean per exact series-quarter leaf'},'expanded_coverage_version':'original64049_plus_unique_prefix_H1_5_10_21_v1','short_alias_policy':plan['short_score_alias_policy'],'weighting':'Equal requests within series-quarter; equal native leads within each request after trainSD/sqrt(lead); unchanged higher hierarchy'}
    write(destination/'combined_score_review.json',result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','verify','handoff','smoke','run','combine'])
    parser.add_argument('--directory',type=Path,default=D);parser.add_argument('--original-directory',type=Path,default=O/'short_horizon_optimized_v2')
    parser.add_argument('--root-admission',type=Path);parser.add_argument('--workers',type=int,default=12);parser.add_argument('--authorization',type=Path,default=O/'final240_authorization_v3.json');args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.original_directory,args.directory,args.authorization)
    elif args.command=='verify':result={'verified':True,'series':len(verify(read(args.directory/'plan.json'))['series'])}
    elif args.command=='handoff':result=make_handoff(args.directory)
    elif args.command=='combine':result=combine(args.directory)
    else:
        if args.root_admission is None:parser.error('Explicit root admission required')
        result=run(args.directory,args.root_admission,args.command=='smoke',args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_hashes')},sort_keys=True))
if __name__=='__main__':main()
