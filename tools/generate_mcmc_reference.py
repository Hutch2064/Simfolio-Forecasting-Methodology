"""Full canonical paired runner with exact evaluator/denominator gates.
Uses bounded spawned processes and training/config-only memoization.
"""
import argparse
import hashlib
import json
import multiprocessing
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"mcmc_runtime"))
from predictive_sv_candidates import fast_rebalanced, fast_uniforms
from research_jit_pilot import make_accelerators

from simfolio_forecasting_methodology.evaluation import CellAccumulator
from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
    FilteredInnovationFixedMeanMomentSV,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg
from simfolio_forecasting_methodology.runner import evaluate_origin_task, task_identity

HERE=Path(__file__).resolve().parent
import os

DATA=Path(os.environ.get('SIMFOLIO_OOS_DATA',str(Path.cwd()/'.simfolio-oos-data')))
MODELS=None

def initialize():
    global MODELS
    from parameter_mcmc_tuned import CANDIDATES, install
    install();MODELS=(FilteredInnovationFixedMeanMomentSV(),CANDIDATES[0])
    _,_,dlm,sv=make_accelerators();bd._dlm_ar1_loglik=dlm;bd._sv_kalman_filter=sv
    dg.simulate_future_gaussian_uniforms=fast_uniforms;dg.rebalanced_portfolio_log_paths=fast_rebalanced
    from compiled_smoothers import make
    rts,ewma=make();rts(np.array([.1,.2]),.1,.9,.2);ewma(np.array([.1,.2]),.1)
    bd._sv_kalman_rts_smoother_mean=rts;bd._bdes_ewma=ewma
    original_fit=bd.fit_bdes_fastmap
    @lru_cache(maxsize=4096)
    def cached(data,filtered,fixed):return original_fit(np.frombuffer(data,np.float64),filtered_innovations=filtered,fixed_mean=fixed)
    def fit(x,*,filtered_innovations=False,fixed_mean=False):return cached(np.asarray(x,np.float64).tobytes(),filtered_innovations,fixed_mean)
    bd.fit_bdes_fastmap=fit

def task_run(index,task):
    result=[]
    for model in MODELS:
        start=time.perf_counter();loss=evaluate_origin_task(model,task,simulations=240);result.append((loss,time.perf_counter()-start))
    return index,result

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--output',required=True);args=parser.parse_args();assert 1<=args.workers<=4
    plan=build_experiment_plan(DATA);assert plan.task_count==4080 and plan.cell_capacity==701280
    tasks=list(plan.tasks)
    from parameter_mcmc_tuned import CANDIDATES
    models=(FilteredInnovationFixedMeanMomentSV(),CANDIDATES[0]);out=Path(args.output);out.mkdir(exist_ok=True)
    sources=[Path(__file__)]+list((HERE/'mcmc_runtime').glob('*.py'))+list((HERE.parent/'src').rglob('*.py'))
    manifest={"scope": 'full_canonical',"simulations": 240,"models": [asdict(m) for m in models],"research_revision": '40928c1364630dc736e8681b508766a0f0a631a2',"python": platform.python_version(),"numpy": np.__version__,"workers": args.workers,"scoring": 'unchanged public evaluate_origin_task and CellAccumulator, full origin-count denominator gates',"sources": {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},"tasks": [[asdict(task_identity(t,m.model_id,240)) for m in models] for t in tasks]}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    else:atomic(out/'manifest.json',manifest)
    pending=[];hashes={}
    for i,t in enumerate(tasks):
        paths=[out/f'task-{i:04d}-model-{j}.npz' for j in range(2)]
        if all(p.exists() for p in paths):
            for p in paths:
                r=np.load(p);assert str(r['manifest'])==digest;hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        else:assert not any(p.exists() for p in paths);pending.append(i)
    completed=len(tasks)-len(pending);it=iter(pending);wall=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures={}
        def submit():
            try:i=next(it)
            except StopIteration:return
            futures[pool.submit(task_run,i,tasks[i])]=i
        for _ in range(args.workers):submit()
        while futures:
            done,_=wait(futures,return_when=FIRST_COMPLETED)
            for f in done:
                i=futures.pop(f);returned,result=f.result();assert i==returned
                for j,(loss,seconds) in enumerate(result):
                    assert loss.size==tasks[i].horizon_days and np.all(np.isfinite(loss))
                    p=out/f'task-{i:04d}-model-{j}.npz';temp=p.with_suffix('.tmp.npz');np.savez_compressed(temp,losses=loss,seconds=seconds,manifest=digest);temp.replace(p);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
                completed+=1;atomic(out/'progress.json',{"completed": completed,"required": len(tasks),"manifest_sha256": digest,"elapsed_seconds": time.perf_counter()-wall});print(json.dumps({"completed": completed,"required": len(tasks)}),flush=True);submit()
    counts={};acc=[CellAccumulator(),CellAccumulator()];times=[0.,0.]
    for i,t in enumerate(tasks):
        for h in range(1,t.horizon_days+1):counts[(t.portfolio_id,h)]=counts.get((t.portfolio_id,h),0)+1
        for j in range(2):
            r=np.load(out/f'task-{i:04d}-model-{j}.npz');assert str(r['manifest'])==digest;acc[j].add_vector(t.portfolio_id,r['losses']);times[j]+=float(r['seconds'])
    assert len(counts)==701280
    keys=sorted(counts);arrays={"portfolio": np.asarray([k[0] for k in keys]),"horizon": np.asarray([k[1] for k in keys]),"origin_count": np.asarray([counts[k] for k in keys])};scores={}
    for j,m in enumerate(models):
        value=acc[j].fixed_denominator_score(expected_cells=701280,expected_tasks=len(tasks),completed_tasks=completed,expected_origin_counts=counts);means=acc[j].cell_means();arrays[f'model_{j}']=np.array([means[k] for k in keys]);scores[m.model_id]={"crps": value,"tasks": completed,"cells": 701280,"summed_worker_seconds": times[j]}
    np.savez_compressed(out/'paired-cells.npz',**arrays);atomic(out/'checkpoint-hashes.json',hashes);atomic(out/'results.json',{"scope": manifest['scope'],"scores": scores,"manifest_sha256": digest,"elapsed_seconds": time.perf_counter()-wall,"all_denominator_gates_passed": True});print(json.dumps(scores,indent=2),flush=True)
if __name__=='__main__':main()
