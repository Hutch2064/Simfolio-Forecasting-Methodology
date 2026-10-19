"""All 80 portfolios, 51 origins: complete trace/path parity and unchanged CRPS."""
import hashlib
import json
import multiprocessing
import os
import platform
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optimized_mcmc as opt
from runtime_setup import DATA, initialize

from simfolio_forecasting_methodology.evaluation import CellAccumulator
from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.runner import evaluate_origin_task, task_identity

HERE=Path(__file__).resolve().parent
REFERENCE=Path(os.environ['SIMFOLIO_MCMC_REFERENCE'])
OUT=Path(os.environ.get('SIMFOLIO_MCMC_OUTPUT','optimized-mcmc-full-parity'))
MODEL=None

def setup():
 global MODEL
 initialize(True);opt.install(True);MODEL=opt.OptimizedMCMC()

def run(index,task):
 start=time.perf_counter();before=len(opt.CHAIN_CHECKS)
 losses=evaluate_origin_task(MODEL,task,simulations=240)
 reference=np.load(REFERENCE/f'task-{index:04d}-model-1.npz')['losses']
 assert losses.tobytes()==reference.tobytes(),('scored losses',index,float(np.max(abs(losses-reference))))
 return index,losses,time.perf_counter()-start,opt.PATH_CHECKS[-1],opt.CHAIN_CHECKS[before:]

def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)

def main():
 plan=build_experiment_plan(DATA);assert len(plan.portfolios)==80 and plan.task_count==4080 and plan.cell_capacity==701280
 OUT.mkdir(exist_ok=True)
 sources=list(HERE.glob('*.py'))+list((HERE.parents[1]/'src').rglob('*.py'))
 model=opt.OptimizedMCMC();manifest={'scope': 'full_80_portfolio_51_origin_canonical_panel','reference_manifest_sha256': json.loads((REFERENCE/'results.json').read_text())['manifest_sha256'],'research_revision': '40928c1364630dc736e8681b508766a0f0a631a2','python': platform.python_version(),'numpy': np.__version__,'simulations': 240,'model': asdict(model),'chains': 2,'burn_per_chain': 1024,'kept_per_chain': 2048,'posterior_nodes': 16,'scoring': 'unchanged public evaluator and CellAccumulator','source_sha256': {str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'tasks': [asdict(task_identity(t,model.model_id,240)) for t in plan.tasks],'workers': 4,'fit_workers_per_process': "automatic 2/6, capped by available CPUs and chain lanes; maximum twelve active numerical threads per process; four bounded processes",'parity': 'all sampled parameter draws and acceptance rates, complete multiscale fit dictionaries, dependence posterior arrays, all sorted marginal values, mapped asset paths, daily portfolio paths, and horizon loss vectors'}
 digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
 if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
 else:atomic(OUT/'manifest.json',manifest)
 done_indices={int(p.stem.split('-')[1]) for p in OUT.glob('task-*.npz') if (OUT/(p.stem+'-parity.json')).exists()};todo=iter(i for i in range(4080) if i not in done_indices);completed=len(done_indices);start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=setup) as pool:
  futures={}
  def submit():
   try:i=next(todo)
   except StopIteration:return
   futures[pool.submit(run,i,plan.tasks[i])]=i
  for _ in range(4):submit()
  while futures:
   ready,_=wait(futures,return_when=FIRST_COMPLETED)
   for f in ready:
    index,losses,seconds,paths,chains=f.result();assert futures.pop(f)==index
    file=OUT/f'task-{index:04d}.npz';temp=file.with_suffix('.tmp.npz');np.savez_compressed(temp,losses=losses,validation_seconds=seconds,manifest=digest);temp.replace(file)
    atomic(OUT/f'task-{index:04d}-parity.json',{'task': index,'paths': paths,'new_chain_checks': chains,'all_checks_byte_exact': True})
    completed+=1;atomic(OUT/'progress.json',{'completed': completed,'required': 4080,'elapsed_seconds': time.perf_counter()-start});print(json.dumps({'completed': completed,'required': 4080}),flush=True);submit()
 accumulator=CellAccumulator();counts={};asset_values=daily_values=0;chains={};hashes={}
 for i,t in enumerate(plan.tasks):
  file=OUT/f'task-{i:04d}.npz';r=np.load(file);assert str(r['manifest'])==digest;accumulator.add_vector(t.portfolio_id,r['losses']);hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
  for h in range(1,t.horizon_days+1):counts[(t.portfolio_id,h)]=counts.get((t.portfolio_id,h),0)+1
  parity=json.loads((OUT/f'task-{i:04d}-parity.json').read_text());assert parity['all_checks_byte_exact'];asset_values+=parity['paths']['asset_values'];daily_values+=parity['paths']['daily_values']
  for chain in parity['new_chain_checks']:
   key=(chain['observations_sha256'],chain['seed']);old=chains.get(key)
   if old is not None:assert old==chain
   chains[key]=chain
 score=accumulator.fixed_denominator_score(expected_cells=701280,expected_tasks=4080,completed_tasks=4080,expected_origin_counts=counts)
 expected=json.loads((REFERENCE/'results.json').read_text())['scores'][model.model_id]['crps'];assert score==expected
 atomic(OUT/'results.json',{'scope': manifest['scope'],'portfolios': 80,'origins_per_portfolio': 51,'tasks': 4080,'cells': len(counts),'crps': score,'all_complete_outputs_byte_exact': True,'all_denominator_gates_passed': True,'asset_path_values_checked': asset_values,'daily_portfolio_values_checked': daily_values,'unique_chain_traces_checked': len(chains),'manifest_sha256': digest,'validation_elapsed_seconds': time.perf_counter()-start,'runtime_claim': 'verification timing includes original computation and comparisons; use separate matched benchmarks'})
 atomic(OUT/'unique-chain-checks.json',list(chains.values()));atomic(OUT/'checkpoint-hashes.json',hashes);print((OUT/'results.json').read_text(),flush=True)
if __name__=='__main__':main()
