"""Training-only mechanism checks; no panel outcomes or model selection."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,datetime,concurrent.futures,multiprocessing
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from asset_paths import JointAssetBootstrap

def case(args):
 n,assets,replacement=args;rng=np.random.default_rng(391+n+assets);x=rng.normal(.0001,.012,(n,assets));weights=np.ones(assets)/assets
 model=JointAssetBootstrap(replacement=replacement);terminals={};horizon=3*n
 for start,terminal in model.terminal_chunks(x,weights,horizon,1000,511,width=n):
  terminals[(start+terminal.shape[1])//n]=terminal[:,-1].copy()
 daily=np.log1p(np.sum(np.expm1(x)*weights,axis=1));expected=float(daily.sum())
 result={'history_rows':n,'assets':assets,'replacement':replacement,'paths':1000,'worker_pid':os.getpid(),'terminal_sd_at_complete_cycles':{str(k):float(np.std(v,ddof=1))for k,v in terminals.items()},'max_error_vs_exact_cycle_total':{str(k):float(np.max(np.abs(v-k*expected)))for k,v in terminals.items()}}
 if not replacement:assert max(result['max_error_vs_exact_cycle_total'].values())<1e-12
 else:assert min(result['terminal_sd_at_complete_cycles'].values())>0
 return result

def main():
 cases=[(n,a,r)for n in [24,53,101]for a in [1,3]for r in [True,False]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'))as pool:records=list(pool.map(case,cases))
 # Long-run row exposure for uniform valid nonwrapping starts, conditional on
 # a new block. Expected visits per block / expected length is the renewal share.
 exposures=[]
 for n in [24,53,101]:
  weights=np.zeros(n)
  for length in range(1,n+1):
   for start in range(n-length+1):weights[start:start+length]+=1/(n*(n-length+1))
  weights/=(n+1)/2;assert abs(weights.sum()-1)<1e-12
  exposures.append({'n':n,'edge_exposure_vs_uniform':float(weights[0]*n),'center_exposure_vs_uniform':float(weights[n//2]*n),'definition':'renewal long-run row occupancy, not a finite forecast score or observed error'})
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_only_synthetic_mechanism_check':True,'forecast_cases':len(records),'workers':12,'panel_outcomes_or_scores_used':False,'records':records,'nonwrapping_row_exposure':exposures,'findings':['Without replacement complete cycles have deterministic cumulative constant-weight portfolio log return conditional on the finite history; shuffling changes order but not this terminal total. This alone does not prove worse CRPS.','With-replacement nonwrapping blocks give nonuniform long-run row exposure; circular starts would remove this edge effect. This alone does not prove an accuracy improvement.'],'next_tests_pending_full_baseline_diagnosis':['Training-fitted joint mean uncertainty for long-horizon finite-history uncertainty.','Circular joint row sampling as a separate edge-exposure contrast if supported by new baseline results.'],'no_candidate_admitted':True,'no_reference_selected':True}
 (O/'asset_block_mechanism_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='records'}),flush=True)
if __name__=='__main__':main()
