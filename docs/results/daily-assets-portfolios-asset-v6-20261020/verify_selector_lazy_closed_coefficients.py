"""Read-only numerical study of completed historical coefficient checkpoints."""
from pathlib import Path
import argparse,datetime,hashlib,json,statistics,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
from candidates.asset_selector_lazy_coefficients.solver import fit
from candidates.asset_selector_universe_certified.energy import fit as dense_fit
from candidates.asset_selector_stacking.energy import even_allocation

D=V/'candidates/asset_selector_path_objective/development1000'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=O/'selector_lazy_closed_coefficients.json');args=parser.parse_args()
files=sorted((D/'coefficients').glob('*.npz'));records=[]
for p in files:
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 with np.load(p,allow_pickle=False) as data:
  for mode in ('terminal','path'):
   A=data['A_'+mode];B=data['B_'+mode];seen=[];started=time.process_time()
   def column(j):seen.append(j);return B[:,j]
   result=fit(A,np.diag(B),column);cpu=time.process_time()-started;dense=dense_fit(A,B)
   difference=abs(result['objective']-dense['objective']);bound=result['KKT_gap']+dense['KKT_gap']+128*np.finfo(float).eps*result['scale']
   assert difference<=bound and len(seen)==len(set(seen))
   records.append({'coefficient_checkpoint':str(p.relative_to(O)),'checkpoint_sha256':digest,'criterion':mode,'options':len(A),'requested_pairwise_columns':len(seen),'selected_nonzero_weights':int(np.count_nonzero(result['weights'])),'columns':seen,'objective':result['objective'],'dense_objective':dense['objective'],'objective_difference':difference,'certified_difference_upper_bound':bound,'scaled_KKT_gap':result['scaled_KKT_gap'],'tolerance':result['tolerance'],'weights_exactly_equal':bool(np.array_equal(result['weights'],dense['weights'])),'outer1000_path_allocation_exactly_equal':bool(np.array_equal(even_allocation(result['weights'],1000),even_allocation(dense['weights'],1000))),'solver_cpu_seconds':cpu})
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'closed_checkpoints_read':len(files),'coefficient_objectives_verified':len(records),'all_original_coefficient_checkpoints_unchanged':True,'global_all_option_certificates_verified':True,'requested_columns_minimum':min(r['requested_pairwise_columns']for r in records),'requested_columns_median':statistics.median(r['requested_pairwise_columns']for r in records),'requested_columns_maximum':max(r['requested_pairwise_columns']for r in records),'allocation_mismatches_against_dense':sum(not r['outer1000_path_allocation_exactly_equal']for r in records),'records':records,'scope':'Precomputed history-only coefficient matrices; no historical path generation, outer forecasting or runtime speedup established','not_bit_exact_solver_replacement':True,'no_full_request_speedup_claim':True,'no_accuracy_or_full_panel_admission':True,'new_scientific_workers':0,'integration_requires_column_generation_recompute_or_bounded_storage_cost_measurement':True}
summary['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
args.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items()if k!='records'},indent=2),flush=True)
