"""Own one finite shared-worker stage and its independent saved-loss audit."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
O=Path(__file__).resolve().parent;D=O/'asset_level_v6/candidates/asset_selector_design/study1000'
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
read=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,separators=(',',':'))+'\n')
write(O/'selector_design_stage_active.json',{'pid':os.getpid(),'status':'running','utc':now()})
try:
 for source,log in [('run_selector_design72.py','selector_design_live.log'),('verify_selector_design72.py','selector_design_independent_audit.log')]:
  with (O/log).open('w')as out:subprocess.run([sys.executable,str(O/source)]+(['--run']if source.startswith('run_')else[]),cwd=O,stdout=out,stderr=subprocess.STDOUT,check=True)
 audit=read(D/'verification_receipt.json');assert audit['verified']
 g=read(O/'goal_progress.json');assert g['goal_status']=='active';g['active_pilot']['status']='completed_independently_verified';g['stage']='selector_menu_replication_evidence_review';g['selector_design']={'verification':str(D/'verification_receipt.json'),'forecasts_verified':audit['candidate_forecasts_verified'],'original_controls_byte_exact':audit['original_control_loss_vectors_byte_exact'],'source_hashes_verified':True,'working_best_not_automatically_promoted':True,'next':'Review precision, validation support and pooling contrasts, confirm promising changes beyond these reused development cases, then test supported interactions with flattened/PPW menu. Stopping condition not established.'};g['updated_at']=now();write(O/'goal_progress.json',g)
 write(O/'selector_design_stage_active.json',{'pid':os.getpid(),'status':'terminal_verified','utc':now(),'candidate_forecasts_verified':audit['candidate_forecasts_verified'],'goal_remains_active':True})
except Exception as error:
 write(O/'selector_design_stage_active.json',{'pid':os.getpid(),'status':'failed','utc':now(),'error_type':type(error).__name__,'message':str(error),'goal_not_completed':True});raise
