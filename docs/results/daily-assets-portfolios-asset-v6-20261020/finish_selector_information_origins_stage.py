"""Observe the existing scientific owner, then run the coverage-corrected audit."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;D=O/'asset_level_v6/candidates/asset_selector_information_origins/study1000'
for key in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
read=lambda p:json.loads(p.read_text());now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,data):
 temporary=p.with_name(p.name+'.owner.tmp');temporary.write_text(json.dumps(data,separators=(',',':'))+'\n');temporary.replace(p)
owner=read(D/'active.json')['pid'];write(O/'selector_information_origins_stage_active.json',{'pid':os.getpid(),'status':'observing_existing_scientific_owner','scientific_owner_pid':owner,'scientific_job_not_restarted':True,'utc':now()})
try:
 while True:
  command=subprocess.run(['ps','-p',str(owner),'-o','args='],text=True,stdout=subprocess.PIPE).stdout
  if 'run_selector_information_origins355.py'not in command:break
  time.sleep(2)
 assert(D/'summary.json').exists(),'Scientific owner exited without a complete summary; do not restart automatically'
 with(O/'selector_information_origins_independent_audit_v2.log').open('w')as log:subprocess.run([sys.executable,str(O/'verify_selector_information_origins355_v2.py')],cwd=O,stdout=log,stderr=subprocess.STDOUT,check=True)
 audit=read(D/'verification_receipt.json');assert audit['verified']
 g=read(O/'goal_progress.json');assert g['goal_status']=='active';g['active_pilot']['status']='completed_independently_verified';g['stage']='selector_information_origins_evidence_review';g['selector_information_origins']={'verification':str(D/'verification_receipt.json'),'forecasts_verified':audit['candidate_forecasts_verified'],'known_control_loss_vectors_byte_exact':audit['original_control_loss_vectors_byte_exact'],'next':'Review355matched scores and measured fitting cost; continue supported component/origin interactions and methodology alternatives. Paired8request canonical scores unavailable if quarter support is incomplete; do not replace the main score.'};g['updated_at']=now();write(O/'goal_progress.json',g)
 write(O/'selector_information_origins_stage_active.json',{'pid':os.getpid(),'status':'terminal_verified','utc':now(),'candidate_forecasts_verified':audit['candidate_forecasts_verified'],'goal_remains_active':True,'scientific_job_not_restarted':True})
except Exception as error:
 write(O/'selector_information_origins_stage_active.json',{'pid':os.getpid(),'status':'verification_failed','utc':now(),'error_type':type(error).__name__,'message':str(error),'goal_not_completed':True,'scientific_job_not_restarted':True});raise
