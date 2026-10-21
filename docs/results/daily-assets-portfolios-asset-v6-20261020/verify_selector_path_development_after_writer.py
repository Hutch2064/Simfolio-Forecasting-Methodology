"""Wait on scientific ownership, then audit closed development without promotion."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys
O=Path(__file__).resolve().parent;D=O/'asset_level_v6/candidates/asset_selector_path_objective/development1000';source=O/'verify_selector_path_development355.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();expected=sha(source)
def write(status,**extra):
 p=O/'selector_path_development_audit_active.json';q=p.with_suffix('.tmp');q.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':status,'source_sha256':expected,**extra},indent=2)+'\n');q.replace(p)
write('waiting_for_shared_scientific_writer')
try:
 with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert sha(source)==expected
  assert json.loads((D/'active.json').read_text())['status']=='terminal','Writer released pool without terminal stage; inspect, do not rerun'
  write('auditing_closed_development');subprocess.run([sys.executable,str(source)],cwd=O,check=True)
 write('verified_terminal_no_promotion',receipt=str(D/'verification_receipt.json'))
except Exception as e:
 write('audit_requires_inspection',error_type=type(e).__name__,error=str(e));raise
