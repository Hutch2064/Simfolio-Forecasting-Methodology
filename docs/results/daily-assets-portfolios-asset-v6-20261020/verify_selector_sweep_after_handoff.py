"""Verify next-stage completion without admission, restart or promotion."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;D=O/'asset_level_v6/candidates/asset_selector_coefficient_sweep/validation_and_reserved';source=O/'verify_selector_sweep_reserved.py';expected=hashlib.sha256(source.read_bytes()).hexdigest();owner=json.loads((O/'selector_sweep_handoff_active.json').read_text())['pid'];out=O/'selector_sweep_audit_active.json'
def live(pid):
 p=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);return p.returncode==0 and p.stdout.strip()and not p.stdout.strip().startswith('Z')
def write(status,**extra):
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':status,'waiting_handoff_pid':owner,'verifier_sha256':expected,**extra},indent=2)+'\n');tmp.replace(out)
try:
 while live(owner):write('waiting_for_qualified_scientific_handoff',actual_owner_live=True);time.sleep(30)
 if not D.exists():
  status=json.loads((O/'selector_sweep_handoff_active.json').read_text())['status'];assert status=='not_admitted_development_did_not_improve',status;write('no_new_stage_admitted',goal_complete=False)
 else:
  a=json.loads((D/'active.json').read_text())
  while a['status']!='terminal'and live(a['pid']):write('waiting_for_actual_scientific_writer',writer_pid=a['pid']);time.sleep(30);a=json.loads((D/'active.json').read_text())
  assert a['status']=='terminal','Missing writer without terminal state; inspect instead of restarting'
  with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);assert hashlib.sha256(source.read_bytes()).hexdigest()==expected;write('auditing_closed_replay_and_reserved');subprocess.run([sys.executable,str(source)],cwd=O,check=True)
  write('verified_terminal_no_promotion',receipt=str(D/'verification_receipt.json'),goal_complete=False)
except Exception as e:write('audit_requires_inspection',error_type=type(e).__name__,error=str(e));raise
