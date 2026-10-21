"""Wait on verified current handles; admit only the qualified unchanged method."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;PRE=O/'asset_level_v6/candidates/asset_selector_path_objective/development1000';driver=O/'run_selector_sweep_validation_reserved.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();expected=sha(driver)
read=lambda p:json.loads(p.read_text())
writer=read(PRE/'active.json')['pid'];auditor=read(O/'selector_path_development_audit_active.json')['pid'];status=O/'selector_sweep_handoff_active.json'
def live(pid):
 r=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);return r.returncode==0 and r.stdout.strip()and not r.stdout.strip().startswith('Z')
def write(stage,**extra):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':stage,'waiting_writer_pid':writer,'waiting_auditor_pid':auditor,'driver_sha256':expected,**extra},indent=2)+'\n');p.replace(status)
try:
 while not(PRE/'verification_receipt.json').exists():
  w=bool(live(writer));a=bool(live(auditor));write('waiting_for_development_and_independent_audit',actual_writer_live=w,actual_auditor_live=a)
  if not w and not a:raise RuntimeError('Both expected handles ended without a verification receipt; inspect, do not restart')
  time.sleep(30)
 audit=read(PRE/'verification_receipt.json');assert audit['verified']and audit['summary_sha256']==sha(PRE/'summary.json')and sha(driver)==expected
 summary=read(PRE/'summary.json');path=next(v for v in summary['models'].values()if v['config']['criterion']=='path')
 if not path['beats_matched_terminal_pool']:
  write('not_admitted_development_did_not_improve',goal_complete=False)
 else:
  assert read(O/'goal_progress.json')['goal_status']=='active';r=read(O/'selector_coefficient_sweep_public_recovery.json');assert r['remote_main_verified']and r['recovered_tests_passed']==6
  write('launching_qualified_replay_and_reserved');subprocess.run([sys.executable,str(driver)],cwd=O,check=True);write('stage_terminal_requires_independent_audit',goal_complete=False)
except Exception as e:
 write('handoff_requires_inspection',error_type=type(e).__name__,error=str(e));raise
