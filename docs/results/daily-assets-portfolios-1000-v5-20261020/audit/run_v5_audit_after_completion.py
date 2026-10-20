"""Read-only full-baseline audit handoff; no forecast workers or admission."""
from pathlib import Path
import datetime,json,os,subprocess,time,hashlib
O=Path(__file__).resolve().parent;V=O/'simulation_1000_v5';A=O/'v5_audit_active.json'
read=lambda p:json.loads(Path(p).read_text())
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(**d):
 p=A.with_suffix('.tmp');p.write_text(json.dumps({'pid':os.getpid(),'utc':now(),**d},indent=2)+'\n');p.replace(A)
try:
 assert not A.exists(),'Inspect existing audit instead of duplicating it'
 expected=read(O/'v5_independent_audit_preflight.json')['audit_source_sha256']
 assert hashlib.sha256((O/'verify_v5_full_suite.py').read_bytes()).hexdigest()==expected
 write(status='waiting_for_complete_retained_baseline_report',forecast_workers_used=0)
 while True:
  if read(O/'goal_progress.json')['status']!='active' or read(V/'control.json').get('stop'):
   raise RuntimeError('Authorization/control changed; no automatic audit launch')
  a=read(V/'controller_active.json');s=read(V/'results/status.json')
  if a['status']=='terminal':
   assert s['completed_active_tasks']==s['expected_active_tasks']==256196 and s['failed_active_tasks']==0
   r=read(V/'results/scoreboard.json')
   assert r['scoring_integrity_errors']==r['native_failures']==0
   if not r['reference_selection_preliminary'] and len(r['complete_coverage_leaderboard'])==4:break
  else:os.kill(a['pid'],0)
  time.sleep(15)
 write(status='running_independent_raw_loss_score_audit',forecast_workers_used=0)
 assert hashlib.sha256((O/'verify_v5_full_suite.py').read_bytes()).hexdigest()==expected
 for name,logname in [('verify_v5_full_suite.py','v5_full_suite_independent_audit.log'),('audit_v5_stable_parity.py','v5_stable_parity_audit.log')]:
  with (O/logname).open('ab') as log:
   r=subprocess.run(['/Users/aidanhutchison/simfolio/tmp/catalogue52-20261018/venv/bin/python','-u',name],cwd=O,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{name} failed: inspect saved log, no automatic restart')
 assert read(O/'v5_full_suite_independent_audit.json')['total_checkpoints_verified']==256196
 assert read(O/'v5_stable_parity_audit.json')['full_panel_audit_complete']
 write(status='terminal_verified',all_retained_checkpoints_audited=256196,no_model_promotion_or_goal_completion=True)
except Exception as e:
 write(status='terminal_needs_review',exception=type(e).__name__,message=str(e),no_automatic_restart=True)
 raise
