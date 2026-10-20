"""Guarded handoff to the frozen GJR pilot; no admission/restart of baseline jobs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import datetime,json,subprocess,time,sys,hashlib
O=Path(__file__).resolve().parent
V=O/'simulation_1000_v5'
P=V/'research_gjr'
A=O/'gjr_handoff_active.json'
PY='/Users/aidanhutchison/simfolio/tmp/catalogue52-20261018/venv/bin/python'
read=lambda p:json.loads(Path(p).read_text())
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(d):
 t=A.with_suffix('.tmp');t.write_text(json.dumps({'pid':os.getpid(),'utc':now(),**d},indent=2)+'\n');t.replace(A)
def guard_authorization():
 if read(O/'goal_progress.json')['status']!='active' or read(V/'control.json').get('stop'):
  raise RuntimeError('Research or shared control no longer authorizes automatic handoff')
 if (P/'development/summary.json').exists():
  raise RuntimeError('Development already has a result; do not duplicate or retune')
try:
 assert not A.exists(), 'Existing handoff state must be inspected, not overwritten'
 from importlib import import_module
 sys.path.insert(0,str(V))
 pilot=import_module('research_gjr.pilot_run')
 plan=pilot.guard()
 write({'status':'waiting_for_exact_baseline_completion','baseline_controller_pid':read(V/'controller_active.json')['pid'],
        'plan_sha256':hashlib.sha256((P/'plan.json').read_bytes()).hexdigest(),
        'forecast_workers_used_by_waiter':0,'will_run_development_then_reserved_only_if_gate_passes':True})
 while True:
  guard_authorization()
  active=read(V/'controller_active.json')
  if active['status']=='terminal':
   try: pilot.wait_for_baseline_completion()
   except (AssertionError,RuntimeError,FileNotFoundError):
    state=read(V/'results/status.json')
    if state['completed_active_tasks']!=256196 or state['failed_active_tasks']:
     raise RuntimeError('Baseline terminated without the complete failure-free required coverage')
   else: break
  else:
   os.kill(active['pid'],0)
  time.sleep(15)
 guard_authorization()
 for phase in ['development','reserved']:
  if phase=='reserved' and not read(P/'development/summary.json')['advance_to_next_stage']:
   break
  if read(O/'goal_progress.json')['status']!='active' or read(V/'control.json').get('stop'):
   raise RuntimeError('Authorization/control changed before next phase')
  write({'status':'running_'+phase,'forecast_workers_requested':12,'no_baseline_job_restart':True})
  with (P/(phase+'.log')).open('ab') as log:
   result=subprocess.run([PY,'-u','-m','research_gjr.pilot_run',phase],cwd=V,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   raise RuntimeError(f'{phase} pilot exited with code {result.returncode}; inspect log and checkpoints; no automatic restart')
  summary=read(P/phase/'summary.json')
  if not summary['complete']:
   raise RuntimeError(f'{phase} pilot coverage incomplete; do not silently promote')
 write({'status':'terminal_pilots_completed','development_summary':str(P/'development/summary.json'),
        'reserved_summary':str(P/'reserved/summary.json') if (P/'reserved/summary.json').exists() else None,
        'no_full_admission_or_goal_completion':True})
except Exception as exc:
 write({'status':'terminal_needs_research_review','exception_type':type(exc).__name__,'message':str(exc),
        'no_automatic_restart_or_promotion':True})
 raise
