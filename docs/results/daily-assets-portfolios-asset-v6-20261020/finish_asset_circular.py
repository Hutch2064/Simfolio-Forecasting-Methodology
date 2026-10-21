"""Verify, conditionally promote, preserve and catalogue the full circular candidate."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time,traceback
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_circular_blocks';F=C/'full'
read=lambda p:json.loads(p.read_text())
def state(stage,**details):
    p=O/'asset_circular_finish_active.json';t=p.with_suffix('.json.new')
    t.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'stage':stage,'goal_complete':False,**details},indent=2)+'\n');os.replace(t,p)
def live(pid,script):
    return script in subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout

def main():
    gate=read(O/'asset_circular_full_gate_receipt.json');assert gate['admitted']
    while not (F/'verification_receipt.json').exists():
        assert live(gate['controller_pid'],'run_asset_circular_full.py'),'Full owner ended without verification; inspect actual state rather than restarting'
        state('waiting_live_full_forecasts_or_audit',owner_pid=gate['controller_pid']);time.sleep(10)
    result=read(F/'verification_receipt.json')
    assert result['complete'] and result['successful_tasks']==64049 and result['series']==1264
    assert result['native_failures']==result['scoring_integrity_errors']==0
    while True:
        reporter=read(F/'reporter_active.json')
        if reporter['status']=='terminal' and not live(gate['reporter_pid'],'report_asset_circular_full.py'):break
        assert live(gate['reporter_pid'],'report_asset_circular_full.py'),'Reporter ended without terminal receipt; inspect'
        state('waiting_live_reporter_terminal',reporter_pid=gate['reporter_pid']);time.sleep(5)
    actual=read(O/'goal_progress.json')
    if result['main_score']<actual['current_best_main_score']:
        state('promoting_verified_full_improvement')
        subprocess.run([sys.executable,str(O/'promote_asset_circular.py')],cwd=O,check=True)
    else:
        state('keeping_verified_incumbent',candidate_main_Q=result['main_score'],incumbent_main_Q=actual['current_best_main_score'])
    # Bulk archive IO is coordinated with all live science via the same lease.
    with (O/'asset_v6_forecast_slots.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        state('preserving_complete_full_result')
        subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.8','--artifact-root',str(O),'--budget-gib','12'],check=True)
        subprocess.run([sys.executable,str(O/'preserve_asset_circular_closed.py')],cwd=O,check=True)
    state('publishing_verified_catalogue_and_source')
    subprocess.run([sys.executable,str(O/'publish_asset_v6_prepared.py')],cwd=O,check=True)
    state('terminal_full_stage_preserved_research_continues',winner=read(O/'goal_progress.json')['current_best_model_id'],scientific_originals_removed=False)

if __name__=='__main__':
    try:main()
    except Exception:
        state('attention_required',traceback=traceback.format_exc());raise
