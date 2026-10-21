"""Admit the strongest frozen broad-qualified fade, never a horizon-picked model."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates/asset_fading_vol'
read=lambda p:json.loads(p.read_text())
def write(p,data):
    temp=p.with_suffix('.json.new');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,p)
def main():
    while not (C/'broad/summary.json').exists():
        state=read(O/'asset_fading_vol_handoff_active.json')
        command=subprocess.run(['ps','-p',str(state['pid']),'-o','command='],capture_output=True,text=True).stdout
        if 'continue_asset_fading_vol.py' not in command:
            assert state['status']=='terminal_rejected_this_frozen_contrast',state
            write(O/'asset_fading_full_gate_receipt.json',{'admitted':False,'reason':'Earlier frozen contrast did not earn broad escalation','goal_complete':False});return
        time.sleep(5)
    broad=read(C/'broad/summary.json')
    if not broad['advancing_model_ids']:
        write(O/'asset_fading_full_gate_receipt.json',{'admitted':False,'reason':'Neither frozen recipe improves broad overall main score','goal_complete':False});return
    selected=min(broad['advancing_model_ids'],key=lambda mid:broad['sole_main_scores'][mid])
    for phase,count in [('development',355),('reserved',355),('broad',7418)]:
        result=read(C/phase/'summary.json')
        assert result['complete'] and result['cases_per_model']==count
        assert result['native_failures']==result['scoring_integrity_errors']==0
        assert selected in result['advancing_model_ids']
    assert not (C/'full/active.json').exists(),'Inspect an existing job rather than restarting'
    mode='ewma' if '_EWMA_' in selected else 'garch'
    write(C/'full_selection.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_id':selected,'mode':mode,'rule':'Lowest unchanged overall main Q among frozen recipes passing all three matched gates; no horizon/family score enters selection.','broad_summary_sha256':hashlib.sha256((C/'broad/summary.json').read_bytes()).hexdigest(),'all_broad_scores':broad['sole_main_scores'],'paths':1000,'official_promotion_requires_complete_independent_full_audit':True,'goal_complete':False})
    env={**os.environ,**{k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']}}
    owner=subprocess.Popen(['/usr/bin/nice','-n','15',sys.executable,str(O/'run_asset_fading_full.py')],cwd=O,env=env,stdout=(O/'asset_fading_full.log').open('ab',buffering=0),stderr=subprocess.STDOUT,start_new_session=True)
    launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':owner.pid,'model_id':selected,'paths':1000,'expected_tasks':64049,'shared_forecast_workers':12}
    write(O/'asset_fading_full_launch.json',launch)
    reporter=subprocess.Popen(['/usr/bin/nice','-n','19',sys.executable,str(O/'report_asset_fading_full.py')],cwd=O,env=env,stdout=(O/'asset_fading_full_reporter.log').open('ab',buffering=0),stderr=subprocess.STDOUT,start_new_session=True)
    write(O/'asset_fading_full_gate_receipt.json',{'admitted':True,'utc':launch['utc'],'controller_pid':owner.pid,'reporter_pid':reporter.pid,'model_id':selected,'full_promotion_requires_independent_complete_audit':True,'goal_complete':False})
    finisher=subprocess.Popen(['/usr/bin/nice','-n','19',sys.executable,str(O/'finish_asset_fading.py')],cwd=O,env=env,stdout=(O/'asset_fading_finish.log').open('ab',buffering=0),stderr=subprocess.STDOUT,start_new_session=True)
    write(O/'asset_fading_finish_launch.json',{'utc':launch['utc'],'pid':finisher.pid,'goal_complete':False})
    p=O/'goal_progress.json';j=read(p);j.update(stage='fading_volatility_full_candidate',active_pilot={'phase':'full_candidate','pid':owner.pid,'reporter_pid':reporter.pid,'paths':1000,'expected_tasks':64049,'status_path':str(C/'full/results/status.json'),'verification_receipt':str(C/'full/verification_receipt.json'),'workers_requested':12},current_control_path=str(C/'full/control.json'),current_registry_path=str(C/'full/registry.json'),current_launch_path=str(C/'full/active.json'),current_research_processes={'full_candidate':{'pid':owner.pid,'active_path':str(C/'full/active.json')},'progress_reporter':{'pid':reporter.pid,'active_path':str(C/'full/reporter_active.json')},'full_finisher':{'pid':finisher.pid}},scientific_forecast_workers_currently_active=0,stopping_requirement_met=False)
    write(p,j)
    print(json.dumps({'owner':owner.pid,'reporter':reporter.pid,'finisher':finisher.pid,'selected':selected,'goal_complete':False}),flush=True)
if __name__=='__main__':main()
