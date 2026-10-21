"""Launch the frozen full candidate only after both355 gates and7418-case broad gate."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates/asset_circular_blocks'
read=lambda p:json.loads(p.read_text())
def write(name,data):
    p=O/name;t=p.with_suffix('.json.new');t.write_text(json.dumps(data,indent=2)+'\n');os.replace(t,p)

def main():
    while not (C/'broad/summary.json').exists():
        launch=read(O/'asset_circular_handoff_launch.json')
        process=subprocess.run(['ps','-p',str(launch['pid']),'-o','command='],capture_output=True,text=True)
        if 'continue_asset_circular.py' not in process.stdout:
            state=read(O/'asset_circular_handoff_active.json')
            assert state['status']=='terminal_rejected_this_frozen_contrast',state
            write('asset_circular_full_gate_receipt.json',{'admitted':False,'reason':'Prior frozen pilot did not earn broad/full escalation','goal_complete':False});return
        time.sleep(5)
    for phase,count in [('development',355),('reserved',355),('broad',7418)]:
        result=read(C/phase/'summary.json')
        if not result['advance_to_next_stage']:
            write('asset_circular_full_gate_receipt.json',{'admitted':False,'failed_phase':phase,'goal_complete':False});return
        assert result['complete'] and result['cases_per_model']==count
        assert result['native_failures']==result['scoring_integrity_errors']==0
    assert not (C/'full/active.json').exists(),'Inspect existing scientific job; never blindly restart'
    env={**os.environ,**{k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']}}
    owner=subprocess.Popen([sys.executable,str(O/'run_asset_circular_full.py')],cwd=O,env=env,stdout=(O/'asset_circular_full.log').open('ab',buffering=0),stderr=subprocess.STDOUT,start_new_session=True)
    launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':owner.pid,'paths':1000,'expected_tasks':64049,'shared_forecast_workers':12}
    write('asset_circular_full_launch.json',launch)
    reporter=subprocess.Popen(['/usr/bin/nice','-n','19',sys.executable,str(O/'report_asset_circular_full.py')],cwd=O,env=env,stdout=(O/'asset_circular_full_reporter.log').open('ab',buffering=0),stderr=subprocess.STDOUT,start_new_session=True)
    write('asset_circular_full_gate_receipt.json',{'admitted':True,'utc':launch['utc'],'controller_pid':owner.pid,'reporter_pid':reporter.pid,'full_promotion_requires_independent_complete_audit':True,'goal_complete':False})
    print(json.dumps({'owner':owner.pid,'reporter':reporter.pid,'goal_complete':False}),flush=True)

if __name__=='__main__':main()
