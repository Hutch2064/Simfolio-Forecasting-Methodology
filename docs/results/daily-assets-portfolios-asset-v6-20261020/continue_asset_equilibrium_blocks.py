"""Queue the qualified equilibrium contrast behind the live predecessor's verification."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time,traceback
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_equilibrium_blocks';P=V/'candidates/asset_fading_vol'
read=lambda p:json.loads(p.read_text())
def state(stage,**extra):
    p=O/'asset_equilibrium_handoff_active.json';tmp=p.with_suffix('.json.new')
    tmp.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'stage':stage,'goal_complete':False,**extra},indent=2)+'\n');os.replace(tmp,p)
def live(pid,name):
    return name in subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout

def main():
    assert not (V/'candidates/asset_mean_uncertainty/equilibrium_blocks_screen100/active.json').exists(),'Inspect existing run rather than repeat'
    while True:
        if (P/'full/verification_receipt.json').exists():
            proof=read(P/'full/verification_receipt.json')
            assert proof['complete'] and proof['native_failures']==proof['scoring_integrity_errors']==0
            if not proof['eligible_for_current_best_promotion']:
                state('terminal_not_admitted_parent_not_promoted',reason='This frozen prototype extends GARCH fade. Reassess the starting model after full evidence; no field forecasts started.');return
            if (O/'archives/asset-v6-fading-1000-20261021/receipt.json').exists():
                assert read(O/'archives/asset-v6-fading-1000-20261021/receipt.json')['complete'];break
        assert live(read(O/'asset_fading_finish_launch.json')['pid'],'finish_asset_fading.py'),'Predecessor finished without required full verification/preservation; inspect without restarting'
        state('waiting_actual_live_predecessor_full_verification_and_preservation',owner_pid=read(O/'asset_fading_full_launch.json')['pid']);time.sleep(10)
    state('exploratory100path_screen',paths=100,selected_tasks=355,workers=12)
    subprocess.run([sys.executable,str(O/'screen_asset_equilibrium_blocks.py')],cwd=O,check=True)
    screen=read(V/'candidates/asset_mean_uncertainty/equilibrium_blocks_screen100/summary.json')
    assert screen['complete'] and screen['native_failures']==0
    if not screen['promising_arms_for1000path_confirmation']:
        state('terminal_rejected100path_contrast',reason='No overall screen improvement; keep all evidence. Goal remains active.');return
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.3','--artifact-root',str(O),'--budget-gib','12'],check=True)
    for phase in ('development','reserved','broad'):
        assert not (C/phase/'active.json').exists(),'Inspect existing phase rather than restart'
        state(phase,paths=1000,workers=12)
        subprocess.run([sys.executable,str(O/'run_asset_equilibrium_pilot.py'),phase],cwd=O,check=True)
        result=read(C/phase/'summary.json')
        assert result['complete'] and result['native_failures']==result['scoring_integrity_errors']==0
        if not result['advancing_model_ids']:
            state('terminal_rejected_frozen_contrast',failed_phase=phase,reason='Does not improve the same matched current-best overall main Q; no retuning or escalation.');return
    state('terminal_qualified_for_full_admission_review',qualified_model_ids=result['advancing_model_ids'],full_admitted=False)
if __name__=='__main__':
    try:main()
    except Exception:
        state('attention_required',traceback=traceback.format_exc());raise
