"""Advance a frozen circular-block candidate only through matched pilot gates."""
from pathlib import Path
import datetime,json,os,subprocess,sys
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_circular_blocks'
read=lambda p:json.loads(p.read_text())
def state(phase,**extra):
    p=O/'asset_circular_handoff_active.json';temp=p.with_suffix('.json.new')
    temp.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'phase':phase,'goal_complete':False,**extra},indent=2)+'\n');os.replace(temp,p)

def main():
    assert read(C/'development/summary.json')['advance_to_next_stage']
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.3','--artifact-root',str(O),'--budget-gib','12'],check=True)
    for phase in ('reserved','broad'):
        assert not (C/phase/'active.json').exists(),'Inspect existing phase instead of restarting'
        state(phase,status='running',selected_tasks=355 if phase=='reserved' else 7418)
        subprocess.run([sys.executable,'-m','candidates.asset_circular_blocks.pilot_run',phase],cwd=V,check=True)
        result=read(C/phase/'summary.json')
        assert result['complete'] and result['native_failures']==result['scoring_integrity_errors']==0
        if not result['advance_to_next_stage']:
            state(phase,status='terminal_rejected_this_frozen_contrast',reason='Matched main Q did not improve current verified best; preserve evidence, no escalation or parameter retuning.')
            return
    state('broad',status='terminal_qualified_for_full_admission_review')

if __name__=='__main__':main()
