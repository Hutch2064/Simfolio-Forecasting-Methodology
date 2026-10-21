"""Advance only empirically qualified modes through fixed reserved and broad gates."""
from pathlib import Path
import datetime,json,os,subprocess,sys
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_empirical_mean'
read=lambda p:json.loads(p.read_text())
def state(phase,**extra):
    p=O/'asset_empirical_mean_handoff_active.json';t=p.with_suffix('.json.new')
    t.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'phase':phase,'goal_complete':False,**extra},indent=2)+'\n');os.replace(t,p)
def main():
    assert read(C/'development/summary.json')['advancing_model_ids']
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.3','--artifact-root',str(O),'--budget-gib','12'],check=True)
    for phase in ('reserved','broad'):
        assert not (C/phase/'active.json').exists(),'Inspect existing phase rather than restarting'
        state(phase,status='running')
        subprocess.run([sys.executable,'-m','candidates.asset_empirical_mean.pilot_run',phase],cwd=V,check=True)
        result=read(C/phase/'summary.json')
        assert result['complete'] and result['native_failures']==result['scoring_integrity_errors']==0
        if not result['advancing_model_ids']:
            state(phase,status='terminal_rejected_this_frozen_contrast',reason='No mode improves the exact matched current-best main Q; preserve all records and do not escalate.')
            return
    state('broad',status='terminal_qualified_for_full_admission_review',qualified_model_ids=result['advancing_model_ids'])
if __name__=='__main__':main()
