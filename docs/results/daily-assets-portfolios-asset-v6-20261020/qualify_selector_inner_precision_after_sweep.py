"""Admit the prepared precision test only after verified prerequisite work."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_inner_precision/study1000';S=V/'candidates/asset_selector_coefficient_sweep/validation_and_reserved';OUT=O/'selector_inner_precision_handoff_active.json'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
driver=O/'run_selector_inner_precision72.py';verifier=O/'verify_selector_inner_precision72.py';bound={str(p.relative_to(O)):sha(p)for p in(driver,verifier)};owner=read(O/'selector_sweep_audit_active.json')['pid']
def actual(pid):
    return subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout.strip()
expected=actual(owner)
assert 'verify_selector_sweep_after_handoff.py'in expected,'Dependency auditor is not currently live'
def write(status,**more):
    temp=OUT.with_suffix('.tmp');temp.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':status,'waiting_auditor_pid':owner,'waiting_auditor_command':expected,'bound_sources':bound,**more},indent=2)+'\n');temp.replace(OUT)
try:
    while actual(owner)==expected:
        if read(O/'goal_progress.json')['goal_status']!='active':write('deferred_goal_not_active');sys.exit(0)
        write('waiting_for_actual_prerequisite_auditor',actual_owner_live=True);time.sleep(20)
    prerequisite=read(O/'selector_sweep_audit_active.json')
    if prerequisite['status']=='no_new_stage_admitted':write('not_admitted_path_development_needs_review',goal_complete=False);sys.exit(0)
    assert prerequisite['status']=='verified_terminal_no_promotion',prerequisite
    audit=read(S/'verification_receipt.json');summary=read(S/'summary.json');assert audit['verified']and audit['summary_sha256']==sha(S/'summary.json')and summary['qualification_valid']and not summary['errors']
    sys.path.insert(0,str(V));from candidates.asset_selector_coefficient_sweep.candidate import identity
    path=summary['models'][identity({'criterion':'path','inner_paths':1000})]
    if not path['beats_matched_terminal_pool']:write('not_admitted_path_reserved_needs_review',goal_complete=False);sys.exit(0)
    assert all(sha(O/p)==h for p,h in bound.items());p=read(O/'selector_inner_precision_public_recovery.json');q=read(O/'selector_precision_worker_public_recovery.json');assert p['remote_main_verified']and p['recovered_tests_passed']==6 and q['remote_main_verified']and q['recovered_worker_tests_passed']==1
    if read(O/'goal_progress.json')['goal_status']!='active':write('deferred_goal_not_active');sys.exit(0)
    write('running_precision_study_shared12worker_budget');subprocess.run([sys.executable,'-u',str(driver),'--run'],cwd=O,check=True)
    assert read(D/'active.json')['status']=='terminal'and read(D/'summary.json')['qualification_valid'];assert all(sha(O/p)==h for p,h in bound.items());write('verifying_all_saved_precision_forecasts');subprocess.run([sys.executable,'-u',str(verifier)],cwd=O,check=True);assert read(D/'verification_receipt.json')['verified'];write('verified_precision_study_no_promotion',receipt=str(D/'verification_receipt.json'),goal_complete=False)
except BaseException as error:
    if isinstance(error,SystemExit)and error.code==0:raise
    write('failed_requires_owner_review',error_type=type(error).__name__,error=str(error),goal_complete=False);raise
