"""Publish qualified new criterion source and diagnostic evidence, not raw data."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;V=O/'asset_level_v6';C=V/'candidates/asset_selector_path_objective';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
qualification=read(C/'qualification_receipt.json');assert qualification['tests_passed']==7 and all(sha(O/p)==h for p,h in qualification['source_hashes'].items())
changed=[]
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);changed.append(str(target.relative_to(D)))
for name in ('screen_selector_path_objective.py','verify_selector_batched_reserved.py','diagnose_selector_pool_development.py','test_selector_path_coefficients.py','test_selector_path_candidate.py','selector_path_full_qualification.log','publish_selector_path_prototype.py'):
    copy(O/name,D/name)
for p in C.iterdir():
    if p.is_file()and(p.suffix in ('.py','.md')or p.name=='qualification_receipt.json'):copy(p,D/'candidates/asset_selector_path_objective'/p.name)
copy(V/'candidates/asset_selector_universe_certified/diagnostics.json',D/'candidates/asset_selector_universe_certified/development_diagnostics.json')
doc=D/'PATH_OBJECTIVE_STUDY.md';doc.write_text('''# Requested-horizon path criterion: qualified prototype

The frozen355case development pool result is13.772057%better than the fixed
with-replacement reference at1000inner/final paths. This is development evidence,
not a full-panel or independent-validation claim. The additive decomposition
reconciles exactly to the unchanged mainQ. Relative to the prior working model,
the gain is concentrated in long-horizon equities, particularly short-term
reversal portfolios; bonds, mixed portfolios and the shortest leads regress.
Diagnostic bands do not change score weights, impose subgroup promotion gates,
or route forecasts.

This motivates a separate methodological contrast: retain the same166component
menu and historical folds, but fit one joint-path mixture using the forecast
path through the requested horizon, with the canonical positive log-calendar
lead masses and sqrt(nativelead)normalization. The control fits only the terminal
lead. There is no manually chosen short/long percentage, no removal of leads or
options, and no change to the outer main score. Only known training calendar
dates enter fitting. Long-horizon support limitations remain disclosed.

Seven qualification tests passed: finite-ensemble scoring algebra, convex
optimality, temporal prefix isolation, date/request binding, short-history
eligibility, chunk invariance and exact endpoint-control output parity on the
qualified fixture. A matched100inner/100final-path72case screen is queued behind
the existing shared12-worker validation pool. It will check its endpoint-control
losses against the preserved prior100path run. No financial result or promotion
is claimed for the new criterion. A lower-simulation screen cannot replace an
official1000path result or add to the full-panel catalogue count.

The active reserved validation and all prior full-panel evidence remain intact.
The goal remains active, and its stopping requirement has not been reached.
Source and compact evidence are preserved here; raw histories and losses remain
in their existing private repository-linked archives or active local ownership.
''');changed.append(doc.name)
manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
run(['git','add','--',REL]);run(['git','commit','-m','Preserve qualified path-criterion prototype and reconciled selector diagnostics']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'v6_public_recovery/18c4f734da2e'
old=read(O/'selector_precision_public_recovery.json');bound={r['path']:r['sha256']for r in old['files']}
assert not set(changed)&set(bound),'Do not overwrite sources bound by pending earlier recovered tests'
def recover(name):
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as r:data=r.read()
    assert hashlib.sha256(data).hexdigest()==manifest[name];p=restore/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[name]
    return {'path':name,'sha256':manifest[name],'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,sorted(set(changed))))
assert all(sha(restore/name)==digest for name,digest in bound.items())
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'previous_bound44_source_files_unchanged':True,'recovered_tests_pending_shared_capacity':True,'goal_active':True,'full_panel_catalogue_increment':0,'no_original_scientific_outputs_removed':True}
(O/'selector_path_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
