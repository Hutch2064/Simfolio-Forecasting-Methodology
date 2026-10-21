"""Preserve qualified optional-component source; no private histories or scores."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;C=O/'asset_level_v6/candidates/asset_selector_ppw';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main']);q=read(C/'qualification_receipt.json');assert q['tests_passed']==9 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
changed=[]
def copy(p,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);changed.append(str(target.relative_to(D)))
for p in C.iterdir():
    if p.is_file()and p.suffix in('.py','.md','.json'):copy(p,D/'candidates/asset_selector_ppw'/p.name)
for name in ('test_selector_ppw_components.py','test_selector_ppw_pool.py','selector_ppw_shared_contrast_qualification.log','screen_selector_ppw_menu.py','publish_selector_ppw_prototype.py'):copy(O/name,D/name)
manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
run(['git','add','--',REL]);run(['git','commit','-m','Qualify optional correctedPPW components without changing running selector']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'v6_public_recovery/18c4f734da2e'
def recover(name):
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==manifest[name];p=restore/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[name];return {'path':name,'sha256':manifest[name],'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,changed))
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_sha256_verified':len(files),'files':files,'recovered_tests_pending':True,'financial_panel_admission_or_score_claim':False,'goal_active':True,'full_panel_catalogue_increment':0,'no_originals_removed':True}
(O/'selector_ppw_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='files'},indent=2),flush=True)
