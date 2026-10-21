"""Publish only scoped adaptive-selector source and closed screen evidence."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,concurrent.futures,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;V=O/'asset_level_v6';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda a:subprocess.check_output(a,cwd=R,env=ENV,text=True)
lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
assert not run(['git','status','--porcelain']).strip()
run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
changed=[]
def copy(p,q):
 q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);changed.append(str(q.relative_to(D)))
for name in ['test_asset_selector_stacking.py','test_asset_selector_universe.py','test_asset_selector_universe_precision.py','screen_asset_selector_stacking.py','screen_asset_selector_universe.py','run_asset_selector_universe_precision.py','verify_asset_selector_stacking.py','verify_asset_selector_universe.py','preserve_asset_selector_stacking.py','preserve_asset_selector_universe.py','selector_component_review.json','publish_selector_pool_research.py']:
 copy(O/name,D/name)
for folder in ['asset_selector_stacking','asset_selector_universe','asset_selector_universe_precision']:
 C=V/'candidates'/folder
 for p in C.iterdir():
  if p.is_file()and(p.suffix in('.py','.md')or p.name in('qualification_receipt.json','validation_receipt.json','decision.json','screen_weight_diagnostics.json')):copy(p,D/'candidates'/folder/p.name)
 for phase in('screen100','screen1000'):
  p=C/phase/'summary.json'
  if p.exists():
   s=read(p);s.pop('records',None);q=D/'candidates'/folder/(phase+'_summary.json');q.write_text(json.dumps(s,indent=2)+'\n');changed.append(str(q.relative_to(D)))
for tag in ['asset-v6-horizon-pool-screen-20261021','asset-v6-broad-selector-screen-20261021']:
 p=O/'archives'/tag/'receipt.json';assert read(p)['complete'];copy(p,D/(tag+'-archive-receipt.json'))
q=D/'SELECTOR_POOL_RESEARCH.md';q.write_text('''# Adaptive selector and learned-pool research

The actual goal remains active: maximize canonical full-panel accuracy regardless of compute cost/complexity, with runtime secondary. Frozen1264series, main scoring, fixed worse asset reference and stopping/catalogue requirements are unchanged. No production change.

The initial19-option horizon-matched terminal-CRPS screen completed144forecasts(72cases each) at1000inner and outer paths. Hard selection Q=.8820768417915955; convex pooling Q=.8709965039754616; matched working parent .8214554222471507; fixed reference .9660201133857566. Both lose the parent and are not promoted. Nine qualification tests plus independent144candidate/144reference checkpoint verification and fresh multivariate replay passed. All closed source, seeds, losses, weights, traces and harness evidence are privately archived with full fresh-download/disk-restore/SHA256verification.

A separate expanded menu has166options: unchanged working parent, three raw samplers, and162pure filtered combinations of9volatility families,3innovation laws,3mean treatments and mean uncertainty on/off. See bank source and DESIGN.md; this is not every possible method. Both algorithms adapt for each request from historical rolling forecasts, not outer outcomes. Six qualification tests check every option, inherited numerical parity and joint-path accounting. The frozen72-case100path exploratory screen completed144forecasts with zero forecast or scoring errors. Hard selection Q=.8120426969079203, learned pooling .7741087467314782; matched parent .8235702227265657, fixed reference .967834337489039. Pooling is20.0163998376%better than the reference on this sample, compared with14.9058686156%for the parent. No full-panel or independent-validation claim. The precision version uses1000inner and outer paths with a distinct source/model identity;355matched development cases per policy are currently being evaluated. Nothing promoted from partial evidence.

The last completed official full-panel model remains unchanged; the newer research working parent has broader7418-case evidence rather than a full-panel score. All full-panel catalogue counts stay unchanged by these screens. Terminal long-horizon validation may provide only two historical blocks and very early fitted prefixes; this weakness remains an explicit research question. No stopping judgment is reached. Future corrected plug-in block selection, EGARCH and other component/validation alternatives are researched, not claimed implemented.

Archived screen receipts identify recovery releases and checksums; raw market histories remain private. Diagnostics have zero additional score/selection weight. Local numerical timings are not production speed or browser latency evidence.
''');changed.append(str(q.relative_to(D)))
manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
run(['git','add','--',REL]);run(['git','commit','-m','Qualify broad adaptive selector universe and preserve matched screens']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip()
# Recover all new files into the already sealed, previously recovered parent tree.
restore=O/'v6_public_recovery/18c4f734da2e';checked=[]
def recover(name):
 digest=manifest[name];url=f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}'
 with urllib.request.urlopen(url,timeout=60)as response:data=response.read()
 assert hashlib.sha256(data).hexdigest()==digest
 p=restore/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==digest
 return {'path':name,'sha256':digest,'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:checked=list(pool.map(recover,sorted(set(changed))))
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(checked),'recovered_tree':str(restore),'files':checked,'recovered_numerical_tests':'Pending until shared scientific slots free; local new tests21passed and prior167recoveredtests unchanged','full_panel_catalogue_increment':0,'scientific_originals_removed':False}
(O/'selector_pool_source_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
