"""Publish closed audited stages and versioned ongoing precision orchestration."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;V=O/'asset_level_v6';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
changed=[]
def copy(p,q):
    q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);changed.append(str(q.relative_to(D)))
for name in ('verify_selector_batched_reserved.py','verify_selector_path_stage.py','run_selector_path_precision72.py','preserve_selector_reserved.py','preserve_selector_path_screen.py','publish_selector_reserved_path_progress.py','selector_precision_recovered_tests.json','selector_path_public_recovery.json','percentage_metadata_correction_receipt.json'):
    copy(O/name,D/name)
for folder in ('asset_selector_batched/validation_and_reserved','asset_selector_path_objective/screen100'):
    source=V/'candidates'/folder;assert read(source/'verification_receipt.json')['verified']
    for name in ('verification_receipt.json','decision.json'):
        if(source/name).exists():copy(source/name,D/'candidates'/folder/name)
    summary=read(source/'summary.json');summary.pop('records',None);summary.pop('validation_receipts',None);summary.pop('case_timings',None)
    target=D/'candidates'/folder/'summary.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(summary,indent=2)+'\n');changed.append(str(target.relative_to(D)))
private_files=0
for tag in ('asset-v6-selector-reserved-20261021','asset-v6-selector-pathscreen-20261021'):
    source=O/'archives'/tag/'receipt.json';receipt=read(source);assert receipt['complete'];private_files+=receipt['files_fresh_downloaded_disk_restored_sha256_verified'];copy(source,D/(tag+'-archive-receipt.json'))
text='''# Current selector: reserved validation and path-criterion progress

Goal active. The frozen1,264series panel, mainQ and fixed worse joint-asset
with-replacement reference are unchanged. LowerQ is better. Official full-panel
results remain both references plus the verified circular/GARCH-fade incumbent,
3.205447%better than the fixed reference on all64049matched1000path forecasts.
No development result below is a new full-panel winner or catalogue entry.

## Completed matched research evidence

* The166-option learned pool completes355development cases at1000inner/final
  paths:13.772057%better than fixedWR. Hard selection does not advance.
* On the separate355case reserved research slice at1000inner/final paths, the
  pool is5.196151%better than fixedWR(Q.664693682834264 versus.701125207979668).
  The old working parent is4.037528%better on exactly those cases(Q.672817083580487).
  Independent audit checks355candidate+710reference checkpoints, all training
  support/signatures, seeds, path allocations and main-score reductions.
* Shared simulation passes all355development request replays:710identical outer
  loss vectors and3704788inner component terminal ensembles. Fourteen tests also
  pass from the freshly downloaded precision/batched source. This qualifies the
  shared backend; it does not change the financial algorithm or claim production
  latency savings.
* The isolated whole-path versus endpoint criterion screen completes72cases per
  method at100inner/final paths. Whole-path fitting is21.599394%better than fixedWR
  (Q.758787985311656), endpoint fitting20.016400%better(Q.774108746731478).
  All72endpoint loss vectors match the preserved control exactly. Independent
  audit verifies144candidate+216reference checkpoints, historical prefixes/dates
  and all main scores. Seven numerical/causality tests pass from downloaded source.

Each bullet has its own matched comparison. These are adaptively reused research
slices, not independent validation or a mixed-coverage leaderboard. Percentages
are100*(1-Qmodel/QfixedWR). The verified working and full-panel incumbents are not
replaced from the exploratory screen.

## Running next

The whole-path criterion has advanced to the same72cases at1000inner/final paths.
It reuses verified existing reference checkpoints, and replays the endpoint
control at1000paths. The first request must pass before the shared12worker pool
opens. Numerical sources, data, scoring and the166-option menu remain fixed.
No final1000path result for the new path criterion is claimed here. Broader
representative evaluation remains necessary before any full-panel conclusion.

The changed criterion learns one whole joint-path mixture from historical
forecasts through the requested horizon, using the canonical positive log-calendar
lead masses and sqrt(nativelead)normalization. Only known historical dates enter
fitting. It introduces no manual short/long mix, diagnostic selection weight,
outer-outcome fitting, automatic previous-winner fallback or capped option count.

## Preservation and correction

The two newly linked private archives contain1528files freshly downloaded,
fully disk-restored andSHA256verified:1201closed parity/reserved files and327
closed criterion-screen/source files. Active1000path outputs are excluded and
retained locally with their readers/inputs. No scientific original was removed.
Full-panel catalogue counts remain23unique methods/28executions.

PERCENTAGE_METADATA_ERRATA.json corrects13derived percentage fields across5older
compact summaries that used a1000rather than100multiplier. Original compact
summaries are retained beside corrected copies. Canonical mainQ, raw losses,
frozen local evidence, fitting, ranks and the correctly reported working-incumbent
percentages are unchanged. See the erratum for the historical100path field-name
caveat in1000path training-block stages.
'''
f=D/'SELECTOR_RESERVED_AND_PATH_UPDATE.md';f.write_text(text);changed.append(f.name)
f=D/'CURRENT_SELECTOR_STATUS.md';f.write_text('Latest dated evidence and running stage: [SELECTOR_RESERVED_AND_PATH_UPDATE.md](SELECTOR_RESERVED_AND_PATH_UPDATE.md). The earlier snapshot below is retained.\n\n'+f.read_text());changed.append(f.name)
manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
run(['git','add','--',REL]);run(['git','commit','-m','Record audited reserved gain and qualify whole-path precision evaluation']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_reserved_public_recovery'/commit[:12]
def recover(name):
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==manifest[name];p=restore/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[name];return {'path':name,'sha256':manifest[name],'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,changed))
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_sha256_verified':len(files),'files':files,'new_private_files_freshly_restored_verified':private_files,'goal_active':True,'full_panel_catalogue_increment':0,'no_scientific_originals_removed':True}
(O/'selector_reserved_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='files'},indent=2),flush=True)
