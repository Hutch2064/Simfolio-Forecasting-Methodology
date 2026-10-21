"""Publish closed audited criterion results, corrections and active-stage source."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;V=O/'asset_level_v6';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
changed=[]
def copy(p,q):
 q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);changed.append(str(q.relative_to(D)))
def save(q,obj):
 q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(obj,indent=2)+'\n');changed.append(str(q.relative_to(D)))
def compact(obj):return {k:v for k,v in obj.items()if k not in('records','case_timings','validation_receipts')}
for name in ('run_selector_path_development355.py','score_percentage.py','test_score_percentage.py','score_percentage_regression.log','correct_closed_selector_percentages.py','selector_path_precision_percentage_correction.log','verify_selector_ppw_stage.py','selector_ppw_independent_verification.log','selector_path_coefficient_runtime_diagnostic.json','preserve_selector_path_precision_and_ppw.py','publish_selector_path_precision_results.py'):
 copy(O/name,D/name)
P=V/'candidates/asset_selector_path_objective/screen1000';Q=V/'candidates/asset_selector_ppw/screen100'
assert read(P/'verification_receipt.json')['verified']and read(Q/'verification_receipt.json')['verified']
assert read(P/'percentage_metadata_erratum.json')['corrected_summary_sha256']==sha(P/'summary.corrected.json')
for source in (P,Q):
 target=D/'candidates'/source.relative_to(V/'candidates')
 for name in ('verification_receipt.json','decision.json','REPORT_AUTHORITY.json','percentage_metadata_erratum.json'):
  if(source/name).exists():copy(source/name,target/name)
 if source==P:
  save(target/'summary.original_percentage_metadata.json',compact(read(source/'summary.json')))
  save(target/'summary.corrected.json',compact(read(source/'summary.corrected.json')))
  save(target/'summary.json',compact(read(source/'summary.corrected.json')))
 else:save(target/'summary.json',compact(read(source/'summary.json')))
archive=O/'archives/asset-v6-path-precision-and-PPW-screen-20261021/receipt.json';a=read(archive);assert a['complete']and a['files_fresh_downloaded_disk_restored_sha256_verified']==782
copy(archive,D/'asset-v6-path-precision-and-PPW-screen-20261021-archive-receipt.json')
status='''# Whole-path precision results and broader development

Goal remains active. Panel, canonical main score, sole fixed worse asset-level
with-replacement reference and 1,000-path official protocol remain unchanged.
The official full-panel winner remains the circular/GARCH-fade model, 3.205447%
better than the reference on all 64,049 matched forecasts. No result below
promotes a new full-panel winner. Catalogue remains 23 unique methods/28 executions.

## Completed 1,000-path precision comparison

On the same 72 requests, with 1,000 internal and final paths for both methods:

| Method | Main Q, lower is better | Better than fixed reference |
| --- | ---: | ---: |
| Fixed with-replacement reference | 0.9660201133857566 | 0% |
| Endpoint-criterion pool | 0.767592367846421 | 20.540747% |
| Whole-path-criterion pool | 0.7601663127925109 | 21.309474% |

All 144 evaluations completed with zero native failures or canonical scoring
errors. Independent verification checks all 144 candidate and 216 reference
checkpoints, main-score reductions, historical-prefix/date causality and all 72
exact endpoint-control loss replays. These are adaptively reused research cases,
not independent validation or evidence of a full-panel improvement.

The whole-path criterion learns one joint-path mixture from historical forecast
scores across all available native leads through the requested horizon, using
positive log-calendar masses and sqrt(native-lead) normalization. It keeps the
166-option universe and historical validation schedule fixed. It introduces no
manual percentage blend or outer-outcome fitting.

## Completed exploratory block-menu comparison

At 100 internal/final paths on the same 72 requests, adding stationary and
circular corrected-PPW block-length options to the 166-option menu worsened
both whole-path and endpoint variants slightly. Whole-path original/expanded
scores were 0.7587879853116555/0.7600849889235529, respectively 21.599394% and
21.465383% better than their matched reference. Endpoint original/expanded
scores were 0.7741087467314783/0.7756244427417525, respectively 20.016400% and
19.859793% better. All 288 evaluations and 144 original-control trace/loss replays
passed independent audit, including percentage calculations, with zero errors.

The additions do not advance from this screen. The small differences at 100
paths do not establish that no precision reversal or useful block-based method
exists. Do not rank these 100-path results against the 1,000-path results above.

## Current broader check

The qualified whole-path method is running the original 355-request development
slice at 1,000 internal/final paths with 12 shared workers. All 72 verified
precision pairs are reused, leaving 283 new requests/566 new outer evaluations.
Every new endpoint control must reproduce its preserved complete loss vector.
Internal A/B coefficient matrices are saved for the 283 new requests only,
allowing later solver audits and compatible numerical reuse without repeating
historical simulation. No stopping decision or full-panel admission is implied.

Profiling 13 selected first requests at 100 paths attributes about 79% of measured
preparation CPU to dense CRPS coefficient assembly. This is a local bottleneck
measurement, not a complete-workload, 1,000-path or production speedup claim.
Any optimization must retain all options, leads, origins and simulation members.

## Percentage correction and preservation

The original one-time precision driver accidentally changed its percentage
multiplier from 100 to 1,000 while changing the simulation count. Its original
summary consequently contains incorrect 213.09%/205.41% displays. Saved losses,
main Q, coverage and ranking were unaffected. Correct values are 21.309474% and
20.540747%. Original source/report are retained; REPORT_AUTHORITY.json and
percentage_metadata_erratum.json identify immutable originals and corrected
reports. Public summary.json is the corrected compact view; raw full summaries
are in the private archive. Percentage calculation now has a separate helper
and regression tests preventing simulation count from changing its scale.

The linked private archive preserves 782 closed results, sources, profiles and
audit files, freshly downloaded, fully disk-restored and SHA256-verified. Active
355-request outputs are excluded. Scientific originals remain local while active
readers need them. The public compact files are derived views; private manifests
bind original full evidence. Historical percentage corrections remain documented
separately in PERCENTAGE_METADATA_ERRATA.json.
'''
f=D/'SELECTOR_PATH_PRECISION_UPDATE.md';f.write_text(status);changed.append(f.name)
f=D/'CURRENT_SELECTOR_STATUS.md';f.write_text('Latest evidence: [SELECTOR_PATH_PRECISION_UPDATE.md](SELECTOR_PATH_PRECISION_UPDATE.md). Earlier dated updates remain available as history.\n');changed.append(f.name)
manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
run(['git','add','--',REL]);run(['git','commit','-m','Preserve corrected path precision results and audited PPW screen']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_path_precision_public_recovery'/commit[:12]
def recover(name):
 with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as response:data=response.read()
 assert hashlib.sha256(data).hexdigest()==manifest[name];p=restore/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[name];return {'path':name,'sha256':manifest[name],'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,sorted(set(changed))))
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_sha256_verified':len(files),'files':files,'new_private_files_freshly_restored_verified':782,'goal_active':True,'full_panel_catalogue_increment':0,'scientific_originals_removed':False}
(O/'selector_path_precision_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='files'},indent=2),flush=True)
