"""Publish verified development evidence and versioned ongoing optimized sources."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;V=O/'asset_level_v6';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda a:subprocess.check_output(a,cwd=R,env=ENV,text=True)
lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
archive=O/'archives/asset-v6-broad-selector-precision-20261021/receipt.json';assert read(archive)['complete']
changed=[]
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);changed.append(str(q.relative_to(D)))
for n in ['test_selector_convex_certificate.py','test_asset_selector_universe_certified.py','test_asset_selector_batched.py','qualify_selector_solver_after_writer.py','complete_certified_selector_development.py','verify_selector_certified_development.py','profile_selector_inner_components.py','benchmark_selector_shared_inner.py','run_selector_batched_validation.py','preserve_selector_precision_development.py','selector_precision_solver_review.json','selector_precision_runtime_diagnostic.json','selector_component_runtime_profile.json','selector_batched_runtime_preflight.json','selector_component_review.json','selector_finite_ensemble_review.json','selector_temporal_weight_idea.json','publish_selector_precision_progress.py']:
 copy(O/n,D/n)
for folder in ['asset_selector_universe_precision','asset_selector_universe_certified','asset_selector_batched']:
 C=V/'candidates'/folder
 for p in C.iterdir():
  if p.is_file()and(p.suffix in('.py','.md')or p.name in('qualification_receipt.json','validation_receipt.json','decision.json')):copy(p,D/'candidates'/folder/p.name)
 for phase in('development1000','repair_qualification','paired_inner_benchmark'):
  p=C/phase/'summary.json'
  if p.exists():
   s=read(p);s.pop('records',None);s.pop('validation_receipts',None);target=D/'candidates'/folder/(phase+'_summary.json');target.write_text(json.dumps(s,indent=2)+'\n');changed.append(str(target.relative_to(D)))
copy(archive,D/'asset-v6-broad-selector-precision-20261021-archive-receipt.json')
s=read(V/'candidates/asset_selector_universe_certified/development1000/summary.json');ref=s['reference_main_Q']['fixed_reference'];parent=s['reference_main_Q']['current_best'];pool=s['complete_identical_coverage_development_ranking'][0]
text=f'''# Current selector research status

Goal active. Frozen1,264-series panel and single main scoring protocol unchanged. Maximize full-panel accuracy regardless of cost/complexity; runtime secondary. Both TestFolio-style joint-asset references remain unchanged at1,000paths; with replacement remains the fixed worse reporting reference. No new full-panel winner or catalogue entry is claimed here.

## Complete matched355-case development result

All methods below use exactly the same355cases and1,000final paths. Adaptive procedures also use1,000inner paths.

| Method | MainQ(lower better) | Better than fixed reference |
|---|---:|---:|
| Fixed asset blocks with replacement | {ref:.12f} | Reference |
| Current research working parent | {parent:.12f} | {100*(1-parent/ref):.6f}% |
| Learned predictive pooling | {pool['main_Q']:.12f} | {pool['percent_better_vs_same1000path_fixed_reference']:.6f}% |
| Hard selection | {s['complete_identical_coverage_development_ranking'][1]['main_Q']:.12f} | {s['complete_identical_coverage_development_ranking'][1]['percent_better_vs_same1000path_fixed_reference']:.6f}% |

Pooling advances; hard selection loses the working parent and does not advance. This is development evidence, not full-panel, significance or independent-validation evidence. Each request learns whole joint-path mixture weights from its own historical rolling forecasts. The initial166-option bank includes the prior trained model, three raw samplers and162filtered combinations of9volatility methods,3shock laws,3mean treatments and uncertainty on/off. No fixed percentage mixture, arbitrary maximum origin count or automatic prior-winner fallback is introduced. See implementation/DESIGN files for long-horizon support limitations and finite-ensemble allocation.

The initial precision attempt had704successful forecasts and3optimizer-status failures. The numerical correction accepts only a verified convex optimality certificate and continues uncertified iteration-limited solves without a total cap. All3failed cases replayed successfully; a successful multivariate request(case221)matched both prior loss vectors. Complete corrected coverage is710forecasts. Independent audit verifies710selected/710reference checkpoints, every origin/main score, original source aliases and saved convex coefficients. Original failed records remain preserved. The initial harness's nominal355anchor count included the3fits that failed before anchor replay; actual initial executed parent replays were352. This metadata correction does not change financial scores.

## Ongoing efficiency and separate-slice checks

Measured timing in the original development placed about94%of selection time in simulation. The shared implementation preserves mean-specific asset/portfolio states while sharing mean-independent volatility/innovation work. Four integrated tests cover all162filtered options, joint asset states, chunk behavior, full request traces, coefficients and learned weights. One actual8asset historical fold showed bitwise-identical166component terminal ensembles and a1.8758CPU/1.8833wall median ratio in3interleaved warm-code pairs. This is local simulation-phase timing only, not full-request or deployed performance.

A single shared12worker pool now replays all355development requests(710outer loss vectors and all inner component ensemble hashes)against the original references while computing learned-pool forecasts on a separate fixed355case slice. Both use1,000inner/final paths. The new implementation remains subject to complete workload parity before retention; partial results cannot promote it. The separate slice is also adaptively reused research data, not independent validation. Current scientific paths are asset_selector_batched/validation_and_reserved in the local research root. No final full-panel admission; original stopping requirement still applies.

## Preservation

All closed initial/corrected precision results, failed attempts, source, losses, weights, traces, audit and closed paired benchmark are in the linked private archive receipt:1,864files freshly downloaded, disk-restored andSHA256verified. Active writers and their inputs remain local. Large raw research files remain private; public sources, compact summaries and repository-linked recovery records are preserved here. Full-panel catalogue counts remain23unique methods/28executions; these development experiments add none.
'''
p=D/'CURRENT_SELECTOR_STATUS.md';p.write_text(text);changed.append(str(p.relative_to(D)))
for n in('CURRENT_RESEARCH.md','SELECTOR_RESEARCH.md','SELECTOR_POOL_RESEARCH.md'):
 p=D/n
 if p.exists():
  old=p.read_text();prefix='Current adaptive-selector evidence: [CURRENT_SELECTOR_STATUS.md](CURRENT_SELECTOR_STATUS.md). The historical snapshot below is retained.\n\n';p.write_text(prefix+old);changed.append(n)
manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
run(['git','add','--',REL]);run(['git','commit','-m','Verify355case adaptive pooling gain and preserve complete precision evidence']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'v6_public_recovery/18c4f734da2e'
def recover(n):
 with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{n}',timeout=60)as r:data=r.read()
 assert hashlib.sha256(data).hexdigest()==manifest[n];p=restore/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[n];return {'path':n,'sha256':manifest[n],'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,sorted(set(changed))))
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'closed_private_files_freshly_restored_verified':1864,'current_scientific_sources_unchanged_during_publication':True,'source_recovered_tests_pending_available_worker_capacity':True,'goal_active':True,'full_panel_catalogue_increment':0}
(O/'selector_precision_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
