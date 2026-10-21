"""Preserve history-only sequential prototype and matched diagnostic evidence."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';V=O/'asset_level_v6';C=V/'candidates/asset_selector_online_pool';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;ENV={**os.environ,'GODEBUG':'http2client=0'}
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[k]='1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
with(O/'asset_v6_publication.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main']);q=read(C/'qualification_receipt.json');assert q['tests_passed']==6 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
 names=['test_selector_online_pool.py','selector_online_pool_qualification.log','diagnose_selector_path_precision.py','diagnose_asset_v6_pair.py','selector_path_precision_diagnostics.json','selector_path_precision_diagnostics.log','project_selector_path_work.py','selector_path_development_workload_projection.json','selector_path_workload_projection.log','publish_selector_online_prototype.py']+[str(p.relative_to(O))for p in C.iterdir()if p.is_file()and p.suffix in('.py','.md','.json')]
 changed=[]
 for name in names:
  target=D/name.removeprefix('asset_level_v6/');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/name,target);changed.append(str(target.relative_to(D)))
 doc=D/'SELECTOR_WEIGHT_STABILITY_UPDATE.md';doc.write_text('''# Whole-path diagnostic trade-offs and sequential prototype

The official full-panel result and active355-case study are unchanged. The
closed72-case1000path whole-path learner is21.309474%better than fixedWR, versus
20.540747%for its endpoint control. Its0.768727percentage-point incremental gain
is not uniform. An exact216checkpoint decomposition reconciles all dimensions
to the same unchanged canonicalQ. Equities andFX improve; fixed income and mixed
portfolios regress. Descriptive horizon contributions improve through31days,
32to366days,367to1826days and over7305days, but regress over1827to7305days.
These are additive contributions on adaptively reused research cases, not
within-class percentage skills, independent evidence, new scores, forecast
routes or model promotion gates. Diagnostics have zero additional score weight.

The separate scale-free sequential prototype replaces the batch-optimal mixture
with historical-fold CRPS-gradient updates. Its learning-rate scale is derived
from observed gradient magnitudes. It keeps the166option bank, asset paths,
validation schedule and outer score. It introduces no manual blend percentage,
fixed learning-rate cap, gradient clipping or previous-winner fallback. The
minimum-norm initial simplex state is an uninformed symmetry rule; final weights
are learned. This is SOLO-FTRL adapted to a convex CRPS mixture, not quantile BOAG
or a proof of future financial accuracy. See itsDESIGN.md for primary references,
fixed-bank eligibility assumptions and limitations.

Six prototype tests pass, including independent projection/gradient checks,
chronological weight prefixes, scale/common-loss-shift invariance, edge cases,
and exact output preservation of both joint-path batch controls. No real-panel
accuracy result or new forecast pool is admitted. A future matched100path72case
screen can share historical component simulation between both batch controls
and the sequential rule, after the current and queued validations are reviewed.
No full-panel catalogue entry is added.

A separate read-only workload projection, timestamped2026-10-21T13:23:11Z, uses
59nonwarm1kprecision request CPU observations with the exact same numerical
source closure. For281queued/inflight requests, mean/median ideal12core remaining
compute projections are14025/14176seconds. They include the entire cost of
already running requests rather than deducting elapsed progress, and exclude
setup,IO,contention and failed-fit repair. They are workload projections, not
calibrated wall completion times, original broad-market probes or forecast
speedup claims. The active study's live estimator remains unset until enough
within-stage nonwarm requests complete.

The exact-reducer replay/reserved stage remains queued behind the current
independent audit, subject to development improvement. No current numerical
source, scoring weight, preserved outcome or goal stopping rule has changed.
''');changed.append(doc.name)
 manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
 run(['git','add','--',REL]);run(['git','commit','-m','Qualify sequential CRPS prototype and reconcile path-score tradeoffs']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_online_public_recovery'/commit[:12]
 source_map=read(O/'selector_coefficient_sweep_public_recovery.json')['verified_source_hashes'];source_map.update(q['source_hashes']);original_by_public={p.removeprefix('asset_level_v6/'):p for p in source_map};all_names=sorted(set(changed)|set(original_by_public));assert all(manifest[p]==source_map[original_by_public[p]]for p in original_by_public)
 def recover(name):
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==manifest[name];target=restore/original_by_public.get(name,name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);assert sha(target)==manifest[name];return {'public_path':name,'recovered_path':str(target.relative_to(restore)),'sha256':manifest[name],'bytes':len(data)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,all_names))
 with(restore/'recovered_tests.log').open('w')as output:subprocess.run([sys.executable,'-m','unittest','test_selector_online_pool'],cwd=restore,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=True)
 tests=(restore/'recovered_tests.log').read_text();assert 'Ran 6 tests'in tests and tests.rstrip().endswith('OK')
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'recovered_tests_passed':6,'recovered_test_log_sha256':sha(restore/'recovered_tests.log'),'source_scope':'Published research numerical sources; sealed dependencies retain existing private-archive provenance','no_real_panel_accuracy_result_for_new_prototype':True,'no_new_forecast_pool_started':True,'goal_remains_active':True,'scientific_originals_removed':False};(O/'selector_online_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='files'},indent=2),flush=True)
