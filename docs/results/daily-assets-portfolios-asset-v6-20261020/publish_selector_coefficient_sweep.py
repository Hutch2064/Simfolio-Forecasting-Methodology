"""Preserve qualified exact-reducer prototype, next-stage source and recovery tests."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';V=O/'asset_level_v6';C=V/'candidates/asset_selector_coefficient_sweep';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;ENV={**os.environ,'GODEBUG':'http2client=0'}
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[k]='1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
with(O/'asset_v6_publication.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main']);q=read(C/'qualification_receipt.json');assert q['tests_passed']==6 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
 names=['test_selector_coefficient_sweep.py','test_selector_coefficient_sweep_integration.py','benchmark_selector_coefficient_sweep.py','selector_coefficient_sweep_qualification.log','selector_coefficient_sweep_benchmark.json','selector_coefficient_sweep_runtime_preflight.json','selector_weight_stability_research.md','verify_selector_path_development355.py','verify_selector_path_development_after_writer.py','run_selector_sweep_validation_reserved.py','qualify_selector_sweep_after_development.py','publish_selector_coefficient_sweep.py']+[str(p.relative_to(O))for p in C.iterdir()if p.is_file()and p.suffix in('.py','.json')]
 changed=[]
 for name in names:
  target=D/name.removeprefix('asset_level_v6/');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/name,target);changed.append(str(target.relative_to(D)))
 doc=D/'COEFFICIENT_SWEEP_PROTOTYPE.md';doc.write_text('''# Exact shared score-coefficient sweep

This is a research backend prototype. It does not add a forecasting method,
change the166-option universe, criterion, validation folds, members, outer score,
full-panel catalogue count or current official winner. The running355-case
path study continues with its original frozen numerical source.

The existing reducer independently merges sorted samples for every option pair
at every known historical lead. The prototype sorts all labelled members once
per lead, shares per-option cumulative counts and sums, and retains the original
summation order for each pair. Equal-value events are all incorporated before
scoring each event, preserving the original <= convention. Memory remains
bounded by current-lead members plus the dense pair matrix. No leads, members,
options or origins are dropped and no numerical precision is relaxed.

Six tests pass: random166option/1000member coefficient bytes, ties/point masses/
scale extremes, chunk continuation, certified weight equality, all-fold dated
joint-history coefficient/weight/forecast-path bytes at100and1000members, and
request-binding rejection. Three interleaved warm-code synthetic pairs show
median paired coefficient CPU ratios of6.557940at100members and5.827371at1000.
All compared coefficients are byte-identical. Twelve active scientific workers
confound wall timing. These are coefficient-only synthetic timings; no whole
request, production or browser speedup is claimed. The earlier13-request local
profile attributes about79%of selected100path preparation CPU to this reducer,
which motivates testing the improvement on real complete workloads.

If the current355-case development result passes independent audit and improves
its matched endpoint control, the next planned shared12-worker stage will replay
all355development requests and run both criteria on355separate reserved research
requests at1000inner/final members. The replay must reproduce all710complete
outer loss vectors, all recorded component traces, learned weights and1132stored
coefficient arrays from283new development requests. The other72requests have
no saved coefficient matrices; their traces, weights and outer losses still
must match. Reserved endpoint controls must also exactly match preserved losses.
Any parity failure invalidates acceptance of that stage. Reserved slices are
adaptively reused research evidence, not independent validation. No partial
result promotes a full-panel winner or meets the goal stopping condition.

The separate weight-stability research note documents further credible ideas;
it does not admit a model or prescribe arbitrary mixture percentages.
''');changed.append(doc.name)
 manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
 run(['git','add','--',REL]);run(['git','commit','-m','Qualify exact shared CRPS reducer and prepare matched reserved study']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_coefficient_sweep_public_recovery'/commit[:12]
 source_map={p:h for p,h in read(V/'candidates/asset_selector_path_objective/development1000/plan.json')['source_hashes'].items()if p.endswith('.py')};source_map.update(q['source_hashes'])
 original_by_public={p.removeprefix('asset_level_v6/'):p for p in source_map}
 all_names=sorted(set(changed)|set(original_by_public));assert all(manifest[p]==source_map[original_by_public[p]]for p in original_by_public)
 def recover(name):
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==manifest[name];local=original_by_public.get(name,name);target=restore/local;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);assert sha(target)==manifest[name];return {'public_path':name,'recovered_path':local,'sha256':manifest[name],'bytes':len(data)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,all_names))
 with(restore/'recovered_tests.log').open('w')as output:subprocess.run([sys.executable,'-m','unittest','test_selector_coefficient_sweep','test_selector_coefficient_sweep_integration'],cwd=restore,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=True)
 tests=(restore/'recovered_tests.log').read_text();assert 'Ran 6 tests' in tests and tests.rstrip().endswith('OK')
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'verified_source_hashes':source_map,'recovered_tests_passed':6,'recovered_test_log_sha256':sha(restore/'recovered_tests.log'),'source_recovery_scope':'Published research numerical source closure; sealed original financial dependencies remain bound to existing verified archives and active local dependency roots','all_real355_request_backend_replay_pending':True,'new_financial_full_panel_models':0,'goal_active':True,'scientific_originals_removed':False};(O/'selector_coefficient_sweep_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('files','verified_source_hashes')},indent=2),flush=True)
