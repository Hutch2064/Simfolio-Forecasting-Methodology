"""Preserve the precision auditor and reproduce the frozen usage diagnostic."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,urllib.request

O=Path(__file__).resolve().parent;V=O/'asset_level_v6';R=O/'repository'
C=V/'candidates/asset_selector_inner_precision_batched'
REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL
ENV={**os.environ,'GODEBUG':'http2client=0'}
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[k]='1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)

with (O/'asset_v6_publication.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not run(['git','status','--porcelain']).strip()
 run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
 q=read(C/'qualification_receipt.json')
 assert q['tests_passed']==6 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
 source_map={f['recovered_path']:f['sha256'] for f in read(O/'selector_online_public_recovery.json')['files'] if f['recovered_path'].startswith('asset_level_v6/') or f['recovered_path'].startswith('test_selector_')}
 source_map.update(q['source_hashes'])
 source_map.update({'asset_level_v6/'+p:h for p,h in read(V/'candidates/asset_selector_native_adapter/numerical_sources.json')['source_files'].items()})
 names=set(source_map)|{'score_percentage.py','selector_inner_precision_batched_qualification.log','publish_selector_precision_audit.py','verify_selector_inner_precision_batched72.py','test_selector_precision_audit.py','reproduce_selector_selection_usage.py','selector_selection_usage_diagnostic.json','selector_selection_usage_reproduction.json','selector_fast_transition_receipt.json','run_selector_inner_precision_batched72.py','asset_level_v6/candidates/asset_selector_inner_precision_batched/study1000/plan.json','runtime_target_20261021.json'}|{str(p.relative_to(O)) for p in C.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md')}
 original_by_public={p.removeprefix('asset_level_v6/'):p for p in names}
 for public,original in original_by_public.items():
  source=O/original;target=D/public
  if original in source_map:assert sha(source)==source_map[original]
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='PUBLIC_FILES.json' and '__pycache__' not in p.parts}
 (D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
 run(['git','add','--',REL]);run(['git','commit','-m','Preserve precision audit and reproduce frozen selector usage diagnostic'])
 run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip()
 restore=O/'selector_precision_audit_public_recovery'/commit[:12]
 def recover(public):
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{public}',timeout=60) as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==manifest[public]
  target=restore/original_by_public[public];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  assert sha(target)==manifest[public]
  return {'public_path':public,'recovered_path':str(target.relative_to(restore)),'sha256':manifest[public],'bytes':len(data)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(recover,sorted(original_by_public)))
 with (restore/'recovered_tests.log').open('w') as output:
  subprocess.run([sys.executable,'-m','unittest','test_selector_inner_precision_batched','test_selector_inner_precision_batched_worker','test_selector_precision_audit'],cwd=restore,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=True)
 tests=(restore/'recovered_tests.log').read_text();assert 'Ran 9 tests' in tests and tests.rstrip().endswith('OK')
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'recovered_tests_passed':6,'recovered_worker_tests_passed':1,'total_recovered_tests_passed':9,'precision_audit_tests_passed':2,'recovered_test_log_sha256':sha(restore/'recovered_tests.log'),'verified_source_hashes':source_map,'source_scope':'Controlled internal-precision implementation and frozen numerical source closure','no_real_panel_accuracy_result':True,'controlled_study_is_running':True,'reproduced_usage_scope':'Frozen135-request diagnostic; actual allocation and checkpoint identities verified locally, no remote private-loss reexecution claim','no_new_financial_model_or_forecast_pool_admitted':True,'goal_remains_active':True,'scientific_originals_removed':False}
 (O/'selector_precision_audit_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ('files','verified_source_hashes')},indent=2),flush=True)
