"""Preserve isolated prefix-scale methodology and run recovered qualification."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,urllib.request
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';R=O/'repository';C=V/'candidates/asset_selector_fold_scale_batched'
REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL
ENV={**os.environ,'GODEBUG':'http2client=0'}
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[k]='1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(args):return subprocess.check_output(args,cwd=R,env=ENV,text=True)
with(O/'asset_v6_publication.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
 q=read(C/'qualification_receipt.json');assert q['tests_passed']==8 and all(sha(O/p)==h for p,h in q['source_hashes'].items())
 base=read(O/'selector_inner_precision_batched_public_recovery.json');assert base['remote_main_verified']and base['recovered_tests_passed']==6
 sources={**base['verified_source_hashes'],**q['source_hashes']}
 names=set(sources)|{'publish_selector_fold_scale_batched.py','selector_fold_scale_batched_qualification.log','selector_fold_scale_batched_worker_qualification.log','selector_fold_scaling_diagnostic.json','selector_fold_scaling_diagnostic.log','asset_level_v6/candidates/asset_selector_fold_scale_batched/qualification_receipt.json'}
 original_by_public={p.removeprefix('asset_level_v6/'):p for p in names}
 for public,original in original_by_public.items():
  source=O/original;target=D/public
  if original in sources:assert sha(source)==sources[original]
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 assert all(line[3:].startswith(REL+'/')for line in run(['git','status','--porcelain']).splitlines())
 run(['git','add','--',REL]);run(['git','commit','-m','Validate matched fold-scale worker and checkpoint resume guards']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_fold_scale_batched_public_recovery'/commit[:12]
 def recover(public):
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{public}',timeout=60)as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==manifest[public];target=restore/original_by_public[public];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);assert sha(target)==manifest[public]
  return {'public_path':public,'recovered_path':str(target.relative_to(restore)),'sha256':manifest[public],'bytes':len(data)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,sorted(original_by_public)))
 with(restore/'recovered_tests.log').open('w')as output:subprocess.run([sys.executable,'-m','unittest','test_selector_fold_scale_batched','test_selector_fold_scale_batched_worker'],cwd=restore,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=True)
 tests=(restore/'recovered_tests.log').read_text();assert 'Ran 9 tests'in tests and tests.rstrip().endswith('OK')
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'recovered_tests_passed':8,'recovered_worker_tests_passed':1,'total_recovered_tests_passed':9,'recovered_test_log_sha256':sha(restore/'recovered_tests.log'),'verified_source_hashes':sources,'preserved_canonical_scoring_sha256':q['preserved_canonical_scoring_sha256'],'recovery_tests_load_preserved_scoring_formula':True,'no_real_panel_accuracy_result':True,'scientific_stage_admitted':False,'no_new_forecast_pool_started':True,'goal_remains_active':True,'scientific_originals_removed':False}
 (O/'selector_fold_scale_batched_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k not in('files','verified_source_hashes')},indent=2),flush=True)
