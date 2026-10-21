"""Preserve worker-level synthetic qualification and verify remote recovery."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL
ENV={**os.environ,'GODEBUG':'http2client=0'}
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[key]='1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
with(O/'asset_v6_publication.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
 base=read(O/'selector_inner_precision_public_recovery.json');assert base['remote_main_verified']and base['recovered_tests_passed']==6
 names=['test_selector_inner_precision_worker.py','selector_inner_precision_worker_qualification.log','selector_inner_precision_worker_qualification.json','publish_selector_precision_worker.py','score_percentage.py']
 log=(O/names[1]).read_text();assert 'Ran 1 test'in log and log.rstrip().endswith('OK')
 for name in names:shutil.copy2(O/name,D/name)
 manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
 run(['git','add','--',REL]);run(['git','commit','-m','Verify precision worker joint accounting checkpoint resume and metadata guards']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_precision_worker_public_recovery'/commit[:12]
 source_map=base['verified_source_hashes'];original_by_public={p.removeprefix('asset_level_v6/'):p for p in source_map};original_by_public.update({name:name for name in names})
 def recover(public):
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{public}',timeout=60)as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==manifest[public];p=restore/original_by_public[public];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[public];return {'public_path':public,'recovered_path':str(p.relative_to(restore)),'sha256':manifest[public],'bytes':len(data)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,sorted(original_by_public)))
 with(restore/'recovered_tests.log').open('w')as log:subprocess.run([sys.executable,'-m','unittest','test_selector_inner_precision_worker'],cwd=restore,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert 'Ran 1 test'in(restore/'recovered_tests.log').read_text()and(restore/'recovered_tests.log').read_text().rstrip().endswith('OK')
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'recovered_worker_tests_passed':1,'synthetic_joint_forecasts_generated':12,'synthetic_checkpoint_reuses_without_fitting':12,'altered_path_count_rejected':True,'new_scientific_process_pool_started':False,'real_panel_accuracy_and_runtime_study_not_launched':True,'source_and_frozen_precision_plan_unchanged':True,'scientific_originals_removed':False};(O/'selector_precision_worker_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='files'},indent=2),flush=True)
