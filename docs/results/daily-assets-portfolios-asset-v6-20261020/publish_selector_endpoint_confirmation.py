"""Preserve the completed information-origin study, its source and measured control stop."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,urllib.request,time

O=Path(__file__).resolve().parent;V=O/'asset_level_v6';R=O/'repository'
C=V/'candidates/asset_selector_endpoint_precision'
REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL
ENV={**os.environ,'GODEBUG':'http2client=0'}
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[k]='1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)

plan=read(C/'study1000/plan.json');assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
with (O/'asset_v6_publication.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not run(['git','status','--porcelain']).strip()
 run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
 q=read(C/'qualification_receipt.json');assert q['tests_passed']==3 and q['worker_tests_passed']==1
 source_map={**plan['source_hashes'],**read(V/'candidates/asset_selector_endpoint_components/study1000/plan.json')['source_hashes']}
 names=set(source_map)|{'publish_selector_endpoint_confirmation.py','selector_component_coverage_review_v2.md','selector_coverage_document_version_repair.json','selector_information_endpoint_results.md','selector_information_endpoint_verification.json','selector_endpoint_precision_results.md','selector_endpoint_precision_verification.json','selector_endpoint_precision_qualification.log','selector_endpoint_components_qualification_complete.log','asset_level_v6/candidates/asset_selector_endpoint_precision/qualification_receipt.json','asset_level_v6/candidates/asset_selector_endpoint_precision/study1000/plan.json','asset_level_v6/candidates/asset_selector_endpoint_components/qualification_receipt.json','asset_level_v6/candidates/asset_selector_endpoint_components/study1000/plan.json','selector_endpoint_evidence_public_recovery.json'}

 original_by_public={p.removeprefix('asset_level_v6/'):p for p in names}
 for public,original in original_by_public.items():
  source=O/original;target=D/public
  if original in source_map:assert sha(source)==source_map[original]
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='PUBLIC_FILES.json' and '__pycache__' not in p.parts}
 (D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
 run(['git','add','--',REL]);run(['git','commit','-m','Preserve repeated-seed endpoint precision evidence and qualified component interaction study'])
 run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip()
 restore=O/'selector_endpoint_confirmation_public_recovery'/commit[:12]
 def recover(public):
  error=None
  for attempt in range(4):
   try:
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{public}',timeout=30)as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==manifest[public]
    target=restore/original_by_public[public];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    assert sha(target)==manifest[public]
    return {'public_path':public,'recovered_path':str(target.relative_to(restore)),'sha256':manifest[public],'bytes':len(data)}
   except Exception as e:error=e;time.sleep(attempt+1)
  raise error
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(recover,sorted(original_by_public)))
 with (restore/'recovered_tests.log').open('w') as output:
  subprocess.run([sys.executable,'-m','unittest','test_selector_endpoint_cache','test_selector_endpoint_precision_worker','test_selector_endpoint_components','test_selector_endpoint_components_worker'],cwd=restore,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=True)
 tests=(restore/'recovered_tests.log').read_text();assert 'Ran 10 tests' in tests and tests.rstrip().endswith('OK')
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'recovered_tests_passed':10,'recovered_test_log_sha256':sha(restore/'recovered_tests.log'),'verified_source_hashes':source_map,'source_scope':'Verified4260forecast repeated-seed precision study and qualified3240forecast component interaction source; exact historical documentation restored and revised review versioned separately','goal_remains_active':True,'scientific_originals_removed':False,'official_score_changed':False,'full_panel_catalogue_increment':0}
 (O/'selector_endpoint_confirmation_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ('files','verified_source_hashes')},indent=2),flush=True)
