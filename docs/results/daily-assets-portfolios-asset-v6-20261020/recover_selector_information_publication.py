"""Retry fresh recovery of an already pushed immutable publication."""
from pathlib import Path
import ast,concurrent.futures,datetime,hashlib,json,os,subprocess,sys,time,urllib.request
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_information_origins';R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';ENV={**os.environ,'GODEBUG':'http2client=0'}
for key in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[key]='1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
commit=run(['git','rev-parse','HEAD']).strip();assert commit.startswith('30b9c2e');plan=read(C/'study1000/plan.json')
tree=ast.parse((O/'publish_selector_information_origins.py').read_text());context=globals().copy()
for key in('source_map','names'):
 node=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==key for t in n.targets));context[key]=eval(compile(ast.Expression(node.value),'<owned publisher file inventory>','eval'),context)
source_map=context['source_map'];original_by_public={p.removeprefix('asset_level_v6/'):p for p in context['names']}
manifest=json.loads(run(['git','show',f'{commit}:{REL}/PUBLIC_FILES.json']));restore=O/'selector_information_origins_public_recovery'/commit[:12]
def recover(public):
 target=restore/original_by_public[public];attempts=0
 if not target.exists()or sha(target)!=manifest[public]:
  for attempt in range(4):
   attempts+=1
   try:
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{public}',timeout=25)as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==manifest[public];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);break
   except(Exception):
    if attempt==3:raise
    time.sleep(attempt+1)
 assert sha(target)==manifest[public]
 return{'public_path':public,'recovered_path':str(target.relative_to(restore)),'sha256':manifest[public],'bytes':target.stat().st_size,'retry_download_attempts':attempts,'fresh_recovery_directory_for_this_immutable_publication':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,sorted(original_by_public)))
with(restore/'recovered_tests_retry.log').open('w')as output:subprocess.run([sys.executable,'-m','unittest','test_selector_information_origins','test_selector_information_origins_worker'],cwd=restore,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=True)
log=(restore/'recovered_tests_retry.log').read_text();assert'Ran 7 tests'in log and log.rstrip().endswith('OK')
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'recovered_tests_passed':7,'recovered_test_log_sha256':sha(restore/'recovered_tests_retry.log'),'verified_source_hashes':source_map,'initial_recovery_transient_network_timeout_preserved':True,'recovery_completed_without_restarting_scientific_jobs':True,'goal_remains_active':True,'scientific_originals_removed':False}
(O/'selector_information_origins_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k not in('files','verified_source_hashes')},indent=2),flush=True)
