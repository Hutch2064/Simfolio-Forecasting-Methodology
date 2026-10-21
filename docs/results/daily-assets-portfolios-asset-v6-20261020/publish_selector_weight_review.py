"""Preserve a methodology review and expanded saved-coefficient audit."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL
ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(args):return subprocess.check_output(args,cwd=R,env=ENV,text=True)
names=['selector_weight_uncertainty_review.md','verify_selector_lazy_closed_coefficients.py','selector_lazy_closed_coefficients_expanded.json','selector_lazy_closed_coefficients_expanded.log','publish_selector_weight_review.py']
audit=read(O/names[2]);assert audit['closed_checkpoints_read']>14 and audit['global_all_option_certificates_verified']and audit['all_original_coefficient_checkpoints_unchanged']
assert audit['verifier_sha256']==sha(O/names[1])
assert all(sha(O/r['coefficient_checkpoint'])==r['checkpoint_sha256']for r in audit['records'])
with(O/'asset_v6_publication.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
 for name in names:shutil.copy2(O/name,D/name)
 manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 assert all(line[3:].startswith(REL+'/')for line in run(['git','status','--porcelain']).splitlines())
 run(['git','add','--',REL]);run(['git','commit','-m','Review fold-weight uncertainty and extend certified saved-coefficient study']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_weight_review_public_recovery'/commit[:12]
 def recover(name):
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==sha(D/name);p=restore/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==sha(D/name)
  if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
  return {'public_path':name,'sha256':sha(p),'bytes':len(data)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,names+['PUBLIC_FILES.json']))
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'saved_coefficient_objectives_verified':audit['coefficient_objectives_verified'],'all_option_optimization_certificates':True,'source_compilation_only_after_recovery':True,'remote_reexecution_of_private_active_coefficients_claimed':False,'no_new_method_or_forecast_pool_admitted':True,'goal_remains_active':True,'scientific_originals_removed':False}
 (O/'selector_weight_review_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
