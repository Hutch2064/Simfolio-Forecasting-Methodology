"""Publish future independent audit source and verify its exact remote bytes."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;ENV={**os.environ,'GODEBUG':'http2client=0'};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
with(O/'asset_v6_publication.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
 names=['verify_selector_sweep_reserved.py','verify_selector_sweep_after_handoff.py','publish_selector_sweep_audit.py','selector_coefficient_sweep_public_recovery.json']
 for name in names:shutil.copy2(O/name,D/name)
 manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
 run(['git','add','--',REL]);run(['git','commit','-m','Preserve independent all-work replay and reserved-score verifier']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'selector_sweep_audit_recovery'/commit[:12];files=[]
 for name in names:
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as r:data=r.read()
  assert hashlib.sha256(data).hexdigest()==manifest[name];p=restore/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[name];files.append({'path':name,'sha256':sha(p),'bytes':len(data)})
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'future_real_data_audit_still_pending':True,'scientific_originals_removed':False};(O/'selector_sweep_audit_public_recovery.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
