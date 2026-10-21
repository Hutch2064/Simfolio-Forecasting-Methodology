from pathlib import Path
import json,hashlib,subprocess,datetime,tempfile
O=Path(__file__).resolve().parent;D=O/'archives/asset-v6-circular-raw-retirement-20261021';repo='Hutch2064/simfolio-engine';tag='research-asset-v6-circular-1000-20261021';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[D/'receipt.json',D/'pre_retirement_manifest.json'];assert json.loads(files[0].read_text())['complete']
# Distinct metadata asset names do not replace any scientific archive or existing manifests.
names=['raw-checkpoint-retirement-receipt.json','retired-raw-checkpoint-manifest.json'];rows=[]
with tempfile.TemporaryDirectory(dir=D) as temp:
 T=Path(temp)
 for p,name in zip(files,names):
  dst=T/name;dst.write_bytes(p.read_bytes());subprocess.run(['gh','release','upload',tag,str(dst),'--repo',repo,'--clobber'],check=True)
  Q=T/'fresh';Q.mkdir(exist_ok=True);subprocess.run(['gh','release','download',tag,'--repo',repo,'--pattern',name,'--dir',str(Q)],check=True);assert sha(Q/name)==sha(p)
  rows.append({'asset':name,'sha256':sha(p),'bytes':p.stat().st_size,'fresh_downloaded_disk_written_sha256_verified':True})
(D/'metadata_remote_recovery.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'repository':repo,'release_tag':tag,'files':rows},indent=2)+'\n')
print('Verified retirement metadata remote recovery',len(rows))
