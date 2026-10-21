"""Retire only closed dependence/weight fit caches after verified private recovery."""
from pathlib import Path
import json,subprocess,hashlib,shutil,datetime,os
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates';A=O/'archives/asset-v6-sample-refocus-20261021';D=O/'archives/asset-v6-sample-fit-cache-retirement-20261021';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();env={**os.environ,'GODEBUG':'http2client=0'}
assert not(D/'receipt.json').exists();a=read(A/'receipt.json');assert a['complete']and a['files_fresh_downloaded_disk_restored_sha256_verified']==49387
phases=[C/'asset_dynamic_correlation/screen100',*[C/'asset_component_weights'/name for name in('screen120','development1000','reserved1000')]]
for phase in phases:assert read(phase/'active.json')['status']=='terminal'
processes=subprocess.check_output(['ps','-axo','command='],text=True)
for runner in('screen_asset_dynamic_correlation_v2.py','screen_asset_component_weights.py','run_asset_component_confirmation.py','run_asset_component_reserved.py'):
 assert str(O/runner)not in processes,'A closed cache still has an active reader'
releases=json.loads(subprocess.check_output(['gh','api',f"repos/{a['repository']}/releases?per_page=100"],text=True,env=env));release=next(r for r in releases if r['tag_name']==a['release_tag']);assets={x['name']:x for x in release['assets']}
D.mkdir(parents=True);fresh=D/'fresh';fresh.mkdir();entries={};proof=[]
for part in a['archives']:
 assert part['remote_download_sha256_verified'] and part['all_files_disk_restored_and_sha256_verified']
 for kind in ('archive','manifest'):assert assets[part[kind+'_name']]['digest']=='sha256:'+part[kind+'_sha256']
 subprocess.run(['gh','release','download',a['release_tag'],'--repo',a['repository'],'--pattern',part['manifest_name'],'--dir',str(fresh)],env=env,check=True)
 p=fresh/part['manifest_name'];assert sha(p)==part['manifest_sha256']
 for e in read(p)['entries']:
  path=O/e['path']
  if path.suffix=='.npz' and any(path.is_relative_to(phase/'fits')for phase in phases):
   assert not path.is_symlink() and path.stat().st_size==e['bytes'] and sha(path)==e['sha256'];entries[e['path']]=e
 proof.append({'archive':part['archive_name'],'server_digest':assets[part['archive_name']]['digest'],'fresh_manifest_sha256':sha(p)})
actual={str(p.relative_to(O))for phase in phases for p in(phase/'fits').glob('*.npz')};assert set(entries)==actual and len(entries)>0
before=shutil.disk_usage(O).free
(D/'manifest.json').write_text(json.dumps({'entries':entries,'fresh_online_proof':proof,'archive_receipt_sha256':sha(A/'receipt.json')},indent=2)+'\n')
for name,e in entries.items():
 p=O/name;assert p.stat().st_size==e['bytes'];p.unlink()
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'files_retired':len(entries),'bytes_retired':sum(e['bytes']for e in entries.values()),'disk_free_before_bytes':before,'disk_free_after_bytes':shutil.disk_usage(O).free,'raw_losses_sources_plans_and_summaries_preserved_locally':True,'active_broad_fits_ephemeral_not_affected':True,'private_release':a['release_tag'],'repository':a['repository'],'manifest_sha256':sha(D/'manifest.json'),'fresh_remote_digests_manifests_prior_complete_disk_restore_verified':True,'restore':'Download the private development release archive parts and manifests, verify all SHA256 values, extract at owner root and verify each recovered member against the manifest.'}
(D/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
for phase in phases:(phase/'ARCHIVED_FIT_CACHE.json').write_text(json.dumps(r,indent=2)+'\n')
shutil.rmtree(fresh);print(json.dumps(r),flush=True)
