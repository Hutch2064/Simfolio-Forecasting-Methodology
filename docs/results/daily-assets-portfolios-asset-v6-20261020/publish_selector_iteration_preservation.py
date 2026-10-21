"""Link verified private scientific archives from the public research repository."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';A=O/'archives/selector-iteration-studies-20261021T1810Z';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
a=read(A/'receipt.json');assert a['complete']and a['files_fresh_downloaded_disk_restored_sha256_verified']==18961
assert all(r['remote_download_sha256_verified']and r['all_files_disk_restored_and_sha256_verified']for r in a['archives'])
release=json.loads(run(['gh','release','view',a['release_tag'],'--repo',a['repository'],'--json','isDraft']));assert not release['isDraft']
with(O/'asset_v6_publication.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
 name='archives/selector-iteration-studies-20261021T1810Z';target=D/name;target.mkdir(parents=True,exist_ok=True)
 mapping={name+'/receipt.json':A/'receipt.json',name+'/local_duplicate_retirement.json':A/'local_duplicate_retirement.json','preserve_selector_iteration_studies.py':O/'preserve_selector_iteration_studies.py','recover_selector_information_publication.py':O/'recover_selector_information_publication.py','publish_selector_iteration_preservation.py':Path(__file__),'selector_next_learner_research.md':O/'selector_next_learner_research.md','selector_information_origins_public_recovery.json':O/'selector_information_origins_public_recovery.json'}
 for public,source in mapping.items():p=D/public;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,p)
 restore=target/'RESTORE.md';restore.write_text('''# Verified private research recovery

Private release: https://github.com/Hutch2064/simfolio-engine/releases/tag/research-asset-v6-selector-iteration-studies-20261021T1810Z

The public receipt records11archive/manifest pairs. Each archive and manifest was freshly downloaded;18,961members were disk-restored and checked for byte length, SHA256and file mode. Source publication30b9c2ec8f476a19f854195b25bbc32ce431435bhas its own175file fresh-recovery receipt and7passing recovered tests.

1. Authenticate gh with access to the private repository. Download the exact named release into a new empty staging directory:

```sh
gh release download research-asset-v6-selector-iteration-studies-20261021T1810Z --repo Hutch2064/simfolio-engine --dir /chosen/new-empty-directory
```

2. Verify every archive/manifest SHA256against receipt.json before extraction. The per-part manifest supplies every member's relative path, byte length, mode and SHA256.
3. Restore into a separate empty research root, preserving relative paths and modes; reject absolute paths, traversal, symlinks, duplicates or unknown members. Check every restored file against its manifest. The preservation script in this source snapshot demonstrates the exact streamed verification procedure.
4. Recover referenced baseline inputs, fixed-reference checkpoints and sealed parent dependencies from the receipt's dependency releases before reproducing forecasts. Recorded absolute paths reflect the original research owner; preserve their relative layout or explicitly map them to the recovered root.
5. Copy into an existing research root only after matching existing files or preserving differences. Never overwrite active or unique work. Do not automatically restart a run or recompute scores merely because archived duplicate files are absent.

The8request all-origin control is intentionally stopped. Other included finite studies are complete. The information-originV2auditor corrects the auxiliary unsupported-quarter subset report while preserving the frozen original verifier. Its355request main scores are fully audited. Neither stage completion nor archive recovery completes the active accuracy-maximization goal.

Individual loss checkpoints, records, summaries and reference files remain locally needed for active comparisons. Completed per-case aggregate duplicates and regenerable source-recovery compiler caches were retired after verification; local_duplicate_retirement.jsonrecords exact paths and evidence. This changes no numerical source or score. Archives preserve rejected models and all generated evidence, not only winners.
''')
 mapping[name+'/RESTORE.md']=restore
 manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and'__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
 run(['git','add','--',REL]);run(['git','commit','-m','Link recovery-verified selector iteration archives and preserve open learner research']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();recovered=O/'selector_iteration_preservation_public_recovery'/commit[:12]
 def recover(public):
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{public}',timeout=60)as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==manifest[public];p=recovered/public;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==manifest[public];return{'public_path':public,'sha256':manifest[public],'bytes':len(data)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,sorted(mapping)))
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_remote_files_disk_restored_SHA256_verified':len(files),'files':files,'private_files_verified':18961,'private_release_tag':a['release_tag'],'full_panel_catalogue_increment':0,'goal_remains_active':True,'scientific_checkpoints_retained':True};(O/'selector_iteration_preservation_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
