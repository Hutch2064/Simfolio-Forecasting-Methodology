"""Archive the closed parity/reserved stage; never retire active scientific inputs."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,tarfile
OWN=Path(__file__).resolve().parent;V=OWN/'asset_level_v6'
TAG='research-asset-v6-selector-reserved-20261021';REPO='Hutch2064/simfolio-engine';DEST=OWN/'archives/asset-v6-selector-reserved-20261021';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());run=lambda a:subprocess.check_output(a,env=ENV,text=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
subprocess.run([os.sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.6','--artifact-root',str(OWN),'--budget-gib','16'],check=True)
B=V/'candidates/asset_selector_batched';D=B/'validation_and_reserved';audit=read(D/'verification_receipt.json');plan=read(D/'plan.json')
assert read(D/'active.json')['status']=='terminal'and audit['verified']and audit['development_parity_loss_vectors_verified']==710
assert audit['reserved_candidate_checkpoints_verified']==355 and audit['reserved_reference_checkpoints_verified']==710
assert all(sha(OWN/n)==h for n,h in plan['source_hashes'].items())
assert read(OWN/'selector_precision_recovered_tests.json')['tests_passed']==14
assert json.loads(run(['gh','repo','view',REPO,'--json','isPrivate']))['isPrivate']
DEST.mkdir(parents=True,exist_ok=True)
if not(DEST/'release_created.json').exists():
 notes=DEST/'notes.md';notes.write_text('Private closed adaptive selector parity/reserved evidence:355historical requests exactly replayed,710outerloss vectors and3704788innercomponent ensembles verified.355distinct reserved1000path forecasts and710reference losses independently audited. Pool5.196151%better than fixedWR on the reserved slice, versus priorworking4.037528%; no full-panel promotion or independence claim. All14recovered precision/batch tests and7recovered newcriterion tests passed. Active path-objective screen outputs excluded. Original scientific inputs retained.\n')
 run(['gh','release','create',TAG,'--repo',REPO,'--draft','--title','Joint selector parity and reserved validation','--notes-file',str(notes)])
 (DEST/'release_created.json').write_text(json.dumps({'tag':TAG,'repo':REPO})+'\n')
files=lambda p:[f for f in sorted(p.rglob('*'))if f.is_file()and '__pycache__'not in f.parts and f.suffix not in('.nbc','.nbi')]
groups={'closed_complete_parity_and_reserved':files(D),'pinned_source_closure':[OWN/n for n in sorted(plan['source_hashes'])]}
C=V/'candidates/asset_selector_path_objective'
groups['new_path_criterion_qualified_sources']=[f for f in C.iterdir()if f.is_file()and(f.suffix in('.py','.md')or f.name=='qualification_receipt.json')]
names=['preserve_selector_reserved.py','verify_selector_batched_reserved.py','selector_batched_reserved_independent_audit.log','selector_batched_validation.log','selector_precision_recovered_tests.json','selector_precision_recovered_tests.log','selector_path_public_recovery.json','selector_path_recovered_tests.log','selector_path_full_qualification.log','test_selector_path_candidate.py','test_selector_path_coefficients.py','screen_selector_path_objective.py','diagnose_selector_pool_development.py','correct_historical_percentage_metadata.py']
groups['closed_audit_and_source_evidence']=[OWN/n for n in names]+[V/'candidates/asset_selector_universe_certified/diagnostics.json']
seen=set()
for group,paths in groups.items():
 unique=[]
 for f in paths:
  if f not in seen:unique.append(f);seen.add(f)
 groups[group]=sorted(unique)

allreceipts=[]
for group,paths in groups.items():
    chunks=[];current=[];size=0
    for p in paths:
        assert not p.is_symlink()
        n=p.stat().st_size
        if current and size+n>128*1024**2:
            chunks.append(current);current=[];size=0
        current.append(p);size+=n
    if current:chunks.append(current)
    for i,files in enumerate(chunks):
        name=group+f'-part-{i+1:02d}'
        receipt_path=DEST/(name+'-receipt.json')
        if receipt_path.exists():
            allreceipts.append(read(receipt_path));continue
        entries=[{'path':str(p.relative_to(OWN)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
        mp=DEST/(name+'-manifest.json');mp.write_text(json.dumps({'schema':'v6-closed-feature-development-evidence-v1','group':group,'part':i+1,'parts':len(chunks),'entries':entries},indent=2)+'\n')
        archive=DEST/(name+'.tar.gz')
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for e in entries:tar.add(OWN/e['path'],arcname=e['path'],recursive=False)
        assert archive.stat().st_size<1900000000
        digest=sha(archive);manifest_digest=sha(mp)
        run(['gh','release','upload',TAG,str(archive),str(mp),'--repo',REPO,'--clobber'])
        download=DEST/'download';download.mkdir(exist_ok=False)
        run(['gh','release','download',TAG,'--repo',REPO,'--pattern',archive.name,'--pattern',mp.name,'--dir',str(download)])
        assert sha(download/archive.name)==digest and sha(download/mp.name)==manifest_digest
        restore=DEST/'restore';restore.mkdir(exist_ok=False)
        expected={e['path']:e for e in entries};seen=set()
        with tarfile.open(download/archive.name,'r:gz') as tar:
            for member in tar:
                assert member.isfile() and member.name in expected and member.name not in seen
                target=restore/member.name
                assert target.resolve().is_relative_to(restore.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
                e=expected[member.name]
                assert target.stat().st_size==e['bytes'] and sha(target)==e['sha256']
                assert sha(OWN/e['path'])==e['sha256'], 'Closed scientific bytes changed during preservation'
                seen.add(member.name);target.unlink()
        assert seen==set(expected)
        receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':REPO,'release_tag':TAG,
            'group':group,'part':i+1,'parts':len(chunks),'archive_name':archive.name,'archive_sha256':digest,
            'manifest_name':mp.name,'manifest_sha256':manifest_digest,'archive_bytes':archive.stat().st_size,
            'files':len(entries),'remote_download_sha256_verified':True,'all_files_disk_restored_and_sha256_verified':True,
            'remote_release_url':f'https://github.com/{REPO}/releases/tag/{TAG}','scientific_originals_removed':False}
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');allreceipts.append(receipt)
        shutil.rmtree(download);shutil.rmtree(restore);archive.unlink()
        print(json.dumps(receipt),flush=True)
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'repository':REPO,'release_tag':TAG,'archives':allreceipts,'files_fresh_downloaded_disk_restored_sha256_verified':sum(r['files']for r in allreceipts),'scientific_originals_removed':False,'full_panel_catalogue_increment':0,'dependency_releases':['research-asset-v6-broad-selector-precision-20261021','research-asset-v6-horizon-pool-screen-20261021','research-asset-v6-broad-selector-screen-20261021','research-asset-v6-baselines-1000-20261020','research-asset-v6-feature-development-20261021','research-asset-v6-feature-broad-20261021','research-asset-v6-fading-1000-20261021','research-asset-v6-learned-pool-broad-20261021','research-asset-v6-learned-pool-pilots-20261021','research-asset-v6-sample-refocus-20261021'],'recovery':'Download and SHA256verify all archive and manifest pairs; disk restore at research owner root and check every per-file manifest hash. Restore referenced baseline inputs and sealed parent dependencies first.'}
(DEST/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='archives'}),flush=True)
