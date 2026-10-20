"""Preserve the closed joint-asset baseline pair; never retire active dependencies."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,tarfile
OWN=Path(__file__).resolve().parent;V4=OWN/'asset_level_v6'
TAG='research-asset-v6-baselines-1000-20261020'
REPO='Hutch2064/simfolio-engine'
DEST=OWN/'archives/asset-v6-baselines-1000-20261020'
ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text())
run=lambda a:subprocess.check_output(a,env=ENV,text=True)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

assert read(V4/'full_pair_independent_audit.json')['complete_pair_audit']
assert read(V4/'reporter_active.json')['status']=='terminal'
assert read(V4/'controller_active.json')['status']=='terminal'
assert read(V4/'results/scoreboard.json')['scoring_integrity_errors']==0
assert read(V4/'results/status.json')['completed_active_tasks']==128098
assert json.loads(run(['gh','repo','view',REPO,'--json','isPrivate']))['isPrivate']
# Only the closed derived index is changed; scientific checkpoints stay immutable.
import sqlite3
index=V4/'results/native-loss-index.sqlite3'
with sqlite3.connect(index) as connection:
    assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]==0
assert not any(p.stat().st_size for p in (V4/'results').glob('*sqlite3-wal'))
DEST.mkdir(parents=True,exist_ok=True)
if not (DEST/'release_created.json').exists():
    notes=DEST/'notes.md';notes.write_text('Private closed joint-asset V6 baseline pair at1000simulations. Two completed methods,1264series,64049tasks/model,128098successes,zero native failures and zero scoring integrity errors. All checkpoint identities/hashes and unchanged single-score reduction independently audited. This release preserves complete input histories, matrices, frozen implementation/control/profile and both raw loss trees. Candidate research is separate, active and not included as a completed baseline. Existing sealed numerical-parent archives remain necessary dependencies. No production changes.\n')
    run(['gh','release','create',TAG,'--repo',REPO,'--draft','--title','Joint-asset V6 baselines 1000 paths: complete evidence','--notes-file',str(notes)])
    (DEST/'release_created.json').write_text(json.dumps({'tag':TAG,'repo':REPO})+'\n')
models=sorted(p for p in (V4/'results/models').iterdir() if p.is_dir())
groups={p.name:sorted(f for f in p.rglob('*') if f.is_file()) for p in models}
groups['shared']=[p for p in sorted(V4.rglob('*')) if p.is_file() and not p.is_relative_to(V4/'results/models') and not p.is_relative_to(V4/'candidates') and '__pycache__' not in p.parts]
groups['audit']=[OWN/name for name in ['audit_asset_v6_full_pair.py','bind_asset_prior_parent.py','continue_asset_v6_research.py','publish_asset_v6_prepared.py','test_asset_v6_decomposition.py','diagnose_asset_v6_pair.py']]

allreceipts=[]
for group,paths in groups.items():
    chunks=[];current=[];size=0
    for p in paths:
        assert not p.is_symlink()
        n=p.stat().st_size
        if current and size+n>256*1024**2:
            chunks.append(current);current=[];size=0
        current.append(p);size+=n
    if current:chunks.append(current)
    for i,files in enumerate(chunks):
        name=group+f'-part-{i+1:02d}'
        receipt_path=DEST/(name+'-receipt.json')
        if receipt_path.exists():
            allreceipts.append(read(receipt_path));continue
        entries=[{'path':str(p.relative_to(OWN)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
        mp=DEST/(name+'-manifest.json');mp.write_text(json.dumps({'schema':'v6-closed-asset-baseline-pair-evidence-v1','group':group,'part':i+1,'parts':len(chunks),'entries':entries},indent=2)+'\n')
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
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'repository':REPO,'release_tag':TAG,
    'archives':allreceipts,'files_fresh_downloaded_disk_restored_sha256_verified':sum(r['files'] for r in allreceipts),
    'scientific_originals_removed':False,'inputs_and_index_remain_active_research_dependencies':True,
    'recovery':'Download each archive and its manifest with gh release download, verify both SHA256 values, extract at the research owner root, verify every per-file manifest size/hash; restore sealed numerical-parent archives as referenced in execution identities and prior repository RESTORE documentation.'}
(DEST/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='archives'}),flush=True)
