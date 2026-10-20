"""Preserve the closed v5 suite losslessly; do not remove scientific originals."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,tarfile
OWN=Path(__file__).resolve().parent;V4=OWN/'simulation_1000_v5'
TAG='research-portfolio-v5-1000-20261020'
REPO='Hutch2064/simfolio-engine'
DEST=OWN/'archives/portfolio-v5-1000-20261020'
ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text())
run=lambda a:subprocess.check_output(a,env=ENV,text=True)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

assert read(OWN/'v5_full_suite_independent_audit.json')['complete_retained_suite_audit']
assert read(OWN/'v5_stable_parity_audit.json')['full_panel_audit_complete']
assert read(V4/'reporter_active.json')['status']=='terminal'
assert read(V4/'controller_active.json')['status']=='terminal'
assert read(OWN/'v5_stable_parity_audit.json')['mismatches']==0
assert json.loads(run(['gh','repo','view',REPO,'--json','isPrivate']))['isPrivate']
assert not list((V4/'results').glob('*sqlite3-wal'))
DEST.mkdir(parents=True,exist_ok=True)
if not (DEST/'release_created.json').exists():
    notes=DEST/'notes.md';notes.write_text('Private complete historical v5 daily assets/portfolios 1000-simulation panel. Four active completed methods plus the original positive-wealth-overflow execution,1264series,64049attempts/model. All256196 active checkpoints independently audited;64048 repaired/native loss vectors bitwise exact and known failed origin separately verified. One historical generation failure,zero active failures,zero scoring integrity errors. Complete data/source/checkpoints/index. Future asset-level V6 is a separate architecture. Wider accuracy research remains active. Existing sealed numerical-parent archives are necessary recovery parents. No production changes.\n')
    run(['gh','release','create',TAG,'--repo',REPO,'--draft','--title','Historical portfolio v5 1000 paths: complete reproducible evidence','--notes-file',str(notes)])
    (DEST/'release_created.json').write_text(json.dumps({'tag':TAG,'repo':REPO})+'\n')
models=sorted(p for p in (V4/'results/models').iterdir() if p.is_dir())
groups={p.name:sorted(f for f in p.rglob('*') if f.is_file()) for p in models}
groups['shared']=[p for p in sorted(V4.rglob('*')) if p.is_file() and not p.is_relative_to(V4/'results/models') and '__pycache__' not in p.parts]
groups['audit']=[OWN/name for name in ['verify_v5_full_suite.py','v5_full_suite_independent_audit.json','audit_v5_stable_parity.py','v5_stable_parity_audit.json','v5_derived_index_close_receipt.json','v5_native_failure_diagnostic.json','v5_stable_failed_origin_replay.json']]+sorted((OWN/'v5_audits').glob('*'))
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
        mp=DEST/(name+'-manifest.json');mp.write_text(json.dumps({'schema':'v5-complete-private-evidence-v1','group':group,'part':i+1,'parts':len(chunks),'entries':entries},indent=2)+'\n')
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
