"""Preserve completed asset-level circular hybrid; baseline and hybrid archives supply frozen dependencies."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,tarfile
OWN=Path(__file__).resolve().parent;V4=OWN/'asset_level_v6'
C=V4/'candidates/asset_empirical_mean';F=V4/'candidates/asset_mean_uncertainty/empirical_mean_screen100'
TAG='research-asset-v6-empirical-mean-pilot-20261021'
REPO='Hutch2064/simfolio-engine'
DEST=OWN/'archives/asset-v6-empirical-mean-pilot-20261021'
ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text())
run=lambda a:subprocess.check_output(a,env=ENV,text=True)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

result=read(C/'reserved/summary.json')
assert result['complete'] and result['cases_per_model']==355 and not result['advancing_model_ids']
assert result['native_failures']==0
assert read(F/'active.json')['status']=='terminal'
assert read(C/'decision.json')['not_full_panel'] is True
baseline_archive=read(OWN/'archives/asset-v6-baselines-1000-20261020/receipt.json')
assert baseline_archive['complete']
assert json.loads(run(['gh','repo','view',REPO,'--json','isPrivate']))['isPrivate']
DEST.mkdir(parents=True,exist_ok=True)
if not (DEST/'release_created.json').exists():
    notes=DEST/'notes.md';notes.write_text('Private closed empirical joint batch-mean uncertainty prototype. Complete710case100path screen,7101000path development forecasts and355reserved forecasts, frozen source, unit tests and rejection evidence. Zero forecast errors; no full-panel admission or catalogue increment. Restore baseline inputs and circular current-best dependencies before replay. No production changes.\n')
    run(['gh','release','create',TAG,'--repo',REPO,'--draft','--title','Joint-asset empirical mean pilot: complete rejected evidence','--notes-file',str(notes)])
    (DEST/'release_created.json').write_text(json.dumps({'tag':TAG,'repo':REPO})+'\n')
groups={'candidate': [p for p in sorted(C.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('-wal','-shm'))]}
groups['screen']=[p for p in sorted(F.rglob('*')) if p.is_file()]
groups['audit']=[OWN/name for name in ['screen_asset_empirical_mean.py','test_asset_empirical_mean.py','preserve_asset_adaptive_pilot.py']]

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
        mp=DEST/(name+'-manifest.json');mp.write_text(json.dumps({'schema':'v6-closed-asset-empirical-mean-pilot-evidence-v1','group':group,'part':i+1,'parts':len(chunks),'entries':entries},indent=2)+'\n')
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
    'baseline_input_archive_receipt_sha256':sha(OWN/'archives/asset-v6-baselines-1000-20261020/receipt.json'),
    'baseline_input_archive_release':baseline_archive['release_tag'],
    'hybrid_parent_archive_release':'research-asset-v6-hybrid-1000-20261021',
    'hybrid_parent_archive_receipt_sha256':sha(OWN/'archives/asset-v6-hybrid-1000-20261021/receipt.json'),
    'pilot_confirmation_summary_sha256':sha(F/'summary.json'),
    'circular_parent_archive_release':'research-asset-v6-circular-1000-20261021',
    'full_panel_candidate':False,
    'recovery':'Download each archive and its manifest with gh release download, verify both SHA256 values, extract at the research owner root, verify every per-file manifest size/hash; restore sealed numerical-parent archives as referenced in execution identities and prior repository RESTORE documentation.'}
(DEST/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='archives'}),flush=True)
