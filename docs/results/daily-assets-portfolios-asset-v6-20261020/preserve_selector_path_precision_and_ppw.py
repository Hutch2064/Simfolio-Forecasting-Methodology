"""Preserve closed1kcriterion and100pathmenu results with their display erratum."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,tarfile
OWN=Path(__file__).resolve().parent;V=OWN/'asset_level_v6'
TAG='research-asset-v6-path-precision-and-PPW-screen-20261021';REPO='Hutch2064/simfolio-engine';DEST=OWN/'archives/asset-v6-path-precision-and-PPW-screen-20261021';ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text());run=lambda a:subprocess.check_output(a,env=ENV,text=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
subprocess.run([os.sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.8','--artifact-root',str(OWN),'--budget-gib','16'],check=True)
P=V/'candidates/asset_selector_path_objective/screen1000';Q=V/'candidates/asset_selector_ppw/screen100'
for stage in (P,Q):
 assert read(stage/'active.json')['status']=='terminal'and read(stage/'verification_receipt.json')['verified']
 assert all(sha(OWN/n)==h for n,h in read(stage/'plan.json')['source_hashes'].items())
assert sha(P/'summary.json')==read(P/'verification_receipt.json')['summary_sha256']
assert sha(P/'summary.corrected.json')==read(P/'percentage_metadata_erratum.json')['corrected_summary_sha256']
assert json.loads(run(['gh','repo','view',REPO,'--json','isPrivate']))['isPrivate']
DEST.mkdir(parents=True,exist_ok=True)
if not(DEST/'release_created.json').exists():
 notes=DEST/'notes.md';notes.write_text('Private completed studies:72cases x2criteria at1000inner/final paths,144loss vectors and216reference checkpoints audited.Whole-path21.309474%better than fixedWR vs20.540747%endpoint,positive precision gate.No full-panel claim.Original driver/raw summary had a1000rather than100display multiplier;immutable originals and corrected sidecars/erratum preserve this reporting-only correction.Two regression tests cover display scale.In the4way100pathmenu screen all288forecasts and288reference checkpoints audited,144original controls exactly replayed;PPWmenu variants did not improve this exploratory slice.ThirteenCPU profiles retained.Active355case broader outputs excluded.\n')
 run(['gh','release','create',TAG,'--repo',REPO,'--draft','--title','Whole-path precision and shared PPW menu screen','--notes-file',str(notes)])
 (DEST/'release_created.json').write_text(json.dumps({'tag':TAG,'repo':REPO})+'\n')
files=lambda p:[f for f in sorted(p.rglob('*'))if f.is_file()and '__pycache__'not in f.parts and f.suffix not in('.nbc','.nbi')]
closure={**read(P/'plan.json')['source_hashes'],**read(Q/'plan.json')['source_hashes']}
groups={'closed_1000path_criterion_precision':files(P),'closed_100path_four_way_menu':files(Q),'pinned_sources':[OWN/n for n in sorted(closure)]}
names=['preserve_selector_path_precision_and_ppw.py','run_selector_path_precision72.py','run_selector_path_development355.py','screen_selector_ppw_menu.py','preflight_selector_ppw_screen.py','verify_selector_path_stage.py','verify_selector_ppw_stage.py','score_percentage.py','correct_closed_selector_percentages.py','test_score_percentage.py','score_percentage_regression.log','selector_path_precision_percentage_correction.log','selector_path_precision72.log','selector_ppw_screen_bound.log','selector_ppw_independent_verification.log','selector_ppw_preflight.log','selector_ppw_public_recovery.json','selector_ppw_public_recovery.initial8.json','selector_ppw_public_recovery.qualified9.json','selector_ppw_component_qualification.log','selector_ppw_integrated_qualification.log','selector_ppw_shared_contrast_qualification.log','selector_ppw_recovered_tests.log','selector_ppw_recovered_tests.initial8.log','selector_path_coefficient_runtime_diagnostic.json','selector_path_worker_profile.txt','selector_path_worker_profile_receipt.json','selector_path_column_research_preflight.json']
groups['closed_audits_and_new_broad_driver_source']=[OWN/n for n in names]+[f for f in(V/'candidates/asset_selector_ppw').iterdir()if f.is_file()and(f.suffix in('.py','.md')or f.name in('qualification_receipt.json','research_basis.json','SCREEN_PROTOCOL.json','preflight_receipt.json','queue_startup_binding_revision.json'))]
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
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'repository':REPO,'release_tag':TAG,'archives':allreceipts,'files_fresh_downloaded_disk_restored_sha256_verified':sum(r['files']for r in allreceipts),'scientific_originals_removed':False,'full_panel_catalogue_increment':0,'dependency_releases':['research-asset-v6-selector-pathscreen-20261021','research-asset-v6-selector-reserved-20261021','research-asset-v6-broad-selector-precision-20261021','research-asset-v6-horizon-pool-screen-20261021','research-asset-v6-broad-selector-screen-20261021','research-asset-v6-baselines-1000-20261020','research-asset-v6-feature-development-20261021','research-asset-v6-feature-broad-20261021','research-asset-v6-fading-1000-20261021','research-asset-v6-learned-pool-broad-20261021','research-asset-v6-learned-pool-pilots-20261021','research-asset-v6-sample-refocus-20261021'],'recovery':'Download and SHA256verify all archive and manifest pairs; disk restore at research owner root and check every per-file manifest hash. Restore referenced baseline inputs and sealed parent dependencies first.'}
(DEST/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='archives'}),flush=True)
