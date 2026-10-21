"""Retire the replaced hybrid's raw checkpoints only after fresh remote proof."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_prior_best'
ARCH=O/'archives/asset-v6-hybrid-1000-20261021';DEST=O/'archives/asset-v6-hybrid-raw-retirement-20261021'
ENV={**os.environ,'GODEBUG':'http2client=0'}
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    assert not (DEST/'receipt.json').exists(),'Retirement already recorded; inspect rather than repeat'
    current=read(V/'official_scoreboard.json')
    assert current['current_best_model_id']=='model_testing_asset_equal_circular_blocks_FHS_smooth_joint_mean_uncertainty_v1'
    assert current['same_successful_tasks_per_model']==64049
    assert read(C/'full/active.json')['status']=='terminal' and read(C/'full/reporter_active.json')['status']=='terminal'
    processes=subprocess.check_output(['ps','-axo','command='],text=True)
    assert str(O/'run_asset_prior_full.py') not in processes and str(O/'report_asset_prior_full.py') not in processes
    archive=read(ARCH/'receipt.json');assert archive['complete'] and archive['files_fresh_downloaded_disk_restored_sha256_verified']==72258
    releases=json.loads(subprocess.check_output(['gh','api',f"repos/{archive['repository']}/releases?per_page=100"],env=ENV,text=True))
    matching=[d for d in releases if d['tag_name']==archive['release_tag']]
    assert len(matching)==1, 'Current private draft release must be present in authenticated listing'
    release=matching[0]
    assets={a['name']:a for a in release['assets']};entries={};verified=[]
    DEST.mkdir(parents=True,exist_ok=True);download=DEST/'fresh-manifests';download.mkdir(exist_ok=False)
    before=shutil.disk_usage(O).free
    for part in archive['archives']:
        assert part['remote_download_sha256_verified'] and part['all_files_disk_restored_and_sha256_verified']
        for key in ('archive','manifest'):
            online=assets[part[key+'_name']]
            assert online['digest']=='sha256:'+part[key+'_sha256']
        subprocess.run(['gh','release','download',archive['release_tag'],'--repo',archive['repository'],'--pattern',part['manifest_name'],'--dir',str(download)],env=ENV,check=True)
        manifest=download/part['manifest_name'];assert sha(manifest)==part['manifest_sha256']
        for entry in read(manifest)['entries']:
            relative=Path(entry['path']);p=O/relative
            if p.suffix!='.npz' or not p.is_relative_to(C):continue
            assert p.resolve().is_relative_to(C.resolve()) and not p.is_symlink()
            assert relative.as_posix() not in entries
            assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
            entries[relative.as_posix()]=entry
        verified.append({'archive_name':part['archive_name'],'server_digest':assets[part['archive_name']]['digest'],'manifest_name':part['manifest_name'],'fresh_manifest_sha256':sha(manifest)})
    actual={str(p.relative_to(O)) for p in C.rglob('*.npz')}
    assert set(entries)==actual and len(entries)==72177
    proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(entries),'bytes':sum(e['bytes'] for e in entries.values()),
           'archive_receipt_sha256':sha(ARCH/'receipt.json'),'fresh_online_proof':verified,
           'paths':entries,'current_best_preserved':current['current_best_model_id'],'current_best_raw_losses_not_touched':True,
           'active_dependencies_preserved':'Both baseline raw trees/inputs, current circular raw losses, and every old hybrid source/kernel/metadata/index remain local. Active future screens use separately preserved100-path losses; no active reader needs these closed1000-path raw files.'}
    (DEST/'pre_retirement_manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    for name in sorted(entries):
        p=O/name;assert p.stat().st_size==entries[name]['bytes'];p.unlink()
    after=shutil.disk_usage(O).free
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'raw_checkpoint_files_retired':len(entries),'raw_bytes_retired':proof['bytes'],
             'raw_scientific_outputs_already_fresh_downloaded_disk_restored_SHA256_verified':True,
             'fresh_server_archive_digests_and_downloaded_manifests_verified':True,'archive_release':archive['release_tag'],
             'archive_repository':archive['repository'],'pre_retirement_manifest_sha256':sha(DEST/'pre_retirement_manifest.json'),
             'disk_free_before_bytes':before,'disk_free_after_bytes':after,'sources_inputs_current_best_and_metadata_preserved':True,
             'restore':'Download candidate-part-*.tar.gz and matching manifests from the private release, verify archive/manifest SHA256, extract at the research owner root and verify each file size/hash. Restore sealed dependencies as described in the original archive receipt before rerunning anything.'}
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (C/'ARCHIVED_LOSSES.json').write_text(json.dumps(receipt,indent=2)+'\n')
    shutil.rmtree(download)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
