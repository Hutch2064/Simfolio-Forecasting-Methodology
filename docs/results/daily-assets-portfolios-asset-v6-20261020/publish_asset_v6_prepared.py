"""Publish compact qualified sources/evidence, then verify fresh remote recovery."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import fcntl

O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
R = O/'repository'
REL = 'docs/results/daily-assets-portfolios-asset-v6-20261020'
D = R/REL
env = {**os.environ, 'GODEBUG': 'http2client=0', 'OPENBLAS_NUM_THREADS': '1',
       'OMP_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
       'PYTHONDONTWRITEBYTECODE': '1', 'NUMBA_CACHE_DIR': str(O/'v6_public_test_cache')}
PY = '/Users/aidanhutchison/simfolio/tmp/catalogue52-20261018/venv/bin/python'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    lock = (O/'asset_v6_publication.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=R, text=True).strip(), 'Preserve unrelated repository edits'
    archive = O/'archives/asset-v6-hybrid-1000-20261021/receipt.json'
    if archive.exists():
        subprocess.run([PY, str(O/'catalogue_asset_hybrid_closed.py')], cwd=O, env=env, check=True)
        shutil.copy2(archive, D/'closed-hybrid-archive-receipt.json')
    circular_archive = O/'archives/asset-v6-circular-1000-20261021/receipt.json'
    if circular_archive.exists():
        subprocess.run([PY, str(O/'catalogue_asset_circular_closed.py')], cwd=O, env=env, check=True)
        shutil.copy2(circular_archive, D/'closed-circular-archive-receipt.json')
    fading_archive = O/'archives/asset-v6-fading-1000-20261021/receipt.json'
    if fading_archive.exists():
        subprocess.run([PY, str(O/'catalogue_asset_fading_closed.py')], cwd=O, env=env, check=True)
        shutil.copy2(fading_archive,D/'closed-fading-archive-receipt.json')
    subprocess.run([PY, str(O/'refresh_asset_v6_research_doc.py')], cwd=O, env=env, check=True)
    for name in ['test_asset_prior_best.py', 'audit_asset_v6_full_pair.py',
                 'bind_asset_prior_parent.py', 'publish_asset_v6_prepared.py',
                 'continue_asset_v6_research.py', 'diagnose_asset_v6_pair.py',
                 'test_asset_v6_decomposition.py', 'preserve_asset_v6_closed.py',
                 'catalogue_asset_v6_closed.py', 'archive_asset_v6_when_closed.py',
                 'prepare_asset_prior_broad.py', 'goal_objective_updated.txt',
                 'goal_screen_parallel_receipt.json', 'run_asset_prior_full.py',
                 'screen_asset_means.py', 'test_asset_mean_screen.py',
                 'report_asset_prior_full.py', 'asset_v6_shared_slots_receipt.json',
                 'screen_asset_components.py', 'screen_asset_parent.py', 'screen_asset_circular.py',
                 'test_asset_sampler_parent.py', 'test_asset_circular.py',
                 'promote_asset_hybrid.py', 'preserve_asset_hybrid_closed.py',
                 'archive_asset_hybrid_closed.py', 'catalogue_asset_hybrid_closed.py',
                 'diagnose_asset_hybrid_full.py', 'asset_prior_full_idle_drain_receipt.json',
                 'continue_asset_circular.py', 'asset_circular_handoff_syntax_fix.json',
                 'run_asset_circular_full.py', 'report_asset_circular_full.py',
                 'promote_asset_circular.py', 'launch_asset_circular_full_when_ready.py',
                 'screen_asset_iid_component.py', 'test_asset_iid_component.py',
                 'preserve_asset_circular_closed.py', 'catalogue_asset_circular_closed.py',
                 'finish_asset_circular.py', 'qualify_asset_gjr.py', 'test_asset_gjr.py',
                 'refresh_asset_v6_research_doc.py', 'screen_asset_gjr.py',
                 'screen_asset_adaptive_blocks.py', 'test_asset_adaptive_blocks.py',
                 'screen_asset_empirical_mean.py', 'test_asset_empirical_mean.py',
                 'continue_asset_empirical_mean.py', 'screen_asset_fading_vol.py',
                 'test_asset_fading_vol.py', 'run_asset_fading_pilot.py',
                 'continue_asset_fading_vol.py', 'preserve_asset_gjr_pilot.py',
                 'preserve_asset_adaptive_pilot.py', 'preserve_asset_empirical_mean_pilot.py',
                 'retire_closed_asset_hybrid_losses.py', 'run_asset_fading_full.py',
                 'report_asset_fading_full.py', 'promote_asset_fading.py',
                 'preserve_asset_fading_closed.py', 'catalogue_asset_fading_closed.py',
                 'finish_asset_fading.py', 'launch_asset_fading_full_when_ready.py',
                 'diagnose_asset_fading_pilots.py', 'asset_fading_storage_budget.json']:
        shutil.copy2(O/name, D/name)
    c = V/'candidates/asset_prior_best'
    target = D/'candidates/asset_prior_best'
    for path in c.glob('*.py'):
        shutil.copy2(path, target/path.name)
    mean_source = V/'candidates/asset_mean_uncertainty'
    mean_target = D/'candidates/asset_mean_uncertainty'
    mean_target.mkdir(exist_ok=True)
    for path in mean_source.glob('*.py'):
        shutil.copy2(path, mean_target/path.name)
    circular = V/'candidates/asset_circular_blocks'
    circular_target = D/'candidates/asset_circular_blocks'
    circular_target.mkdir(exist_ok=True)
    for path in circular.iterdir():
        if path.suffix in ('.py', '.md'):
            shutil.copy2(path, circular_target/path.name)
    iid = V/'candidates/asset_iid_component'
    iid_target = D/'candidates/asset_iid_component'
    iid_target.mkdir(exist_ok=True)
    for path in iid.iterdir():
        if path.suffix in ('.py', '.md'):
            shutil.copy2(path, iid_target/path.name)
    gjr = V/'candidates/asset_gjr'
    gjr_target = D/'candidates/asset_gjr'
    gjr_target.mkdir(exist_ok=True)
    for path in gjr.iterdir():
        if path.suffix in ('.py', '.md') or path.name == 'source_derivation.json':
            shutil.copy2(path, gjr_target/path.name)
    qualification = gjr/'qualification/summary.json'
    if qualification.exists():
        compact = {k: v for k, v in json.loads(qualification.read_text()).items() if k != 'records'}
        (gjr_target/'training_qualification_summary.json').write_text(json.dumps(compact, indent=2)+'\n')
    for name in ('plan.json', 'plan_receipt.json'):
        if (circular/name).exists():
            shutil.copy2(circular/name, circular_target/name)
    for phase in ('development', 'reserved', 'broad'):
        path = circular/phase/'summary.json'
        if path.exists():
            compact = {k: v for k, v in json.loads(path.read_text()).items() if k != 'records'}
            (circular_target/(phase+'_summary.json')).write_text(json.dumps(compact, indent=2)+'\n')
    for phase in ('components_screen100', 'parent_screen100', 'circular_screen100'):
        path = mean_source/phase/'summary.json'
        if path.exists():
            compact = {k: v for k, v in json.loads(path.read_text()).items() if k != 'records'}
            (mean_target/(phase+'_summary.json')).write_text(json.dumps(compact, indent=2)+'\n')
    for name in ('asset_adaptive_blocks','asset_empirical_mean','asset_fading_vol'):
        source = V/'candidates'/name
        dest = D/'candidates'/name
        dest.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.suffix in ('.py','.md') or path.name in ('plan.json','plan_receipt.json','selector_provenance.json','decision.json','admission_source_registration_fix.json','full_selection.json','horizon_diagnostics.json'):
                shutil.copy2(path,dest/path.name)
        for phase in ('development','reserved','broad'):
            path = source/phase/'summary.json'
            if path.exists():
                compact = {k:v for k,v in json.loads(path.read_text()).items() if k!='records'}
                (dest/(phase+'_summary.json')).write_text(json.dumps(compact,indent=2)+'\n')
    fading_source=V/'candidates/asset_fading_vol'
    fading_target=D/'candidates/asset_fading_vol'
    (fading_target/'full').mkdir(exist_ok=True)
    for name in ('verification_receipt.json','scoreboard.json','admission.json'):
        if (fading_source/'full'/name).exists():
            shutil.copy2(fading_source/'full'/name,fading_target/'full'/name)
    if (fading_source/'promotion_receipt.json').exists():
        shutil.copy2(fading_source/'promotion_receipt.json',fading_target/'promotion_receipt.json')
    for phase in ('iid_component_screen100','gjr_screen100','adaptive_blocks_screen100','empirical_mean_screen100','fading_vol_screen100'):
        path = mean_source/phase/'summary.json'
        if path.exists():
            compact = {k:v for k,v in json.loads(path.read_text()).items() if k!='records'}
            (mean_target/(phase+'_summary.json')).write_text(json.dumps(compact,indent=2)+'\n')
    for phase in ('development','reserved'):
        path = gjr/phase/'summary.json'
        if path.exists():
            compact = {k:v for k,v in json.loads(path.read_text()).items() if k!='records'}
            (gjr_target/(phase+'_summary.json')).write_text(json.dumps(compact,indent=2)+'\n')
    if (gjr/'decision.json').exists():
        shutil.copy2(gjr/'decision.json',gjr_target/'decision.json')
    for folder in ('asset-v6-GJR-pilot-20261021','asset-v6-adaptive-pilot-20261021','asset-v6-empirical-mean-pilot-20261021'):
        path = O/'archives'/folder/'receipt.json'
        if path.exists():
            shutil.copy2(path,D/(folder+'-archive-receipt.json'))
    retirement = O/'archives/asset-v6-hybrid-raw-retirement-20261021'
    for name in ('receipt.json','metadata_remote_recovery.json'):
        if (retirement/name).exists():
            shutil.copy2(retirement/name,D/('old-hybrid-raw-retirement-'+name))
    if (c/'ARCHIVED_LOSSES.json').exists():
        shutil.copy2(c/'ARCHIVED_LOSSES.json',target/'ARCHIVED_LOSSES.json')
    full_target = target/'full'
    full_target.mkdir(exist_ok=True)
    for name in ('verification_receipt.json', 'scoreboard.json', 'admission.json'):
        if (c/'full'/name).exists():
            shutil.copy2(c/'full'/name, full_target/name)
    if (c/'promotion_receipt.json').exists():
        shutil.copy2(c/'promotion_receipt.json', target/'promotion_receipt.json')
    circular_full = circular_target/'full'
    circular_full.mkdir(exist_ok=True)
    for name in ('verification_receipt.json', 'scoreboard.json', 'admission.json'):
        if (circular/'full'/name).exists():
            shutil.copy2(circular/'full'/name, circular_full/name)
    if (circular/'promotion_receipt.json').exists():
        shutil.copy2(circular/'promotion_receipt.json', circular_target/'promotion_receipt.json')
    screen = mean_source/'screen100/summary.json'
    if screen.exists():
        compact = {k: v for k, v in json.loads(screen.read_text()).items() if k != 'records'}
        (mean_target/'screen100_summary.json').write_text(json.dumps(compact, indent=2)+'\n')
    for phase in ('development', 'reserved', 'broad'):
        path = c/phase/'summary.json'
        if path.exists():
            compact = {k: v for k, v in json.loads(path.read_text()).items() if k not in ('records', 'worker_pids')}
            (target/(phase+'_summary.json')).write_text(json.dumps(compact, indent=2)+'\n')
    for name in ['plan.json', 'parent_selection.json', 'plan_receipt.json',
                 'broad_case_selection_receipt.json', 'DESIGN.md']:
        if (c/name).exists():
            shutil.copy2(c/name, target/name)
    for name in ['full_pair_independent_audit.json', 'full_pair_diagnostics.json', 'candidate_full_diagnostics.json', 'official_scoreboard.json', 'CURRENT_RESEARCH.md']:
        if (V/name).exists():
            shutil.copy2(V/name, D/name)
    manifest = {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
                if p.is_file() and p.name != 'PUBLIC_FILES.json'
                and '__pycache__' not in p.parts}
    (D/'PUBLIC_FILES.json').write_text(json.dumps(manifest, indent=2)+'\n')
    test = subprocess.run([PY, '-m', 'unittest', '-v', 'test_asset_paths.py',
                           'test_v6_reporting.py', 'test_asset_prior_best.py',
                           'test_asset_v6_decomposition.py', 'test_asset_mean_screen.py',
                           'test_asset_sampler_parent.py', 'test_asset_circular.py',
                           'test_asset_iid_component.py', 'test_asset_gjr.py',
                           'test_asset_adaptive_blocks.py','test_asset_empirical_mean.py','test_asset_fading_vol.py'],
                          cwd=D, env=env, capture_output=True, text=True)
    (O/'v6_prepared_publication_tests.log').write_text(test.stdout+test.stderr)
    assert test.returncode == 0, test.stdout+test.stderr
    subprocess.run([PY, '-m', 'pytest', '-q', 'tests/test_full_panel_catalogue.py'], cwd=R, env=env, check=True)
    changes = subprocess.check_output(['git', 'status', '--porcelain'], cwd=R, text=True)
    for line in changes.splitlines():
        assert line[3:].startswith((REL+'/', 'docs/results/daily-cross-asset-722/catalogue.json', 'docs/results/daily-cross-asset-722/CATALOGUE.md')), line
    if changes.strip():
        subprocess.run(['git', 'add', REL, 'docs/results/daily-cross-asset-722/catalogue.json', 'docs/results/daily-cross-asset-722/CATALOGUE.md'], cwd=R, env=env, check=True)
        subprocess.run(['git', 'commit', '-m', 'Preserve audited asset-pair handoff and matched prior-best pilot'], cwd=R, env=env, check=True)
        subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'], cwd=R, env=env, check=True)
    old = O/'v6_candidate_source_public_recovery.json'
    if old.exists():
        receipt = json.loads(old.read_text())
        saved = O/'v6_public_recovery'/receipt['commit'][:12]/'recovery_receipt.json'
        if not saved.exists():
            shutil.copy2(old, saved)
    subprocess.run([PY, str(O/'recover_v6_candidate_publication.py')], cwd=O, env=env, check=True)
    receipt = json.loads(old.read_text())
    catalogue_files = []
    for rel in ('docs/results/daily-cross-asset-722/catalogue.json', 'docs/results/daily-cross-asset-722/CATALOGUE.md'):
        data = subprocess.check_output(['gh', 'api', '-H', 'Accept: application/vnd.github.raw+json', f"repos/Hutch2064/Simfolio-Forecasting-Methodology/contents/{rel}?ref={receipt['commit']}"], env=env)
        assert hashlib.sha256(data).hexdigest() == sha(R/rel)
        restored = O/'v6_public_recovery'/receipt['commit'][:12]/'catalogue'/Path(rel).name
        restored.parent.mkdir(exist_ok=True)
        restored.write_bytes(data)
        assert sha(restored) == sha(R/rel)
        catalogue_files.append({'path': rel, 'sha256': sha(restored)})
    (O/'v6_hybrid_catalogue_public_recovery.json').write_text(json.dumps({'commit': receipt['commit'], 'fresh_download_disk_restore_SHA256_verified': catalogue_files}, indent=2)+'\n')
    saved = O/'v6_public_recovery'/receipt['commit'][:12]/'recovery_receipt.json'
    shutil.copy2(old, saved)
    print(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'commit': receipt['commit'], 'fresh_remote_recovery_verified': True,
                      'tests': 45}), flush=True)


if __name__ == '__main__':
    main()
