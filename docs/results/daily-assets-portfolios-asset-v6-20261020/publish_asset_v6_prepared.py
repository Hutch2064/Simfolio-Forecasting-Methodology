"""Publish compact qualified sources/evidence, then verify fresh remote recovery."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import fcntl
import sys

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
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True)
    if dirty.strip():
        assert '--resume-owned-prepared' in sys.argv,'Preserve unrelated repository edits'
        for line in dirty.splitlines():
            assert line[3:].startswith((REL+'/', 'docs/results/daily-cross-asset-722/catalogue.json','docs/results/daily-cross-asset-722/CATALOGUE.md')), 'Partial publication includes unrelated edits'

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
                 'diagnose_asset_fading_pilots.py', 'asset_fading_storage_budget.json',
                 'screen_asset_equilibrium_blocks.py','run_asset_equilibrium_pilot.py',
                 'continue_asset_equilibrium_blocks.py','test_asset_equilibrium_blocks.py',
                 'diagnose_asset_training_memory.py','test_asset_training_memory.py',
                 'screen_asset_AR1_mean.py','test_asset_AR1_mean.py','preserve_asset_AR1_pilot.py',
                 'run_asset_feature_batch.py','test_asset_feature_batch.py',
                 'prepare_asset_feature_confirmation.py','run_asset_feature_confirmation.py',
                 'prepare_asset_feature_reserved.py','run_asset_feature_reserved.py','preserve_asset_feature_development.py',
                 'prepare_asset_feature_broad.py','run_asset_feature_broad.py','run_asset_feature_broad_v2.py','asset_feature_storage_budget.json',
                 'retire_closed_feature_fit_caches.py','preserve_asset_feature_cache_retirement_metadata.py',
                 'run_asset_feature_full.py','report_asset_feature_full.py','promote_asset_feature.py',
                 'continue_asset_feature_batch.py','bind_asset_feature_transitive_sources.py','test_asset_feature_admission.py',
                 'prepare_asset_dynamic_correlation_screen.py','screen_asset_dynamic_correlation.py','screen_asset_dynamic_correlation_v2.py','test_asset_dynamic_correlation.py','preserve_asset_feature_broad.py',
                 'screen_asset_component_weights.py','prepare_asset_component_confirmation.py','run_asset_component_confirmation.py','prepare_asset_component_reserved.py','run_asset_component_reserved.py','test_asset_component_weights.py',
                 'screen_asset_training_blocks.py','run_asset_training_blocks_validation.py','test_asset_training_blocks.py','preserve_asset_sample_refocus.py',
                 'screen_asset_likelihood_sv.py','test_asset_likelihood_sv.py','preserve_asset_training_laws.py',
                 'screen_asset_trained_pool.py','run_asset_trained_pool_validation.py','run_asset_trained_pool_broad.py','test_asset_trained_pool.py','screen_asset_regime_rows.py','test_asset_regime_rows.py','screen_asset_random_measure.py','test_asset_random_measure.py','preserve_asset_learned_pool_pilots.py','preserve_asset_learned_pool_broad.py','verify_asset_trained_pool.py','recover_v6_candidate_publication.py','retire_closed_sample_fit_caches.py','preserve_sample_fit_retirement_metadata.py',
                 'screen_asset_latent_mean.py','run_asset_latent_mean_validation.py','test_asset_latent_mean_fit.py','test_asset_latent_mean.py','latent_mean_prototype_public_recovery.json',
                 'research_workflow_policy.json','working_incumbent.json',
                 'retire_closed_asset_circular_losses.py','preserve_asset_circular_retirement_metadata.py']:
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
    for name in ('asset_adaptive_blocks','asset_empirical_mean','asset_fading_vol','asset_equilibrium_blocks','asset_AR1_mean','asset_feature_batch','asset_dynamic_correlation','asset_component_weights','asset_training_blocks','asset_likelihood_sv','asset_trained_pool','asset_regime_rows','asset_random_measure','asset_latent_mean'):
        source = V/'candidates'/name
        dest = D/'candidates'/name
        dest.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.suffix in ('.py','.md') or path.name in ('plan.json','plan_receipt.json','selector_provenance.json','decision.json','admission_source_registration_fix.json','full_selection.json','horizon_diagnostics.json','prior_evidence.json','confirmation_selection.json','development_selection.json','confirmation_preforecast_fix.json','reserved_selection.json','broad_selection.json','transitive_source_receipt.json','full_selection.json','source_derivation.json','qualification_receipt.json','combined710_diagnostic.json','validation_label_note.json','validation_receipt.json','working_promotion.json','prior_working_incumbent.json','source_replay_aliases.json'):
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
    for phase in ('iid_component_screen100','gjr_screen100','adaptive_blocks_screen100','empirical_mean_screen100','fading_vol_screen100','equilibrium_blocks_screen100','AR1_mean_screen100'):
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
    for folder in ('asset-v6-GJR-pilot-20261021','asset-v6-adaptive-pilot-20261021','asset-v6-empirical-mean-pilot-20261021','asset-v6-equilibrium-pilot-20261021','asset-v6-AR1-pilot-20261021','asset-v6-feature-development-20261021','asset-v6-feature-broad-20261021','asset-v6-sample-refocus-20261021','asset-v6-training-laws-20261021','asset-v6-learned-pool-pilots-20261021','asset-v6-learned-pool-broad-20261021'):
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
    cache_retirement=O/'archives/asset-v6-feature-fit-cache-retirement-20261021/receipt.json'
    cache_recovery=cache_retirement.parent/'metadata_remote_recovery.json'
    if cache_recovery.exists():shutil.copy2(cache_recovery,D/'feature-fit-cache-retirement-remote-recovery.json')
    if cache_retirement.exists():shutil.copy2(cache_retirement,D/'feature-fit-cache-retirement-receipt.json')
    startup_fix=O/'asset_feature_broad_startup_fix/receipt.json'
    if startup_fix.exists():shutil.copy2(startup_fix,D/'feature-broad-startup-fix-receipt.json')
    retirement_circular=O/'archives/asset-v6-circular-raw-retirement-20261021'
    for name in ('receipt.json','metadata_remote_recovery.json'):
        if (retirement_circular/name).exists():
            shutil.copy2(retirement_circular/name,D/('circular-raw-retirement-'+name))
    if (circular/'ARCHIVED_LOSSES.json').exists():
        shutil.copy2(circular/'ARCHIVED_LOSSES.json',circular_target/'ARCHIVED_LOSSES.json')
    for phase in ('screen100','confirmation1000','reserved1000','broad1000'):
        source=V/'candidates/asset_feature_batch'/phase
        destination=D/'candidates/asset_feature_batch'/phase
        destination.mkdir(exist_ok=True)
        for name in ('plan.json','summary.json','ARCHIVED_FIT_CACHE.json'):
            path=source/name
            if path.exists():
                compact={k:v for k,v in json.loads(path.read_text()).items() if k not in ('records','fit_packets')}
                (destination/name).write_text(json.dumps(compact,indent=2)+'\n')
    for candidate,phases in [('asset_dynamic_correlation',('screen100',)),('asset_component_weights',('screen120','development1000','reserved1000')),('asset_training_blocks',('screen100','development1000','reserved1000')),('asset_likelihood_sv',('screen100',)),('asset_trained_pool',('screen100','development1000','reserved1000','broad1000')),('asset_regime_rows',('screen100',)),('asset_random_measure',('screen100',)),('asset_latent_mean',('screen100','development1000','reserved1000'))]:
        for phase in phases:
            source=V/'candidates'/candidate/phase
            destination=D/'candidates'/candidate/phase
            if not source.exists():continue
            destination.mkdir(parents=True,exist_ok=True)
            for name in ('plan.json','summary.json','working_incumbent_snapshot.json'):
                path=source/name
                if path.exists():
                    compact={k:v for k,v in json.loads(path.read_text()).items()if k not in('records','fit_packets','reference_records')}
                    (destination/name).write_text(json.dumps(compact,indent=2)+'\n')
    receipt=O/'development_workflow_refocus/checkpoint_receipt.json'
    if receipt.exists():shutil.copy2(receipt,D/'feature-partial-full-checkpoint-receipt.json')
    for name in ('receipt.json','metadata_remote_recovery.json'):
        path=O/'archives/asset-v6-sample-fit-cache-retirement-20261021'/name
        if path.exists():shutil.copy2(path,D/('sample-fit-cache-retirement-'+name))
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
    for name in ['full_pair_independent_audit.json', 'full_pair_diagnostics.json', 'candidate_full_diagnostics.json','training_short_memory_diagnostic.json', 'official_scoreboard.json', 'CURRENT_RESEARCH.md','OPEN_RESEARCH.md']:
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
                           'test_asset_adaptive_blocks.py','test_asset_empirical_mean.py','test_asset_fading_vol.py','test_asset_equilibrium_blocks.py','test_asset_training_memory.py','test_asset_AR1_mean.py','test_asset_feature_batch.py','test_asset_feature_admission.py','test_asset_dynamic_correlation.py','test_asset_component_weights.py','test_asset_training_blocks.py','test_asset_likelihood_sv.py','test_asset_trained_pool.py','test_asset_regime_rows.py','test_asset_random_measure.py','test_asset_latent_mean_fit.py','test_asset_latent_mean.py'],
                          cwd=D, env=env, capture_output=True, text=True)
    (O/'v6_prepared_publication_tests.log').write_text(test.stdout+test.stderr)
    assert test.returncode == 0, test.stdout+test.stderr
    subprocess.run([PY, '-m', 'pytest', '-q', 'tests/test_full_panel_catalogue.py'], cwd=R, env=env, check=True)
    changes = subprocess.check_output(['git', 'status', '--porcelain'], cwd=R, text=True)
    for line in changes.splitlines():
        assert line[3:].startswith((REL+'/', 'docs/results/daily-cross-asset-722/catalogue.json', 'docs/results/daily-cross-asset-722/CATALOGUE.md')), line
    if changes.strip():
        subprocess.run(['git', 'add', REL, 'docs/results/daily-cross-asset-722/catalogue.json', 'docs/results/daily-cross-asset-722/CATALOGUE.md'], cwd=R, env=env, check=True)
        subprocess.run(['git', 'commit', '-m', 'Record sample-first joint-asset research and training-derived block candidates'], cwd=R, env=env, check=True)
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
                      'tests': receipt['recovered_source_tests_passed']}), flush=True)


if __name__ == '__main__':
    main()
