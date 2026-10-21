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
    for name in ['test_asset_prior_best.py', 'audit_asset_v6_full_pair.py',
                 'bind_asset_prior_parent.py', 'publish_asset_v6_prepared.py',
                 'continue_asset_v6_research.py', 'diagnose_asset_v6_pair.py',
                 'test_asset_v6_decomposition.py', 'preserve_asset_v6_closed.py',
                 'catalogue_asset_v6_closed.py', 'archive_asset_v6_when_closed.py',
                 'prepare_asset_prior_broad.py', 'goal_objective_updated.txt',
                 'goal_screen_parallel_receipt.json']:
        shutil.copy2(O/name, D/name)
    c = V/'candidates/asset_prior_best'
    target = D/'candidates/asset_prior_best'
    for path in c.glob('*.py'):
        shutil.copy2(path, target/path.name)
    for name in ['plan.json', 'parent_selection.json', 'plan_receipt.json',
                 'broad_case_selection_receipt.json', 'DESIGN.md']:
        if (c/name).exists():
            shutil.copy2(c/name, target/name)
    for name in ['full_pair_independent_audit.json', 'full_pair_diagnostics.json']:
        if (V/name).exists():
            shutil.copy2(V/name, D/name)
    manifest = {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
                if p.is_file() and p.name != 'PUBLIC_FILES.json'
                and '__pycache__' not in p.parts}
    (D/'PUBLIC_FILES.json').write_text(json.dumps(manifest, indent=2)+'\n')
    test = subprocess.run([PY, '-m', 'unittest', '-v', 'test_asset_paths.py',
                           'test_v6_reporting.py', 'test_asset_prior_best.py',
                           'test_asset_v6_decomposition.py'],
                          cwd=D, env=env, capture_output=True, text=True)
    (O/'v6_prepared_publication_tests.log').write_text(test.stdout+test.stderr)
    assert test.returncode == 0, test.stdout+test.stderr
    changes = subprocess.check_output(['git', 'status', '--porcelain'], cwd=R, text=True)
    for line in changes.splitlines():
        assert line[3:].startswith(REL+'/'), line
    if changes.strip():
        subprocess.run(['git', 'add', REL], cwd=R, env=env, check=True)
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
    saved = O/'v6_public_recovery'/receipt['commit'][:12]/'recovery_receipt.json'
    shutil.copy2(old, saved)
    print(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'commit': receipt['commit'], 'fresh_remote_recovery_verified': True,
                      'tests': 21}), flush=True)


if __name__ == '__main__':
    main()
