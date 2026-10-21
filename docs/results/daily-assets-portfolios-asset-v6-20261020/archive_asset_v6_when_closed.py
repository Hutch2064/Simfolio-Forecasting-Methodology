"""Preserve the closed baseline pair in parallel with research; no local retirement."""
from pathlib import Path
import base64
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
import sys

O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
R = O/'repository'
REL = 'docs/results/daily-assets-portfolios-asset-v6-20261020'
D = R/REL
PY = '/Users/aidanhutchison/simfolio/tmp/catalogue52-20261018/venv/bin/python'
env = {**os.environ, 'GODEBUG': 'http2client=0', 'OPENBLAS_NUM_THREADS': '1',
       'OMP_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1'}
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def state(stage, **details):
    p = O/'asset_v6_archive_active.json'
    tmp = p.with_suffix('.json.new')
    tmp.write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'pid': os.getpid(), 'stage': stage, **details}, indent=2)+'\n')
    os.replace(tmp, p)


def main():
    lock = (O/'asset_v6_archive.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resume_publication = sys.argv[1:] == ['--finish-publication']
    while not resume_publication and not (V/'full_pair_independent_audit.json').exists():
        handoff = read(O/'asset_v6_research_handoff_active.json')
        assert handoff['stage'] != 'attention_required', handoff
        proc = subprocess.run(['ps', '-p', str(handoff['pid']), '-o', 'command='], capture_output=True, text=True)
        assert 'continue_asset_v6_research.py' in proc.stdout, 'Audit owner ended before full-pair proof'
        state('waiting_for_live_audit_owner', owner_pid=handoff['pid'])
        time.sleep(20)
    if resume_publication:
        receipt = read(O/'archives/asset-v6-baselines-1000-20261020/receipt.json')
        assert receipt['complete'] and receipt['files_fresh_downloaded_disk_restored_sha256_verified'] == 134653
    else:
        state('preserving_closed_pair')
        subprocess.run([PY, '/Users/aidanhutchison/simfolio/storage_preflight.py',
                        '--expected-gib', '1', '--artifact-root', str(O), '--budget-gib', '12'], check=True)
        subprocess.run([PY, str(O/'preserve_asset_v6_closed.py')], cwd=O, env=env, check=True)
    publication = (O/'asset_v6_publication.lock').open('a')
    fcntl.flock(publication, fcntl.LOCK_EX)
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=R, text=True)
    if resume_publication:
        assert all(line[3:].startswith((REL+'/', 'docs/results/daily-cross-asset-722/')) for line in dirty.splitlines())
    else:
        assert not dirty.strip()
    subprocess.run([PY, str(O/'catalogue_asset_v6_closed.py')], cwd=O, env=env, check=True)
    shutil.copy2(O/'archives/asset-v6-baselines-1000-20261020/receipt.json', D/'closed-baseline-archive-receipt.json')
    for name in ['preserve_asset_v6_closed.py', 'catalogue_asset_v6_closed.py', 'archive_asset_v6_when_closed.py']:
        shutil.copy2(O/name, D/name)
    manifest = {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
                if p.is_file() and p.name != 'PUBLIC_FILES.json' and '__pycache__' not in p.parts}
    (D/'PUBLIC_FILES.json').write_text(json.dumps(manifest, indent=2)+'\n')
    subprocess.run([PY, '-m', 'pytest', '-q', 'tests/test_full_panel_catalogue.py'], cwd=R, env=env, check=True)
    allowed = [REL+'/', 'docs/results/daily-cross-asset-722/catalogue.json',
               'docs/results/daily-cross-asset-722/CATALOGUE.md']
    changes = subprocess.check_output(['git', 'status', '--porcelain'], cwd=R, text=True)
    assert all(any(line[3:].startswith(p) for p in allowed) for line in changes.splitlines())
    subprocess.run(['git', 'add', REL, *allowed[1:]], cwd=R, env=env, check=True)
    subprocess.run(['git', 'commit', '-m', 'Catalogue both complete joint-asset baselines with verified private recovery'], cwd=R, env=env, check=True)
    subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'], cwd=R, env=env, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip()
    remote = subprocess.check_output(['git', '-c', 'http.version=HTTP/1.1', 'ls-remote', 'origin', 'refs/heads/main'], cwd=R, text=True).split()[0]
    assert remote == commit
    recovered = O/'v6_catalogue_public_recovery'/commit[:12]
    rows = []
    for rel in [*allowed[1:], *[str(p.relative_to(R)) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]]:
        data = subprocess.check_output(['gh', 'api', '-H', 'Accept: application/vnd.github.raw+json',
                                        f'repos/Hutch2064/Simfolio-Forecasting-Methodology/contents/{rel}?ref={commit}'], env=env)
        assert hashlib.sha256(data).hexdigest() == sha(R/rel)
        path = recovered/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        assert sha(path) == sha(R/rel)
        rows.append({'path': rel, 'sha256': sha(path)})
    receipt = {'commit': commit, 'fresh_public_files_downloaded_disk_restored_SHA256_verified': len(rows),
               'files': rows, 'complete_private_recovery_receipt': str(O/'archives/asset-v6-baselines-1000-20261020/receipt.json'),
               'scientific_originals_removed': False}
    (O/'v6_catalogue_public_recovery.json').write_text(json.dumps(receipt, indent=2)+'\n')
    state('terminal', commit=commit, public_files_verified=len(rows), scientific_originals_removed=False)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        state('attention_required', traceback=traceback.format_exc(), scientific_originals_removed=False)
        raise
