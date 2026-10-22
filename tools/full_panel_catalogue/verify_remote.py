"""Check remote archive digests and freshly downloaded manifest hashes.

This checks preservation without downloading all historical multi-GiB archives.
It deliberately does NOT certify a new full recovery or permit local retirement.
Use the recorded full disk-restore receipts, or perform a fresh full restoration,
before retiring any corresponding completed local output.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def fetch(url, token, *, binary=False):
    config = 'header = "Authorization: Bearer ' + token + '"\n'
    if binary:
        config += 'header = "Accept: application/octet-stream"\n'
    result = subprocess.run(
        ['curl', '--fail', '--silent', '--show-error', '--location', '--max-time', '60',
         '--config', '-', url], input=config.encode(), capture_output=True, check=True)
    return result.stdout


def verify(ledger, token):
    refs = {a['archive_sha256']: a for model in ledger['models'].values()
            for execution in model['executions'] for a in execution['archives']}
    repositories = {a['repository'] for a in refs.values()}
    releases = {}
    for repository in repositories:
        page = 1
        while True:
            items = json.loads(fetch(f'https://api.github.com/repos/{repository}/releases?per_page=100&page={page}', token))
            for item in items:
                releases[repository, item['tag_name']] = item
            if len(items) < 100:
                break
            page += 1
    checked = []
    asset_lists = {}
    for archive in refs.values():
        release = releases[archive['repository'], archive['release_tag']]
        release_key = (archive['repository'], release['id'])
        if release_key not in asset_lists:
            assets = {}; page = 1
            while True:
                items = json.loads(fetch(f'https://api.github.com/repos/{archive["repository"]}/releases/{release["id"]}/assets?per_page=100&page={page}', token))
                assets.update((a['name'], a) for a in items)
                if len(items) < 100:
                    break
                page += 1
            asset_lists[release_key] = assets
        assets = asset_lists[release_key]
        payload = assets[archive['archive_name']]
        if payload.get('digest') != 'sha256:' + archive['archive_sha256']:
            raise ValueError('Remote archive digest differs or is unavailable: ' + archive['archive_name'])
        manifest = assets[archive['manifest_asset']]
        data = fetch(manifest['url'] + '?catalogue_manifest_check=1', token, binary=True)
        if hashlib.sha256(data).hexdigest() != archive['manifest_sha256']:
            raise ValueError('Downloaded manifest hash differs: ' + archive['manifest_asset'])
        json.loads(data)
        checked.append({'repository': archive['repository'], 'release_tag': archive['release_tag'],
            'release_url': release['html_url'], 'archive_name': archive['archive_name'],
            'archive_asset_id': payload['id'], 'archive_sha256': archive['archive_sha256'],
            'archive_server_digest_verified': True, 'manifest_asset': archive['manifest_asset'],
            'manifest_sha256': archive['manifest_sha256'], 'manifest_fresh_download_sha256_verified': True,
            'prior_full_disk_recovery_files': archive['prior_files_disk_restored_and_sha256_verified'],
            'new_full_archive_download_or_recovery_performed': False})
    return {'verified_utc': datetime.datetime.now(datetime.UTC).isoformat(),
            'scope': 'Remote archive existence/server digest plus fresh manifest download. Historical full-recovery receipts remain separate.',
            'local_outputs_removed': 0, 'archives': checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalogue', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    token = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True,
                           check=True, timeout=15).stdout.strip()
    receipt = verify(json.loads(args.catalogue.read_text()), token)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'archives_checked': len(receipt['archives']), 'local_outputs_removed': 0}))


if __name__ == '__main__':
    main()
