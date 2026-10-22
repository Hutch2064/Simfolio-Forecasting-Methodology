"""Project pinned saved bridge80 proof into the unchanged publication contract.

No study/auditor execution, NumPy, native imports, forecasts, or network. Run
with --evidence-root WORKSPACE; --check deterministically checks saved outputs.
Public files contain aggregate scores/counts, hashes and provenance only.
"""
import argparse
import ast
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

U = Path('tmp/catalogue52-20261018/short-horizon-mechanisms/refinement52/unified_frontier')
OUT = U/'rough_only_conventional_bridge_publication_normalization'
RUN = U/'rough_only_conventional_bridge/full-4ce66184d7db04fd'
REVIEW = U/'diagnostics/rough_only_conventional_bridge_full_actual_audit/review.json'
REFERENCE = Path('tmp/catalogue52-20261018/short-horizon-mechanisms/rough-off80/manifest.json')
ARCHIVE = U/'archive_corrected_nodes_bridge/corrected-nodes-bridge-2534707642eb5804c5af.manifest.json'
DATA_SHA = '3825258f753ba0a4ec653c56403ef7e163e80ea89eb1c06856b6415160672a76'
PINS = {str(REVIEW): '434cadd8ac94b9a74dc784b0efeb5a7d2ec29455be5d432a8114c9192592c1c1',
        str(ARCHIVE): 'fa1a964cc84e17ccad0f426ca3f62ba7163e0a9a5dd209eed0729025a989d193',
        str(REFERENCE): 'bc4ad30824e8c23d924da3e7de7beb07b1131b3845ae10dd1b6758e16b50c4c2',
        str(RUN/'80/results.json'): '90bc4a1735c352a10b837c82edba0d4979284668ecd05fbccb2b91966ddcd544'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def build(root):
    def read(path):
        return json.loads((root/path).read_text())
    def ref(path):
        path = Path(path)
        return {'path': str(path), 'sha256': sha((root/path).read_bytes())}
    for name, expected in PINS.items():
        assert ref(name)['sha256'] == expected, name
    prior, archive, manifest = read(REVIEW), read(ARCHIVE), read(RUN/'manifest.json')
    archive_members = dict(archive['members'])
    prior_manifests = []
    for dependency in archive['prior_archives']:
        path = Path(dependency['manifest_ref']['path'])
        path = path.relative_to(root) if path.is_absolute() else path
        assert ref(path)['sha256'] == dependency['manifest_asset']['sha256']
        document = read(path)
        archive_members.update(document['members'])
        prior_manifests.append(ref(path))
    assert prior['passed'] and prior['checks']['fresh80_tasks'] == 4080
    assert prior['checks']['fresh52_forecasts'] == 0
    assert prior['checks']['all13_saved_control_cells_bytes_equal']
    assert prior['checks']['independent_ascending_origin_sum_count_cells_bytes_equal']
    for name, expected in prior['artifacts'].items():
        path = Path(name).relative_to(root)
        assert ref(path)['sha256'] == expected
    for group in ('source_hashes', 'snapshot_sources'):
        for name, expected in manifest[group].items():
            path = Path(name)
            path = path.relative_to(root) if path.is_absolute() else path
            assert ref(path)['sha256'] == expected, name
    identities = manifest['task_identities']['80']
    reference = read(REFERENCE)
    assert reference['data_manifest_sha256'] == DATA_SHA
    assert identities == reference['task_identities'] and len(identities) == 4080
    assert len({item['task_id'] for item in identities}) == 4080
    raw_completion = read(RUN/'completion.json')
    assert raw_completion['complete_origins'] == raw_completion['required_origins'] == 4080
    assert raw_completion['failures'] == {} and raw_completion['stop_reason'] is None
    assert raw_completion['new52_forecasts'] == manifest['fresh_forecast_tasks']['52'] == 0
    analysis, results = read(RUN/'analysis.json'), read(RUN/'80/results.json')
    scores = {label: item['CRPS'] for label, item in analysis['80']['models'].items()}
    assert scores == results['scores'] and set(scores) == set(manifest['labels']['80'])
    assert results['complete_origins'] == results['required_origins'] == 4080
    assert results['complete_portfolios'] == 80 and manifest['simulations'] == 240
    # Hash the saved counts payload, not a new reduction/reconstruction.
    with zipfile.ZipFile(root/RUN/'80/cells.npz') as z:
        data = z.read('counts.npy')
    assert data[:6] == b'\x93NUMPY'
    offset = 10 if data[6] == 1 else 12
    size = struct.unpack('<H' if data[6] == 1 else '<I', data[8:offset])[0]
    header = ast.literal_eval(data[offset:offset+size].decode('latin1'))
    counts = data[offset+size:]
    assert header['descr'] == '<i8' and header['shape'] == (80, 8766) and not header['fortran_order']
    assert len(counts) == 701280*8
    values = [item[0] for item in struct.iter_unpack('<q', counts)]
    assert min(values) > 0 and sum(values) == 6006480
    denominators = {'origins': 4080, 'portfolios': 80, 'simulations': 240, 'cells_per_model': 701280,
                        'origin_horizon_contributions': 6006480, 'horizons': 8766, 'counts_sha256': sha(counts)}
    assert all(analysis['80'][k] == v for k, v in denominators.items() if k not in ('horizons', 'counts_sha256'))
    controls = {}
    for control in manifest['control_sources']['80']:
        path = Path(control['path']).relative_to(root)
        assert ref(path)['sha256'] == control['sha256']
        controls.update({label: ref(path) for label in control['labels']})
    dependencies = [ref(U/'rough_only_conventional_bridge/full_audit.py'), ref(REVIEW)]
    auditor = ref(OUT/'normalize.py')
    required = [auditor, *dependencies]
    # Closed bridge raw artifacts are in the pinned archive; canonical reference is separately pinned.
    raw_refs = [ref(RUN/n) for n in ('manifest.json', 'completion.json', 'analysis.json', '80/results.json', '80/cells.npz')]
    for row in [*raw_refs, ref(REFERENCE), ref(REVIEW), *controls.values(), dependencies[0]]:
        assert archive_members[row['path']]['sha256'] == row['sha256']
    completion = {'complete_origins': {'80': 4080}, 'required_origins': {'80': 4080},
                  'failures': {'80': {}}, 'new52_forecasts': 0,
                  'projection_of_original_completion': ref(RUN/'completion.json')}
    outputs = {'80/normalized-completion.json': encode(completion)}
    evidence = {role: ref(path) for role, path in {'manifest': RUN/'manifest.json',
        'reference_manifest': REFERENCE, 'results': RUN/'80/results.json', 'analysis': RUN/'analysis.json',
        'cells': RUN/'80/cells.npz'}.items()}
    evidence['completion'] = {'path': str(OUT/'80/normalized-completion.json'), 'sha256': sha(outputs['80/normalized-completion.json'])}
    epsilon = sys.float_info.epsilon
    gamma = 701280*epsilon/(1-701280*epsilon)
    aggregation = {'schema': 'unified-frontier-scalar-aggregation-v1', 'method': 'float64_gamma_n_mean_absolute_cells',
        'dtype': 'float64', 'canonical_order': 'C', 'n': 701280, 'epsilon': epsilon, 'gamma_n': gamma,
        'canonical_scores': scores, 'raw_results_scores': results['scores'], 'per_label': {label: {
            'mean_absolute_cells': score, 'absolute_difference': 0., 'error_bound': gamma*score,
            'ulp_difference': 0, 'within_bound': True} for label, score in scores.items()}}
    audit = {'schema': 'unified-frontier-publication-audit-v1', 'panel': '80', 'independent_from_runner': True,
        'data_manifest_sha256': DATA_SHA, 'complete_origins': 4080, 'failures': 0,
        'all_task_identities_match_reference': True, 'all_cells_and_counts_rebuilt_byte_exactly': True,
        'all_saved_controls_byte_equal': True, 'all_bound_sources_and_controls_match': True,
        'task_identities_sha256': digest(identities), 'counts_sha256': sha(counts),
        'denominator_sha256': digest(denominators), 'canonical_denominators': denominators,
        'artifact_sha256': {role: row['sha256'] for role, row in evidence.items()},
        'auditor_source': auditor, 'auditor_dependencies': dependencies,
        'required_publication_source_refs': required, 'score_aggregation': aggregation,
        'original_independent_audit': ref(REVIEW), 'original_runner_audit': ref(RUN/'audit.json'),
        'original_private_archive_manifest': ref(ARCHIVE), 'prior_archive_manifests': prior_manifests, 'fresh52_forecasts': 0,
        'scope': 'Deterministic schema projection of pinned completed independent saved80 proof434c; no new cell reconstruction, model or numerical execution.'}
    outputs['80/publication-audit.json'] = encode(audit)
    evidence['audit'] = {'path': str(OUT/'80/publication-audit.json'), 'sha256': sha(outputs['80/publication-audit.json'])}
    final = {'schema': 'unified-frontier-publication-audit-completion-v1', 'no_fits_or_forecasts': True,
        'panel_audits': {'80': evidence['audit']}, 'normalized_completion': evidence['completion'],
        'normalized_analysis': evidence['analysis'], 'original_independent_audit': ref(REVIEW),
        'source_hashes': {row['path']: row['sha256'] for row in required},
        'required_publication_source_refs': required, 'auditor_source': auditor,
        'raw_checkpoints_preserved': True, 'original_private_archive_manifest': ref(ARCHIVE),
        'schema_projection_only': True, 'fresh_forecasts': {'52': 0, '80': 4080},
        'scope': 'Existing verified private raw evidence plus Git-versioned public normalization source/metadata; no extra private asset required.'}
    outputs['publication-audit-completion.json'] = encode(final)
    fixture = {'schema': 'bridge80-normalized-publication-panel-fixture-v1', 'evidence_root': '.',
        'canonical_data_root': 'tmp/catalogue52-20261018', 'sources': required,
        'completion_ref': {'path': str(OUT/'publication-audit-completion.json'), 'sha256': sha(outputs['publication-audit-completion.json'])},
        'panels': {'52': {'status': 'not_evaluated', 'reason': 'Zero fresh52 forecasts; inherited single-history identity is a separate historical annotation.'},
                '80': {'status': 'complete_scored_and_audited', 'evidence': evidence, 'controls': controls,
                    'runner_sources': [ref(U/'rough_only_conventional_bridge/full_run.py')], 'scores': scores,
                    'raw_results_scores': results['scores'], 'score_aggregation': aggregation}},
        'historical_annotation': {'panel': '52', 'new_forecasts': 0, 'inherited_CRPS': .6022154610545161,
            'scope': 'Historical one-history identity only; not a newly evaluated52 panel.'},
        'scope': 'Public-safe aggregate scores, counts, hashes and provenance only; no provider returns or path arrays.'}
    outputs['actual-consumer-panel-fixture.json'] = encode(fixture)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    for name, content in build(root).items():
        path = root/OUT/name
        if args.check:
            assert path.read_bytes() == content, name
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(content)
    print('Saved-only deterministic bridge80 publication projection verified; zero new forecasts.')


if __name__ == '__main__':
    main()
