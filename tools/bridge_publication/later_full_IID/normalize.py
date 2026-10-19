"""Saved-only projection of two fixed completed IID studies; no scientific runs.

Reuse the bridge normalizer's stdlib encoding/hashing. No NumPy, native,
forecast, scorer, archive or network calls. Raw later-batch closure is pending
and remains a separate parent publication gate. --check reproduces all JSON.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zipfile

U = Path('tmp/catalogue52-20261018/short-horizon-mechanisms/refinement52/unified_frontier')
OUT = U/'full_IID_publication_normalization'
BRIDGE = U/'rough_only_conventional_bridge_publication_normalization/normalize.py'
BRIDGE_SHA = '1d60c45411a9e3af4bb4a96bdfa6c5d1ae4e664fb888ad46c3425f380a8663a7'
DATA_SHA = '3825258f753ba0a4ec653c56403ef7e163e80ea89eb1c06856b6415160672a76'
STUDIES = {
    'rough_IID': ('rough_iid_conventional', 'full-b67722c7aaebc226',
        '03ce9b566ec71021dc40f1236e338060238347211a23fd5bc3855985f9a809b9',
        'catalogue_support/drafts_rough_IID/seal-v2.json',
        '9be37f24170a7e3dd129dda0354e8175dba97454eb6d1c4103e22ca3d53b98fc'),
    'original_nodes_IID': ('iid_original_nodes', 'full-52d4c26b608ac1ac',
        '601dcd72bfd541533ad428e65d92cdae37de72e6aee30a0602aa920da3120347',
        'catalogue_support/drafts_original_nodes_IID/seal-v3.json',
        '765206aa4ec39f4bcf523fe61b65f92ff96df52589275481ac0bb2a2afb1842c')}
PANELS = {'52': (52, 2652, 455832, 3904212, 'state-anchor52',
    '3b94398acc00f8b44a71391d515a3d917095364ce5c58fe3101d9d40dadf9cdc'),
    '80': (80, 4080, 701280, 6006480, 'rough-off80',
    'bc4ad30824e8c23d924da3e7de7beb07b1131b3845ae10dd1b6758e16b50c4c2')}


def build(root):
    spec = importlib.util.spec_from_file_location('saved_bridge_projection_helpers', root/BRIDGE)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    sha, encode, digest = bridge.sha, bridge.encode, bridge.digest
    assert sha((root/BRIDGE).read_bytes()) == BRIDGE_SHA
    def read(path):
        return json.loads((root/path).read_text())
    def ref(path):
        path = Path(path)
        return {'path': str(path), 'sha256': sha((root/path).read_bytes())}
    def relative(path):
        path = Path(path)
        return path.relative_to(root) if path.is_absolute() else path
    outputs = {}
    for name, (study, full, audit_sha, draft_path, draft_sha) in STUDIES.items():
        run, dest = U/study/full, OUT/name
        review_path = U/'diagnostics'/(study+'_full_actual_audit/review.json')
        assert ref(review_path)['sha256'] == audit_sha
        draft_path = U/draft_path
        assert ref(draft_path)['sha256'] == draft_sha
        for row in read(draft_path)['files']:
            assert ref(relative(row['path']))['sha256'] == row['sha256']
        prior = read(review_path)
        assert prior['passed']
        for path, expected in prior['artifacts'].items():
            assert ref(relative(path))['sha256'] == expected, path
        manifest, completion = read(run/'manifest.json'), read(run/'completion.json')
        local_audit, analysis = read(run/'audit.json'), read(run/'analysis.json')
        assert local_audit['passed'] and local_audit['all_cells_counts_controls_reconstruct']
        assert local_audit['failures'] == 0 and completion['stop_reason'] is None
        assert manifest['fresh_forecast_tasks'] == {'52': 2652, '80': 4080}
        assert manifest['simulations'] == 240
        assert len(local_audit['checkpoint_sha256']) == len(local_audit['metadata_sha256']) == 6732
        for group in ('source_hashes', 'snapshot_sources'):
            for path, expected in manifest[group].items():
                assert ref(relative(path))['sha256'] == expected, path
        auditor = ref(OUT/'normalize.py')
        dependencies = [ref(BRIDGE), ref(U/study/'full_audit.py'), ref(review_path), ref(draft_path)]
        required = [auditor, *dependencies]
        panel_rows, audits = {}, {}
        for panel, (portfolios, origins, cells, contributions, reference_name, reference_sha) in PANELS.items():
            reference_path = U.parents[1]/reference_name/'manifest.json'
            assert ref(reference_path)['sha256'] == reference_sha
            reference = read(reference_path)
            identities = manifest['task_identities'][panel]
            assert reference['data_manifest_sha256'] == DATA_SHA
            assert identities == reference['task_identities'] and len(identities) == origins
            assert len({item['task_id'] for item in identities}) == origins
            assert completion['complete_origins'][panel] == completion['required_origins'][panel] == origins
            assert completion['failures'][panel] == {} and local_audit['complete_origins'][panel] == origins
            p = prior['panels'][panel]
            if name == 'rough_IID':
                assert p['tasks'] == origins and p['reducer_exact'] and p['controls_exact'] == len(manifest['labels'][panel])-1
                assert p['count_sum'] == contributions
            else:
                assert p['fresh_origins'] == origins and p['canonical_cellbytes_reconstructed']
                assert p['saved_control_rows_exact'] == len(manifest['labels'][panel])-1 and p['sum_counts'] == contributions
            results = read(run/panel/'results.json')
            assert results == completion['results'][panel]
            scores = {label: value['CRPS'] for label, value in analysis[panel]['models'].items()}
            assert scores == results['scores'] and set(scores) == set(manifest['labels'][panel])
            assert len(manifest['new_variants']) == 1
            candidate = manifest['new_variants'][0]
            assert prior['scores'][panel]['candidate_CRPS'] == scores[candidate]
            assert results['complete_origins'] == results['required_origins'] == origins
            assert results['complete_portfolios'] == portfolios
            with zipfile.ZipFile(root/run/panel/'cells.npz') as z:
                data = z.read('counts.npy')
            assert data[:6] == b'\x93NUMPY'
            offset = 10 if data[6] == 1 else 12
            size = struct.unpack('<H' if data[6] == 1 else '<I', data[8:offset])[0]
            header = ast.literal_eval(data[offset:offset+size].decode('latin1'))
            counts = data[offset+size:]
            assert header['descr'] == '<i8' and header['shape'] == (portfolios, 8766) and not header['fortran_order']
            assert len(counts) == cells*8
            total = 0
            for (value,) in struct.iter_unpack('<q', counts):
                assert value > 0
                total += value
            assert total == contributions
            denominators = dict(origins=origins, portfolios=portfolios, simulations=240,
                cells_per_model=cells, origin_horizon_contributions=contributions,
                horizons=8766, counts_sha256=sha(counts))
            assert all(analysis[panel][k] == v for k, v in denominators.items() if k not in ('horizons', 'counts_sha256'))
            controls = {}
            assert len(manifest['control_sources'][panel]) == 1
            for control in manifest['control_sources'][panel]:
                path = relative(control['path'])
                assert ref(path)['sha256'] == control['sha256']
                controls.update({label: ref(path) for label in scores if label != candidate})
            epsilon = sys.float_info.epsilon
            gamma = cells*epsilon/(1-cells*epsilon)
            aggregation = dict(schema='unified-frontier-scalar-aggregation-v1', method='float64_gamma_n_mean_absolute_cells',
                dtype='float64', canonical_order='C', n=cells, epsilon=epsilon, gamma_n=gamma,
                canonical_scores=scores, raw_results_scores=results['scores'], per_label={label: dict(
                    mean_absolute_cells=score, absolute_difference=0., error_bound=gamma*score,
                    ulp_difference=0, within_bound=True) for label, score in scores.items()})
            evidence = {role: ref(path) for role, path in dict(manifest=run/'manifest.json', reference_manifest=reference_path,
                results=run/panel/'results.json', analysis=run/'analysis.json', cells=run/panel/'cells.npz',
                completion=run/'completion.json').items()}
            audit = dict(schema='unified-frontier-publication-audit-v1', panel=panel, independent_from_runner=True,
                data_manifest_sha256=DATA_SHA, complete_origins=origins, failures=0,
                all_task_identities_match_reference=True, all_cells_and_counts_rebuilt_byte_exactly=True,
                all_saved_controls_byte_equal=True, all_bound_sources_and_controls_match=True,
                task_identities_sha256=digest(identities), counts_sha256=sha(counts),
                denominator_sha256=digest(denominators), canonical_denominators=denominators,
                artifact_sha256={role: row['sha256'] for role, row in evidence.items()},
                auditor_source=auditor, auditor_dependencies=dependencies,
                required_publication_source_refs=required, score_aggregation=aggregation,
                original_independent_audit=ref(review_path), original_runner_audit=ref(run/'audit.json'),
                scope='Schema projection of the pinned accepted independent completed saved study, not a new raw reconstruction or financial run.')
            filename = name+'/'+panel+'/publication-audit.json'
            outputs[filename] = encode(audit)
            evidence['audit'] = {'path': str(dest/panel/'publication-audit.json'), 'sha256': sha(outputs[filename])}
            audits[panel] = evidence['audit']
            panel_rows[panel] = dict(status='complete_scored_and_audited', evidence=evidence, controls=controls,
                runner_sources=[ref(U/study/'full_run.py')], scores=scores,
                raw_results_scores=results['scores'], score_aggregation=aggregation)
        final = dict(schema='unified-frontier-publication-audit-completion-v1', no_fits_or_forecasts=True,
            panel_audits=audits, normalized_analysis=ref(run/'analysis.json'), original_independent_audit=ref(review_path),
            source_hashes={row['path']: row['sha256'] for row in required}, required_publication_source_refs=required,
            auditor_source=auditor, raw_checkpoints_preserved=True, schema_projection_only=True,
            archive_requirement='Original later raw-batch verified private closure is pending parent; public wrapper/metadata Git-preserved separately.',
            preserved_draft=ref(draft_path), fresh_forecasts={'52': 2652, '80': 4080},
            scope='Preserves exact existing scores, canonical denominators and validation scope; no new eligibility rules or scientific claims.')
        filename = name+'/publication-audit-completion.json'
        outputs[filename] = encode(final)
        fixture = dict(schema='full-IID-normalized-publication-panel-fixture-v1', evidence_root='.',
            canonical_data_root='tmp/catalogue52-20261018', sources=required, panels=panel_rows,
            completion_ref={'path': str(dest/'publication-audit-completion.json'), 'sha256': sha(outputs[filename])},
            scope='Public-safe aggregate scores/counts, hashes and provenance; completed BOTH panels, not new work. Private raw closure remains a parent publication gate.')
        outputs[name+'/actual-consumer-panel-fixture.json'] = encode(fixture)
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
    print('Both fixed IID saved-publication projections verified; zero new forecasts.')


if __name__ == '__main__':
    main()
