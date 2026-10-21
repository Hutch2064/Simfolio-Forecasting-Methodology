"""Reproduce the frozen, superseded 135-request selection diagnostic."""
from pathlib import Path
import datetime, hashlib, json, sys
import numpy as np

O = Path(__file__).resolve().parent
V = O / 'asset_level_v6'
sys.path.insert(0, str(O)); sys.path.insert(0, str(V))
import common
from candidates.asset_selector_universe.bank import BANK
from candidates.asset_selector_universe_certified import energy

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def family(spec):
    if spec['kind'] == 'filtered':
        return spec['family']
    if spec['kind'] == 'parent':
        return 'SV/RMS parent mixture'
    if spec['sampling'] == 'IID':
        return 'Raw IID'
    return 'Raw blocks replacement' if spec['replacement'] else 'Raw blocks history cycles'


def main():
    folder = V / 'candidates/asset_selector_path_objective/development1000'
    assert read(folder / 'active.json')['status'] == 'superseded_by_user_fast_transition'
    files = sorted((folder / 'case_records').glob('*.json'))
    old = read(O / 'selector_selection_usage_diagnostic.json')
    assert len(files) == old['completed_requests'] == 135
    records = [r for p in files for r in read(p)['records']]
    models = {}
    for j, criterion in enumerate(('terminal', 'path')):
        selected = [r for r in records if r['config_index'] == j]
        assert len(selected) == 135 and len({r['case'] for r in selected}) == 135
        for r in selected:
            allocation = np.asarray(r['selection']['path_counts'])
            weights = np.asarray(r['selection']['ideal_weights'])
            assert allocation.sum() == 1000 and r['paths'] == r['inner_paths'] == 1000
            assert np.array_equal(allocation, energy.even_allocation(weights, 1000))
            cp = folder / 'losses' / f"{j:03d}-{r['case']:03d}.npz"
            assert sha(cp) == r['checkpoint_sha256']
        counts = np.asarray([r['selection']['path_counts'] for r in selected])
        families = []
        for name in sorted({family(spec) for spec in BANK}):
            values = counts[:, [i for i, spec in enumerate(BANK) if family(spec) == name]].sum(axis=1)
            families.append({'family': name, 'requests_present': int((values > 0).sum()),
                             'percent_requests_present': float(100 * (values > 0).mean()),
                             'mean_percent_final_paths': float(values.mean() / 10)})
        families.sort(key=lambda r: -r['mean_percent_final_paths'])
        active = (counts > 0).sum(axis=1)
        result = {'requests': len(selected), 'unique_allocation_vectors': len({tuple(x) for x in counts}),
                  'unique_active_component_sets': len({tuple(np.flatnonzero(x)) for x in counts}),
                  'active_components_min': int(active.min()), 'active_components_median': float(np.median(active)),
                  'active_components_max': int(active.max()), 'components_used': int((counts.sum(axis=0) > 0).sum()),
                  'families': families}
        for key in result:
            if key == 'families':
                for fresh, previous in zip(result[key], old['models'][criterion][key]):
                    assert fresh['family'] == previous['family'] and fresh['requests_present'] == previous['requests_present']
                    assert np.isclose(fresh['percent_requests_present'], previous['percent_requests_present'], rtol=0, atol=1e-12)
                    assert np.isclose(fresh['mean_percent_final_paths'], previous['mean_percent_final_paths'], rtol=0, atol=1e-12)
            else:
                assert result[key] == old['models'][criterion][key]
        models[criterion] = result
    out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'verified': True,
           'diagnostic_only': True, 'zero_score_or_selection_weight': True,
           'scope': '135 frozen completed requests from the user-superseded slow study; no new forecasts or full-panel frequency claim',
           'original_diagnostic_sha256': sha(O / 'selector_selection_usage_diagnostic.json'),
           'case_record_sha256': {p.name: sha(p) for p in files}, 'verifier_sha256': sha(Path(__file__)),
           'models': models}
    (O / 'selector_selection_usage_reproduction.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'case_record_sha256'}, indent=2))


if __name__ == '__main__':
    main()
