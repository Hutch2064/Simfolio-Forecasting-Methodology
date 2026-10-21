"""Describe component usage and allocation changes; no score/selection weight."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
O = Path(__file__).resolve().parent; V = O / 'asset_level_v6'
sys.path.insert(0, str(V)); sys.path.insert(0, str(O))
import common
from candidates.asset_selector_universe.bank import BANK
from candidates.asset_selector_universe_certified import energy
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d = V / 'candidates/asset_selector_inner_precision_batched/study1000'
    s = read(d / 'summary.json'); a = read(d / 'verification_receipt.json')
    assert a['verified'] and a['summary_sha256'] == sha(d / 'summary.json')
    refs = read(d / 'references.json')['cases']; stability = []; selected = None
    for n in (10, 30, 100, 300):
        rows = [r for r in s['records'] if r['inner_paths'] == n and r['configuration']['criterion'] == 'path']
        assert len(rows) == 216 and len({r['case'] for r in rows}) == 72
        counts = np.asarray([r['selection']['path_counts'] for r in rows])
        for r, c in zip(rows, counts):
            assert np.array_equal(c, energy.even_allocation(np.asarray(r['selection']['ideal_weights']), 1000))
        high = np.asarray([refs[str(r['case'])]['precision1000_path']['selection']['path_counts'] for r in rows])
        stability.append({'inner_paths': n, 'executions': len(rows),
                          'mean_allocation_total_variation_vs1000inner': float(np.abs(counts-high).sum(axis=1).mean()/2000),
                          'dominant_component_matches1000inner_percent': float(100*(counts.argmax(axis=1) == high.argmax(axis=1)).mean()),
                          'identical_allocations_percent': float(100*np.all(counts == high, axis=1).mean()),
                          'median_active_components': float(np.median((counts > 0).sum(axis=1)))})
        if n == 30:
            selected = counts
    families = {}
    for i, spec in enumerate(BANK):
        name = (spec['family'] if spec['kind'] == 'filtered' else 'parent mixture' if spec['kind'] == 'parent'
                else 'raw IID' if spec['sampling'] == 'IID' else 'raw with replacement' if spec['replacement'] else 'raw history cycles')
        families.setdefault(name, []).append(i)
    family_rows = []
    for name, indices in families.items():
        totals = selected[:, indices].sum(axis=1)
        family_rows.append({'family': name, 'percent_executions_present': float(100*(totals > 0).mean()),
                            'mean_percent_final_paths': float(totals.mean()/10)})
    family_rows.sort(key=lambda r: -r['mean_percent_final_paths'])
    presence = (selected > 0).sum(axis=0)
    out = {'verified_summary_sha256': sha(d / 'summary.json'), 'audit_sha256': sha(d / 'verification_receipt.json'),
           'diagnostic_only': True, 'zero_score_or_selection_weight': True, 'full_panel_frequencies': False,
           'requests': 72, 'Monte_Carlo_repetitions': 3, 'outer_paths': 1000,
           'no_independence_or_significance_claim': True, 'stability': stability,
           'usage_scope': 'Path-scoring selector at30internal paths on the72reused development requests;216executions',
           'components_used': int((presence > 0).sum()), 'families': family_rows,
           'components_unused': [{'index': i, 'specification': BANK[i]} for i in np.flatnonzero(presence == 0).tolist()],
           'components_present_in_at_most_one_percent_of_executions': int((presence/len(selected) <= .01).sum()),
           'one_percent_is_descriptive_not_a_pruning_rule': True,
           'component_usage': [{'index': i, 'specification': BANK[i], 'executions_present': int(presence[i]),
                                'mean_percent_final_paths': float(selected[:, i].mean()/10)} for i in range(len(BANK))],
           'family_costs_not_profiled_here': True,
           'pruning_requires_matched_main_score_ablation_and_broader_confirmation': True,
           'verifier_sha256': sha(Path(__file__))}
    (O / 'selector_precision_selection_diagnostic.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('component_usage', 'components_unused')}, indent=2))


if __name__ == '__main__':
    main()
