"""Independently aggregate persisted, hashed task losses with the public evaluator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from simfolio_forecasting_methodology.evaluation import CellAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.output/'manifest.json').read_text())
    records = json.loads((args.output/'records.json').read_text())
    arms = list(manifest['model_ids'])
    for group in manifest['tasks']:
        assert len(group) == len(arms)
        shared = {k: v for k, v in group[0].items() if k not in ('task_id', 'model_id')}
        for arm, identity in zip(arms, group, strict=True):
            assert identity['model_id'] == manifest['model_ids'][arm]
            assert {k: v for k, v in identity.items() if k not in ('task_id', 'model_id')} == shared
    expected = {(t[0]['portfolio_id'], t[0]['origin_label']): t[0] for t in manifest['tasks']}
    assert len(expected) == len(manifest['tasks'])
    seen = set(); counts = {}; acc = {arm: CellAccumulator() for arm in arms}; splits = {}
    baseline_matches = 0
    for r in records:
        arm = r['arm']; key = (r['portfolio'], r['origin']); task = expected[key]
        assert (key, arm) not in seen
        seen.add((key, arm))
        p = args.output / f"task-{r['canonical_task']:04d}-{arm}.npz"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['checkpoint_sha256']
        losses = np.load(p)['losses']
        assert len(losses) == task['horizon_days']
        assert hashlib.sha256(losses.tobytes()).hexdigest() == r['losses_sha256']
        if arm == arms[0]:
            for h in range(1, len(losses)+1): counts[(r['portfolio'], h)] = counts.get((r['portfolio'], h), 0)+1
        acc[arm].add_vector(r['portfolio'], losses)
        split = 'rolling_origin' if r['origin'].startswith('rolling_') else r['origin']
        splits.setdefault(split, {a: CellAccumulator() for a in arms})[arm].add_vector(r['portfolio'], losses)
        if args.reference and arm in ('frontier', 'mcmc'):
            j = 0 if arm == 'frontier' else 1
            ref = np.load(args.reference / f"task-{r['canonical_task']:04d}-model-{j}.npz")['losses']
            assert losses.tobytes() == ref.tobytes(), (r['canonical_task'], arm)
            baseline_matches += 1
    assert len(seen) == len(expected)*len(arms)
    keys = sorted(counts)
    arrays = {'portfolio': np.asarray([k[0] for k in keys]), 'horizon': np.asarray([k[1] for k in keys]),
              'origin_count': np.asarray([counts[k] for k in keys])}
    scores = {}
    for arm in arms:
        score = acc[arm].fixed_denominator_score(expected_cells=len(keys), expected_tasks=len(expected),
                    completed_tasks=len(expected), expected_origin_counts=counts)
        means = acc[arm].cell_means(); arrays[arm] = np.array([means[k] for k in keys])
        scores[arm] = {'crps': score, 'seconds': sum(r['seconds'] for r in records if r['arm']==arm),
                       'cpu_seconds': sum(r['cpu_seconds'] for r in records if r['arm']==arm),
                       'cells': len(keys), 'tasks': len(expected)}
    diagnostics = {}
    for arm in arms:
        rows = [d for r in records if r['arm']==arm for d in r['diagnostics']]
        if not rows: continue
        info = {'asset_fits': len(rows), 'laplace_optimizer_failures': sum(not d['laplace_optimizer_success'] for d in rows)}
        info['unique_training_histories'] = len({d['training_sha256'] for d in rows})
        if arm.startswith('hmc'):
            info.update({'max_rhat': max(max(d['rhat'].values()) for d in rows),
                         'min_bulk_ess': min(min(d['bulk_ess'].values()) for d in rows),
                         'min_tail_ess': min(min(d['tail_ess'].values()) for d in rows),
                         'fits_rhat_above_1_01': sum(max(d['rhat'].values())>1.01 for d in rows),
                         'fits_bulk_or_tail_ess_below_400': sum(min(*d['bulk_ess'].values(),*d['tail_ess'].values())<400 for d in rows),
                         'energy_errors': sum(d['energy_error_count'] for d in rows)})
        else:
            info.update({'min_importance_ess': min(d['importance_ess'] for d in rows),
                         'max_pareto_k': max(d['pareto_k'] for d in rows),
                         'fits_pareto_k_above_0_7': sum(d['pareto_k']>.7 for d in rows),
                         'fits_importance_ess_below_400': sum(d['importance_ess']<400 for d in rows),
                         'max_normalized_weight': max(d['max_normalized_weight'] for d in rows)})
            if 'vi_optimizer_success' in rows[0]: info['vi_optimizer_failures'] = sum(not d['vi_optimizer_success'] for d in rows)
        failures = []
        for d in rows:
            reasons = []
            proposals = d.get('mode_search', [d])
            if any(not p['laplace_optimizer_success'] or
                   p['laplace_gradient_max'] > .01 or
                   p['laplace_min_hessian_eigenvalue'] <= 0 for p in proposals):
                reasons.append('proposal_stationarity')
            numeric = [p[k] for p in proposals for k in
                       ('laplace_gradient_max', 'laplace_min_hessian_eigenvalue')]
            if arm.startswith('hmc'):
                numeric += [*d['rhat'].values(), *d['bulk_ess'].values(), *d['tail_ess'].values()]
                if max(d['rhat'].values()) > 1.01:
                    reasons.append('rhat')
                if min(*d['bulk_ess'].values(), *d['tail_ess'].values()) < 400:
                    reasons.append('chain_ess')
                if d['energy_error_count']:
                    reasons.append('retained_energy_error')
            else:
                numeric += [d['pareto_k'], d['importance_ess'], d['max_normalized_weight']]
                if d['pareto_k'] > .7:
                    reasons.append('pareto_k')
                if d['importance_ess'] < 400:
                    reasons.append('importance_ess')
                if d['max_normalized_weight'] > .02:
                    reasons.append('concentrated_weight')
                if ('vi_optimizer_success' in d and
                    (not d['vi_optimizer_success'] or d['vi_gradient_max'] > .01)):
                    reasons.append('vi_stationarity')
            if not np.all(np.isfinite(numeric)):
                reasons.append('nonfinite_diagnostic')
            if reasons:
                failures.append({'training_sha256': d['training_sha256'],
                                 'reasons': reasons})
        info['conservative_diagnostic_gate_passed'] = not failures
        info['failed_asset_fit_calls'] = len(failures)
        info['failed_unique_training_histories'] = len({d['training_sha256'] for d in failures})
        info['diagnostic_failure_reasons'] = {reason: sum(reason in d['reasons'] for d in failures)
                                              for reason in sorted({r for d in failures for r in d['reasons']})}
        diagnostics[arm] = info
    result = {'scope': manifest['scope'], 'scores': scores, 'all_denominator_gates_passed': True,
              'baseline_loss_vectors_byte_exact': baseline_matches,
              'split_scores': {s: {a: v.aggregate_score() for a,v in group.items()} for s,group in splits.items()},
              'diagnostics': diagnostics, 'manifest_file_sha256': hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()}
    np.savez_compressed(args.output/'paired-cells.npz', **arrays)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
