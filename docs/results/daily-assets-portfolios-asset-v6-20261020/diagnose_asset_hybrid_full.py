"""Additive decomposition of the unchanged main Q; never a selection score."""
from collections import defaultdict
from pathlib import Path
import datetime
import hashlib
import json
import sys
import numpy as np

O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
sys.path.insert(0, str(V))
import common
import sweep_v6 as api

read = lambda p: json.loads(p.read_text())


def origin_weights(roster):
    """Flatten the frozen quarter/branch/class/family/series/origin average."""
    groups = defaultdict(list)
    families, classes, branches, quarters = [defaultdict(set) for _ in range(4)]
    for key, m, date in roster:
        b, c, f, sid = [m[k] for k in ('panel_branch', 'asset_class', 'family', 'series_id')]
        q = f'{date[:4]}-Q{(int(date[5:7])-1)//3+1}'
        groups[b, q, c, f, sid].append(key)
        families[b, q, c, f].add(sid)
        classes[b, q, c].add(f)
        branches[b, q].add(c)
        quarters[q].add(b)
    assert quarters
    weights = {}
    for (b, q, c, f, sid), keys in groups.items():
        assert 'assets' in quarters[q] and quarters[q] <= {'assets', 'portfolios'}
        value = 1./(len(quarters)*len(quarters[q])*len(branches[b, q])*
                    len(classes[b, q, c])*len(families[b, q, c, f])*len(keys))
        for key in keys:
            assert key not in weights
            weights[key] = value
    assert abs(sum(weights.values())-1.) < 1e-10
    return weights


def main():
    baseline = read(V/'full_pair_independent_audit.json')
    full = V/'candidates/asset_prior_best/full'
    result = read(full/'verification_receipt.json')
    assert result['complete'] and result['successful_tasks'] == 64049
    mid = result['model_id']
    audit = {'models': {mid: {'main_score': result['main_score'], 'checkpoint_commitment': result['checkpoint_commitment']}},
             'worse_reference_model_id': baseline['worse_reference_model_id']}
    panel = api.load_panel(V/'inputs', verify_files=True)
    roster, series = [], []
    for i, row in enumerate(panel['series']):
        dates, values, origins = api.load_series(V/'inputs', row, verify_files=False)
        series.append((i, row, dates, values, origins))
        roster.extend(((i, j), row, o['origin_date']) for j, o in enumerate(origins))
    weights = origin_weights(roster)
    labels = ['at_most_31_calendar_days', '32_to_366_calendar_days',
              '367_to_1826_calendar_days', '1827_to_7305_calendar_days',
              'over_7305_calendar_days']
    summaries = read(V/'full_pair_diagnostics.json')['main_Q_additive_contributions']
    for mid in audit['models']:
        root = api.model_directory(full/'results', mid)
        fingerprint = read(root/'execution.json')['fingerprint']
        pieces = {name: defaultdict(float) for name in ('horizon', 'family', 'class', 'split', 'branch')}
        verified, commitment = 0, hashlib.sha256()
        for i, row, dates, values, origins in series:
            for j, origin in enumerate(origins):
                cp, _ = api.task_paths(root, (i, j))
                loss, _ = api.original._read_checkpoint(cp, api.task_identity(fingerprint, mid, row['series_id'], origin))
                n, h = origin['position']+1, origin['max_horizon']
                days = (dates[n:n+h].astype('datetime64[D]')-np.datetime64(origin['origin_date'][:10], 'D')).astype(int)
                edges = np.log1p(days.astype(float))
                masses = np.diff(np.concatenate(([0.], edges)))/edges[-1]
                cells = weights[i, j]*loss*masses/(np.std(values[:n], ddof=1)*np.sqrt(np.arange(1, h+1)))
                total = float(cells.sum())
                by_band = np.bincount(np.searchsorted([31, 366, 1826, 7305], days, side='left'), weights=cells, minlength=5)
                for label, value in zip(labels, by_band):
                    pieces['horizon'][label] += float(value)
                pieces['branch'][row['panel_branch']] += total
                pieces['class'][f"{row['panel_branch']}/{row['asset_class']}"] += total
                pieces['family'][f"{row['panel_branch']}/{row['asset_class']}/{row['family']}"] += total
                pieces['split'][origin['evaluation_split']] += total
                verified += 1
                commitment.update(f'{i}/{j}:{hashlib.sha256(cp.read_bytes()).hexdigest()}\n'.encode())
        assert verified == 64049 and commitment.hexdigest() == audit['models'][mid]['checkpoint_commitment']
        for dimension, groups in pieces.items():
            assert abs(sum(groups.values())-audit['models'][mid]['main_score']) < 2e-12, dimension
        summaries[mid] = {name: dict(groups) for name, groups in pieces.items()}
    reference = audit['worse_reference_model_id']
    denominator = baseline['models'][reference]['main_score']
    contributions = {mid: {dimension: {group: 100*(summaries[reference][dimension][group]-value)/denominator
                        for group, value in groups.items()} for dimension, groups in dims.items()}
                        for mid, dims in summaries.items()}
    receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'same_successful_tasks_each_model': 64049, 'main_Q_additive_contributions': summaries,
               'percent_better_contributions_vs_worse_reference': contributions,
               'reference_model_id': reference, 'canonical_Q_reconciled_each_dimension': True,
               'all_checkpoint_hashes_match_independent_audit': True,
               'diagnostic_only_zero_additional_score_or_selection_weight': True,
               'interpretation': 'Each dimension separately sums to the single main Q; percentage contributions sum to the whole-panel improvement versus the worse reference. These are not additional scores or significance tests.'}
    api.original._atomic_json(V/'candidate_full_diagnostics.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('main_Q_additive_contributions', 'percent_better_contributions_vs_worse_reference')}), flush=True)


if __name__ == '__main__':
    main()
