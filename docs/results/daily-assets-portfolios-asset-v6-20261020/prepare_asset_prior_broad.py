"""Freeze broad follow-through from calendar metadata before any pilot outcomes."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
O = Path(__file__).resolve().parent
V = O/'asset_level_v6'
C = V/'candidates/asset_prior_best'
sys.path.insert(0, str(V))
import common
import sweep_v6 as api
read = lambda p: json.loads(p.read_text())
key = lambda c: (c['series_id'], c['origin']['origin_label'])
quarter = lambda o: f"{o['origin_date'][:4]}-Q{(int(o['origin_date'][5:7])-1)//3+1}"


def main():
    assert not (C/'parent_selection.json').exists(), 'Do not change an already bound field plan'
    plan = read(C/'plan.json')
    assert not (C/'development/active.json').exists()
    panel = api.load_panel(V/'inputs', verify_files=False)
    metadata = {r['series_id']: r for r in panel['series']}
    used = {key(c) for phase in ('development', 'reserved') for c in plan['phases'][phase]}
    selected, asset_support = {}, {}
    for row in panel['series']:
        sid = row['series_id']
        origins = api.schedule(V/'inputs', row)
        for origin in origins:
            if row['panel_branch'] == 'assets':
                asset_support.setdefault(quarter(origin), []).append({'series_id': sid, 'origin': origin})
        remaining = [o for o in origins if (sid, o['origin_label']) not in used]
        rolling = [o for o in remaining if o['evaluation_split'] == 'rolling_origin']
        assert rolling
        picks = [rolling[0], rolling[len(rolling)//2], rolling[-1]]
        picks.extend(o for o in remaining if o['evaluation_split'] != 'rolling_origin')
        for origin in picks:
            c = {'series_id': sid, 'origin': origin}
            selected[key(c)] = c
    assets = {quarter(c['origin']) for c in selected.values() if metadata[c['series_id']]['panel_branch'] == 'assets'}
    portfolios = {quarter(c['origin']) for c in selected.values() if metadata[c['series_id']]['panel_branch'] == 'portfolios'}
    support_replays = []
    for q in sorted(portfolios-assets):
        pool = sorted(asset_support[q], key=lambda c: (key(c) in used, key(c)))
        c = pool[0]
        selected[key(c)] = c
        if key(c) in used:
            support_replays.append(c)
    cases = [selected[k] for k in sorted(selected)]
    assert len({c['series_id'] for c in cases}) == 1264
    if 'broad' in plan['phases']:
        assert plan['phases']['broad'] == cases
    else:
        plan['phases']['broad'] = cases
        api.original._atomic_json(C/'plan.json', plan)
    receipt = {'case_count': len(cases), 'series': 1264,
               'families': len({(metadata[c['series_id']]['panel_branch'], metadata[c['series_id']]['asset_class'], metadata[c['series_id']]['family']) for c in cases}),
               'branches': dict(Counter(metadata[c['series_id']]['panel_branch'] for c in cases)),
               'selection': 'First,middle,last remaining rolling origins plus remaining temporal splits for every series; deterministic asset-quarter support where required by the fixed main score.',
               'selection_uses_no_forecast_scores': True,
               'prior_pilot_case_replays_for_required_asset_quarter_support': support_replays,
               'only_run_if_both_predeclared_pilot_phases_improve_main_score': True,
               'independent_validation_claim': False,
               'case_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
    api.original._atomic_json(C/'broad_case_selection_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
