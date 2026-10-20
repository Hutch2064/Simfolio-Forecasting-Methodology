"""Audit every closed pilot loss and retain the frozen additive horizon decomposition."""
from pathlib import Path
import json
import hashlib
import numpy as np
import run
api, sc = run.api, run.sc
OWN = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
BANDS = [('1_30_days', 1, 30), ('31_365_days', 31, 365),
         ('366_1826_days', 366, 1826), ('1827_7305_days', 1827, 7305), ('7306_plus_days', 7306, 10**9)]


def main():
    plan = read(OWN/'plan.json')
    out = {}
    for phase in ('development', 'reserved'):
        summary = read(OWN/(phase+'_summary.json'))
        cases = plan['phases'][phase]
        metadata = {c['series']['series_id']: c['series'] for c in cases}
        expected = {a: {(r['series_id'], r['origin_label']): r for r in summary['records'][a]} for a in plan['arms']}
        prior = read(OWN.parent/'mean_shrinkage'/(phase+'_summary.json'))
        prior_rows = {a: {(r['series_id'], r['origin_label']): r for r in prior['records']['Candidate' if a == 'Incumbent' else a]}
                      for a in ('Public', 'Incumbent')}
        fingerprint = read(OWN/'results'/phase/'execution.json')['fingerprint']
        records = {a: [] for a in plan['arms']}
        hashes = checked = 0
        for case in cases:
            row = case['series']
            sid = row['series_id']
            dates, returns, _ = api.load_series(plan['input_root'], row)
            for origin in case['origins']:
                n = origin['position']+1
                H = origin['max_horizon']
                future = dates[n:n+H]
                elapsed = (future-np.datetime64(origin['origin_date'], 'D')).astype(int)
                h = np.arange(1, H+1, dtype=float)
                sd = float(np.std(returns[:n], ddof=1))
                w = sc.calendar_log_weights(origin['origin_date'], future)
                scales = (w/(sd*np.sqrt(h+h*h/n)), w/(sd*np.sqrt(h)))
                for arm, mid in plan['arms'].items():
                    path = OWN/'results'/phase/arm/sid/(origin['origin_label']+'.npz')
                    loss, _ = api.original._read_checkpoint(path, api.task_identity(fingerprint, mid, sid, origin))
                    r = expected[arm][sid, origin['origin_label']]
                    assert hashlib.sha256(loss.tobytes()).hexdigest() == r['loss_sha256']
                    if arm in prior_rows:
                        assert r['loss_sha256'] == prior_rows[arm][sid, origin['origin_label']]['loss_sha256']
                        hashes += 1
                    vals = {}
                    for profile, scale in zip(('primary', 'sensitivity'), scales):
                        for name, lo, hi in BANDS:
                            mask = (elapsed >= lo) & (elapsed <= hi)
                            vals[profile+'_'+name] = float(np.dot(loss[mask], scale[mask]))
                        assert abs(sum(vals[profile+'_'+x[0]] for x in BANDS)-r[profile+'_score']) < 1e-11
                    records[arm].append({**r, **vals})
                    checked += 1
        bands = {}
        for profile in ('primary', 'sensitivity'):
            values = {arm: {name: sc.aggregate_records(rr, metadata, score_key=profile+'_'+name)['score']
                            for name, _, _ in BANDS} for arm, rr in records.items()}
            for arm in values:
                assert abs(sum(values[arm].values())-summary['arms'][arm][profile+'_score']) < 1e-11
            bands[profile] = [{'band': name, 'contributions': {arm: values[arm][name] for arm in values},
                'candidate_pct_better_vs_Public': 100*(1-values['Candidate'][name]/values['Public'][name]) if values['Public'][name] else None,
                'incumbent_pct_better_vs_Public': 100*(1-values['Incumbent'][name]/values['Public'][name]) if values['Public'][name] else None}
                for name, _, _ in BANDS]
        out[phase] = {'checked_native_checkpoint_and_loss_hashes': checked,
            'historical_baseline_loss_hashes_exactly_reproduced': hashes, 'bands': bands,
            'split_scores': {arm: {split: {profile: sc.aggregate_records(
                [r for r in rr if r['evaluation_split'] == split], metadata, score_key=profile+'_score')['score']
                for profile in ('primary', 'sensitivity')} for split in sorted({r['evaluation_split'] for r in rr})}
                for arm, rr in records.items()}}
    out['scope'] = 'Original positive calendar log weights retained. Additive band contributions reconcile both aggregate profiles. Adaptive pilot evidence, not independent validation.'
    run.write(OWN/'diagnostics.json', out)
    print(json.dumps({phase: {k: v for k, v in out[phase].items() if k not in ('bands', 'split_scores')}
                      for phase in ('development', 'reserved')}))


if __name__ == '__main__':
    main()
