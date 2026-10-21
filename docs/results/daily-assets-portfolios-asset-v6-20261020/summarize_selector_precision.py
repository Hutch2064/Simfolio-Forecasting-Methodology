"""Publish-safe summaries of verified precision evidence, not raw histories."""
from pathlib import Path
import hashlib, json, statistics
import verify_selector_inner_precision_batched72 as audit
from score_percentage import percent_better

O = Path(__file__).resolve().parent
D = audit.D
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    verified = read(D / 'verification_receipt.json')
    summary = read(D / 'summary.json'); plan = read(D / 'plan.json')
    assert verified['verified'] and verified['summary_sha256'] == sha(D / 'summary.json')
    assert verified['verifier_sha256'] == sha(O / 'verify_selector_inner_precision_batched72.py')
    records = summary['records']; configs = plan['configs']
    lookup = {(r['case'], r['config_index']): r for r in records}
    groups = []
    for seed_summary in verified['seed_variation']:
        criterion, n = seed_summary['criterion'], seed_summary['inner_paths']
        selected = [r for r in records if r['configuration']['criterion'] == criterion and r['inner_paths'] == n]
        costs = []
        for r in selected:
            j = next(j for j, cfg in enumerate(configs) if cfg == {**r['configuration'], 'criterion': 'terminal'})
            costs.append(audit.standalone_cost(r, lookup[r['case'], j]))
        groups.append({**seed_summary,
                       'mean_percent_better_vs_fixed_reference': percent_better(seed_summary['mean_main_Q'], verified['main_reference_Q']),
                       'repeated_execution_count': len(costs),
                       'attributed_cpu_seconds_median': statistics.median(costs),
                       'attributed_cpu_seconds_mean': statistics.mean(costs),
                       'attributed_cpu_seconds_maximum': max(costs)})
    compact = {k: v for k, v in summary.items() if k != 'records'}
    compact.update({'groups': groups, 'complete_private_summary_sha256': sha(D / 'summary.json'),
                    'independent_verification_sha256': sha(D / 'verification_receipt.json'),
                    'summarizer_sha256': sha(Path(__file__)),
                    'complete_local_evidence_retained_for_active_followup': True,
                    'private_raw_evidence_remote_recovery_not_claimed_by_this_publication': True})
    (O / 'selector_inner_precision_batched_results.json').write_text(json.dumps(compact, indent=2) + '\n')
    (O / 'selector_inner_precision_batched_verification.json').write_text(json.dumps({k: v for k, v in verified.items() if k != 'verified_checkpoint_identities'}, indent=2) + '\n')
    lines = ['# Controlled internal-precision study', '',
             'All 1,728 candidate forecasts and 216 reference forecasts were independently checked from saved losses. There were no native forecast, scoring-integrity or infrastructure errors. All final forecasts use 1,000 paths. The 72 representative requests are adaptively reused development evidence, not the full panel or independent validation.', '',
             'The table shows the path-scoring criterion. Each score is the fixed canonical main score, averaged descriptively across the three predeclared Monte Carlo seed executions. Lower Q is better; every improvement references the same with-replacement baseline on identical coverage.', '',
             '| Internal paths | Mean Q | Better than baseline | Seed SD of Q | Median attributed CPU seconds | Maximum CPU seconds |',
             '| --- | --- | --- | --- | --- | --- |']
    for g in groups:
        if g['criterion'] == 'path':
            lines.append(f"| {g['inner_paths']} | {g['mean_main_Q']:.9f} | {g['mean_percent_better_vs_fixed_reference']:.5f}% | {g['sample_standard_deviation_of_main_Q']:.6f} | {g['attributed_cpu_seconds_median']:.3f} | {g['attributed_cpu_seconds_maximum']:.3f} |")
    high = verified['preserved1000inner_reference_Q_by_criterion']['path']
    lines += ['', f'The preserved 1,000-internal-path path-scoring reference has Q={high:.9f}, {percent_better(high, verified["main_reference_Q"]):.5f}% better than the baseline. It has only one internal seed replication and an older implementation, so this is neither a repeated-seed stability certificate nor an isolated timing comparison.', '',
              'Thirty internal paths had the lowest mean path-scoring Q among this initial grid. Use 30 for the next matched per-fold training-SD normalization screen, with replication zero fixed in advance rather than selecting the best seed. This is an exploratory research setting, not a production cap, a full-panel promotion, or proof that further precision cannot help.', '',
              'Runtime attribution charges the shared two-criterion preparation to each standalone output. It includes both objective optimizations, union refits, final simulation, scoring, checkpoint compression/hashing and warmup; it excludes blocked IO wait and record writing after measurement. Some requests exceed ten seconds, so the runtime target has not been met. Full runtime scope is recorded in the verification JSON.', '',
              'The next contrast changes only historical-fold training-SD normalization. Keep the bank, folds, seeds, final 1,000 paths, outer score, panel and with-replacement reference fixed. Neither diagnostics nor usage frequency add score weight. The goal remains active; no stopping condition is claimed.', '',
              'Implementations, compact results and verification are published in the research repository. Complete local records and loss checkpoints remain in use by the followup and have not been removed. This compact public publication is not a claim of remote restoration of private raw evidence.']
    (O / 'selector_inner_precision_batched_results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'verified': True, 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()
