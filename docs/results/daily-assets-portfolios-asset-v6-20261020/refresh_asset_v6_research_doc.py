"""Render the public current-result note from the verified retained scoreboard."""
from pathlib import Path
import json
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'
read=lambda p:json.loads(p.read_text())

def main():
    official=read(V/'official_scoreboard.json')
    assert official['official_retained_set_only'] and official['same_successful_tasks_per_model']==64049
    names={'model_testing_asset_circular_raw_EWMA_fade_FHS_mean_v1':'Asset EWMA Fade / Equal Mix / Gaussian Mean Uncertainty',
           'model_testing_asset_circular_raw_GARCH_fade_FHS_mean_v1':'Asset GARCH Fade / Equal Mix / Gaussian Mean Uncertainty',
           'model_testing_asset_joint_blocks_replacement_v6':'Asset Blocks — With Replacement',
           'model_testing_asset_joint_blocks_history_partition_v6':'Asset Blocks — Without Replacement',
           'model_testing_asset_equal_joint_blocks_FHS_smooth_joint_mean_uncertainty_v1':'Asset Equal Mix / Smooth Gaussian Mean Uncertainty',
           'model_testing_asset_equal_circular_blocks_FHS_smooth_joint_mean_uncertainty_v1':'Asset Circular Equal Mix / Smooth Gaussian Mean Uncertainty'}
    reference=official['sole_reference_model_id'];best=official['current_best_model_id']
    order=[reference,*[m for m in official['models'] if m not in (reference,best)],best]
    lines=['# Current asset-level accuracy research','','The accuracy-maximization goal remains active. A verified winner is not the research stopping point.','',
           '## Official retained set: one main score','',
           'All retained methods below cover the identical64,049 successful forecasts across1,264streams at1,000simulations. Lower main Q is better. The fixed reporting reference is the worse completed asset-level TestFolio-style configuration, With Replacement.','',
           '| Method | Main Q | Better vs fixed reference | Recorded worker compute seconds |','|---|---:|---:|---:|']
    for mid in order:
        d=official['models'][mid]
        assert d['successful_tasks']==64049 and d['native_failures']==d['scoring_integrity_errors']==0
        delta='reference' if mid==reference else f"{d['percent_better_vs_fixed_reference']:.7f}%"
        lines.append(f"|{names.get(mid,mid)}|{d['main_score']:.16f}|{delta}|{d['worker_compute_seconds']:.4f}|")
    lines+=['','All retained methods have zero native failures and zero scoring integrity errors. The complete baseline pair and each promoted winner have independently verified source-bound checkpoint identities and canonical score reduction. Recorded worker compute includes input preparation, fitting, joint simulation and CRPS, including identity-verified reused pilot forecasts. It excludes IO/setup/queue and is not a controlled speedup comparison. These adaptively reused cases do not support independence or significance claims.','',
            '## Cohesive model and continuing research','',
            'The winner is a single fixed asset-level procedure. Its frozen implementation and full-panel catalogue specify the raw joint-row sampler, filtered variance law, mean treatment, joint shocks and portfolio accounting. Constituent paths and holdings evolve inside every scenario; no test-score-selected asset-class, family or horizon routing is used.','',
            'The circular raw-block contrast passed355-case development,355-case reserved and7,418-case broad checks at1,000paths before full admission. Official promotion occurs only after complete matched full results and independent verification. Low-simulation component and sampler screens remain exploratory and separate from official results and full-panel catalogue counts. The additive diagnostic reconciles to canonical Q and has zero additional score or model-selection weight.','',
            'Closed exploratory contrasts include an IID raw-component simplification, a nested sign-asymmetric variance filter, training-adaptive stationary blocks and empirical joint batch-mean uncertainty. None earned full admission. The current fading-volatility contrast conditions raw circular deviations using either the existing GARCH state/persistence or a training-fitted EWMA state/decay, returning toward historical variance without scaling the mean. Its development/reserved/broad evidence remains separate from official results; improvements on one horizon cannot qualify a candidate. All scientific forecast/qualification pools share12worker slots; no small win or a few rejected variants closes the goal.','',
            '## Preservation and recovery','']
    for name in ['asset-v6-baselines-1000-20261020','asset-v6-hybrid-1000-20261021','asset-v6-circular-1000-20261021','asset-v6-GJR-pilot-20261021','asset-v6-adaptive-pilot-20261021','asset-v6-empirical-mean-pilot-20261021','asset-v6-fading-1000-20261021']:
        p=O/'archives'/name/'receipt.json'
        if p.exists():
            d=read(p);assert d['complete']
            lines.append(f"- Private release {d['release_tag']}: {d['files_fresh_downloaded_disk_restored_sha256_verified']:,}files fresh-downloaded, physically disk-restored and SHA256verified.")
    lines+=['','Complete raw losses, implementations, fitting/mean kernels, controls/registry, seeds, admission and verification evidence are preserved privately. Public compact receipts provide all archive/manifest digests and dependency releases. Restore baseline shared/input archives and sealed numerical parents, then the hybrid parent source and selected candidate archives at the research owner root; verify every manifest path, size and SHA256 before use. The prior hybrid raw checkpoints were intentionally retired only after complete fresh remote recovery and file-level SHA256 verification; the separate retirement receipt and ARCHIVED_LOSSES marker document this. The baseline inputs and checkpoints and the current circular winner remain local active research dependencies. Their archived prior versions must not be treated as missing data or silently recomputed.','',
            'The catalogue retains all fully attempted implementation versions, including losers; identical reruns count once. Source and compact evidence are pushed to the research repository and separately fresh-download/disk-restore verified. No production implementation or deployment is changed.','']
    (V/'CURRENT_RESEARCH.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
