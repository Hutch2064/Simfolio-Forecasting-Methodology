# Clip-order and remaining-branch audit

**Scope:** bounded review of frozen source and saved diagnostics only. No forecast, fit, build, benchmark, or candidate run was performed for this note. This is not an admission or recommendation.

## Finding on a pre-finish IID contrast

The all-volatility IID implementation applies its scale to the output of `finish(mapped, mean, None)`, then clips the scaled return. `finish` recenters with multiplier 1 and clips to `[-1, 1]`. The frozen mapper already clips both interpolation knots to that interval; linear interpolation therefore returns a bounded value. On this path the `finish` clip is mathematically idempotent, apart from floating-point edge behavior. Moving the IID scale across only that clip is not a meaningful new contrast. Moving it before knot clipping would change the mapping law and is not supported by saved evidence.

The saved IID clip diagnostics cannot establish a pre-finish clipping mechanism: per-task mapped-boundary counts are taken after `finish`, while full-audit counts describe final post-scale clipping. The audit has no raw pre-finish exceedance count. Across the all-volatility IID panel, the full audit records 1,668 final clips out of 937,010,880 path-days for 52 and 18,145 out of 8,649,331,200 for 80; these counts describe the implemented post-scale clip, not a counterfactual order effect. The six-asset 8766-day spot check has zero finished-base boundary observations and 265 final clips for one asset. It is illustrative, not panelwide evidence.

The rough-only IID implementation fuses its scale before one final clip, but it also changes which volatility component is integrated versus simulated. Its results cannot isolate clip order. Thus no separate pre-finish-scale branch is justified by the available clip evidence.

## Remaining specific contrasts

Two fixed methodological branches remain evidence-backed and are already in the parent’s queue:

1. **Original empirical nodes × all-volatility IID scale.** This is a real untested interaction with the node law, not a clip-order cleanup. Its six-case correctness pilot passed. Full canonical panels and the controlled speed gate remain necessary; do not infer admission from the pilot.
2. **Scalar Markov rough factor × actual conventional future states × corrected nodes.** This tests whether preserving the fitted rough marginal variance and adjacent covariance adds value after IID results, while avoiding a full rough future-factor simulation. The source is being prepared; its lag-one match would not restore longer-lag rough memory. It is conditional on residual evidence and a credible speed gate, not a parameter or horizon grid.

Other obvious residuals are already covered or unsupported: the total-mean / total-covariance and IID branches have been tested; the conventional future factor already has a conditional unit-second-moment correction, so a symmetric extra normalizer has no demonstrated residual to target; growth/mean-reversion alternatives were tested. The latest rough-IID × conventional-ON both-panel result (52: 1.39524% better than M261 but 1.60011% worse than selected fast; 80: 0.719494% worse than M261, with a 72.1% deficit over 5041–8766) is evidence against treating marginal scale alone as sufficient. It does not, by itself, validate another unmeasured mechanism.

## Stopping boundary

A first joint score winner is not sufficient by itself. The remaining specific branches should be judged against both canonical panels, matched speed evidence (at least M261 speed under the current user constraint), and independent/seed robustness. A six-case correctness pilot, a quick timing gate, a single fresh timing context, or a nominal two-panel score is not full robustness. If both queued branches fail their score/speed gates, continue only if a saved diagnostic identifies a new residual that maps to one fixed, no-tuning contrast. Otherwise stop this local methodology search and report that no tested candidate met the joint objective; do not create another variant merely to fill an unscored cell. If new controlled evidence identifies a residual, record that evidence and reopen only the corresponding contrast.

## Evidence pins

- `iid_conditional_scale_candidates/models.py` — SHA-256 `fecfceae74414f1dbebe4c7a6625e8d9b275bd4279bf812ed13a282da6103800`.
- `conditional_candidates/marginal.py` — SHA-256 `9acd310070c57efa0ef16df425ed93cca105a116ba3867c06fb4f5bf440f4b8b`.
- `tools/mcmc_runtime/streamed_paths.py` — SHA-256 `8d32098b16e318e1379f82df06ff1814d536edaa8891afd0210fc09a9512d1fa`.
- `iid_conditional_scale_candidates/scale.py` — SHA-256 `5f5c1f761f52af6f3eb3b891f13f50b542deae74883ebfbb4be438fc21db7c98`.
- `iid_conditional_scale_candidates/full_audit.py` — SHA-256 `e59efe7b30147f31e3b59aa60b95726f9771960811d90a9bc9ee5d39dd95e099`.
- Saved all-vol IID diagnostics `full-8544ad97dc7caf39/IID-scale-diagnostics.json` — SHA-256 `7ca8d181d576e43d3811abadbb76016a53f2176329deccea56ca920f5e39804c`.
- Rough-IID both-panel decision `decisions/rough-IID-both-panels-final.json` — SHA-256 `f2f120c86129f90aa9c9870b8215dd7b7bd9c382dc4da04573f192a2adc209e5`; independent audit `diagnostics/rough_iid_conventional_full_actual_audit/review.json` — SHA-256 `03ce9b566ec71021dc40f1236e338060238347211a23fd5bc3855985f9a809b9`.
- Original-node IID correctness pilot `iid_original_nodes/pilot-3b4bebc896c44b63/completion.json` — SHA-256 `4274277724ca3be5771a967329628759571af333225856595623b9dd75134f38`.
