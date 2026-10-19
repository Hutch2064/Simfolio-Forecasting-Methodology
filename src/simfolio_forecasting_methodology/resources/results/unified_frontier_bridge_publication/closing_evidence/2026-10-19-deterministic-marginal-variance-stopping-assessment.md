# Deterministic marginal-variance correction: remaining-evidence assessment

Date: 2026-10-19. Bounded review of frozen source and saved diagnostics only. No forecast, model import or initialization, fit, build, numeric experiment, network call, or source implementation was performed.

## Decision

No additional deterministic conventional-variance/tail-shape law is justified by the current saved evidence. The main analytic correction paths are already tested, and the stochastic scale laws that most directly expose higher moments fail the selected-fast runtime gate. A deterministic univariate marginalization of the IID lognormal scale is mathematically definable, but it would not preserve the joint portfolio law, has no isolated tail-calibration evidence in the saved diagnostics, and has no demonstrated runtime advantage. Do not launch another score panel or introduce a Cornish–Fisher, sinh, strength, or horizon grid.

## What is already covered

- **Continuous node variance:** the selected variance-only node map uses `OLD.continuous_moments(nodes)` and `(nodes-mean)/sqrt(variance)+mean`, normalizing the continuous piecewise-linear uniform map's pre-knot-clip variance while retaining its continuous mean. This is already the live selected RMS basis; original-node controls and node interactions are also saved. The normalization does not guarantee emitted variance after knot/CDF endpoint/final-return clips.
- **First moment:** `conditional_candidates/marginal.py` computes the clipped-knot continuous bias `b` and `b*expm1(-v/8)`. The total conditional mean full panel scored +2.6934% versus M261 on 52 and −2.1305% on 80; it changes location, not higher-moment shape. This is already the deterministic counterpart to the omitted-scale first moment.
- **Dependence and covariance:** conventional-only `covariance_only`, `total_covariance_only`, and full-mean/covariance cells are in the saved factorial. Their latent copula adjustments are not a univariate tail calibration and do not integrate mapped-return covariance exactly.
- **Higher moments:** the full IID lognormal scale on corrected and original nodes tests the random scale-mixture shape but changes cross-asset dependence as well. In the original-node full result, aggregate 52 improves M261 by 2.1505% but wins only 22/52 portfolios and remains 0.8219% behind selected RMS; 80 improves M261 by only 0.0840% and wins 32/80. Its 80 central-90% width ratio to M261 is 0.9423 at 8,766 days (0.9866 relative to selected), but that width alone does not identify tail miscalibration. The candidate gains 3.752% in the 2,521–5,040-day band and loses 1.240% in the 5,041–8,766-day band. Those mixed patterns do not identify a stable tail-shape target or causal marginal-scale benefit.

The conventional-only IID original-node variant has a six-case pilot but no full-panel score. Its independent 18-row actual timing audit explicitly fails strict FAST: candidate/selected time ratios are 1.5746 fresh and 1.8507 cached on SPY 8,766, and 1.0506 fresh and 1.3868 cached on the multi-asset annual-126 context. The full all-volatility original-node IID branch is slower still in its saved quick-speed audit (2.69×/3.28× selected on SPY fresh/cached and 2.09×/2.00× on the multi-asset context). The vc-only source has no rough teacher or future SV-state draws, so the cost cannot be dismissed as merely rough preparation.

## Exact marginalization is not a supported next candidate

For `S=exp(0.5*sqrt(v_c)*Z-v_c/4)`, the existing stochastic law has `E[S]=d=exp(-v_c/8)` and `E[S^2]=1`. If the mapped return is `g=m+X`, with continuous map bias `b=E[X]` and variance `sigma^2`, then before final clipping

```
E[m + X*S]   = m + d*b
Var[m + X*S] = sigma^2 + (1-d^2)*b^2.
```

Thus the stochastic scale preserves the mapped raw second moment around `m`, not both the mapped mean and central variance when `b!=0`. The already-tested mean shift recovers the first-moment attenuation for a law omitting `S`; stacking it with `S` double-adjusts that bias. A recentered or re-standardized mixture could force selected mean and variance, but it would define a different, non-identified transformation. The existing `[-1,1]` mapper and final clip add boundary atoms, so ideal preclip moments do not establish postclip moment or tail parity.

Integrating out `Z` in one asset's conditional mixture and remapping its copula uniform would preserve that marginal mixture only. It would remove the independent group-scale contribution to cross-asset dependence that the stochastic multiplier creates, so it cannot be called an exact deterministic replacement of the joint portfolio law. The full IID score receipts do not separate this marginal effect from the dependence effect. Computing a mixture CDF/inverse at every fitted variance step could also cost more than generating one Gaussian per path/day/history group; no source-level cost proof or paired speed evidence exists. This is a possible mathematical construction, not a causally supported or speed-qualified next test.

## Stop boundary

No untested interaction with both a saved directional signal and a plausible selected-fast path remains in this branch. Keep the already completed mean, continuous-node variance, latent covariance, original-node, and IID evidence as distinct laws. Any return to deterministic tail-shape research requires a new saved diagnostic that isolates a mapped marginal tail error from dependence and clipping, plus a concrete exact-marginal method that preserves the declared point-mean/variance contract and an actual speed gate before any complete panel. Until then, no score panel is justified.

## Evidence pins

- Selected continuous node normalization: `../innovation_mean_variance/models.py`, SHA-256 `3de91419f60a9fd3e6d5c218303ac5000e18b28eaf3aab7d9693ec5b5a347254`.
- First-moment compensation: `conditional_candidates/marginal.py`, SHA-256 `9acd310070c57efa0ef16df425ed93cca105a116ba3867c06fb4f5bf440f4b8b`; full total-mean score receipt `total_mean_candidates/full-1aff29ce9e6500cb/completion.json`, SHA-256 `7491403a77868607248b3cdcba39f928e3e5ca3b1dabaef9079cbf9ef68ae94b`.
- IID lognormal law and clipping/moment diagnostics: `iid_conditional_scale_candidates/scale.py`, SHA-256 `5f5c1f761f52af6f3eb3b891f13f50b542deae74883ebfbb4be438fc21db7c98`.
- Full original-node all-volatility panel: `iid_original_nodes/full-52d4c26b608ac1ac/completion.json`, SHA-256 `6448ad508f8aa29144b833752d053d67f2b78dc05321d740e480934a4cf8f2bd`; saved width/growth audit `diagnostics/iid_original_nodes_full_actual_audit/review.json`, SHA-256 `601dcd72bfd541533ad428e65d92cdae37de72e6aee30a0602aa920da3120347`.
- Conventional-only IID law: `vc_only_iid_original_nodes/models.py`, SHA-256 `d56f3c9644f046c684934548b07df62b2086c0af2f380c405f9dbe9b73f69cd2`; actual speed audit `diagnostics/vc_only_iid_quick_speed_actual_audit/review.json`, SHA-256 `24ec8970d930f56621aef0899e9e011177f05da8e353e7c2a0a42b5fe3ea9570`.
- All-volatility original-node speed audit: `diagnostics/iid_original_nodes_quick_speed_actual_audit/review.json`, SHA-256 `1cb95278c16da038935e0e377035a74ddc8b65755b6500adc1ccf58d6fbc6af7`.
