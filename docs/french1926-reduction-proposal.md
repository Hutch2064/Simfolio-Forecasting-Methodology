# Proposed reduction from 29 to 20 methods

**Proposal only. The current 29-method list is unchanged.** M190, M192, M177 and M222 were the explicit removals already applied.

Preserve both benchmarks, FAST, IID, current production, inexpensive controls, and distinct volatility, inference, mean and dependence families. Flag redundant variants where a retained peer has at least as good saved scores on both panels and less recorded work; interpret small differences cautiously. This is a curated research selection, not a combined-panel ranking or proof of statistical superiority.

| Current method | Proposal |
| --- | --- |
| FAST | Keep |
| IID | Keep |
| M261 — Current production Frontier | Keep |
| M247 — stationary AR1 rough model with leverage proxy | Keep |
| M254 — stationary AR1 rough model with Student LHS copula | Keep |
| M255 — stationary AR1 with corrected DCC innovations | Keep |
| M246 — stationary AR1 with matched offset | Omit |
| M240 — adaptive predictive loading with full-Hurst multiscale offset | Keep |
| M176 — moment-mixture SV MCMC | Keep |
| M233 — single learned untruncated multiscale rate | Keep |
| M202 — Gamma supOU Whittle MAP | Keep |
| M242 — adaptive stable-coupled multiscale rough model | Keep |
| M237 — adaptive predictive loading, full-Hurst untruncated | Omit |
| M238 — single predictive loading, full-Hurst untruncated | Omit |
| M244 — stationary AR1 with OU residual volatility | Keep |
| M195 — multiscale empirical-noise rough MAP | Omit |
| M198 — untrimmed conditional-residual empirical rough MAP | Keep |
| M205 — conditional-empirical Whittle MLE | Omit |
| M193 — predecessor dynamic-rough MAP | Omit |
| M208 — full-Hurst-domain Whittle rough MAP | Omit |
| M203 — conditional-empirical REML rough MAP | Omit |
| M194 — causal-residual dynamic-rough MAP | Omit |
| M182 — three-state HMM with full R-vine SV | Keep |
| M183 — rough-jump HMM with R-vine SV | Keep |
| M220 — Student-return rough model with Laplace-implied noise | Keep |
| M201 — differenced conditional-empirical Whittle rough MAP | Keep |
| M001 — FastMAP Kalman with dynamic Gaussian factors | Keep |
| Website Default | Keep |
| Competing Public Model | Keep |

Proposed omissions: M193, M194, M195, M203, M205, M208, M237, M238 and M246. No further exclusions have been applied.

M183 remains because the jump/regime structure is distinct and the historical80 comparison is missing. M233 retains a simpler single-rate mean model; M201 preserves a different rough-inference trade-off.

The proposed list contains **18 candidates plus two mandatory benchmarks**. Estimated completion is **8–12 hours with 12 workers**, extrapolated from historical timing records, not measured on the new French panel. Full score/work comparisons, peer identities and ETA assumptions are in `resources/french1926/candidate_reduction_proposal.json`.
