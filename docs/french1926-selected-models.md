# Models selected for the French panel

M190, M192 and M177 have been removed from the planned run list. **28 candidates + two required benchmarks = 30 methods.** No models have been wired or run on the French panel.

The table reports saved **52-asset** and **historical 80-portfolio** comparisons. It contains no scores from the new French panel. Production means M261. Percentages use `(baseline - model) / baseline`; lower CRPS is better.

| Model | 52 vs production | 80 vs production | 80 vs website Default |
| --- | ---: | ---: | ---: |
| FAST | 2.95% better | 3.46% worse | 19.89% better |
| IID | 2.15% better | 0.08% better | 22.64% better |
| M261 — Current production Frontier | same | same | 22.57% better |
| M247 — stationary AR1 rough model with leverage proxy | 0.23% worse | 2.31% worse | 20.78% better |
| M254 — stationary AR1 rough model with Student LHS copula | 0.29% worse | 1.53% worse | 21.39% better |
| M255 — stationary AR1 with corrected DCC innovations | 0.29% worse | 0.15% worse | 22.46% better |
| M246 — stationary AR1 with matched offset | 0.43% worse | 2.76% worse | 20.43% better |
| M240 — adaptive predictive loading with full-Hurst multiscale offset | 0.65% worse | 7.05% worse | 17.12% better |
| M176 — moment-mixture SV MCMC | 0.67% worse | 11.36% worse | 13.78% better |
| M233 — single learned untruncated multiscale rate | 0.85% worse | 7.09% worse | 17.09% better |
| M202 — Gamma supOU Whittle MAP | 0.90% worse | 10.65% worse | 14.32% better |
| M242 — adaptive stable-coupled multiscale rough model | 0.92% worse | 6.44% worse | 17.59% better |
| M237 — adaptive predictive loading, full-Hurst untruncated | 0.96% worse | 6.60% worse | 17.46% better |
| M238 — single predictive loading, full-Hurst untruncated | 0.99% worse | 6.69% worse | 17.39% better |
| M244 — stationary AR1 with OU residual volatility | 1.03% worse | 3.20% worse | 20.10% better |
| M195 — multiscale empirical-noise rough MAP | 1.17% worse | 10.68% worse | 14.30% better |
| M198 — untrimmed conditional-residual empirical rough MAP | 1.19% worse | 10.22% worse | 14.66% better |
| M205 — conditional-empirical Whittle MLE | 1.25% worse | 10.26% worse | 14.63% better |
| M193 — predecessor dynamic-rough MAP | 1.33% worse | 11.10% worse | 13.98% better |
| M208 — full-Hurst-domain Whittle rough MAP | 1.36% worse | 10.52% worse | 14.43% better |
| M203 — conditional-empirical REML rough MAP | 1.38% worse | 10.47% worse | 14.46% better |
| M194 — causal-residual dynamic-rough MAP | 1.56% worse | 11.51% worse | 13.66% better |
| M182 — three-state HMM with full R-vine SV | 1.75% worse | — | — |
| M183 — rough-jump HMM with R-vine SV | 2.49% worse | — | — |
| M220 — Student-return rough model with Laplace-implied noise | 2.53% worse | 7.72% worse | 16.59% better |
| M222 — joint Student copula and pathwise leverage rough model | 2.59% worse | 11.92% worse | 13.35% better |
| M201 — differenced conditional-empirical Whittle rough MAP | 2.71% worse | 10.14% worse | 14.72% better |
| M001 — FastMAP Kalman with dynamic Gaussian factors | 4.59% worse | 12.79% worse | 12.67% better |
| Website Default | — | 29.15% worse | same |
| Competing Public Model | — | — | — |

The website Default has no standalone completed52 score attached. HMM models M182/M183 have no matched80 score attached. The Competing Public row is retained as a mandatory benchmark; the current paper reports approximately 0.392 on80, but an exact matched score is not attached here, so it is not substituted with a different older bootstrap audit. Historical catalogue figures remain historical evidence, distinct from new matched validation.

Full precision CRPS and numeric percentage columns are in the [scorecard CSV](../src/simfolio_forecasting_methodology/resources/french1926/selected_model_scorecard.csv) and [JSON](../src/simfolio_forecasting_methodology/resources/french1926/selected_model_scorecard.json). No general runtime-percentile exclusion has been applied.
