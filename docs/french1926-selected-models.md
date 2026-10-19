# Approved French-panel model set

**15 methods replace the previous19: 13 candidates and two exact product benchmarks.** All fifteen adapters passed the bounded native equity probes; the full French551 sweep has not started.

| Method | Historical52 versus production | Historical80 versus production |
| --- | ---: | ---: |
| IID | 2.15% better | 0.08% better |
| M261 — Current production Frontier | same | same |
| FAST | 2.95% better | 3.46% worse |
| M001 — FastMAP Kalman with dynamic Gaussian factors | 4.59% worse | 12.79% worse |
| M035 — Constant-Mean Student-t | 34.97% worse | 48.30% worse |
| M052 — portfolio gjr garch eb sampler stationary sbb optimal | 30.43% worse | 67.54% worse |
| M176 — moment-mixture SV MCMC | 0.67% worse | 11.36% worse |
| M198 — untrimmed conditional-residual empirical rough MAP | 1.19% worse | 10.22% worse |
| M202 — Gamma supOU Whittle MAP | 0.90% worse | 10.65% worse |
| M233 — single learned untruncated multiscale rate | 0.85% worse | 7.09% worse |
| M244 — stationary AR1 with OU residual volatility | 1.03% worse | 3.20% worse |
| M247 — stationary AR1 rough model with leverage proxy | 0.23% worse | 2.31% worse |
| Website Default | unknown | 29.15% worse |
| Competing Public Model | unknown | unknown |
| Standalone fitted-decay EWMA | unknown | unknown |

The likelihood-fitted standalone EWMA has no old-panel score. Historical52 and historical80 results are separate from new French validation. The product benchmarks preserve their exact Default/CompetingPublic identities; no older approximate bootstrap substitutes are admitted.

See [method comparison and selection rationale](french1926-efficiency-reassessment.md). Only this15-method list is current; prior selection decisions survive as historical Git evidence.

The execution control is populated with all fifteen factories and starts paused. RF has been removed; the panel contains551 equity series. Native readiness is separate from completed-panel scoring. See [panel preparation and runtime](french1926-panel.md).
