# Efficiency and method-coverage reassessment

**One proposal: 15 methods. The approved 19-method selection is unchanged.** No new-panel model adapters or forecasts have been started.

The review covers all 52 original candidates (49 sweep methods plus FAST, IID and M261) and the two exact required benchmarks. Historical 52 scores drive the efficiency calculation; historical 80 scores stay separate. The French panel remains unscored.

## Proposed methods

| Method | 52 versus production | Recorded worker minutes | Why retain |
| --- | ---: | ---: | --- |
| FAST | 2.95% better | unknown | Mandatory FAST/IID/production reference with unknown comparable timing. |
| IID | 2.15% better | unknown | Mandatory FAST/IID/production reference with unknown comparable timing. |
| M261 — Current production Frontier | same | unknown | Mandatory FAST/IID/production reference with unknown comparable timing. |
| Website Default | unknown | unknown | Mandatory exact product benchmark; identity and adapter validation remain separate from historical comparisons. |
| Competing Public Model | unknown | unknown | Mandatory exact product benchmark; identity and adapter validation remain separate from historical comparisons. |
| M001 — FastMAP Kalman with dynamic Gaussian factors | 4.59% worse | 4.07 | Fixed inexpensive historical MAP reference. |
| M035 — Constant-Mean Student-t | 34.97% worse | 0.70 | Cheap Student-t heavy-tail reference; intentionally a structural control despite weak old accuracy. |
| M052 — portfolio gjr garch eb sampler stationary sbb optimal | 30.43% worse | 1.64 | Cheap asymmetric GJR volatility with stationary historical-shock resampling. |
| M083 — Bayesian SBB + GARCH(1,1) prequential crps shrinkage | 18.71% worse | 1.84 | Cheap GARCH model with prequential CRPS-based shrinkage; strong low-cost point on the original score/work frontier. |
| M176 — moment-mixture SV MCMC | 0.67% worse | 10.30 | Retains parameter posterior uncertainty with moment-based stochastic volatility at much less recorded work than the old portfolio MCMC variants. |
| M198 — untrimmed conditional-residual empirical rough MAP | 1.19% worse | 6.59 | Retains empirical conditional-residual rough inference, distinct from the Gamma-mixed covariance method. |
| M202 — Gamma supOU Whittle MAP | 0.90% worse | 5.24 | Highest positive-gain efficiency against fixed M001 among the original timed methods; distinct Gamma-mixed covariance. |
| M233 — single learned untruncated multiscale rate | 0.85% worse | 11.84 | One learned volatility-memory rate supplies a compact structure instead of multiple nearby adaptive-memory variants. |
| M244 — stationary AR1 with OU residual volatility | 1.03% worse | 6.62 | Exact OU residual volatility is a simpler alternative to fractional rough residual volatility. |
| M247 — stationary AR1 rough model with leverage proxy | 0.23% worse | 10.85 | Lagged return/volatility leverage changes within-series path dynamics and is not merely cross-asset dependence. |

The three restored methods are M035 (Student-t), M052 (GJR-GARCH with stationary bootstrap), and M083 (prequential GARCH calibration). They are inexpensive structural controls, not claims of strong old-panel accuracy. In this proposal only, their role justifies revisiting the prior score filters. No restoration has been applied to the approved list.

Relative to the approved19, the proposal omits M182, M201, M220, M240, M242, M254 and M255. Dependence variants are not treated as distinct marginal mechanisms in this one-input panel. M254 also changes finite-ensemble Latin-hypercube sampling: the omission does not assert exact output or empirical-CRPS equivalence. Nearby inference/memory variants are represented by simpler or more efficient alternatives. M176 preserves posterior marginal uncertainty without the extra regime/vine comparison.

## Efficiency calculation

For positive gains only: `100 * (1 - CRPS_model / CRPS_M001) / worker_minutes`. M001 is a fixed historical reference with CRPS 0.6420341742861375. The ratio is not ranked for nonpositive gains, because dividing a negative gain by a larger cost can reward slow weak models. Unknown costs remain unknown, never zero. Raw accuracy and score/work dominance are retained alongside the ratio; family coverage is a separate explicit selection judgment.

The old M001 score is used only for this historical shortlist screen. A new French campaign must obtain its own matched model scores and timings before interpreting efficiency there. Cheap low-skill controls are kept to test specific mechanisms, not ranked as efficient by a negative ratio.

## Compute estimate

The proposed ten timed methods have 54.84% less recorded worker compute than the fourteen timed methods in the approved19. Five methods in both sets lack comparable timing fields. This is a comparison of known recorded work, not an all-method measured speedup.

With the same rough linear scaling by series count and history length, mean- and median-imputed longest-jobs-first estimates both give about 4.72 elapsed hours with twelve workers. Use 5–8 hours for planning. The estimate includes five imputed costs, does not measure the new panel, and remains sensitive to nonlinear fits and contention. No model order or execution queue has been launched.

## Complete comparison and reproduction

[Full original-shortlist comparison CSV](../src/simfolio_forecasting_methodology/resources/french1926/efficiency_reassessment.csv) includes all 54 rows, raw scores, historical80 evidence, costs, positive-gain efficiency, frontier membership and proposal decisions. [JSON](../src/simfolio_forecasting_methodology/resources/french1926/efficiency_reassessment.json) retains hashes, source scope, protected approved-list hashes and assumptions.

Recompute and verify without changing the active selection:

```sh
python scripts/reassess_french1926_candidates.py --check
```

Long-horizon mean/median CAGR remains a required diagnostic in the panel harness and is not converted into an exclusion rule. No French model validation or production readiness is implied by this shortlist.
