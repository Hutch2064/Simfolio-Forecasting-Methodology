# Efficiency and method-coverage reassessment

**Approved:15 methods (14 reassessed entries plus one inexpensive EWMA comparison), replacing the prior19-method list.** The fifteen adapters are prepared and passed bounded native equity probes; no full new-panel sweep has started.

The review covers all 52 original candidates (49 sweep methods plus FAST, IID and M261), the two exact required benchmarks, and three additional family-coverage proposals. Historical 52 scores drive the efficiency calculation; historical 80 scores stay separate. The French panel remains unscored.

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
| M176 — moment-mixture SV MCMC | 0.67% worse | 10.30 | Retains parameter posterior uncertainty with moment-based stochastic volatility at much less recorded work than the old portfolio MCMC variants. |
| M198 — untrimmed conditional-residual empirical rough MAP | 1.19% worse | 6.59 | Retains empirical conditional-residual rough inference, distinct from the Gamma-mixed covariance method. |
| M202 — Gamma supOU Whittle MAP | 0.90% worse | 5.24 | Highest positive-gain efficiency against fixed M001 among the original timed methods; distinct Gamma-mixed covariance. |
| M233 — single learned untruncated multiscale rate | 0.85% worse | 11.84 | One learned volatility-memory rate supplies a compact structure instead of multiple nearby adaptive-memory variants. |
| M244 — stationary AR1 with OU residual volatility | 1.03% worse | 6.62 | Exact OU residual volatility is a simpler alternative to fractional rough residual volatility. |
| M247 — stationary AR1 rough model with leverage proxy | 0.23% worse | 10.85 | Lagged return/volatility leverage changes within-series path dynamics and is not merely cross-asset dependence. |
| Standalone fitted-decay EWMA | untested design | unknown | Adds standalone recency-weighted variance; reconcile prior EWMA-QMLE implementation. |

The two restored methods are M035 (Student-t) and M052 (GJR-GARCH with stationary bootstrap). They are inexpensive structural controls, not claims of strong old-panel accuracy. In this proposal only, their role justifies revisiting the prior score filters. No restoration has been applied to the approved list.

Relative to the approved19, the proposal omits M182, M201, M220, M240, M242, M254 and M255. Dependence variants are not treated as distinct marginal mechanisms in this one-input panel. M254 also changes finite-ensemble Latin-hypercube sampling: the omission does not assert exact output or empirical-CRPS equivalence. Nearby inference/memory variants are represented by simpler or more efficient alternatives. M176 preserves posterior marginal uncertainty without the extra regime/vine comparison.

## Efficiency calculation

For positive gains only: `100 * (1 - CRPS_model / CRPS_M001) / worker_minutes`. M001 is a fixed historical reference with CRPS 0.6420341742861375. The ratio is not ranked for nonpositive gains, because dividing a negative gain by a larger cost can reward slow weak models. Unknown costs remain unknown, never zero. Raw accuracy and score/work dominance are retained alongside the ratio; family coverage is a separate explicit selection judgment.

The old M001 score is used only for this historical shortlist screen. A new French campaign must obtain its own matched model scores and timings before interpreting efficiency there. Cheap low-skill controls are kept to test specific mechanisms, not ranked as efficient by a negative ratio.

## Compute estimate

The nine timed historical methods in the proposal have 56.24% less recorded worker compute than the fourteen timed methods in the approved19. Five methods in both sets lack comparable timing fields. This is a comparison of known recorded work, not an all-method measured speedup.

For the historical14 subset only, with the same rough linear scaling by series count and history length, mean- and median-imputed longest-jobs-first estimates both give about 4.72 elapsed hours with twelve workers. That subset has a 5–8 hour planning range. The total15 runtime is not estimated until the retained EWMA implementation is matched and profiled. The estimate includes five imputed costs, does not measure the new panel, and remains sensitive to nonlinear fits and contention. No model order or execution queue has been launched.

## Complete comparison and reproduction

[Full original-shortlist comparison CSV](../src/simfolio_forecasting_methodology/resources/french1926/efficiency_reassessment.csv) includes all 57 rows, raw scores, historical80 evidence, costs, positive-gain efficiency, frontier membership and proposal decisions. [JSON](../src/simfolio_forecasting_methodology/resources/french1926/efficiency_reassessment.json) retains hashes, source scope, protected approved-list hashes and assumptions.

Recompute and verify without changing the active selection:

```sh
python scripts/reassess_french1926_candidates.py --check
```

Long-horizon mean/median CAGR remains a required diagnostic in the panel harness and is not converted into an exclusion rule. No French model validation or production readiness is implied by this shortlist.

M083 was explicitly removed from the proposal because its prequential CRPS calibration depends on forecast scores. Per-origin fitting must use only history available at the origin and must not depend on out-of-sample scoring artifacts or score-based calibration, including prequential CRPS. Historical-score screening of this research shortlist is separate from model fitting. Native adapter validation remains pending before registration.

## Family-coverage review and prior tests

| Family considered | Mechanism | Existing catalogue evidence |
| --- | --- | --- |
| ARMA on returns / ARIMA on log wealth | Explicit serial return-mean dynamics, with training-likelihood order selection and coherent simulated paths | 35 historical master ARMA specifications; 28 completed canonical52 tests |
| Standalone fitted-decay EWMA | Direct recency-weighted variance, fitted by training likelihood instead of fixed decay or forecast-score calibration | Six completed52 EWMA variants; `bayesian_sbb_overlay_ewma_qmle` has historical80 test references |

These are considered families, not globally new families. Only EWMA is retained in the proposal; ARMA and regime switching are omitted after reviewing the previous scores and costs. Before implementation, reuse an existing method when the exact fit and simulation definitions match. A new identity requires a documented methodological difference; old scores cannot be attached to an altered model. The retained EWMA design has no exact matched completed52, historical80 or new-French score attached. The rejected ARMA and regime designs are preserved only as considered-and-omitted records. All exact native adapters, convergence safeguards and history-only boundaries remain validation gates before execution.

The ARIMA mechanism is autocorrelation in the observed series, described in [Forecasting: Principles and Practice](https://otexts.com/fpp3/arima.html). Existing latent-mean and volatility states do not make every ARMA specification equivalent. EWMA is the recursive variance family documented by [RiskMetrics](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a); [arch documents an estimated-decay option](https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.EWMAVariance.html). The planned fit uses training likelihood, not RiskMetrics' fixed decay. A scalar regime model changes the return distribution itself, motivated by [Hamilton and Susmel's regime-switching volatility study](https://www.sciencedirect.com/science/article/pii/0304407694900671); this proposal is not an exact reproduction of that paper's switching ARCH model.

A plain extra GARCH or Gaussian-only model is not added solely for name recognition: the proposal already contains conditional GJR volatility and a static parametric heavy-tail comparison. Existing continuous latent-mean adaptation gives adjacent coverage of state-space smoothing. Extra cross-asset copulas are not a distinct test of this single-series protocol, and realized-volatility/factor models require unavailable century-wide covariates. This is a reasoned family screen, not a claim that every possible forecast model has been exhausted.

Final family decision: omit ARMA (best historical52 score 330.60% worse than production, 11.95 worker minutes) and observable Markov/regime alternatives (35.29% worse, 10.83 worker minutes). Keep one standalone EWMA control: its family representative was 30.52% worse but used only 1.94 worker minutes, making the simple recency-volatility mechanism inexpensive to compare. These are family-screening figures, not scores assigned to the unvalidated likelihood-EWMA design.

## Superseding preparation result

The user removed RF, leaving551 unchanged equity series. All135 retained native equity probes passed; the earlier cash failures remain archived. The paused executable registry contains all fifteen methods. The native timing extrapolation supersedes the historical estimate:65.2 worker-hours and about42.5 elapsed hours with the current one-worker-per-model controller, dominated by M001. No full French scores exist. See the panel readiness and runtime resources.
