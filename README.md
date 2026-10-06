# Simfolio Forecasting Methodology

This repository is a standalone, public-safe research package for the
out-of-sample comparison of probabilistic portfolio forecasts. It contains
the canonical catalogue and protocol contracts, source-derived numerical
implementations that have passed bounded parity checks, and an offline frozen
data snapshot. It has no dependency on a production website, customer system,
private datastore, or operational API.

## Current catalogue and score ranking

The canonical catalogue contains **233 models**: the original 175 plus forty-eight
asset-level models with validated full-panel scores, four partially scored
candidates, and six unscored asset-level rough-volatility candidates. The
dynamic-resolution candidate completed the full panel with CRPS
**0.25074679212444156**. The debiased Whittle rough-inference candidate (M200) scored **0.249657155081818**, a validated catalogue score. Six canonical rough-volatility
alternatives remain unscored. Their [implementations, results and diagnostics](docs/rough-bayesian-oos.md) are saved. The current production
Frontier and both previous Frontiers are included. The original model IDs,
historical ranks and score tokens are preserved; additions have canonical
indices 176–233 and no historical source rank.

`simfolio-oos candidate-scores --json` returns all 233 models once, sorted by
exact empirical CRPS, with unscored full-panel rows last and unranked.
[Partial candidate results](docs/rough-jump-vine-oos.md) retain completed
portfolio scores and blank unfinished portfolios. The [complete ranked JSON](docs/results/combined-176-score-ranking.json)
and [model reference](docs/canonical-model-reference.md) are also available.
Existing command names and catalogue filenames remain compatible.

| Model | Exact empirical CRPS | Role |
| --- | ---: | --- |
| Asset MAP Multiscale + Dynamic Rough Volatility | 0.2518812750312293 | Validated fast research candidate |
| Asset Rough Volterra SV — Eight Factors | 0.2511998559613309 | Validated research candidate |
| Asset Parameter MCMC + Filtered Moment SV | 0.25246784959071183 | Current production Frontier |
| Asset Filtered Innovation Moment SV + Fixed Mean | 0.25439860867855635 | Previous production Frontier |
| Asset Filtered Innovation Moment SV + DLM | 0.2547972662964723 | Validated candidate |
| Asset FastMAP + Dynamic Gaussian Factor | 0.2557255171048505 | Earlier production Frontier; retained historical score |
| Asset Gaussian Moment SV + Fixed Mean | 0.2588375710788695 | Validated candidate |

Each fully scored addition was evaluated on 4,080 origin tasks, 701,280 cells and 240
simulations per origin using the whitepaper's exact empirical CRPS and equal
portfolio-horizon cell weighting. Complete statistical specifications, source
hashes, run manifests and validation references are recorded per model.

The public package has **176 explicit executable factories**. Other entries
retain their standalone research entrypoints and source details.
The [parameter-MCMC report](docs/mcmc-runtime-optimization.md) includes the
current Frontier's complete-output replay and optimized numerical sources.
The [alternative SV inference experiments](docs/sv-inference-candidates.md)
retain their screening results separately.

The packaged canonical data snapshot contains 60 manifest-whitelisted files:
52 asset series, `EFFRX`, five additional canonical drift-proxy series,
and the French daily and Q5 factor inputs. Verification is offline and hash based. The frozen common
window is 1979-12-31 through 2026-05-13 with 11,687 dates. The protocol
constructs 80 portfolios, 4,080 origin tasks, and 701,280 scored cells per
model, with 240 simulations per origin.

## Install and inspect

The reference numerical environment is macOS ARM64 with Python 3.11 or 3.12
and `requirements-lock.txt`. Cross-platform byte equality is not established.
The optional `all-models` extra installs the numerical dependencies:

```bash
python -m pip install '.[all-models]'
simfolio-oos data verify --json
simfolio-oos data prepare --destination .simfolio-oos-data --json
simfolio-oos coverage --json
simfolio-oos canonical-175 --plan --json
```

The data commands use the packaged snapshot by default. Preparation writes a
caller-local execution cache after rechecking every source hash and derived
fingerprint; it never downloads or refreshes data. Use
`simfolio-oos scores --json` to inspect retained score evidence without
starting a forecast run. That command includes the expanded canonical records; use `simfolio-oos candidate-scores --json` for the current
combined ranking.

The bounded implementation surface can be exercised with an exact model ID or
the Frontier rank-one selector. A full canonical run requires the preserved
240 simulations per origin and is not implied by a smoke run.

Inspect the new candidate's task plan with
`simfolio-oos candidate --model experimental_filtered_innovation_moment_sv_fixed_mean --plan --json`.
The same exact selector runs it on the canonical dense plan when `--plan` is
omitted.

## Frontier production validation

The leading asset-level Frontier method matches the verified live numerical
source on three bounded cases. With identical random streams and production
storage precision, joint asset paths and rejoined portfolio paths match
exactly. The historical and production default seed schedules differ; the
[production parity report](docs/frontier-production-parity.md) records both
contracts, intermediate comparisons, and source hashes. This is bounded
numerical validation, not a rerun of the retained white-paper score.

## Reproducibility

Read [README_REPRODUCIBILITY.md](README_REPRODUCIBILITY.md) for the numerical
environment, source/data identities, parity checks, and known validation
limits. [docs/quickstart.md](docs/quickstart.md) shows a clean wheel install;
[docs/validation.md](docs/validation.md) describes the CI gates.

M206 integrates uncertainty in the rough observation level analytically (CRPS 0.24984428792635055). M207 uses the analytic tempered fractional covariance during debiased Whittle fitting (0.2495833333032549), resolving the original dynamic lift only for forecasting. Both completed the full 80-portfolio panel; M207 was the previous score leader but is slower than the controlled baseline.

M209 combines analytic rough covariance with differenced debiased Whittle MAP fitting. It scored **0.2493515254197939** on the complete 80-portfolio panel, versus baseline **0.2518812750312293** and previous leader M207 **0.2495833333032549**. An isolated fresh-cache run took **99.2803 seconds**, versus the controlled baseline **125.7270 seconds**. The return mean and forecast shell are unchanged. Full specifications and independent cell reconstruction are retained in the catalogue.

M210 relaxes the Hurst lower bound to 0.01 in M209, scoring **0.24863202500906678** on all 80 portfolios in **97.9834 seconds** with a fresh cache. It improves baseline CRPS by 1.29% and retains the mean model.

M211 permits the full theoretical Hurst domain in the differenced estimator. Its full-panel score is **0.2495417609939298** and fresh-cache time **99.4612 seconds**; M210 led before the M212 refinement.

M212 uses the empirical log-square innovation noise variance for the causal conventional-volatility offset, keeping its fitted parameters unchanged. It scored **0.2484924596795188** in **97.2791 seconds** on the complete panel, the score leader before M216. This two-stage plug-in quasi-likelihood refinement does not complete the research goal.

M213 and M214 retain M212 but learn Student-t and Hansen skew-t forecast innovation distributions, respectively. Full-panel CRPS is **0.25305232973579417** and **0.24975351800017143**, with isolated fresh-cache run times **98.9646** and **101.8385 seconds**. Neither beats M212; both audited full-panel results are retained.

M215 replaces only the forecast innovation quantile nodes with standard Gaussian nodes, with no fitted tail or skew parameters. It scored **0.25198127331647685** in **99.2198 seconds**; it did not beat M212.

M216 learns observation noise jointly with conventional SV parameters, then uses the same value in its dynamic rough layer. It scored **0.24648440999477378** in **101.5047 seconds**, becoming the recorded score leader at that stage.

M217 uses exact uniform-probability moments of the interpolated empirical quantile function on M216. It scored **0.2475449989529358** in **110.8024 seconds**, trailing M216.

M218 replaces the conventional log-square quasi likelihood with sparse raw-return Laplace SV inference, retaining multiscale and dynamic rough forecasting. It scored **0.2525845890080237** in **117.8753 seconds**, trailing M216 and the goal baseline.

M219 learns Student return tails while using sparse Laplace conventional SV inference, retaining multiscale and dynamic rough forecasting. It scored **0.24548488040352726** in **128.6077 seconds**, beating M216 but short of the goal score/runtime tradeoff.

M220 uses analytically implied Student log-square noise moments on M219, preserving the mean, multiscale construction, empirical forecast innovations and dynamic rough overlay. It scored **0.24423173688367905** in **126.5747 seconds**: 3.037% below M193 at 1.0067 times its runtime, meeting the full-panel breakthrough threshold.

M221 simulates the existing multiscale Gaussian states pathwise on M220, with no additional fitted parameters and second-moment-one normalization. Full-panel CRPS: **0.24400259165672855**. The original mean and M220 parameter fitting remain unchanged.

M221 completed in **147.7899 seconds**: **3.1279%** better CRPS than fixed baseline M193 at **1.1755 times** its runtime, and **0.0938%** better than M220. It wins on 55/80 portfolios against M193 and 48/80 against M220. The research goal remains active; production is unchanged by this publication.

M222 adds jointly fitted Student/Gaussian-copula leverage to M221. Full-panel CRPS: **0.2537399472056091**. The original mean and rough methodology remain; the first future volatility innovation conditions on the last observed return rank.

M223 widens only the rough Hurst support on M221 to the full theoretical domain (0, 1/2). Full-panel CRPS: **0.24454535810237815**. Original mean and other methodology are unchanged.

M223 completed in **133.7196 seconds**, about **9.52% faster** than M221 but **0.2224% worse** in CRPS. Its **2.9125%** gain over fixed baseline M193 falls short of the 3% score threshold; M221 remains the completed leader. The research goal remains active.

M224 estimates effective rough observation variance on M221. Full-panel CRPS: **0.24559602973028682**. Original mean and other methodology are unchanged.

M224 completed in **159.8414 seconds**: **2.4953%** better CRPS than fixed baseline M193, but **0.6530% worse** and **8.1544% slower** than M221. It does not meet the breakthrough criteria; M221 remains the completed leader and the research goal remains active.

M225 uses M221 with forecast innovation nodes matched to its learned Student tail shape, without another tail fit. Full-panel CRPS: **0.25121073766012286**. Original mean and all volatility fitting remain unchanged.

M225 completed in **142.1430 seconds**: **0.2662%** better CRPS than fixed baseline M193, but **2.9541% worse** than M221 for only a **3.8209% runtime saving**. It fails the breakthrough criteria. Both volatility fits and observation moments match M221 across every origin; the difference isolates forecast innovation nodes. M221 remains the completed leader and the research goal remains active.

M226 replaces fixed multiscale shrinkage in M221 with conditional Gaussian empirical Bayes REML and AR1 residuals. Full-panel CRPS: **0.7479695843647364**. Original return mean and both Student and rough fits remain unchanged.

M227 retains M221 with proper untruncated Gaussian priors on rough log mean-reversion and amplitude. Full-panel CRPS: **0.24398463100872367**. Original return mean and conventional Student fit remain unchanged.

M228 retains M227 with an existing multiscale one-step prediction as its rough fitting offset. Full-panel CRPS: **0.2452184293792364**. Original return mean and conventional Student fit remain unchanged.

M229 removes percentile clipping from the M227 rough observation proxy, retaining its finite near-zero floor and all other components. Full-panel CRPS: **0.24425777495214046**.

M230 normalizes the M227 interpolated innovation quantile by exact reference-uniform moments. Full-panel CRPS: **0.24838433757645187**.

M231 learns multiscale decay rates and selects their count per asset by forward conditional BIC. Full-panel CRPS: **0.24289420743117274**, cold research elapsed time **145.81819229200482 seconds**.

M232 learns multiscale decay rates while retaining four components to isolate rate-location learning. Full-panel CRPS: **0.41701921590549546**, cold research elapsed time **181.41140141600044 seconds**.

M233 learns multiscale decay rates with exactly one learned EWMA component as a matched control. Full-panel CRPS: **0.2427871001627023**, cold research elapsed time **154.94550479203463 seconds**.
