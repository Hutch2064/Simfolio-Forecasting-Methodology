# Production purpose and portfolio-level research scope

Human clarification,2026-10-20: the purpose of broad tests is a distributionally
accurate production forecast for arbitrary user-specified multi-asset portfolios,
including up to200tickers. Keep the forecasting model at portfolio level: construct
the historical portfolio-return series, then fit one model to that series. Avoid
requiring one forecasting-model fit per ticker. Portfolio data construction and
asset-path reconstruction remain distinct from that one forecast fit.

The user is seeking accurate forecast distributions, not an exhaustive research
programme over every rebalancing configuration. Asset-level joint modeling and
multivariate asset scores are not mandatory additions to this goal. A portfolio
path does not identify constituent asset paths; simplified reconstruction uses an
additional allocation assumption, rather than supplying independent evidence of
asset-level correlation accuracy.

The current722-series inputs and canonical scoring remain frozen. They are a broad
series-level component benchmark, with overlapping histories/families and heavily
overlapping horizons. The scores do not themselves prove transfer to arbitrary
user portfolios, nor independent post-selection performance after repeated model
selection on this panel. These are material interpretation limits, not grounds
for changing scoring after seeing rankings.

Recommended additional transfer validation, NOT yet admitted or claimed complete:
predefined and reproducibly sampled asset combinations, still modeled as one
portfolio series each; concentrated/diversified and unusual exposures; fixed
portfolio construction consistent with the product; portfolio terminal-return
CRPS plus calibration/downside diagnostics; genuinely untouched compositions and
periods where available, and predeclared seed-repeat checks for Monte Carlo
stability. Previously inspected periods are development evidence. The fixed722
benchmark remains the current goal's score; this note does not silently expand
or replace its data, scorer, gates or completed results.

For future reporting the user selected the worst of the three retained Public
comparators as the score reference. On this full frozen panel that is Public
Adaptive Blocks under both profiles(Qprimary.5993442043272177;
Qsensitivity.6140827734648610). Thus existing percentages versus that reference
remain numerically unchanged. Keep exact matched coverage and disclose failures.
The website Default is a different model and is not substituted into this reference.

The current full-panel winner remains Equal Mix—Shrunk Mean/Gaussian Uncertainty,
1.398780530%primary/1.436539825%sensitivity better than that reference,722series,
36813forecasts,0failures. See the versioned smooth-mean-win report. Recorded full
successful-task compute was951.209worker-seconds versus1329.277forPublicAdaptive;
these separate runs are not a controlled speed benchmark. The paired18-case warm
check showed broadly comparable local runtime(median paired ratio.979894 vsPublic).
No deployed production latency, CTV or full-work speedup is claimed.

## Subsequent explicit authorization

The user then authorized the portfolio expansion,up-to50percent portfolio branch,one main score,and rerunning all five retained methods before continued research. The initial proposal above is historical. The implemented version is documented in docs/results/daily-assets-portfolios-v4-20261020. V4uses the former sqrt(h) calculation as its sole objective; diagnostics have zero ranking influence. Existing722source series remain bitwise unchanged.
