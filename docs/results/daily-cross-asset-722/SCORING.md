# Expanded daily cross-asset panel score (v3)

This is a new, frozen panel aggregation protocol. It does not replace or
rewrite the original French-panel cell-first score. The v3 convention is a
benchmark-specific choice made before reading expanded-panel model rankings;
it is not a claim that there is one academically mandatory way to aggregate
across horizons, series, or asset classes.

## Task loss and training-only scale

For series `s`, origin `o`, observation horizon `h`, let `CRPS[s,o,h]` be the
existing exact empirical CRPS on the cumulative terminal log return from 240
paths. The saved task vector remains unchanged. Divide each horizon loss by

```text
a[s,o,h] = sd(training log returns, ddof=1)
            * sqrt(h + h^2 / n_training)
```

Here `h` counts native observed returns, `n_training` is the number of returns
available at that origin, and the standard deviation uses that origin's
training prefix only. The square-root term is the IID reference scale for a
sum of `h` returns when the mean is estimated from `n_training` observations.
It is a common, training-only normalization, not a fitted description of each
model's predictive variance. A nonfinite or zero training standard deviation
is an error; the reporter does not estimate a replacement from held-out data.

## Actual-calendar lead integration

The task stores one loss for every native observation horizon. Scoring maps
those horizons to the series' actual target dates, not to the synthetic
business-day labels that some model adapters receive for accounting. Let
`d_h` be elapsed calendar days from the origin to the `h`-th actual target date,
with `d_0 = 0`. The positive interval mass is

```text
v_h = log((d_h + 1) / (d_(h-1) + 1))
w_h = v_h / sum_j(v_j)
origin_score[s,o] = sum_h w_h * CRPS[s,o,h] / a[s,o,h]
```

This integrates a logarithmic measure over each observed calendar interval.
Weekends and holidays contribute elapsed time between observations; no return
or target is imputed. Every eligible observed horizon has positive mass. The
weights depend only on the frozen date schedule, not on realized returns,
forecast values, or model scores. This gives roughly comparable mass to
multiplicative lead-time bands while retaining the observed market calendar.

Some native adapters require synthetic business-day future labels for their
existing accounting hooks. The scheduler leaves those hooks unchanged. The
scorer always uses actual per-series target dates for elapsed lead time.

## Panel hierarchy

1. Within each actual calendar quarter and series, average the available
   origin scores. Rolling and temporal origins in that quarter are combined at
   this step, so their labels do not create separate weighting units.
2. Within a quarter, average available series equally within each family.
3. Average available families equally within each asset class.
4. Average the asset classes with eligible scored data in that quarter
   equally. A class enters only when its frozen schedule provides eligible
   observations; there are no fixed percentage class weights.
5. Average the available quarter scores equally.

This hierarchy prevents a class from gaining weight just because it contains
more series or families. Older series can contribute in more historical
quarters and can support longer native lead paths; within an overlapping
series-quarter-family cell, each eligible series remains equally weighted.
The denominator at every level is determined by panel metadata, schedule, and
checkpoint coverage, never by score magnitude.

The horizon-normalized origin scalar is aggregated before the panel hierarchy.
That ordering is part of v3. It differs from the original French whitepaper
score, which first averages origin losses within each series-horizon cell and
then gives the resulting cells equal weight.

## Coverage and comparisons

`score_on_own_coverage` is descriptive and must be read with the reported
coverage fraction. A partial score is not a complete leaderboard result.
Pairwise public-baseline results use only the exact intersection of task
checkpoints available for both models and are labelled `preliminary` unless
both models cover the full frozen task set. `pct_better` is the relative score
reduction versus the public score, `100 * (public_score - model_score) /
public_score`; it is not the fraction of origins won or a significance test.
The common leaderboard is emitted only when every included model has the
identical complete frozen task set. No model-specific missing-task
renormalization can silently make a partial result look complete.

The reporter also computes a predeclared sensitivity profile from the same
saved CRPS vectors and task coverage. It replaces the primary scale above with
`sd(training log returns, ddof=1) * sqrt(h)`, retaining the same actual-calendar
weights and panel hierarchy. Its separate protocol ID is
`daily_cross_asset_panel_crps_v3_logcalendar_hierarchical_sqrt_h`. It omits the
IID reference term for uncertainty in the estimated mean; this is a scale
sensitivity, not another forecast or a claim that either scale is uniquely
correct. Both profiles are computed in one checkpoint pass and reported
separately.

Only IDs currently marked enabled in the validated sweep control file are
eligible for a report. Each execution manifest must match that control entry's
factory and implementation digest; disabled, removed, or mismatched histories
cannot enter the leaderboard. The active control-file hash and included IDs
are recorded with the report.

The SQLite cache stores both normalized scalars per task and binds them to the
panel fingerprint, model execution fingerprint, scorer source hash, checkpoint
path, and a SHA-256 of the complete checkpoint bytes. The full checkpoint hash
is checked on every cache hit, so in-place mutation is detected even when file
size and timestamps are preserved. A checkpoint whose bytes drift under the
same execution identity is excluded and reported as failed; it is never
silently rescored into the old result. A checkpoint that disappears is marked
missing for that scan. This reporter reads saved losses only and does not
launch forecast workers.

## Interpretation limits

The CRPS remains a marginal proper score at each terminal horizon. Combining
the marginal scores does not test the joint dependence or pathwise calibration
of the 240 simulated trajectories. Origins and horizons overlap heavily, so
they are not independent observations for standard errors or significance
claims. Treat pairwise score differences as descriptive unless a separate,
predeclared dependence-aware inference procedure is supplied. The scale and
log-calendar weighting are explicit design decisions to improve cross-series
and lead coverage; neither is a universal ranking objective.

This use of positive ex-ante weights is consistent with proper-score
aggregation, while scale-invariant CRPS is a distinct scoring rule and cannot
be recovered from a saved scalar CRPS alone without the additional
distributional terms or forecast replay. See [Gneiting and Raftery (2007)]
(https://doi.org/10.1198/016214506000001437), [Gneiting and Ranjan (2011)]
(https://doi.org/10.1198/jbes.2010.08110), and [Bolin and Wallin (2023)]
(https://doi.org/10.1214/22-STS864).
