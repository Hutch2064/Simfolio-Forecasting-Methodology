# Forward forecast panel standard

Future asset-level forecast panels include the existing long-horizon schedule and native short horizons **H = 1, 5, 10, and 21**, wherever those leads have valid observations. This is a forward default. It does not alter, reinterpret, or rescore any historical frozen panel; a panel with a changed schedule is a separately versioned panel.

## Requests and horizon conditioning

The horizon-adaptive learner receives the actual requested horizon **H** for each forecast. A maximum horizon or another proxy must not stand in for the requested horizon. Each short-horizon request is identified and deduplicated by `(series ID, training prefix, H)`. Existing long-schedule rows and their aliases remain as provenance; aliases do not create extra score votes.

## Score and aggregation

Keep the v7 per-request score unchanged. For a request ending at horizon **H**, average the scores over its valid native leads `h = 1..H`, normalizing each lead's CRPS by the past-only training portfolio standard deviation (`ddof=1`) and `sqrt(h)`:

`request score = mean_h [ CRPS_h / (training-only portfolio SD × sqrt(h)) ]`.

Do not replace this with raw CRPS or add horizon-band reweighting. Give requests equal weight within each series/quarter group, then preserve the v7 equal-weight aggregation through family and class. In the higher-level matched comparison, keep the portfolio branch at no more than 50% weight.

## Schedule reference

The frozen expanded schedule records **64,049 existing requests + 255,968 short-horizon requests = 320,017 combined requests per model**. The short-horizon rows are four requested horizons across 63,992 unique training prefixes; the 57 duplicated original descriptors remain aliases and are not extra short-horizon votes. These counts describe that frozen expansion and do not change prior panel definitions. Its evidence and provenance are in the [expanded short-horizon takeover record](daily-assets-portfolios-asset-v6-20261020/expanded-short-horizon-takeover-v7/README.md).
