# V9 external score: raw CRPS with fixed series and horizon weights

Score each native day of a requested forecast with empirical CRPS on cumulative portfolio log returns. Average those daily losses per request. No volatility, square-root lead, or calendar-log scaling is applied.

Classify requests by their actual requested end date: short through three calendar years, medium over three and below twenty, long from twenty years onward. Build weights once from the entire frozen 320,017-request schedule, independently of losses. Within each bucket, balance the individual-asset and portfolio branches, classes, families and series, then average the planned requests for each series. Portfolios receive at most half the complete score. The headline loss is the mean of the three bucket losses; compare that single loss with the same fixed with-replacement baseline.

Calendar quarters no longer determine weights. Existing observations retain their exact planned coefficients when new forecasts finish. A partial report shows observed weighted support and its conditional loss; it never assigns zero to missing forecasts and never calls an incomplete result final. Partial scores can still change because newly observed outcomes differ.

The metric was selected retrospectively after comparing scaling choices on a fixed saved cohort. On 147,863 matched requests, the raw loss comparison was 14.400961% better than the baseline, versus 11.479587% with training-SD scaling and 5.913199% with training-SD and square-root-lead scaling. These are descriptive partial comparisons. Only about 10.6%, 10.6% and 12.7% of the planned bucket weight was represented, despite 46.2% of request count. This evidence does not establish the final full-panel percentage.

The underlying proper CRPS objective has fixed positive weights. Its relative improvement display is not claimed to be a proper scoring rule. Raw losses measure absolute error in common return units; scaled alternatives answer different questions rather than being academically invalid. The final objective has been fixed here; all earlier scores remain versioned evidence.

The algorithm, its past-only internal endpoint learner, joint asset paths, baseline, horizons and saved forecasts are unchanged. Completed 1,000-path results remain; new 240-path results are paired with 240-path baselines. No data has been excluded: the completed source audit found no objective error supporting removal.
