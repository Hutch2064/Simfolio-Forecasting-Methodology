# IID raw-component simplification

Keep the verified cohesive equal-pool model and every fitted parameter unchanged. Replace only the raw temporal-block stream with independent draws of complete historical constituent rows. Joint cross-asset shocks remain intact; conditional volatility remains in the filtered half, and both halves retain correlated training-batch mean uncertainty. Holdings and portfolio rebalancing evolve inside each scenario. There is no class/horizon routing, new hyperparameter, fitted block length, clipping or path exclusion.

The hypothesis is that the filtered component and correlated mean uncertainty may already capture useful conditional risk, making temporal raw blocks unnecessary. A pure filtered endpoint and ordinary nonwrapping replacement both lost their screens; this distinct simplification earns no presumption of improvement. Use a memory-bounded random row stream rather than constructing one Python block per simulated day.

The planned100-path355-case screen reuses identical-count fixed-reference, verified-hybrid and circular-hybrid losses. It adds355new forecasts and shares the12worker lease. Results are exploratory and cannot change the official leaderboard. Any serious candidate requires1000-path comparisons against the actual verified current best before broader/full admission.
