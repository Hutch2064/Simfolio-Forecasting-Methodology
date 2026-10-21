# Exact shared score-coefficient sweep

This is a research backend prototype. It does not add a forecasting method,
change the166-option universe, criterion, validation folds, members, outer score,
full-panel catalogue count or current official winner. The running355-case
path study continues with its original frozen numerical source.

The existing reducer independently merges sorted samples for every option pair
at every known historical lead. The prototype sorts all labelled members once
per lead, shares per-option cumulative counts and sums, and retains the original
summation order for each pair. Equal-value events are all incorporated before
scoring each event, preserving the original <= convention. Memory remains
bounded by current-lead members plus the dense pair matrix. No leads, members,
options or origins are dropped and no numerical precision is relaxed.

Six tests pass: random166option/1000member coefficient bytes, ties/point masses/
scale extremes, chunk continuation, certified weight equality, all-fold dated
joint-history coefficient/weight/forecast-path bytes at100and1000members, and
request-binding rejection. Three interleaved warm-code synthetic pairs show
median paired coefficient CPU ratios of6.557940at100members and5.827371at1000.
All compared coefficients are byte-identical. Twelve active scientific workers
confound wall timing. These are coefficient-only synthetic timings; no whole
request, production or browser speedup is claimed. The earlier13-request local
profile attributes about79%of selected100path preparation CPU to this reducer,
which motivates testing the improvement on real complete workloads.

If the current355-case development result passes independent audit and improves
its matched endpoint control, the next planned shared12-worker stage will replay
all355development requests and run both criteria on355separate reserved research
requests at1000inner/final members. The replay must reproduce all710complete
outer loss vectors, all recorded component traces, learned weights and1132stored
coefficient arrays from283new development requests. The other72requests have
no saved coefficient matrices; their traces, weights and outer losses still
must match. Reserved endpoint controls must also exactly match preserved losses.
Any parity failure invalidates acceptance of that stage. Reserved slices are
adaptively reused research evidence, not independent validation. No partial
result promotes a full-panel winner or meets the goal stopping condition.

The separate weight-stability research note documents further credible ideas;
it does not admit a model or prescribe arbitrary mixture percentages.
