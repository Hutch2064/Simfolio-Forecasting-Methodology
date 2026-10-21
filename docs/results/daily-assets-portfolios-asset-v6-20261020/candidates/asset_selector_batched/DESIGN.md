# Exact shared inner simulation work: unqualified prototype

Observed phase timing of302completed requests in the live precision development
attributes94.07%of historical selection time to component simulation,2.42%to
fitting,2.94%to CRPS coefficients and0.57%to other selection work. This is local
incomplete-sample timing, not a speedup or production claim. No numerical scorer
or scientific option is removed.

For each volatility-family/innovation-law group, six mean/uncertainty choices
share normalized innovations, fitted volatility parameters, variance recurrences
and random streams. Their mean offsets, AR previous returns, asset paths and
portfolio accounts remain separate. In the qualified original kernel volatility
updates depend on normalized innovations, not the mean offset. Share only that
mean-independent computation. Keep the original AR return expression per option
rather than replacing it with an algebraically equivalent expression that could
change floating-point output. Reuse original random namespaces, draw shapes and
chunk order. All options still receive1,000inner paths and all original folds.

This prototype is not admitted and no tests or speed gain are claimed yet.
Required: every family/law/mean/uncertainty component and joint asset state must
match the unbatched reference bitwise, including chunk invariance. Before
integration, compare complete historical ensemble hashes and selected forecast
outputs across the same development workload, with source-bound recovery and
paired local phase timings. Final simulation/accounting remains unchanged.
