# Closed development contrast: Gaussian hidden volatility regimes

The current shrunk-mean winner remains retained. This candidate replaced its120-path empirical-GARCH/FHS half with a two-state Gaussian HMM. The exact120-path raw random-block half, smooth mean regularizer, batch Gaussian mean uncertainty, total240 paths and equal whole-path pool remain unchanged. This changes both the conditional volatility structure and the filtered component emission law; it does not claim unchanged empirical FHS innovations.

The model was qualified on all245 exact native training prefixes across49families/6classes. Six tests passed: exhaustive state-path likelihood/smoothing/counts, MAP EM monotonicity/analytic gradients, equivariance and independent posterior optimization, explicit failure rejection, canonical native accounting/Normal boundary, and unchanged pool-wrapper identity. One of490 fixed optimizer starts returned an invalid status; the other qualified on that prefix. Every prefix had a converged eligible MAP fit. Priors and initializations were fixed before field scores, with no state-count, prior-width or seed search. See [DESIGN.md](DESIGN.md).

| Same245 development origins | Primary better than worst Public comparator | Sensitivity better than worst Public comparator |
|---|---:|---:|
| Public Adaptive Blocks |0%|0%|
| Current full-panel winner |2.462925955589%|2.903057657891%|
| Gaussian HMM candidate |1.956533162105%|2.437660907908%|

Public Adaptive Blocks is the lowest-scoring of the three retained Public comparators on the complete fixed panel under both profiles. Positive values mean lower canonical Q/CRPS. This table uses exact matched successful coverage, not mixed-coverage ranks or an independent holdout. The candidate regresses against the unchanged current winner on both profiles. Fixed income improves, while real estate regresses substantially; no class-specific router is fitted. The frozen development gate rejects this implementation, with no reserved/broad/full escalation or outer-score retuning. This does not rule out every hidden-state model.

All735 forecasts succeeded with zero native fitting failures and zero scoring integrity errors. Audit checked735 native task identities and loss hashes;490 Public/current loss hashes exactly reproduced their historical references. Original calendar-weighted horizon contributions reconcile both canonical profiles ([diagnostics](diagnostics.json)). These are adaptively reused development cases, not independent post-selection validation or significance evidence.

The complete758-file private archive was freshly downloaded, every file disk-restored and SHA256-verified before735 redundant raw loss files were retired. [archive-receipt.json](archive-receipt.json) records exact recovery and digests; fixed inputs/native parent sources require their existing linked archives. Public source snapshots and compact evidence above do not replace that complete recovery path. This was a pilot, so the permanent full-panel catalogue stays at13 unique fully attempted models. The wider accuracy-maximization goal remains unfinished.
