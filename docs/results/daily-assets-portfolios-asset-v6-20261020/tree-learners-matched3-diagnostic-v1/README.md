# Isolated tree learner research v1

Five actual implementations: LightGBM4.7.0 and XGBoost3.4.1 as first-order boosted mixture-CRPS gates; sklearn1.9.0 pruned CART, RandomForest and ExtraTrees as predicted single-recipe CRPS controls. This is our adaptation, not a literal FFORMA replication. No production registration or official promotion.

All methods use the same matured cached A/B and outcome-free current predictive B, all167 eligible recipes and the shared geometry/MVD feature extraction. No historical simulations, inner paths or current probes are repeated. One simplex covers the entire joint trajectory. Existing screen adapter generates final100 coherent paths, scored by current v7. Old numerical sources/outputs remain untouched.

## Exact choices

The full coupled objective is J=sum m(A'v−v'Bv/2)/(S sum m), v=softmax(logits); S is positive training-only uniform loss. Its exact logit gradient is v*(A−Bv−v'(A−Bv)), with group masses/scaling once. Boosted trees fit −g/||g||infinity as a direction proposal using all expert rows/features. Joint predictions are centered within episode and normalized by ONE training-joint infinity norm. Armijo backtracking checks the original float64 coupled objective; effective tree coefficient=step/rawdirectionnorm is persisted and replayed. No diagonal Newton-Hessian or global optimum claim. Libraries may quantize feature/target representation internally; exact-objective verification, not exact split arithmetic, is authoritative.

Each weak tree has no preset depth cap: maximal leaves and histogram-bin capacity scale with training rows, minleaf1, minsplitgain0, no row/feature subsampling. This is a full finite positive-gain proposal hypothesis, not an assertion of globally optimal CART splits. LGBM min_data_in_bin1 avoids its default3-row bin merge. XGBoost hist/lossguide supports unrestricted depth with sample-supported leaf capacity.

Controls regress vertex CRPS A_i−B_ii/2 normalized by S, with per-row weights m/(K sum m). Their target omits off-diagonal complementarity, so they are explicitly predicted-loss/model-selection controls. Exact predicted ties share recipe mass uniformly. CART uses the full pruning-alpha path from earliest matured training, normalized through the dimensionless response, including root/uniform null and alpha0. Later assessments refit each alpha on their own past; alpha labels are regularization strengths, not matching split structures. Forests use the same alpha candidates; actual fixedseed0 sklearn N-tree fits have identical tree prefixes as N grows. RandomForest bootstrap is proposal diversity only, not independent validation or OOB inference. ExtraTrees uses all rows with randomized splits.

Predictive complexity is a root-approved greedy hypothesis: boosting stage budgets0(null),1,2,4,... and forest tree counts1,2,4,... are assessed on the SAME full matured chronological assessment set. Stop at the first nonimproving pooled full-mixture CRPS; exactties retain earlier/fewer stages. This does not rule out later improvement or claim an unrestricted-ensemble optimum. Every evaluated count/risk/support/seed is retained. Refit the selected count on all mature training, with the same declared structure/seed rule. No outer outcome selects counts, pruning, features or stopping.

An analytic scalar batch blend tau is selected separately for each assessed alpha/count using pooled quadratic CRPS. Exactties prefer stronger pruning/null and fewer stages. Strict tau0 returns separately certified final batch without unused final model refit. F2 has exactly ONE internal tuning episode, even though167 recipe alternatives exist; recipe rows are not167 independent temporal validations.

## Numerical failure and finite paths

Boost target count is always explicit. At each accepted stage require finite coupled risk/gradient and simplex; failure to find resolved descent is explicit unsupported selected stage, never accepted failed weights. A directly checked original logit-gradient certificate (sqrt(float64eps)*numberlogits) is an absorbing stationary endpoint; replaying a larger budget retains that endpoint and records fewer actual stages. This is stationarity only. Backtracking halves until exact Armijo plus entry-resolution decrease succeeds or the step/decrease is unrepresentable; there is no unlimited retry or relaxed tolerance. Positive fixed entry-scale objective envelopes forbid infinitely many resolved descent stages, although no practical runtime bound is promised. Greedy validation early stopping usually selects much sooner, but this requires measurement.

Request-local immutable preparation (basis/scales/x/vertex targets/masses) is reused across alpha/count for the same causal training set; it expires at request end. No persistent fitted-model cache is introduced. Source/input-bound durable runner checkpoints retain each family success/failure independently; no partial leaderboard.

## Limitations and prelaunch evidence

Axis-aligned trees depend on the frozen full-rank geometry/MVD coordinates; repeated PCA eigenspaces may rotate under refit/permutation and change tree partitions. No spherical-basis invariance claim. No arbitrary positive-rank feature truncation, external factors or ticker/quarter labels. More trees, seed sensitivity and alternative greedy stopping rules remain separate research hypotheses.

Synthetic tests verify coupled gradients, off-diagonal dependence versus unchanged vertex targets, exact descent/rejection and scaling, training/predictor weight agreement, mature-only folds/current-outcome independence, seeded true forest prefixes/noOOB, all5 libraries on geometry/MVD, cache prediction identity, finite guards, explicit unsupported stages, exact tie/null rules and simplex output. Financial cached requests require root admission and frozen complete source/input/dependency manifests. The predeclared requests0/35/53 were completed as a matched 100-path development diagnostic after a source-bound three-cell roundoff repair; see RESULTS.md. They do not constitute a full sweep.

Sources: [FFORMA](https://robjhyndman.com/papers/fforma.pdf), [Friedman functional gradient boosting](https://doi.org/10.1214/aos/1013203451), [XGBoost custom-objective constraints](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html), [LightGBM train](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html), [sklearn pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).


## Public distribution and reproducibility

The numerical model, objective, and portable model tests above are the exact frozen source files used for the exploratory qualification. The source-frozen execution harness and its orchestration tests are preserved in the private recovery archive referenced separately. They are omitted from this public package because the harness contains machine-specific path bindings. The complete private source and dependency snapshot is retained for exact recovery.


## Completed diagnostic and source lineage

The repaired matched-three results, independent arithmetic verification, and repair lineage are recorded in [RESULTS.md](RESULTS.md), [RESULTS.json](RESULTS.json), and [REPAIR_LINEAGE.json](REPAIR_LINEAGE.json). The portable uniform-shrinkage adapter and synthetic-only qualification are included under uniform_shrinkage_full/. The full-panel runner is referenced by exact SHA in [RUNNER_AND_UNIFORM_ADAPTER.json](RUNNER_AND_UNIFORM_ADAPTER.json); its frozen file contains local absolute helper paths, so it is retained unchanged in the private archive instead of being edited for publication. The private archive is partial until its completion receipt is produced.
